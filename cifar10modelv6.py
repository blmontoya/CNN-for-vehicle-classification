from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from torch.utils.tensorboard import SummaryWriter

# using gpu over cpu since its faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helps speeds up epochs by enabling enables cuDNN auto-tuner to find the best algorithm 
# for  hardware and input size configuration when using GPU
torch.backends.cudnn.benchmark = True

# Training data gets augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    #transforms.RandomErasing(p=0.25),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # mean for R, G, B
                         (0.2023, 0.1994, 0.2010))  # std for R, G, B
])

# Testing data is only normalized (without augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 training dataset
train_dataset = datasets.CIFAR10(root='./data/',
                                 train=True,
                                 download=True,
                                 transform=train_transform)

# Load CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(root='./data/',
                                train=False,
                                download=True,
                                transform=test_transform)

# DataLoaders to load batches
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=64,
                                           num_workers=4, 
                                           pin_memory=True,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=2)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match shape if needed
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class ResidualBlockSequence(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ResidualBlockSequence, self).__init__()
        blocks = []
        # first block may downsample and change channels
        blocks.append(ResidualBlock(in_channels, out_channels, downsample=(in_channels != out_channels)))
        # subsequent blocks keep out_channels same
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels))
        self.sequence = nn.Sequential(*blocks)

    def forward(self, x):
        return self.sequence(x)
    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (log_probs.size(1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # First layer takes 3 32x32 inputs and applies 10 different 5x5 filters
        # each feature map is 28x28 since 32-5+1=28
        # max pooling 28x28 -> 14x14 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # Same process as above but produces 20 feature maps each map size is 10
        # 14-5+1=10
        # max pool again so 10x10 -> 5x5
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Adding batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        # Adding residual blocks 
        self.res1 = ResidualBlockSequence(in_channels=64, out_channels=64, num_blocks=2)
        self.res2 = ResidualBlockSequence(in_channels=128, out_channels=128, num_blocks=2)
        self.res3 = ResidualBlockSequence(in_channels=256, out_channels=256, num_blocks=2)

        # pooling is 2x2, aka why feature map dimensions are halved (e.g., 10x10 -> 5x5)
        self.mp = nn.MaxPool2d(2)
        # 88 * 5 * 5 = 2200 since 88 outputs of 5x5
        conv_out_size = self._get_conv_output((3, 32, 32))
        self.fc = self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        # added dropout
        self.dropout = nn.Dropout(0.5)

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        self.eval()  # set model to eval mode so batchnorm uses running stats
        with torch.no_grad():
            output_feat = self._forward_features(input)
        self.train()  # set back to train mode
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size
    
    def _forward_features(self, x):
        # applies the first convolutional layer then max pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.res1(x)


        # applies the second convolutional layer then max pooling
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.res2(x)


        # applies the third convolutional layer then max pooling
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.res3(x)


        return x

    def forward(self, x):
        # batch size, meant to be sued later
        in_size = x.size(0)

        # runs the framework of the model
        x = self._forward_features(x)

        # reshapes tensor from an infered dimension (-1) to a 2D tensor
        x = x.view(in_size, -1)  

        # Adding dropout
        x = self.dropout(x)

        # Flatten the tensor (-1 infers the correct dimension size)
        x = self.fc(x)
        return x

model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

warmup_epochs = 5
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20 - warmup_epochs)

scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
loss_fn = LabelSmoothingLoss(smoothing=0.1)

writer = SummaryWriter()


def train(epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # instead of cross entropy loss, helps model not to be overconfident
        # good for smaller training data like cifar-10
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)

    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('LearningRate', current_lr, epoch)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += loss_fn(output, target).item() * data.size(0)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

if __name__ == '__main__':
    for epoch in range(1, 31):
        train(epoch)
        test()
        scheduler.step()
    writer.close()