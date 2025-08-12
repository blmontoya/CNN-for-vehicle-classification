from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder

# Use GPU instead of CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helps speeds up epochs by enabling cuDNN auto-tuner for best algorithm 
# for hardware and input size configuration while using GPU
torch.backends.cudnn.benchmark = True

# Augmentations to the training data:, includes resize, random crop, horizontal flip,
# and color jitter
train_transform = transforms.Compose([
    transforms.Resize(140),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Testing data without augmentations
test_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Paths to training and testing data
# print statements verify training and testing folders align (e.g., motorcycle=0 for both)
train_dataset = ImageFolder(root='./customdata/train', transform=train_transform)
print(train_dataset.class_to_idx)
test_dataset = ImageFolder(root='./customdata/test', transform=test_transform)
print(test_dataset.class_to_idx)

# DataLoader used to load datasets, shuffles training data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

class Inception(nn.Module):

    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

        # 16 + 24 + 24 + 24 = 88 channel output

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #self.dropout = nn.Dropout(0.3)

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
        #out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

'''
ResBlockSeq creates a sequence of n ResBlocks and appends them together
In this program, n is defined to be 2
'''
class ResidualBlockSequence(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ResidualBlockSequence, self).__init__()
        blocks = []
        # Appends first ResBlock to the sequence
        # Note that the first block may downsample and change channels
        blocks.append(ResidualBlock(in_channels, out_channels, downsample=(in_channels != out_channels)))
        # Note that subsequent blocks keep out_channels same
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels))
        # Adds blocks to the sequence
        self.sequence = nn.Sequential(*blocks)

    # Returns the block sequence to the forward method
    def forward(self, x):
        return self.sequence(x)

'''
Defines label smoothing loss for the loss function. Label smooth loss seemed to be more
apropriate as a loss function over cross entropy loss considering the size of the training data.
Instead of cross entropy loss, LSL helps the model not to be overconfident and is good for smaller 
training data (such as CIFAR-10).
'''
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

'''
Defines the structure of the neural network, such as the convolution layers, 
batch normalizations, ResBlock sequences, max pooling, and a 
'''
class Net(nn.Module):
    # __init__ defines the what the components of the CNN will be
    def __init__(self):
        super(Net, self).__init__()

        # Defines ReLU algorithm
        self.relu = nn.ReLU(inplace=True)

        # Defines first convolutional layer with an input of 3 -> 64. Kernel is 3x3 
        # as inspired by ResNet-18 with a padding of 1 for all convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Inputs 64 -> 128, same kernel size and padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Inputs 128 -> 256, same kernel size and padding
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Adding batch normalization that matches the output of each convolutional layer
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # Adding ResBlock sequences of 2 whose dimensions match the output of each convolutional layer
        self.res1 = ResidualBlockSequence(in_channels=32, out_channels=32, num_blocks=2)
        self.res2 = ResidualBlockSequence(in_channels=64, out_channels=64, num_blocks=2)
        self.res3 = ResidualBlockSequence(in_channels=128, out_channels=128, num_blocks=2)
        self.res4 = ResidualBlockSequence(in_channels=256, out_channels=256, num_blocks=2)

        # Adding inception modules
        self.incept1 = Inception(in_channels=32)
        self.incept2 = Inception(in_channels=64)
        self.incept3 = Inception(in_channels=128)
        self.incept4 = Inception(in_channels=256)


        self.reduce1 = nn.Conv2d(88, 32, kernel_size=1)
        self.reduce2 = nn.Conv2d(88, 64, kernel_size=1)
        self.upscale1 = nn.Conv2d(88, 128, kernel_size=1)
        self.upscale2= nn.Conv2d(88, 256, kernel_size=1)




        # Pooling is 2x2, aka why feature map dimensions are halved (e.g., 10x10 -> 5x5)
        self.mp = nn.MaxPool2d(2)
        # 88 * 5 * 5 = 2200 since 88 outputs of 5x5
        conv_out_size = self._get_conv_output((3, 128, 128))

        # Defines rate of dropout
        self.dropout = nn.Dropout(0.5)

        # Defines a nonlinear final layer after the convolutional layers to obtain the logits
        self.fc = self.fc = nn.Sequential(
            # Maps to vector of length 512
            nn.Linear(conv_out_size, 512),
            # Applies batch norm, ReLU, and dropout
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # Maps down from length 512 vector to logits
            nn.Linear(512, 3)
        )

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        self.eval()  # set model to eval mode so batchnorm uses running stats
        with torch.no_grad():
            output_feat = self._forward_features(input)
        self.train()  # set back to train mode
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size
    
    # Forward_features organizes the core structure of the CNN into a method into a 
    # calleable method. Each component goes in the order of convolutional layer -> 
    # batch norm -> ReLU -> Max Pooling -> ResBlock sequence
    def _forward_features(self, x):
        # First component
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.res1(x)
        x = self.incept1(x) 
        x = self.reduce1(x)

        # Second component
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.res2(x)
        x = self.incept2(x) 
        x = self.reduce2(x)

        # Third component
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.res3(x)
        x = self.incept3(x) 
        x = self.upscale1(x)

        # Fourth component
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.res4(x)
        x = self.incept4(x)
        x = self.upscale2(x)

        # Returns the organized structure
        return x

    # Forward defines the CNN itself that will be used in training and testing
    def forward(self, x):
        in_size = x.size(0)

        # Forward_features runs the core framework for the model
        x = self._forward_features(x)

        # Reshapes tensor from an infered dimension (noted by -1) to a 2D tensor
        x = x.view(in_size, -1)  

        # Adding dropout rate of 0.5 as previously defined
        x = self.dropout(x)

        # Flattens the tensor using a final nonlinear layer (-1 infers the correct dimension size)
        x = self.fc(x)

        # Returns the final output
        return x

# Defines the model to the local device
model = Net().to(device)

# Optimizer uses stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Defines warmup epochs and warmup scheduler to stabilize learning rate in earlier epohs
warmup_epochs = 5
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

# Defines cosine scheduler to smooth decay and prevent overshooting
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80 - warmup_epochs)

# Defines the scheduler using the warmup and cosine schedulers
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

# Defines the loss function as Label Smoothing Loss
loss_fn = LabelSmoothingLoss(smoothing=0.1)

# Defines the writer to record data such as learning rate, loss, and accuracy to TensorBoard
writer = SummaryWriter()

'''
Defines the training process for the CNN
'''
def train(epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

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

'''
Defines the testing process for the CNN
'''
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
    for epoch in range(1, 86):
        train(epoch)
        test()
        scheduler.step()
    writer.close()