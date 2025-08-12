'''
CNN may take some time to load before getting started (~10-20s on RTX 3060)
'''
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

'''
This Residual Block class is inspired by the various projects such as 
ResNet-18, which helps to avoid vanishing gradients and train the CNN deeper
'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        # If downsampling, spatial resolution is cut in half
        stride = 2 if downsample else 1

        # Main residual branch
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Another 3x3 conv which keeps the channels the same, then batch norm
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
        # Prepares the shortcut version if needed
        identity = self.shortcut(x)

        # Pass through the first conv -> batch norm -> RelU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.dropout(out)

        # Pass through second conv -> batch norm
        out = self.conv2(out)
        out = self.bn2(out)

        # Adds shortcut branch to the main branch element-wise
        out += identity

        # Final activation and return
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
        # Sets correct label probability to 0.9
        confidence = 1.0 - self.smoothing

        # raw outputs are converted to log probabilities
        log_probs = F.log_softmax(pred, dim=-1)

        # Tensor with the same shape as log_probs filled with zeros
        true_dist = torch.zeros_like(log_probs)

        # Obtains cross-entropy between smoothed target distribution and the model's
        # predicted distribution
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

    # Meant to determine the convolutional output given a shape
    def _get_conv_output(self, shape):
        # Measures the output by creating a "fake" input tensor
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        # Model is set into evaluation mode
        self.eval()
        # Disables gradient and rusn the "fake" input throught the forward features
        # of the model
        with torch.no_grad():
            output_feat = self._forward_features(input)
        # Sets the model back into training mode
        self.train() 
        # Flattens output and gets the number of features
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

        # Second component
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.res2(x)

        # Third component
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
# Defines the model to the local device
model = Net().to(device)

# Optimizer uses stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Defines warmup epochs and warmup scheduler to stabilize learning rate in earlier epohs
warmup_epochs = 5
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

# Defines cosine scheduler to smooth decay and prevent overshooting
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20 - warmup_epochs)

# Defines the scheduler using the warmup and cosine schedulers
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

# Defines the loss function as Label Smoothing Loss
loss_fn = LabelSmoothingLoss(smoothing=0.1)

# Defines the writer to record data such as learning rate, loss, and accuracy to TensorBoard
writer = SummaryWriter()

# Training and testing methods were inspired by the ZeroToAll PyTorch lectures
# from HKUST, which can be found in the Google Doc in the README

'''
Defines the training process for the CNN
'''
def train(epoch):
    # Sets into training mode and sets a count to average loss later
    model.train()
    total_loss = 0
    
    # Loops through the batches from the train loader
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward and backward passes
        optimizer.zero_grad()
        output = model(data)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Accrues loss
        total_loss += loss.item()
        
        # Prints progress after every 100 batches 
        # (this was mainly used in the model trained on CIFAR-10, the dataset here
        # does not have enough data for 100 batches)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    # Computes and tracks avg loss with TensorBoard
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)

    # Retrieves and tracks learning rate using TensorBoard
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('LearningRate', current_lr, epoch)

'''
Defines the testing process for the CNN
'''
def test():
    # Sets into evaluation mode and sets a count for test loss and
    # correct count later
    model.eval()
    test_loss = 0
    correct = 0

    # Loops over the batches from the test loader
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sum up batch loss
            test_loss += loss_fn(output, target).item() * data.size(0)
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # Counts how many predictions match the true labels
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Averages test loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # Prints progress after every 100 batches 
    # (this was mainly used in the model trained on CIFAR-10, the dataset here
    # does not have enough data for 100 batches)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # Tracks test loss and accuracy using TensorBoard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

# Main training loop, which trains for 85 epochs
if __name__ == '__main__':
    for epoch in range(1, 31):
        train(epoch)
        test()
        scheduler.step()
    writer.close()