import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define MnasNet architecture
class SepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SepConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.relu(x)
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expansion_factor != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1))
            layers.append(nn.ReLU(inplace=True))
        
        layers.extend([
            SepConvBlock(hidden_dim, hidden_dim, stride=stride),
            nn.Conv2d(hidden_dim, out_channels, 1)
        ])

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv_block(x)
        else:
            return self.conv_block(x)

class MnasNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MnasNet, self).__init__()
        # Initial Convolution Layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(inplace=True)
        )
        
        # Inverted Residual Blocks
        self.blocks = nn.Sequential(
            InvertedResidualBlock(32, 16, 1, 1),
            InvertedResidualBlock(16, 24, 6, 2),
            InvertedResidualBlock(24, 24, 6, 1),
            InvertedResidualBlock(24, 32, 6, 2),
            InvertedResidualBlock(32, 32, 6, 1),
            InvertedResidualBlock(32, 64, 6, 2),
            InvertedResidualBlock(64, 64, 6, 1),
        )
        
        # Final Classification Layer
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 1280, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer
mnasnet = MnasNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnasnet.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = mnasnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = mnasnet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {(100 * correct / total):.2f}%')
