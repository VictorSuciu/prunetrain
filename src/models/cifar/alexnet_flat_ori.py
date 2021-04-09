""" Flattened version of AlexNet for CIFAR10/100
"""

import torch.nn as nn
import math

__all__ = ['alexnet_flat']

class AlexNet(nn.Module):

    # This should be redefined by the channel count
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5, bias=True)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=True)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu  = nn.ReLU(inplace=True)
        self.fc    = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    # This part of architecture remains the same
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def alexnet_flat(**kwargs):
    model = AlexNet(**kwargs)
    return model
