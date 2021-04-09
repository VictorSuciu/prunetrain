""" Flattened VGG16 for ImageNet
"""

import torch.nn as nn
import math

__all__ = ['vgg16_flat']

class VGG16(nn.Module):

    # This should be redefined by the channel count
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.conv2  = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(64)
        # MaxPool
        self.conv3  = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3    = nn.BatchNorm2d(128)
        self.conv4  = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4    = nn.BatchNorm2d(128)
        # MaxPool
        self.conv5  = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn5    = nn.BatchNorm2d(256)
        self.conv6  = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6    = nn.BatchNorm2d(256)
        self.conv7  = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn7    = nn.BatchNorm2d(256)
        # MaxPool
        self.conv8  = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn8    = nn.BatchNorm2d(512)
        self.conv9  = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn9    = nn.BatchNorm2d(512)
        self.conv10  = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn10    = nn.BatchNorm2d(512)
        # MaxPool
        self.conv11  = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn11    = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn12    = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn13    = nn.BatchNorm2d(512)
        # MaxPool

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # This part of architecture remains the same
    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def vgg16_flat(**kwargs):
    model = VGG16(**kwargs)
    return model
