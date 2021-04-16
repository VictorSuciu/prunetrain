""" Flattened ResNet32 for CIFAR10/100
"""

import torch.nn as nn
import math

__all__ = ['resnet32_flat']

class ResNet32(nn.Module):

    # This should be redefined by the channel count
    def __init__(self, num_classes=10):
        super(ResNet32, self).__init__()
        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn1    = nn.BatchNorm2d(16)

        #1
        self.conv2  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn2    = nn.BatchNorm2d(16)
        self.conv3  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn3    = nn.BatchNorm2d(16)

        #2
        self.conv4  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn4    = nn.BatchNorm2d(16)
        self.conv5  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn5    = nn.BatchNorm2d(16)

        #3
        self.conv6  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn6    = nn.BatchNorm2d(16)
        self.conv7  = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn7    = nn.BatchNorm2d(16)

        #4        
        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn8   = nn.BatchNorm2d(16)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn9   = nn.BatchNorm2d(16)

        #5
        self.conv10 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn10   = nn.BatchNorm2d(16)
        self.conv11 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn11   = nn.BatchNorm2d(16)

        #6 (Stage 2)
        self.conv12 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False, stride=2)
        self.bn12   = nn.BatchNorm2d(32)
        self.conv13 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn13   = nn.BatchNorm2d(32)
        self.conv14 = nn.Conv2d(16, 32, kernel_size=1, padding=0, bias=False, stride=2)
        self.bn14   = nn.BatchNorm2d(32)

        #7
        self.conv15 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn15   = nn.BatchNorm2d(32)
        self.conv16 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn16   = nn.BatchNorm2d(32)

        #8
        self.conv17 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn17   = nn.BatchNorm2d(32)
        self.conv18 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn18   = nn.BatchNorm2d(32)

        #9
        self.conv19 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn19   = nn.BatchNorm2d(32)
        self.conv20 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn20   = nn.BatchNorm2d(32)

        #10
        self.conv21 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn21   = nn.BatchNorm2d(32)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn22   = nn.BatchNorm2d(32)

        #11 (Stage 3)
        self.conv23 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False, stride=2)
        self.bn23   = nn.BatchNorm2d(64)
        self.conv24 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn24   = nn.BatchNorm2d(64)
        self.conv25 = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False, stride=2)
        self.bn25   = nn.BatchNorm2d(64)

        #12
        self.conv26 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn26   = nn.BatchNorm2d(64)
        self.conv27 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn27   = nn.BatchNorm2d(64)

        #13
        self.conv28 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn28   = nn.BatchNorm2d(64)
        self.conv29 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn29   = nn.BatchNorm2d(64)

        #14
        self.conv30 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn30   = nn.BatchNorm2d(64)
        self.conv31 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn31   = nn.BatchNorm2d(64)

        #15
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn32   = nn.BatchNorm2d(64)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn33   = nn.BatchNorm2d(64)

        self.avgpool = nn.AvgPool2d(8)
        self.fc     = nn.Linear(64, num_classes)
        self.relu   = nn.ReLU(inplace=True)

        self.resnet_groups = [
            [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
            [12, 13, 14],
            [15, 16], [17, 18], [19, 20], [21, 22],
            [23, 24, 25],
            [26, 27], [28, 29], [30, 31], [32, 33]
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
  
    # This part of architecture remains the same
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        _x = self.relu(x)

        # forwaard pass through all resnet groups
        for group in self.resnet_groups:

            # if a group has been removed due to pruning, skip it
            if f'conv{group[0]}' not in self._modules:
                continue
            
            if len(group) == 2:
                x = self._modules[f'conv{group[0]}'](_x)
                x = self._modules[f'bn{group[0]}'](x)
                x = self.relu(x)
                x = self._modules[f'conv{group[1]}'](x)
                x = self._modules[f'bn{group[1]}'](x)
                _x = _x + x
                _x = self.relu(_x)
            
            elif len(group) == 3:
                x = self._modules[f'conv{group[0]}'](_x)
                x = self._modules[f'bn{group[0]}'](x)
                x = self.relu(x)
                x = self._modules[f'conv{group[1]}'](x)
                x = self._modules[f'bn{group[1]}'](x)
                _x = self._modules[f'conv{group[2]}'](_x)
                _x = self._modules[f'bn{group[2]}'](_x)
                _x = _x + x
                _x = self.relu(_x)

        x = self.avgpool(_x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet32_flat(**kwargs):
    model = ResNet32(**kwargs)
    return model
