""" Flattened ResNet50 for ImageNet
"""

import torch.nn as nn
import math

__all__ = ['resnet50_flat']

class ResNet50(nn.Module):

    # This should be redefined by the channel count
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.conv1  = nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False, stride=2)
        self.bn1    = nn.BatchNorm2d(64)
        print('INITIALIZEING RESNET-50')
        #1
        self.conv2  = nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn2    = nn.BatchNorm2d(64)
        self.conv3  = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn3    = nn.BatchNorm2d(64)
        self.conv4  = nn.Conv2d(64, 256, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn4    = nn.BatchNorm2d(256)
        self.conv5  = nn.Conv2d(64, 256, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn5    = nn.BatchNorm2d(256)

        #2
        self.conv6  = nn.Conv2d(256, 64, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn6    = nn.BatchNorm2d(64)
        self.conv7  = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn7    = nn.BatchNorm2d(64)
        self.conv8  = nn.Conv2d(64, 256, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn8    = nn.BatchNorm2d(256)

        #3
        self.conv9  = nn.Conv2d(256, 64, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn9    = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn10   = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64, 256, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn11   = nn.BatchNorm2d(256)

        #4 (Stage 2)
        self.conv12 = nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn12   = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, stride=2)
        self.bn13   = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn14   = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(256, 512, kernel_size=1, padding=0, bias=False, stride=2)
        self.bn15   = nn.BatchNorm2d(512)

        #5
        self.conv16 = nn.Conv2d(512, 128, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn16   = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn17   = nn.BatchNorm2d(128)
        self.conv18 = nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn18   = nn.BatchNorm2d(512)

        #6
        self.conv19 = nn.Conv2d(512, 128, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn19   = nn.BatchNorm2d(128)
        self.conv20 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn20   = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn21   = nn.BatchNorm2d(512)
        
        #7
        self.conv22 = nn.Conv2d(512, 128, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn22   = nn.BatchNorm2d(128)
        self.conv23 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn23   = nn.BatchNorm2d(128)
        self.conv24 = nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn24   = nn.BatchNorm2d(512)

        #8 (Stage 3)
        self.conv25 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn25   = nn.BatchNorm2d(256)
        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, stride=2)
        self.bn26   = nn.BatchNorm2d(256)
        self.conv27 = nn.Conv2d(256, 1024, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn27   = nn.BatchNorm2d(1024)
        self.conv28 = nn.Conv2d(512, 1024, kernel_size=1, padding=0, bias=False, stride=2)
        self.bn28   = nn.BatchNorm2d(1024)

        #9
        self.conv29 = nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn29   = nn.BatchNorm2d(256)
        self.conv30 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn30   = nn.BatchNorm2d(256)
        self.conv31 = nn.Conv2d(256, 1024, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn31   = nn.BatchNorm2d(1024)

        #10
        self.conv32 = nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn32   = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn33   = nn.BatchNorm2d(256)
        self.conv34 = nn.Conv2d(256, 1024, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn34   = nn.BatchNorm2d(1024)

        #11
        self.conv35 = nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn35   = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn36   = nn.BatchNorm2d(256)
        self.conv37 = nn.Conv2d(256, 1024, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn37   = nn.BatchNorm2d(1024)

        #12
        self.conv38 = nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn38   = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn39   = nn.BatchNorm2d(256)
        self.conv40 = nn.Conv2d(256, 1024, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn40   = nn.BatchNorm2d(1024)

        #13
        self.conv41 = nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn41   = nn.BatchNorm2d(256)
        self.conv42 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn42   = nn.BatchNorm2d(256)
        self.conv43 = nn.Conv2d(256, 1024, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn43   = nn.BatchNorm2d(1024)

        #14 (Stage 4)
        self.conv44 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn44   = nn.BatchNorm2d(512)
        self.conv45 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False, stride=2)
        self.bn45   = nn.BatchNorm2d(512)
        self.conv46 = nn.Conv2d(512, 2048, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn46   = nn.BatchNorm2d(2048)
        self.conv47 = nn.Conv2d(1024, 2048, kernel_size=1, padding=0, bias=False, stride=2)
        self.bn47   = nn.BatchNorm2d(2048)

        #15
        self.conv48 = nn.Conv2d(2048, 512, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn48   = nn.BatchNorm2d(512)
        self.conv49 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn49   = nn.BatchNorm2d(512)
        self.conv50 = nn.Conv2d(512, 2048, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn50   = nn.BatchNorm2d(2048)

        #16
        self.conv51 = nn.Conv2d(2048, 512, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn51   = nn.BatchNorm2d(512)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn52   = nn.BatchNorm2d(512)
        self.conv53 = nn.Conv2d(512, 2048, kernel_size=1, padding=0, bias=False, stride=1)
        self.bn53   = nn.BatchNorm2d(2048)

        self.avgpool_adt = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc     = nn.Linear(2048, num_classes)
        self.relu   = nn.ReLU(inplace=True)

        self.resnet_groups = [
            [2, 3, 4, 5], [6, 7, 8], [9, 10, 11],
            [12, 13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24],
            [25, 26, 27, 28], [29, 30, 31], [32, 33, 34], [35, 36, 37], [38, 39, 40], [41, 42, 43],
            [44, 45, 46, 47], [48, 49, 50], [51, 52, 53]
        ]

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



    # This part of architecture remains the same
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        _x = self.relu(x)

        # forward pass through all resnet groups
        for group in self.resnet_groups:

            # if a group has been removed due to pruning, skip it
            if f'conv{group[0]}' not in self._modules:
                continue
            
            if len(group) == 3:
                x = self._modules[f'conv{group[0]}'](_x)
                x = self._modules[f'bn{group[0]}'](x)
                x = self.relu(x)
                x = self._modules[f'conv{group[1]}'](x)
                x = self._modules[f'bn{group[1]}'](x)
                x = self.relu(x)
                x = self._modules[f'conv{group[2]}'](_x)
                x = self._modules[f'bn{group[2]}'](_x)
                _x = _x + x
                _x = self.relu(_x)
            
            elif len(group) == 4:
                x = self._modules[f'conv{group[0]}'](_x)
                x = self._modules[f'bn{group[0]}'](x)
                x = self.relu(x)
                x = self._modules[f'conv{group[1]}'](x)
                x = self._modules[f'bn{group[1]}'](x)
                x = self.relu(x)
                x = self._modules[f'conv{group[2]}'](x)
                x = self._modules[f'bn{group[2]}'](x)
                
                _x = self._modules[f'conv{group[3]}'](_x)
                _x = self._modules[f'bn{group[3]}'](_x)
                _x = _x + x
                _x = self.relu(_x)

        x = self.avgpool_adt(_x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    


def resnet50_flat(**kwargs):
    model = ResNet50(**kwargs)
    return model
