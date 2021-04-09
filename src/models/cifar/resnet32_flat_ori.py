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

        #1
        x = self.conv2(_x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        _x = _x + x
        _x = self.relu(_x)

        #2
        x = self.conv4(_x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        _x = _x + x
        _x = self.relu(_x)

        #3
        x = self.conv6(_x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        _x = _x + x
        _x = self.relu(_x)

        #4
        x = self.conv8(_x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        _x = _x + x
        _x = self.relu(_x)

        #5
        x = self.conv10(_x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.conv11(x)
        x = self.bn11(x)
        _x = _x + x
        _x = self.relu(_x)

        #6 (Stage 2)
        x = self.conv12(_x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        _x = self.conv14(_x)
        _x = self.bn14(_x)
        _x = _x + x
        _x = self.relu(_x)

        #7
        x = self.conv15(_x)
        x = self.bn15(x)
        x = self.relu(x)
        x = self.conv16(x)
        x = self.bn16(x)
        _x = _x + x
        _x = self.relu(_x)

        #8
        x = self.conv17(_x)
        x = self.bn17(x)
        x = self.relu(x)
        x = self.conv18(x)
        x = self.bn18(x)
        _x = _x + x
        _x = self.relu(_x)

        #9
        x = self.conv19(_x)
        x = self.bn19(x)
        x = self.relu(x)
        x = self.conv20(x)
        x = self.bn20(x)
        _x = _x + x
        _x = self.relu(_x)

        #10
        x = self.conv21(_x)
        x = self.bn21(x)
        x = self.relu(x)
        x = self.conv22(x)
        x = self.bn22(x)
        _x = _x + x
        _x = self.relu(_x)

        #11 (Stage 3)
        x = self.conv23(_x)
        x = self.bn23(x)
        x = self.relu(x)
        x = self.conv24(x)
        x = self.bn24(x)
        _x = self.conv25(_x)
        _x = self.bn25(_x)
        _x = _x + x
        _x = self.relu(_x)

        #12
        x = self.conv26(_x)
        x = self.bn26(x)
        x = self.relu(x)
        x = self.conv27(x)
        x = self.bn27(x)
        _x = _x + x
        _x = self.relu(_x)

        #13
        x = self.conv28(_x)
        x = self.bn28(x)
        x = self.relu(x)
        x = self.conv29(x)
        x = self.bn29(x)
        _x = _x + x
        _x = self.relu(_x)

        #14
        x = self.conv30(_x)
        x = self.bn30(x)
        x = self.relu(x)
        x = self.conv31(x)
        x = self.bn31(x)
        _x = _x + x
        _x = self.relu(_x)

        #15
        x = self.conv32(_x)
        x = self.bn32(x)
        x = self.relu(x)
        x = self.conv33(x)
        x = self.bn33(x)
        _x = _x + x
        _x = self.relu(_x)

        x = self.avgpool(_x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet32_flat(**kwargs):
    model = ResNet32(**kwargs)
    return model
