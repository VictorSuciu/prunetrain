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
        x = self.relu(x)
        _x = self.maxpool(x)

        #1
        x = self.conv2(_x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        _x = self.conv5(_x)
        _x = self.bn5(_x)
        _x = _x + x
        _x = self.relu(_x)

        #2
        x = self.conv6(_x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        _x = _x + x
        _x = self.relu(_x)

        #3
        x = self.conv9(_x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.conv11(x)
        x = self.bn11(x)
        _x = _x + x
        _x = self.relu(_x)

        #4 (Stage 2)
        x = self.conv12(_x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu(x)
        x = self.conv14(x)
        x = self.bn14(x)
        _x = self.conv15(_x)
        _x = self.bn15(_x)
        _x = _x + x
        _x = self.relu(_x)

        #5
        x = self.conv16(_x)
        x = self.bn16(x)
        x = self.relu(x)
        x = self.conv17(x)
        x = self.bn17(x)
        x = self.relu(x)
        x = self.conv18(x)
        x = self.bn18(x)
        _x = _x + x
        _x = self.relu(_x)

        #6
        x = self.conv19(_x)
        x = self.bn19(x)
        x = self.relu(x)
        x = self.conv20(x)
        x = self.bn20(x)
        x = self.relu(x)
        x = self.conv21(x)
        x = self.bn21(x)
        _x = _x + x
        _x = self.relu(_x)

        #7
        x = self.conv22(_x)
        x = self.bn22(x)
        x = self.relu(x)
        x = self.conv23(x)
        x = self.bn23(x)
        x = self.relu(x)
        x = self.conv24(x)
        x = self.bn24(x)
        _x = _x + x
        _x = self.relu(_x)

        #8
        x = self.conv25(_x)
        x = self.bn25(x)
        x = self.relu(x)
        x = self.conv26(x)
        x = self.bn26(x)
        x = self.relu(x)
        x = self.conv27(x)
        x = self.bn27(x)
        _x = self.conv28(_x)
        _x = self.bn28(_x)
        _x = _x + x
        _x = self.relu(_x)

        #9
        x = self.conv29(_x)
        x = self.bn29(x)
        x = self.relu(x)
        x = self.conv30(x)
        x = self.bn30(x)
        x = self.relu(x)
        x = self.conv31(x)
        x = self.bn31(x)
        _x = _x + x
        _x = self.relu(_x)

        #10
        x = self.conv32(_x)
        x = self.bn32(x)
        x = self.relu(x)
        x = self.conv33(x)
        x = self.bn33(x)
        x = self.relu(x)
        x = self.conv34(x)
        x = self.bn34(x)
        _x = _x + x
        _x = self.relu(_x)

        #11
        x = self.conv35(_x)
        x = self.bn35(x)
        x = self.relu(x)
        x = self.conv36(x)
        x = self.bn36(x)
        x = self.relu(x)
        x = self.conv37(x)
        x = self.bn37(x)
        _x = _x + x
        _x = self.relu(_x)

        #12
        x = self.conv38(_x)
        x = self.bn38(x)
        x = self.relu(x)
        x = self.conv39(x)
        x = self.bn39(x)
        x = self.relu(x)
        x = self.conv40(x)
        x = self.bn40(x)
        _x = _x + x
        _x = self.relu(_x)

        #13
        x = self.conv41(_x)
        x = self.bn41(x)
        x = self.relu(x)
        x = self.conv42(x)
        x = self.bn42(x)
        x = self.relu(x)
        x = self.conv43(x)
        x = self.bn43(x)
        _x = _x + x
        _x = self.relu(_x)

        #14 (Stage 4)
        x = self.conv44(_x)
        x = self.bn44(x)
        x = self.relu(x)
        x = self.conv45(x)
        x = self.bn45(x)
        x = self.relu(x)
        x = self.conv46(x)
        x = self.bn46(x)
        _x = self.conv47(_x)
        _x = self.bn47(_x)
        _x = _x + x
        _x = self.relu(_x)

        #15
        x = self.conv48(_x)
        x = self.bn48(x)
        x = self.relu(x)
        x = self.conv49(x)
        x = self.bn49(x)
        x = self.relu(x)
        x = self.conv50(x)
        x = self.bn50(x)
        _x = _x + x
        _x = self.relu(_x)

        #16
        x = self.conv51(_x)
        x = self.bn51(x)
        x = self.relu(x)
        x = self.conv52(x)
        x = self.bn52(x)
        x = self.relu(x)
        x = self.conv53(x)
        x = self.bn53(x)
        _x = _x + x
        _x = self.relu(_x)

        x = self.avgpool_adt(_x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50_flat(**kwargs):
    model = ResNet50(**kwargs)
    return model
