import torch.nn as nn
import math

__all__ = ['mobilenet_flat']

class MobileNet(nn.Module):                                                                  

    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        self.conv1  = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(32)

        # conv_dw( 32,  64, 1)
        self.conv2  = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32 ,bias=False)
        self.bn2    = nn.BatchNorm2d(32)
        self.conv3  = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3    = nn.BatchNorm2d(64)

        # conv_dw( 64, 128, 2)
        self.conv4  = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64 ,bias=False)
        self.bn4    = nn.BatchNorm2d(64)
        self.conv5  = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5    = nn.BatchNorm2d(128)

        # conv_dw(128, 128, 1)
        self.conv6  = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128 ,bias=False)
        self.bn6    = nn.BatchNorm2d(128)
        self.conv7  = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7    = nn.BatchNorm2d(128)

        # conv_dw(128, 256, 2)
        self.conv8  = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128 ,bias=False)
        self.bn8    = nn.BatchNorm2d(128)
        self.conv9  = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9    = nn.BatchNorm2d(256)

        # conv_dw(256, 256, 1)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256 ,bias=False)
        self.bn10   = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11   = nn.BatchNorm2d(256)

        # conv_dw(256, 512, 2)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256 ,bias=False)
        self.bn12   = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn13   = nn.BatchNorm2d(512)

        # conv_dw(512, 512, 1)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512 ,bias=False)
        self.bn14   = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn15   = nn.BatchNorm2d(512)

        # conv_dw(512, 512, 1)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512 ,bias=False)
        self.bn16   = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn17   = nn.BatchNorm2d(512)

        # conv_dw(512, 512, 1)
        self.conv18 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512 ,bias=False)
        self.bn18   = nn.BatchNorm2d(512)
        self.conv19 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn19   = nn.BatchNorm2d(512)

        # conv_dw(512, 512, 1)
        self.conv20 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512 ,bias=False)
        self.bn20   = nn.BatchNorm2d(512)
        self.conv21 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21   = nn.BatchNorm2d(512)

        # conv_dw(512, 512, 1)
        self.conv22 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512 ,bias=False)
        self.bn22   = nn.BatchNorm2d(512)
        self.conv23 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn23   = nn.BatchNorm2d(512)

        # conv_dw(512, 1024, 2)
        self.conv24 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512 ,bias=False)
        self.bn24   = nn.BatchNorm2d(512)
        self.conv25 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn25   = nn.BatchNorm2d(1024)

        # conv_dw(1024, 1024, 1)
        self.conv26 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, groups=1024 ,bias=False)
        self.bn26   = nn.BatchNorm2d(1024)
        self.conv27 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn27   = nn.BatchNorm2d(1024)

        self.relu   = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, 1000)

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

    def forward(self, x):                                                              
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu(x)

        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu(x)
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.relu(x)

        x = self.conv16(x)
        x = self.bn16(x)
        x = self.relu(x)
        x = self.conv17(x)
        x = self.bn17(x)
        x = self.relu(x)

        x = self.conv18(x)
        x = self.bn18(x)
        x = self.relu(x)
        x = self.conv19(x)
        x = self.bn19(x)
        x = self.relu(x)

        x = self.conv20(x)
        x = self.bn20(x)
        x = self.relu(x)
        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu(x)

        x = self.conv22(x)
        x = self.bn22(x)
        x = self.relu(x)
        x = self.conv23(x)
        x = self.bn23(x)
        x = self.relu(x)

        x = self.conv24(x)
        x = self.bn24(x)
        x = self.relu(x)
        x = self.conv25(x)
        x = self.bn25(x)
        x = self.relu(x)

        x = self.conv26(x)
        x = self.bn26(x)
        x = self.relu(x)
        x = self.conv27(x)
        x = self.bn27(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def mobilenet_flat(**kwargs):
    model = MobileNet(**kwargs)
    return model
