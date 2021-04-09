import torch.nn as nn
import math

__all__ = ['mobilenet075_flat']

class MobileNet(nn.Module):                                                                  

    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        self.conv1  = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(24)

        # conv_dw( 24,  48, 1)
        self.conv2  = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=24 ,bias=False)
        self.bn2    = nn.BatchNorm2d(24)
        self.conv3  = nn.Conv2d(24, 48, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3    = nn.BatchNorm2d(48)

        # conv_dw( 48, 96, 2)
        self.conv4  = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, groups=48 ,bias=False)
        self.bn4    = nn.BatchNorm2d(48)
        self.conv5  = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5    = nn.BatchNorm2d(96)

        # conv_dw(96, 96, 1)
        self.conv6  = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96 ,bias=False)
        self.bn6    = nn.BatchNorm2d(96)
        self.conv7  = nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7    = nn.BatchNorm2d(96)

        # conv_dw(96, 192, 2)
        self.conv8  = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, groups=96 ,bias=False)
        self.bn8    = nn.BatchNorm2d(96)
        self.conv9  = nn.Conv2d(96, 192, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9    = nn.BatchNorm2d(192)

        # conv_dw(192, 192, 1)
        self.conv10 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, groups=192 ,bias=False)
        self.bn10   = nn.BatchNorm2d(192)
        self.conv11 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11   = nn.BatchNorm2d(192)

        # conv_dw(192, 384, 2)
        self.conv12 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, groups=192 ,bias=False)
        self.bn12   = nn.BatchNorm2d(192)
        self.conv13 = nn.Conv2d(192, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn13   = nn.BatchNorm2d(384)

        # conv_dw(384, 384, 1)
        self.conv14 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=384 ,bias=False)
        self.bn14   = nn.BatchNorm2d(384)
        self.conv15 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn15   = nn.BatchNorm2d(384)

        # conv_dw(384, 384, 1)
        self.conv16 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=384 ,bias=False)
        self.bn16   = nn.BatchNorm2d(384)
        self.conv17 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn17   = nn.BatchNorm2d(384)

        # conv_dw(384, 384, 1)
        self.conv18 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=384 ,bias=False)
        self.bn18   = nn.BatchNorm2d(384)
        self.conv19 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn19   = nn.BatchNorm2d(384)

        # conv_dw(384, 384, 1)
        self.conv20 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=384 ,bias=False)
        self.bn20   = nn.BatchNorm2d(384)
        self.conv21 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21   = nn.BatchNorm2d(384)

        # conv_dw(384, 384, 1)
        self.conv22 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=384 ,bias=False)
        self.bn22   = nn.BatchNorm2d(384)
        self.conv23 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn23   = nn.BatchNorm2d(384)

        # conv_dw(384, 768, 2)
        self.conv24 = nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, groups=384 ,bias=False)
        self.bn24   = nn.BatchNorm2d(384)
        self.conv25 = nn.Conv2d(384, 768, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn25   = nn.BatchNorm2d(768)

        # conv_dw(768, 768, 1)
        self.conv26 = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1, groups=768 ,bias=False)
        self.bn26   = nn.BatchNorm2d(768)
        self.conv27 = nn.Conv2d(768, 768, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn27   = nn.BatchNorm2d(768)

        self.relu   = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(768, 1000)

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
        x = x.view(-1, 768)
        x = self.fc(x)
        return x

def mobilenet075_flat(**kwargs):
    model = MobileNet(**kwargs)
    return model
