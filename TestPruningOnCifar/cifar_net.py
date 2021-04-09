import torch
from torch import nn
from torch.nn import functional as F


class CifarNet(nn.Module):

    def __init__(self):
        super(CifarNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=1, padding=5, bias=True)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 100, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(100, 170, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(170, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(2, 2)
        self.relu5 = nn.ReLU()

        self.flatten1 = nn.Flatten()

        self.linear1 = nn.Linear(256, 64)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(64, 10)
        self.softmax1 = nn.Softmax(dim=0)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.relu5(x)

        x = self.flatten1(x)

        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.softmax1(x)

        return x
