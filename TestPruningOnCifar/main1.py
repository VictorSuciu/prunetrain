import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from cifar_net import CifarNet
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    correct = (output.max(1).indices + 1) - target == 0
    return torch.sum(correct).item() / target.shape[0]


cifar_path = '/Users/victor/College/Spring-Quarter/ML-Practice/cifar-10/cifar-10-batches-py'

epochs = 40
batch_size = 16
lr = 0.05
momentum=0.5

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


dataloader = datasets.CIFAR10
trainset = dataloader(root='./dataset/data/torch', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

testset = dataloader(root='./dataset/data/torch', train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=50, shuffle=False, num_workers=1)

device = torch.device('cuda:0')

model = CifarNet().to(device)
# print(f'model device: {model.device}')
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss_func = nn.CrossEntropyLoss()
losses = []

# prune testing
layer_idx = 4
kernel_idx = 25

for ep in range(epochs):
    bcount = 1
    print(f'Epoch {ep+1}')

    for batch_data, batch_labels in trainloader:
        print('\n----------\n')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        for name, param in model.named_parameters():
            print(name, list(param.shape))
        
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        if( bcount % 1 == 0):
            print(f'batch {bcount}')
            with torch.no_grad():
                accuracies = []
                for test_data, test_labels in testloader:
                    test_data, test_labels = test_data.to(device), test_labels.to(device)
                    test_preds = model(test_data)
                    accuracies.append(accuracy(test_preds, test_labels))
                print(np.mean(accuracies))
        
        optimizer.zero_grad()
        preds = model(batch_data)
        loss = loss_func(preds, batch_labels)

        losses.append(loss.item())
        # print(loss.item())
        loss.backward()
        optimizer.step()

        all_params = [param for name, param in model.named_parameters()]
        
        # params to prune
        param_back = all_params[layer_idx]
        param_bias_back = all_params[layer_idx + 1]
        param_front = all_params[layer_idx + 2]

        # indexes to keep
        keep_idxs = np.arange(param_front.shape[1])
        keep_idxs = keep_idxs[keep_idxs != kernel_idx]
        
        # prune desired parameters
        new_back = nn.Conv2d(64, 99, kernel_size=5, stride=1, padding=2)
        new_front = nn.Conv2d(99, 170, kernel_size=3, stride=1, padding=1)

        with torch.no_grad()
            new_back.weight = param_back[keep_idxs,:,:,:]
            new_back.bias = param_bias_back[keep_idxs]
            new_front.weight = param_front[:,keep_idxs,:,:]
        
        # param_back.data = param_back[keep_idxs,:,:,:]
        # param_bias_back.data = param_bias_back[keep_idxs]
        # param_front.data = param_front[:,keep_idxs,:,:]
        
        # list sizes to verify
        for i in range(5):
            print(list(model.state_dict()[f'conv{i+1}.weight'].shape))
        bcount += 1

    

