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
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss_func = nn.CrossEntropyLoss()
losses = []

# delete a kernel from one layer and its corresponding input channel from the layer in front
back_layer_idx = 3
kernel_idx = 25

for ep in range(epochs):
    bcount = 1
    print(f'Epoch {ep+1}')

    for batch_data, batch_labels in trainloader:
        print('\n----------\n')
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
        loss.backward()
        optimizer.step()
        


        
        # params from which to prune
        param_back = model._modules[f'conv{back_layer_idx}'].weight
        param_bias_back = model._modules[f'conv{back_layer_idx}'].bias
        param_front = model._modules[f'conv{back_layer_idx+1}'].weight

        # indexes to keep
        # 0, 1, 2, 3, 4, 5... N delete kernel_idx
        keep_idxs = np.arange(param_front.shape[1])
        keep_idxs = keep_idxs[keep_idxs != kernel_idx]
        
        # new layers to replace
        new_back = nn.Conv2d(64, keep_idxs.shape[0], kernel_size=5, stride=1, padding=2).to(device)
        new_front = nn.Conv2d(keep_idxs.shape[0], 170, kernel_size=3, stride=1, padding=1).to(device)

        # prune from desired parameters
        with torch.no_grad():
            # param_back.data = param_back[keep_idxs,:,:,:]
            # param_bias_back.data = param_bias_back[keep_idxs]
            # param_front.data = param_front[:,keep_idxs,:,:]

            new_back.weight = nn.Parameter(param_back[keep_idxs,:,:,:])
            new_back.bias = nn.Parameter(param_bias_back[keep_idxs])
            new_front.weight = nn.Parameter(param_front[:,keep_idxs,:,:])
        
        # replace original layers with pruned version
        model._modules[f'conv{back_layer_idx}'] = new_back
        model._modules[f'conv{back_layer_idx+1}'] = new_front
        
        # list sizes to verify
        print(f'new channel count: {keep_idxs.shape[0]}')
        for i in range(5):
            print(list(model.state_dict()[f'conv{i+1}.weight'].shape))
        bcount += 1

    

