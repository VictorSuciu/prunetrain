"""
 Copyright 2019 Sangkug Lym
 Copyright 2019 The University of Texas at Austin

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models

from utils import Logger, AverageMeter, accuracy, mkdir_p
from custom import _makeSparse, _genDenseModel, _DataParallel
from custom import get_group_lasso_global, get_group_lasso_group
from custom_arch import *
import numpy as np

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Baseline
parser.add_argument('--data_path', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names, help='model architecture')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--manualSeed', type=int, help='manual seed')

# PruneTrain
parser.add_argument('--schedule-exp', type=int, default=0, help='Exponential LR decay.')
parser.add_argument('--save_checkpoint', default=10, type=int, 
                    help='Interval to save checkpoint')
parser.add_argument('--sparse_interval', default=0, type=int, 
                    help='Interval to force the value under threshold')
parser.add_argument('--threshold', default=0.0001, type=float, 
                    help='Threshold to force weight to zero')
parser.add_argument('--en_group_lasso', default=False, action='store_true',
                    help='Set the group-lasso coefficient')
parser.add_argument('--global_group_lasso', default=True, action='store_true',
                    help='True: use a global group lasso coefficient, '
                    'False: use sqrt(num_params) as a coefficient for each group')
parser.add_argument('--var_group_lasso_coeff', default=0.1, type=float,
                    help='Ratio = group-lasso / (group-lasso + loss)')
parser.add_argument('--grp_lasso_coeff', default=0.0005, type=float,
                    help='claim as a global param')
parser.add_argument('--arch_out_dir1', default=None, type=str,
                    help='directory to store the temporary architecture file')
parser.add_argument('--arch_out_dir2', default=None, type=str,
                    help='directory to architecture files matching to checkpoints ')
parser.add_argument('--arch_name', default='net.py', type=str,
                    help='name of the new architecture')
parser.add_argument('--is_gating', default=False, action='store_true',
                    help='Use gating for residual network')
parser.add_argument('--threshold_type', default='max', choices=['max', 'mean'], type=str,
                    help='Thresholding type')
parser.add_argument('--coeff_container', default='./coeff', type=str,
                    help='Directory to store lasso coefficient')
parser.add_argument('--global_coeff', default=True, action='store_true',
                    help='Use a global group lasso regularizaiton coefficient')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)') 

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

# Sub-sampling dataset
#class LimitDataset(data.Dataset):
#    def __init__(self, dataset, n):
#        self.dataset = dataset
#        self.n = n
#
#    def __len__(self):
#        return self.n
#
#    def __getitem__(self, i):
#        return self.dataset[i]

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    traindir  = os.path.join(args.data_path, 'train')
    valdir    = os.path.join(args.data_path, 'validation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                        transforms.RandomSizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,]))

    # Restrict the number of samples per class
    #train_dataset = LimitDataset(train_dataset, 200)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, 
        shuffle=True,
        num_workers=args.workers, 
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True)

    # Momdel creation
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet'):
        model.features = _DataParallel(model.features)
        model.cuda()
    else:
        model = _DataParallel(model).cuda()

    # Sanity check: print module name and shape
    #for name, param in model.named_parameters():
    #    print("{}, {}".format(name, list(param.shape)))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume from a checkpoint
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch'] +1 
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['LearningRate', 'TrainLoss', 'ValidLoss', 'TrainAcc.', 'ValidAcc.', 'Lasso/Full_loss', 'TrainEpochTime(s)', 'TestEpochTime(s)'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc, test_epoch_time = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs+1):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))

        train_loss, train_acc, lasso_ratio, train_epoch_time = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_epoch_time = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, lasso_ratio, train_epoch_time, test_epoch_time])

        # SparseTrain routine
        if args.en_group_lasso and (epoch % args.sparse_interval == 0):
            # Force weights under threshold to zero
            dense_chs, chs_map = _makeSparse(model, args.threshold, args.arch, 
                                             args.threshold_type,
                                             'imagenet',
                                             is_gating=args.is_gating)
            # Reconstruct architecture
            if args.arch_out_dir2 != None:
                _genDenseModel(model, dense_chs, optimizer, args.arch, 'imagenet')
                _genDenseArch = custom_arch_imagenet[args.arch]
                if 'resnet' in args.arch:
                    _genDenseArch(model, args.arch_out_dir1, args.arch_out_dir2, 
                                args.arch_name, dense_chs, 
                                chs_map, args.is_gating)
                else:
                    _genDenseArch(model, args.arch_out_dir1, args.arch_out_dir2, 
                                args.arch_name, dense_chs, chs_map)

        # Save the checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        print("[INFO] Storing checkpoint...")
        save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),},
                is_best, 
                checkpoint=args.checkpoint)

        # Leave unique checkpoint of pruned models druing training
        if epoch % args.save_checkpoint == 0:
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),},
                    is_best, 
                    checkpoint=args.checkpoint,
                    filename='checkpoint'+str(epoch)+'.tar')

    logger.close()

    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    lasso_ratio = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data_load_time = time.time() - end

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # lasso penalty
        init_batch = batch_idx == 0 and epoch == 1

        if args.en_group_lasso:
            if args.global_coeff:
                lasso_penalty = get_group_lasso_global(model, args.arch)
            else:
                lasso_penalty = get_group_lasso_group(model, args.arch)

            # Auto-tune the group-lasso coefficient @first training iteration
            coeff_dir = os.path.join(args.coeff_container, 'imagenet', args.arch)
            if init_batch:
                args.grp_lasso_coeff = args.var_group_lasso_coeff *loss.item() / (lasso_penalty * (1-args.var_group_lasso_coeff))
                grp_lasso_coeff = torch.autograd.Variable(args.grp_lasso_coeff)

                if not os.path.exists( coeff_dir ):
                    os.makedirs( coeff_dir )
                with open( os.path.join(coeff_dir, str(args.var_group_lasso_coeff)), 'w' ) as f_coeff:
                    f_coeff.write( str(grp_lasso_coeff.item()) )

            else:
                with open( os.path.join(coeff_dir, str(args.var_group_lasso_coeff)), 'r' ) as f_coeff:
                    for line in f_coeff:
                        grp_lasso_coeff = float(line)

            lasso_penalty = lasso_penalty * grp_lasso_coeff
        else:
            lasso_penalty = 0.

        # Group lasso calcution is not performance-optimized => Ignore from execution time
        loss += lasso_penalty

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        lasso_ratio.update(lasso_penalty / loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end - data_load_time)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    epoch_time = batch_time.avg * len(train_loader)    # Time for total training dataset
    return (losses.avg, top1.avg, lasso_ratio.avg, epoch_time)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data_load_time = time.time() - end

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end - data_load_time)
        end = time.time()

    epoch_time = batch_time.avg * len(val_loader)   # Time for total test dataset
    return (losses.avg, top1.avg, epoch_time)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if args.schedule_exp == 0:
        # Step-wise LR decay
        set_lr = args.lr
        for lr_decay in args.schedule:
            if epoch >= lr_decay:
                set_lr *= args.gamma
        state['lr'] = set_lr
    else:
        # Exponential LR decay
        set_lr = args.lr
        exp = int((epoch -1) / args.schedule_exp)
        state['lr'] = set_lr * (args.gamma**exp)

    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
