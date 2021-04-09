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

import os
import glob
import argparse

from feature_size_cifar import *
from scripts_util import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--in_dir', type=str)
parser.add_argument('-a', '--arch', type=str)
parser.add_argument('-g', '--gating', default=False, action='store_true')
args = parser.parse_args()

LARGE = 99999999999999
PFLOPS = 1000 *1000 *1000 *1000 *1000 /2    # MUL + ADD
MFLOPS = 1000 *1000 /2                      # MUL + ADD
WORD = 2
PB = 1024*1024*1024*1024*1024
MB = 1024*1024

reconf_int = 5
mini_batch = 256
epochs = 90

cifar = 50000. / mini_batch
imgnet = 1281167. / mini_batch
iters_per_epoch = imgnet
bn = True

inf_list = glob.glob(args.in_dir)
fmap = imagenet_feature_size[args.arch]
log_tot = {'coeff':[], 'train_cost':[], 'bn_cost':[], 'best_acc':[], 'inf_cost':[]}

""" Calcuate below metrics from the network architecture
# 1. training cost
# 2. inference cost
# 3. memory accesses in BN layers
# 4. activation size
# 5. model size
# 6. output channels
"""
def getTrainingCost(arch, base=False, verbose=True):

    train_cost_acc, bn_cost_acc, inf_cost_acc = 0, 0, 0
    out_act, out_chs_tot, model_size = 0, 0, 0
    
    for lyr_name, lyr in arch.items():
        out_chs = 0

        if base:
            dims = len(lyr)
            if dims == 4:
                k, c, r, s  = lyr[0], lyr[1], lyr[2], lyr[3]
                pad = 0 if (lyr[2] == 1) else 2
                out_chs = k
                if args.arch == 'resnet50' and lyr_name == 'conv1':
                    pad = 3
            else:
                k, c  = lyr[0], lyr[1]
            #print("base, {}, {}, {}, {}, {}, {}".format(lyr_name, c, k, r, s, pad))
        else:
            if lyr['cfg'] == None:
                continue

            dims = len(lyr['cfg'])
            if args.gating:
                c = lyr['cfg'][1] if (lyr['gt'][1] == None) else lyr['gt'][1]
                k = lyr['cfg'][0] if (lyr['gt'][0] == None) else lyr['gt'][0]
                out_chs = k
            else:
                c = lyr['cfg'][1]
                k = lyr['cfg'][0]
                out_chs = k

            if dims == 4:
                r, s = lyr['cfg'][2], lyr['cfg'][3]
                pad = 0 if (r == 1) else 2
                if args.arch == 'resnet50' and lyr_name == 'conv1':
                    pad = 3

        if dims == 4:
            # Inference cost = (CRS)(K)(PQ)
            inf_cost    = (c * r * s) * k * (fmap[lyr_name][1]**2)

            # Train cost = Forward + WGRAD + DGRAD
            # = (CRS)(K)(NPQ) + (CRS)(NPQ)(K) + (NHW)(KRS)(C)
            # = (N x inference_cost x2) + (NHW)(KRS)(C)
            train_cost  = mini_batch *(2*inf_cost + ((fmap[lyr_name][0]+pad)**2) * (k * r * s) * c)
            #train_cost  = mini_batch *(2*inf_cost + fmap[lyr_name][0]**2 * (k * r * s) * c)
            # BN cost = (3 x NKPQ) + (5 x NKPQ)
            bn_cost = 8 * (mini_batch * k * (fmap[lyr_name][1]**2))
            model_size += float(r*s*c*k*WORD)/MB
            if bn:
                model_size += float(k*WORD)/MB
        else:
            # Forward: NCK
            inf_cost    = k * c
            train_cost  = k * c * mini_batch * 3
            bn_cost     = 0
            model_size += float(c*k*WORD)/MB
            model_size += float(k*WORD)/MB

        if 'fc' not in lyr_name:
            out_act += out_chs * (fmap[lyr_name][1]**2)
            out_chs_tot += out_chs

        if verbose: 
            if 'conv' in lyr_name:
                print_name = lyr_name.split('conv')[1]
            else:
                print_name = lyr_name
            print("{}, {}, {}, {}".format(print_name, train_cost, inf_cost, out_chs))
        inf_cost_acc    += inf_cost
        train_cost_acc  += train_cost
        bn_cost_acc     += bn_cost

        #print("{}, {}, {}, {}, {}, {}, {}".format(lyr_name, c, k, r, s, pad, inf_cost))

    train_cost_acc *= iters_per_epoch
    bn_cost_acc *= iters_per_epoch

    if verbose: 
        print("===================")
    return train_cost_acc, bn_cost_acc, inf_cost_acc, out_act, out_chs_tot, model_size


# Training iterations before compression
train_cost_base, bn_cost_base, inf_cost_base, out_act_base, out_chs_base, model_size_base = getTrainingCost(base_archs[args.arch], base=True)

# Base architecture cost
log_tot['coeff'].append(0.)
log_tot['train_cost'].append( train_cost_base * epochs )
log_tot['bn_cost'].append( bn_cost_base * epochs )
log_tot['best_acc'].append( 100 )
log_tot['inf_cost'].append( inf_cost_base )

for inf_name in sorted(inf_list):
    inf = open(inf_name, 'r')
    log = {'epoch':[], 'train_cost':[], 'bn_cost':[], 'inf_cost':[], 'out_act':[], 'out_chs':[], 'model_size':[]}

    log['epoch'].append(1)
    log['train_cost'].append(train_cost_base)
    log['bn_cost'].append(bn_cost_base)
    log['inf_cost'].append(inf_cost_base)
    log['out_act'].append(out_act_base)
    log['out_chs'].append(out_chs_base)
    log['model_size'].append(model_size_base)

    next_line = LARGE

    for line_num, line in enumerate(inf):
        # Get epoch info
        if "Epoch" in line and "LR" in line:
            epoch = int(line.split('|')[0].split('[')[1])
        
        # Create a new architecture info container
        if "Force the sparse filters to zero" in line:
            arch = {}

        if ("weight" in line) and ("bn" not in line):
            layer = line.split('.')[1]
            if layer not in arch:
                arch[layer] = {'cfg':None, 'gt':[None, None]}

            # Get channel info without gating

            #if any([i for i in ['Input_ch', 'Output_ch'] if i in line]):
            if 'Input_ch' in line:
                arch[layer]['gt'][1] = int(line.split(' =>')[0].split(': ')[1])
            elif 'Output_ch' in line:
                arch[layer]['gt'][0] = int(line.split(' =>')[0].split(': ')[1])
            elif '>>' in line:
                tensor = line.split('>> [')[1].split(']')[0].split(',')
                arch[layer]['cfg'] = [int(i) for i in tensor]

        if "Best acc" in line:
            next_line = line_num+1

        if line_num == next_line:
            best_acc = float(line)

        # Register the epoch computation cost
        if "Generating a new dense architecture" in line:
            train_cost, bn_cost, inf_cost, out_act, out_chs, model_size = getTrainingCost(arch)

            log['epoch'].append(epoch)
            log['train_cost'].append(train_cost)
            log['bn_cost'].append(bn_cost)
            log['inf_cost'].append(inf_cost)
            log['out_act'].append(out_act)
            log['out_chs'].append(out_chs)
            log['model_size'].append(model_size)


    print("Total [{}] epochs processed...".format(len(log['epoch'])* reconf_int))
    coeff = inf_name.split('/')[-1].split('.log')[0]
    log_tot['coeff'].append(coeff)
    log_tot['train_cost'].append( sum(log['train_cost']) *reconf_int )
    log_tot['bn_cost'].append( sum(log['bn_cost']) *reconf_int )
    log_tot['best_acc'].append( best_acc )
    log_tot['inf_cost'].append( min(log['inf_cost']) )

    print("======== {} =======".format(coeff))
    print("{}, {}, {}, {}, {}, {}, {}".format('epoch', 'train_cost', 'bn_cost', 'inf_cost', 'out_act', 'out_chs', 'model_size'))
    for idx, e in enumerate(log['epoch']):
        print('{}, {}, {}, {}, {}, {}, {}'.format(e, log['train_cost'][idx], 
                                             log['bn_cost'][idx], 
                                             log['inf_cost'][idx], 
                                             log['out_act'][idx], 
                                             log['out_chs'][idx],
                                             log['model_size'][idx]))

    # Remove the final recfguration info for total training cost
    if epochs % reconf_int == 0:
        for i in log:
            del log[i][-1]


print("======== Total Training cost =======")
print("coeff, best_acc, train_cost, bn_cost")
for idx, e in enumerate(log_tot['coeff']):
    print('{}, {}, {}, {}'.format(e, 
                                  log_tot['best_acc'][idx],
                                  float(log_tot['train_cost'][idx])/PFLOPS,
                                  float(log_tot['bn_cost'][idx])*WORD/PB,
                                  ))


print("======== Inference cost =======")
print("coeff, best_acc, inf_cost")
for idx, e in enumerate(log_tot['coeff']):
    print('{}, {}, {}'.format(e, 
                              log_tot['best_acc'][idx],
                              float(log_tot['inf_cost'][idx])/MFLOPS
                              ))
