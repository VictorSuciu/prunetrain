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

import torch.nn as nn
# from resnet32_flat import ResNet32
# from feature_size_cifar import *
# from scripts_util import *

LARGE = 99999999999999
PFLOPS = 1000 *1000 *1000 *1000 *1000 /2    # MUL + ADD
MFLOPS = 1000 *1000 /2                      # MUL + ADD
WORD = 2
PB = 1024*1024*1024*1024*1024
MB = 1024*1024

reconf_int = 10
mini_batch = 128
epochs = 200

cifar = 50000. / mini_batch
imgnet = 1281167. / mini_batch
iters_per_epoch = cifar
bn = True

# inf_list = glob.glob(args.in_dir)

log_tot = {'coeff':[], 'train_cost':[], 'bn_cost':[], 'best_acc':[], 'inf_cost':[]}

""" Calcuate below metrics from the network architecture
# 1. training cost
# 2. inference cost
# 3. memory accesses in BN layers
# 4. activation size
# 5. model size
# 6. output channels
"""
def getTrainingCost(model, arch, gating=False, base=False, verbose=True):
    fmap = cifar_feature_size[arch]
    layer_size_dict = {}
    
    module_list = [m for m in model.modules()][1:]
    
    
    print('Calculating FLOPS')
    for name, module in zip(model._modules, module_list):
        if 'conv' in name or 'fc' in name:
            size_to_add = list(module.weight.shape)
            if 'fc' in name:
                size_to_add.reverse()
            
            layer_size_dict[name] = size_to_add
            print(name , '-->', layer_size_dict[name])
    
    train_cost_acc, bn_cost_acc, inf_cost_acc = 0, 0, 0
    out_act, out_chs_tot, model_size = 0, 0, 0
    
    arch = layer_size_dict

    for lyr_name, lyr in arch.items():
        out_chs = 0

        if base:
            dims = len(lyr)
            if dims == 4:
                k, c, r, s  = lyr[0], lyr[1], lyr[2], lyr[3]
                pad = 0 if (lyr[2] == 1) else 2
                out_chs = k
                if arch == 'resnet50' and lyr_name == 'conv1':
                    pad = 3
            else:
                k, c  = lyr[0], lyr[1]
            #print("base, {}, {}, {}, {}, {}, {}".format(lyr_name, c, k, r, s, pad))
        else:
            if lyr['cfg'] == None:
                continue

            dims = len(lyr['cfg'])
            if gating:
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
                if arch == 'resnet50' and lyr_name == 'conv1':
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



# model = ResNet32()

# print(getTrainingCost(model, 'resnet32_flat', base=True))

# # Training iterations before compression
# train_cost_base, bn_cost_base, inf_cost_base, out_act_base, out_chs_base, model_size_base = getTrainingCost(model, 'resnet32_flat', base=True)


#feature_size_cifar.py

alexnet = {}
alexnet['conv1'] = (32, 32)
alexnet['conv2'] = (8, 8)
alexnet['conv3'], alexnet['conv4'], alexnet['conv5'] = (4, 4), (4, 4), (4, 4)

vgg8 = {}
vgg8['conv1'] = (32, 32)
vgg8['conv2'] = (16, 16)
vgg8['conv3'] = (8, 8)
vgg8['conv4'] = (4, 4)
vgg8['conv5'] = (2, 2)

vgg11 = {}
vgg11['conv1'] = (32, 32)
vgg11['conv2'] = (16, 16)
vgg11['conv3'], vgg11['conv4'] = (8, 8), (8, 8)
vgg11['conv5'], vgg11['conv6'] = (4, 4), (4, 4)
vgg11['conv7'], vgg11['conv8'] = (2, 2), (2, 2)

vgg13 = {}
vgg13['conv1'], vgg13['conv2'] = (32, 32), (32, 32)
vgg13['conv3'], vgg13['conv4'] = (16, 16), (16, 16)
vgg13['conv5'], vgg13['conv6'] = (8, 8), (8, 8)
vgg13['conv7'], vgg13['conv8'] = (4, 4), (4, 4)
vgg13['conv9'], vgg13['conv10'] = (2, 2), (2, 2)

# ResNet20
resnet20 = dict.fromkeys(['conv'+str(i) for i in range(1,22)])
resnet20.update(dict.fromkeys(['conv'+str(i) for i in range(1,8)],  (32,32)))
resnet20.update(dict.fromkeys(['conv8', 'conv10'],  (32,16)))
resnet20['conv9'] = (16,16)
resnet20.update(dict.fromkeys(['conv'+str(i) for i in range(11,15)], (16,16)))
resnet20.update(dict.fromkeys(['conv15', 'conv17'],  (16,8)))
resnet20['conv16'] = (8,8)
resnet20.update(dict.fromkeys(['conv'+str(i) for i in range(18,22)], (8,8)))

# ResNet32
resnet32 = dict.fromkeys(['conv'+str(i) for i in range(1,34)])
resnet32.update(dict.fromkeys(['conv'+str(i) for i in range(1,12)],  (32,32)))
resnet32.update(dict.fromkeys(['conv12', 'conv14'],  (32,16)))
resnet32['conv13'] = (16,16)
resnet32.update(dict.fromkeys(['conv'+str(i) for i in range(15,23)], (16,16)))
resnet32.update(dict.fromkeys(['conv23', 'conv25'],  (16,8)))
resnet32['conv24'] = (8,8)
resnet32.update(dict.fromkeys(['conv'+str(i) for i in range(26,34)], (8,8)))

# ResNet32_BT
resnet32_bt = dict.fromkeys(['conv'+str(i) for i in range(1,50)])
resnet32_bt.update(dict.fromkeys(['conv'+str(i) for i in range(1,19)],  (32,32)))
resnet32_bt.update(dict.fromkeys(['conv19', 'conv21'],  (32,16)))
resnet32_bt['conv20'] = (16,16)
resnet32_bt.update(dict.fromkeys(['conv'+str(i) for i in range(22,35)], (16,16)))
resnet32_bt.update(dict.fromkeys(['conv35', 'conv37'],  (16,8)))
resnet32_bt['conv36'] = (8,8)
resnet32_bt.update(dict.fromkeys(['conv'+str(i) for i in range(38,50)], (8,8)))

# ResNet50_BT
resnet50_bt = dict.fromkeys(['conv'+str(i) for i in range(1,77)])
resnet50_bt.update(dict.fromkeys(['conv'+str(i) for i in range(1,28)],  (32,32)))
resnet50_bt.update(dict.fromkeys(['conv28', 'conv30'],  (32,16)))
resnet50_bt['conv29'] = (16,16)
resnet50_bt.update(dict.fromkeys(['conv'+str(i) for i in range(31,53)], (16,16)))
resnet50_bt.update(dict.fromkeys(['conv53', 'conv55'],  (16,8)))
resnet50_bt['conv54'] = (8,8)
resnet50_bt.update(dict.fromkeys(['conv'+str(i) for i in range(56,77)], (8,8)))

# ResNet56_BT
resnet56_bt = dict.fromkeys(['conv'+str(i) for i in range(1,86)])
resnet56_bt.update(dict.fromkeys(['conv'+str(i) for i in range(1,31)],  (32,32)))
resnet56_bt.update(dict.fromkeys(['conv31', 'conv33'],  (32,16)))
resnet56_bt['conv32'] = (16,16)
resnet56_bt.update(dict.fromkeys(['conv'+str(i) for i in range(34,59)], (16,16)))
resnet56_bt.update(dict.fromkeys(['conv59', 'conv61'],  (16,8)))
resnet56_bt['conv60'] = (8,8)
resnet56_bt.update(dict.fromkeys(['conv'+str(i) for i in range(62,86)], (8,8)))

######### ImageNet data #########

# ResNet50
resnet50 = dict.fromkeys(['conv'+str(i) for i in range(1,54)])
resnet50['conv1'] = (224,112)
resnet50.update(dict.fromkeys(['conv'+str(i) for i in range(2,13)],  (56,56)))
resnet50.update(dict.fromkeys(['conv13', 'conv15'],  (56,28)))
resnet50['conv14'] = (28,28)
resnet50.update(dict.fromkeys(['conv'+str(i) for i in range(16,26)], (28,28)))
resnet50.update(dict.fromkeys(['conv26', 'conv28'],  (28,14)))
resnet50['conv27'] = (14,14)
resnet50.update(dict.fromkeys(['conv'+str(i) for i in range(29,45)], (14,14)))
resnet50.update(dict.fromkeys(['conv45', 'conv47'],  (14,7)))
resnet50['conv46'] = (7,7)
resnet50.update(dict.fromkeys(['conv'+str(i) for i in range(48,54)], (7,7)))

# MobileNet (224)
mobilenet = dict.fromkeys(['conv'+str(i) for i in range(1, 28)])
mobilenet['conv1'] = (224,112)
mobilenet.update(dict.fromkeys(['conv'+str(i) for i in range(2,4)],  (112,112)))
mobilenet['conv4'] = (112,56)
mobilenet.update(dict.fromkeys(['conv'+str(i) for i in range(5,8)],  (56,56)))
mobilenet['conv8'] = (56,28)
mobilenet.update(dict.fromkeys(['conv'+str(i) for i in range(9,12)],  (28,28)))
mobilenet['conv12'] = (28,14)
mobilenet.update(dict.fromkeys(['conv'+str(i) for i in range(13,24)],  (14,14)))
mobilenet['conv24'] = (14,7)
mobilenet.update(dict.fromkeys(['conv'+str(i) for i in range(25,28)],  (7,7)))

vgg16 = {}                                                                                
vgg16['conv1'],  vgg16['conv2'] = (224, 224), (224, 224)                                  
vgg16['conv3'],  vgg16['conv4'] = (112, 112), (112, 112)                                  
vgg16['conv5'],  vgg16['conv6'],  vgg16['conv7']  = (56, 56), (56, 56), (56, 56)          
vgg16['conv8'],  vgg16['conv9'],  vgg16['conv10'] = (28, 28), (28, 28), (28, 28)          
vgg16['conv11'], vgg16['conv12'], vgg16['conv13'] = (14, 14), (14, 14), (14, 14)

cifar_feature_size = {
    'alexnet'       :alexnet,
    'vgg8'          :vgg8,
    'vgg8_bn_flat'  :vgg8,
    'vgg11'         :vgg11,
    'vgg11_bn_flat' :vgg11,
    'vgg13'         :vgg13,
    'vgg13_bn_flat' :vgg13,
    'resnet20_flat'      :resnet20,
    'resnet32_flat'      :resnet32,
    'resnet32_bt_flat'   :resnet32_bt,
    'resnet32_bt_flat_temp'   :resnet32_bt,
    'resnet50_bt_flat'   :resnet50_bt,
    'resnet56_bt_flat'   :resnet56_bt,
}

imagenet_feature_size = {
    'resnet50'      :resnet50,
    'resnet50_flat'      :resnet50,
    'resnet50_flat_01'      :resnet50,
    'mobilenet'     :mobilenet,
    'vgg16_flat'    :vgg16,
}


#scripts_util.py

# Base architecture

alexnet = {}
alexnet['conv1'] = [64,3,11,11]
alexnet['conv2'] = [192,64,5,5]
alexnet['conv3'] = [384,192,3,3]
alexnet['conv4'] = [256,384,3,3]
alexnet['conv5'] = [256,256,3,3]
alexnet['fc']    = [100,256]

vgg8 = {}
vgg8['conv1'] = [64,3,3,3]
vgg8['conv2'] = [128,64,3,3]
vgg8['conv3'] = [256,128,3,3]
vgg8['conv4'] = [512,256,3,3]
vgg8['conv5'] = [512,512,3,3]
vgg8['fc']    = [100,512]

vgg11 = {}
vgg11['conv1'] = [64,3,3,3]
vgg11['conv2'] = [128,64,3,3]
vgg11['conv3'] = [256,128,3,3]
vgg11['conv4'] = [256,256,3,3]
vgg11['conv5'] = [512,256,3,3]
vgg11['conv6'] = [512,512,3,3]
vgg11['conv7'] = [512,512,3,3]
vgg11['conv8'] = [512,512,3,3]
vgg11['fc']    = [100,512]

vgg13 = {}
vgg13['conv1']  = [64,3,3,3]
vgg13['conv2']  = [64,64,3,3]
vgg13['conv3']  = [128,64,3,3]
vgg13['conv4']  = [128,128,3,3]
vgg13['conv5']  = [256,128,3,3]
vgg13['conv6']  = [256,256,3,3]
vgg13['conv7']  = [512,256,3,3]
vgg13['conv8']  = [512,512,3,3]
vgg13['conv9']  = [512,512,3,3]
vgg13['conv10'] = [512,512,3,3]
vgg13['fc']     = [100,512]

resnet20 = {
    'conv1':[16,3,3,3], 'conv2':[16,16,3,3], 'conv3':[16,16,3,3], 'conv4':[16,16,3,3],
    'conv5':[16,16,3,3], 'conv6':[16,16,3,3], 'conv7':[16,16,3,3], 'conv8':[32,16,3,3],
    'conv9':[32,32,3,3], 'conv10':[32,16,3,3], 'conv11':[32,32,3,3], 'conv12':[32,32,3,3],
    'conv13':[32,32,1,1], 'conv14':[32,32,3,3], 'conv15':[64,32,3,3], 'conv16':[64,64,3,3],
    'conv17':[64,32,1,1], 'conv18':[64,64,3,3], 'conv19':[64,64,3,3], 'conv20':[64,64,3,3],
    'conv21':[64,64,3,3], 'fc':[100,64]
}

resnet32 = {
    'conv1':[16,3,3,3], 'conv2':[16,16,3,3], 'conv3':[16,16,3,3], 'conv4':[16,16,3,3],
    'conv5':[16,16,3,3], 'conv6':[16,16,3,3], 'conv7':[16,16,3,3], 'conv8':[16,16,3,3],
    'conv9':[16,16,3,3], 'conv10':[16,16,3,3], 'conv11':[16,16,3,3], 'conv12':[32,16,3,3],
    'conv13':[32,32,3,3], 'conv14':[32,16,1,1], 'conv15':[32,32,3,3], 'conv16':[32,32,3,3],
    'conv17':[32,32,3,3], 'conv18':[32,32,3,3], 'conv19':[32,32,3,3], 'conv20':[32,32,3,3],
    'conv21':[32,32,3,3], 'conv22':[32,32,3,3], 'conv23':[64,32,3,3], 'conv24':[64,64,3,3],
    'conv25':[64,32,1,1], 'conv26':[64,64,3,3], 'conv27':[64,64,3,3], 'conv28':[64,64,3,3],
    'conv29':[64,64,3,3], 'conv30':[64,64,3,3], 'conv31':[64,64,3,3], 'conv32':[64,64,3,3],
    'conv33':[64,64,3,3], 'fc':[100,64]
}

resnet32_bt = {
    'conv1':[16,3,3,3],   
    'conv2':[16,16,1,1],   'conv3':[16,16,3,3],   'conv4':[64,16,1,1],  'conv5':[64,16,1,1],
    'conv6':[16,64,1,1],   'conv7':[16,16,3,3],   'conv8':[64,16,1,1],
    'conv9':[16,64,1,1],   'conv10':[16,16,3,3],  'conv11':[64,16,1,1],
    'conv12':[16,64,1,1],  'conv13':[16,16,3,3],  'conv14':[64,16,1,1],
    'conv15':[16,64,1,1],  'conv16':[16,16,3,3],  'conv17':[64,16,1,1],
    'conv18':[32,64,1,1],  'conv19':[32,32,3,3],  'conv20':[128,32,1,1], 'conv21':[128,64,1,1],
    'conv22':[32,128,1,1], 'conv23':[32,32,3,3],  'conv24':[128,32,1,1],
    'conv25':[32,128,1,1], 'conv26':[32,32,3,3],  'conv27':[128,32,1,1],
    'conv28':[32,128,1,1], 'conv29':[32,32,3,3],  'conv30':[128,32,1,1],
    'conv31':[32,128,1,1], 'conv32':[32,32,3,3],  'conv33':[128,32,1,1],
    'conv34':[64,128,1,1], 'conv35':[64,64,3,3],  'conv36':[256,64,1,1], 'conv37':[256,128,1,1],
    'conv38':[64,256,1,1], 'conv39':[64,64,3,3],  'conv40':[256,64,1,1],
    'conv41':[64,256,1,1], 'conv42':[64,64,3,3],  'conv43':[256,64,1,1],
    'conv44':[64,256,1,1], 'conv45':[64,64,3,3],  'conv46':[256,64,1,1],
    'conv47':[64,256,1,1], 'conv48':[64,64,3,3],  'conv49':[256,64,1,1], 'fc':[100,256]
}

resnet50_bt = {
    'conv1':[16,3,3,3],   
    'conv2':[16,16,1,1],   'conv3':[16,16,3,3],   'conv4':[64,16,1,1],  'conv5':[64,16,1,1],
    'conv6':[16,64,1,1],   'conv7':[16,16,3,3],   'conv8':[64,16,1,1],
    'conv9':[16,64,1,1],   'conv10':[16,16,3,3],  'conv11':[64,16,1,1],
    'conv12':[16,64,1,1],  'conv13':[16,16,3,3],  'conv14':[64,16,1,1],
    'conv15':[16,64,1,1],  'conv16':[16,16,3,3],  'conv17':[64,16,1,1],
    'conv18':[16,64,1,1],  'conv19':[16,16,3,3],  'conv20':[64,16,1,1],
    'conv21':[16,64,1,1],  'conv22':[16,16,3,3],  'conv23':[64,16,1,1],
    'conv24':[16,64,1,1],  'conv25':[16,16,3,3],  'conv26':[64,16,1,1],
    'conv27':[32,64,1,1],  'conv28':[32,32,3,3],  'conv29':[128,32,1,1], 'conv30':[128,64,1,1],
    'conv31':[32,128,1,1], 'conv32':[32,32,3,3],  'conv33':[128,32,1,1],
    'conv34':[32,128,1,1], 'conv35':[32,32,3,3],  'conv36':[128,32,1,1],
    'conv37':[32,128,1,1], 'conv38':[32,32,3,3],  'conv39':[128,32,1,1],
    'conv40':[32,128,1,1], 'conv41':[32,32,3,3],  'conv42':[128,32,1,1],
    'conv43':[32,128,1,1], 'conv44':[32,32,3,3],  'conv45':[128,32,1,1],
    'conv46':[32,128,1,1], 'conv47':[32,32,3,3],  'conv48':[128,32,1,1],
    'conv49':[32,128,1,1], 'conv50':[32,32,3,3],  'conv51':[128,32,1,1],
    'conv52':[64,128,1,1], 'conv53':[64,64,3,3],  'conv54':[256,64,1,1], 'conv55':[256,128,1,1],
    'conv56':[64,256,1,1], 'conv57':[64,64,3,3],  'conv58':[256,64,1,1],
    'conv59':[64,256,1,1], 'conv60':[64,64,3,3],  'conv61':[256,64,1,1],
    'conv62':[64,256,1,1], 'conv63':[64,64,3,3],  'conv64':[256,64,1,1],
    'conv65':[64,256,1,1], 'conv66':[64,64,3,3],  'conv67':[256,64,1,1],
    'conv68':[64,256,1,1], 'conv69':[64,64,3,3],  'conv70':[256,64,1,1],
    'conv71':[64,256,1,1], 'conv72':[64,64,3,3],  'conv73':[256,64,1,1],
    'conv74':[64,256,1,1], 'conv75':[64,64,3,3],  'conv76':[256,64,1,1], 'fc':[100,256]
}

############## ImageNet ###############

resnet50 = {
        'conv1':[64,3,7,7],
        'conv2':[64,64,1,1],     'conv3':[64,64,3,3],     'conv4':[256,64,1,1],   'conv5':[256,64,1,1],
        'conv6':[64,256,1,1],    'conv7':[64,64,3,3],     'conv8':[256,64,1,1],
        'conv9':[64,256,1,1],    'conv10':[64,64,3,3],    'conv11':[256,64,1,1],
        'conv12':[128,256,1,1],  'conv13':[128,128,3,3],  'conv14':[512,128,1,1], 'conv15':[512,256,1,1],
        'conv16':[128,512,1,1],  'conv17':[128,128,3,3],  'conv18':[512,128,1,1],
        'conv19':[128,512,1,1],  'conv20':[128,128,3,3],  'conv21':[512,128,1,1],
        'conv22':[128,512,1,1],  'conv23':[128,128,3,3],  'conv24':[512,128,1,1],
        'conv25':[256,512,1,1],  'conv26':[256,256,3,3],  'conv27':[1024,256,1,1], 'conv28':[1024,512,1,1],
        'conv29':[256,1024,1,1], 'conv30':[256,256,3,3],  'conv31':[1024,256,1,1],
        'conv32':[256,1024,1,1], 'conv33':[256,256,3,3],  'conv34':[1024,256,1,1],
        'conv35':[256,1024,1,1], 'conv36':[256,256,3,3],  'conv37':[1024,256,1,1],
        'conv38':[256,1024,1,1], 'conv39':[256,256,3,3],  'conv40':[1024,256,1,1],
        'conv41':[256,1024,1,1], 'conv42':[256,256,3,3],  'conv43':[1024,256,1,1],
        'conv44':[512,1024,1,1], 'conv45':[512,512,3,3],  'conv46':[2048,512,1,1], 'conv47':[2048,1024,1,1],
        'conv48':[512,2048,1,1], 'conv49':[512,512,3,3],  'conv50':[2048,512,1,1], 
        'conv51':[512,2048,1,1], 'conv52':[512,512,3,3],  'conv53':[2048,512,1,1], 
        'fc':[1000,2048]
        }

mobilenet = {
        'conv1':[32, 3, 3, 3],     'conv2':[32, 1, 3, 3],     'conv3':[64, 32, 1, 1],    
        'conv4':[64, 1, 3, 3],     'conv5':[128, 64, 1, 1],   'conv6':[128, 1, 3, 3],    
        'conv7':[128, 128, 1, 1],  'conv8':[128, 1, 3, 3],    'conv9':[256, 128, 1, 1],  
        'conv10':[256, 1, 3, 3],    'conv11':[256, 256, 1, 1],  'conv12':[256, 1, 3, 3],    
        'conv13':[512, 256, 1, 1],  'conv14':[512, 1, 3, 3],    'conv15':[512, 512, 1, 1],  
        'conv16':[512, 1, 3, 3],    'conv17':[512, 512, 1, 1],  'conv18':[512, 1, 3, 3],    
        'conv19':[512, 512, 1, 1],  'conv20':[512, 1, 3, 3],    'conv21':[512, 512, 1, 1],  
        'conv22':[512, 1, 3, 3],    'conv23':[512, 512, 1, 1],  'conv24':[512, 1, 3, 3],    
        'conv25':[1024, 512, 1, 1], 'conv26':[1024, 1, 3, 3],   'conv27':[1024, 1024, 1, 1],
        'fc':[1000, 1024],
        }

base_archs = {
        'alexnet'       :alexnet,
        'vgg8'          :vgg8,
        'vgg11'         :vgg11,
        'vgg13'         :vgg13,
        'resnet20_flat'      :resnet20,
        'resnet32_flat'      :resnet32,
        'resnet32_bt_flat'   :resnet32_bt,
        'resnet32_bt_flat_temp'   :resnet32_bt,
        'resnet50_bt_flat'   :resnet50_bt,
        'resnet50'           :resnet50,
        'mobilenet'          :mobilenet,
        }


