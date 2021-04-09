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
from .arch_utils import layerUtil

k7_s2_p3 = [1]
k3_s2_p1 = [13, 26, 45]
k1_s2_p0 = [15, 28, 47]
k3_s1_p1 = [3, 7, 10, 17, 20, 23, 30, 33, 36, 39, 42, 49, 52]

arch = {}
for i in range(1, 54):
  conv_idx = (i-1)*2
  bn_idx   = conv_idx +1

  if i in k7_s2_p3:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':7, 'stride':2, 'padding':3, 'bias':False}
  elif i in k3_s2_p1:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}
  elif i in k1_s2_p0:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':1, 'stride':2, 'padding':0, 'bias':False}
  elif i in k3_s1_p1:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
  else:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':1, 'stride':1, 'padding':0, 'bias':False}
  arch[bn_idx] = {'name':'bn'+str(i)}

arch[106] = {'name':'avgpool_adt', 'num':(1,1)}
arch[107] = {'name':'maxpool', 'kernel_size':3, 'stride':2, 'padding':1}
arch[108] = {'name':'relu'}
arch[109] = {'name':'fc', 'out_chs':'num_classes'}

def _genDenseArchResNet50(model, out_f_dir1, out_f_dir2, arch_name, dense_chs, chs_map, is_gating=False):
  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += 'import torch\n'
  ctx += '__all__ = [\'resnet50_flat\']\n'
  ctx += 'class ResNet50(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=1000):\n'
  ctx += '\t\tsuper(ResNet50, self).__init__()\n'

  lyr = layerUtil(model, dense_chs)

  # Layer definition
  for idx in sorted(arch):
    ctx += lyr.getLayerDef(arch[idx])

  # Architecture sequential
  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1')
  ctx += lyr.forward('bn1')
  ctx += lyr.forward('relu')
  ctx += '\t\t_x = self.maxpool(x)\n'

  if chs_map != None: 
    chs_map0, chs_map1, chs_map2, chs_map3, chs_map4 = chs_map[0], chs_map[1], chs_map[2], chs_map[3], chs_map[4]
  else:               
    chs_map0, chs_map1, chs_map2, chs_map3, chs_map4 = None, None, None, None, None

  if is_gating:
    ctx += lyr.empty_ch(i='_x')
    ctx += lyr.merge('conv1', chs_map0, i='_x', o='_x')

  ctx += lyr.resnet_module_pool(chs_map0, chs_map1, is_gating, 2,3,4,5) #1
  ctx += lyr.resnet_module(chs_map1, is_gating, 6,7,8) #2
  ctx += lyr.resnet_module(chs_map1, is_gating, 9,10,11) #3

  ctx += lyr.resnet_module_pool(chs_map1, chs_map2, is_gating, 12,13,14,15) #9
  ctx += lyr.resnet_module(chs_map2, is_gating, 16,17,18) #10
  ctx += lyr.resnet_module(chs_map2, is_gating, 19,20,21) #11
  ctx += lyr.resnet_module(chs_map2, is_gating, 22,23,24) #12

  ctx += lyr.resnet_module_pool(chs_map2, chs_map3, is_gating, 25,26,27,28) #17
  ctx += lyr.resnet_module(chs_map3, is_gating, 29,30,31) #18
  ctx += lyr.resnet_module(chs_map3, is_gating, 32,33,34) #19
  ctx += lyr.resnet_module(chs_map3, is_gating, 35,36,37) #20
  ctx += lyr.resnet_module(chs_map3, is_gating, 38,39,40) #21
  ctx += lyr.resnet_module(chs_map3, is_gating, 41,42,43) #21

  ctx += lyr.resnet_module_pool(chs_map3, chs_map4, is_gating, 44,45,46,47) #17
  ctx += lyr.resnet_module(chs_map4, is_gating, 48,49,50) #18
  ctx += lyr.resnet_module(chs_map4, is_gating, 51,52,53) #19

  if is_gating:
    ctx += lyr.mask('fc', chs_map2, i='_x', o='_x')

  ctx += '\t\tx = self.avgpool(_x)\n'
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # ResNet50 definition
  ctx += 'def resnet50_flat(**kwargs):\n'
  ctx += '\tmodel = ResNet50(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir2):
      os.makedirs(out_f_dir2)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join(out_f_dir1, 'resnet50_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir2, arch_name),'w')
  f_out2.write(ctx)
