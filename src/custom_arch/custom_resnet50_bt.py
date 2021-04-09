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

k3_s2_p1 = [28, 53]
k1_s2_p0 = [30, 55]
k3_s1_p1 = [1, 3, 7, 10, 13, 16, 19, 22, 25, 28, 32, 35, 38, 41, 44, 47, 50, 
            53, 57, 60, 63, 66, 69, 72, 75]

arch = {}
for i in range(1, 77):
  conv_idx = (i-1)*2
  bn_idx   = conv_idx +1

  if i in k3_s2_p1:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}
  elif i in k1_s2_p0:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':1, 'stride':2, 'padding':0, 'bias':False}
  elif i in k3_s1_p1:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
  else:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':1, 'stride':1, 'padding':0, 'bias':False}
  arch[bn_idx] = {'name':'bn'+str(i)}

arch[152] = {'name':'avgpool', 'num':8}
arch[153] = {'name':'relu'}
arch[154] = {'name':'fc', 'out_chs':'num_classes'}

def _genDenseArchResNet50BT(model, out_f_dir1, out_f_dir2, arch_name, dense_chs, chs_map, is_gating=False):
  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += 'import torch\n'
  ctx += '__all__ = [\'resnet50_bt_flat\']\n'
  ctx += 'class ResNet50BT(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(ResNet50BT, self).__init__()\n'

  lyr = layerUtil(model, dense_chs)

  # Layer definition
  for idx in sorted(arch):
    ctx += lyr.getLayerDef(arch[idx])

  # Architecture sequential
  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1')
  ctx += lyr.forward('bn1')
  ctx += lyr.forward('relu', o='_x')

  if chs_map != None: chs_map0, chs_map1, chs_map2 = chs_map[0], chs_map[1], chs_map[2]
  else:               chs_map0, chs_map1, chs_map2 = None, None, None

  if is_gating:
    ctx += lyr.empty_ch(i='_x')
    ctx += lyr.merge('conv1', chs_map0, i='_x', o='_x')

  ctx += lyr.resnet_module_pool(chs_map0, chs_map1, is_gating, 2,3,4,5) #1
  ctx += lyr.resnet_module(chs_map0, is_gating, 6,7,8) #2
  ctx += lyr.resnet_module(chs_map0, is_gating, 9,10,11) #3
  ctx += lyr.resnet_module(chs_map0, is_gating, 12,13,14) #4
  ctx += lyr.resnet_module(chs_map0, is_gating, 15,16,17) #5
  ctx += lyr.resnet_module(chs_map0, is_gating, 18,19,20) #6
  ctx += lyr.resnet_module(chs_map0, is_gating, 21,22,23) #7
  ctx += lyr.resnet_module(chs_map0, is_gating, 24,25,26) #8

  ctx += lyr.resnet_module_pool(chs_map0, chs_map1, is_gating, 27,28,29,30) #9
  ctx += lyr.resnet_module(chs_map1, is_gating, 31,32,33) #10
  ctx += lyr.resnet_module(chs_map1, is_gating, 34,35,36) #11
  ctx += lyr.resnet_module(chs_map1, is_gating, 37,38,39) #12
  ctx += lyr.resnet_module(chs_map1, is_gating, 40,41,42) #13
  ctx += lyr.resnet_module(chs_map1, is_gating, 43,44,45) #14
  ctx += lyr.resnet_module(chs_map1, is_gating, 46,47,48) #15
  ctx += lyr.resnet_module(chs_map1, is_gating, 49,50,51) #16

  ctx += lyr.resnet_module_pool(chs_map1, chs_map2, is_gating, 52,53,54,55) #17
  ctx += lyr.resnet_module(chs_map2, is_gating, 56,57,58) #18
  ctx += lyr.resnet_module(chs_map2, is_gating, 59,60,61) #19
  ctx += lyr.resnet_module(chs_map2, is_gating, 62,63,64) #20
  ctx += lyr.resnet_module(chs_map2, is_gating, 65,66,67) #21
  ctx += lyr.resnet_module(chs_map2, is_gating, 68,69,70) #22
  ctx += lyr.resnet_module(chs_map2, is_gating, 71,72,73) #23
  ctx += lyr.resnet_module(chs_map2, is_gating, 74,75,76) #24

  if is_gating:
    ctx += lyr.mask('fc', chs_map2, i='_x', o='_x')

  ctx += '\t\tx = self.avgpool(_x)\n'
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # ResNet50BT definition
  ctx += 'def resnet50_bt_flat(**kwargs):\n'
  ctx += '\tmodel = ResNet50BT(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir2):
      os.makedirs(out_f_dir2)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join(out_f_dir1, 'resnet50_bt_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir2, arch_name),'w')
  f_out2.write(ctx)
