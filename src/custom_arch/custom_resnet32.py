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

k3_s2_p1 = [12, 23]
k1_s2_p0 = [14, 25]

arch = {}
for i in range(1, 34):
  conv_idx = (i-1)*2
  bn_idx = conv_idx +1

  if i in k3_s2_p1:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}
  elif i in k1_s2_p0:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':1, 'stride':2, 'padding':0, 'bias':False}
  else:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
  arch[bn_idx] = {'name':'bn'+str(i)}

arch[66] = {'name':'avgpool', 'num':8}
arch[67] = {'name':'relu'}
arch[68] = {'name':'fc', 'out_chs':'num_classes'}

def _genDenseArchResNet32(model, out_f_dir1, out_f_dir2, arch_name, dense_chs, chs_map, is_gating=False):

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += '__all__ = [\'resnet32_flat\']\n'
  ctx += 'class ResNet32(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(ResNet32, self).__init__()\n'

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

  ctx += lyr.resnet_module(chs_map0, is_gating, 2,3) #1
  ctx += lyr.resnet_module(chs_map0, is_gating, 4,5) #2
  ctx += lyr.resnet_module(chs_map0, is_gating, 6,7) #3
  ctx += lyr.resnet_module(chs_map0, is_gating, 8,9) #4
  ctx += lyr.resnet_module(chs_map0, is_gating, 10,11) #5

  ctx += lyr.resnet_module_pool(chs_map0, chs_map1, is_gating, 12,13,14) #6
  ctx += lyr.resnet_module(chs_map1, is_gating, 15,16) #7
  ctx += lyr.resnet_module(chs_map1, is_gating, 17,18) #8
  ctx += lyr.resnet_module(chs_map1, is_gating, 19,20) #9
  ctx += lyr.resnet_module(chs_map1, is_gating, 21,22) #10

  ctx += lyr.resnet_module_pool(chs_map1, chs_map2, is_gating, 23,24,25) #11
  ctx += lyr.resnet_module(chs_map2, is_gating, 26,27) #12
  ctx += lyr.resnet_module(chs_map2, is_gating, 28,29) #13
  ctx += lyr.resnet_module(chs_map2, is_gating, 30,31) #14
  ctx += lyr.resnet_module(chs_map2, is_gating, 32,33) #15

  if is_gating:
    ctx += lyr.mask('fc', chs_map2, i='_x', o='_x')

  ctx += '\t\tx = self.avgpool(_x)\n'
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # ResNet32 definition
  ctx += 'def resnet32_flat(**kwargs):\n'
  ctx += '\tmodel = ResNet32(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir2):
      os.makedirs(out_f_dir2)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join(out_f_dir1, 'resnet32_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir2, arch_name),'w')
  f_out2.write(ctx)
