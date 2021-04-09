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

arch = {}
for i in range(1, 9):
    conv_idx = (i-1)*2
    bn_idx = conv_idx +1
    arch[conv_idx] = {'name':'conv'+str(i) , 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
    arch[bn_idx] = {'name':'bn'+str(i)}

arch[16] = {'name':'pool', 'kernel_size':2, 'stride':2}
arch[17] = {'name':'relu'}
arch[18] = {'name':'fc', 'out_chs':'num_classes'}

def _genDenseArchVGG11BN(model, out_f_dir1, out_f_dir2, arch_name, dense_chs, chs_map=None):

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += '__all__ = [\'vgg11_bn_flat\']\n'
  ctx += 'class VGG11(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(VGG11, self).__init__()\n'

  lyr = layerUtil(model, dense_chs)

  # Layer definition
  for idx in sorted(arch):
    ctx += lyr.getLayerDef(arch[idx])

  # Architecture sequential
  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1')
  ctx += lyr.forward('bn1')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv2')
  ctx += lyr.forward('bn2')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv3')
  ctx += lyr.forward('bn3')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv4')
  ctx += lyr.forward('bn4')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv5')
  ctx += lyr.forward('bn5')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv6')
  ctx += lyr.forward('bn6')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv7')
  ctx += lyr.forward('bn7')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv8')
  ctx += lyr.forward('bn8')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # AlexNet definition
  ctx += 'def vgg11_bn_flat(**kwargs):\n'
  ctx += '\tmodel = VGG11(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir2):
      os.makedirs(out_f_dir2)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join(out_f_dir1, 'vgg11_bn_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir2, arch_name),'w')
  f_out2.write(ctx)

