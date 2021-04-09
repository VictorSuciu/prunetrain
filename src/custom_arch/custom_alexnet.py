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
arch[0] = {'name':'conv1', 'kernel_size':11, 'stride':4, 'padding':5, 'bias':True}
arch[1] = {'name':'conv2', 'kernel_size':5,  'stride':1, 'padding':2, 'bias':True}
arch[2] = {'name':'conv3', 'kernel_size':3,  'stride':1, 'padding':1, 'bias':True}
arch[3] = {'name':'conv4', 'kernel_size':3,  'stride':1, 'padding':1, 'bias':True}
arch[4] = {'name':'conv5', 'kernel_size':3,  'stride':1, 'padding':1, 'bias':True}
arch[5] = {'name':'pool', 'kernel_size':2, 'stride':2}
arch[6] = {'name':'relu'}
arch[7] = {'name':'fc', 'out_chs':'num_classes'}

def _genDenseArchAlexNet(model, out_f_dir1, out_f_dir2, arch_name, dense_chs, chs_map, is_gating=False):

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += '__all__ = [\'alexnet_flat\']\n'
  ctx += 'class AlexNet(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=10):\n'
  ctx += '\t\tsuper(AlexNet, self).__init__()\n'

  lyr = layerUtil(model, dense_chs)

  # Layer definition
  for idx in sorted(arch):
    ctx += lyr.getLayerDef(arch[idx])

  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')
  ctx += lyr.forward('conv2')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')
  ctx += lyr.forward('conv3')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv4')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv5')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += forward('fc')
  ctx += '\t\treturn x\n'

  # AlexNet definition
  ctx += 'def alexnet_flat(**kwargs):\n'
  ctx += '\tmodel = AlexNet(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir2):
      os.makedirs(out_f_dir2)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join(out_f_dir1, 'alexnet_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir2, arch_name),'w')
  f_out2.write(ctx)