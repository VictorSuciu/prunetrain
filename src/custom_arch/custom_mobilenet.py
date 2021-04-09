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

k3_s2_p1 = [4, 8, 12, 24]

arch = {}
arch[0] = {'name':'conv1' , 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}
arch[1] = {'name':'bn1'}

# Sequence of depth-wise convolutions
for i in range(2, 27, 2):
  conv_idx = (i-1)*2
  bn_idx   = conv_idx +1

  # Depth-wise convolution
  if i in k3_s2_p1:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False, 'dw':True}
  else:
    arch[conv_idx] = {'name':'conv'+str(i), 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False, 'dw':True}
  arch[bn_idx] = {'name':'bn'+str(i)}

  # 1x1 convolution
  arch[conv_idx +2] = {'name':'conv'+str(i+1), 'kernel_size':1, 'stride':1, 'padding':0, 'bias':False}
  arch[bn_idx +2] = {'name':'bn'+str(i+1)}


arch[54] = {'name':'avgpool', 'num':7}
arch[55] = {'name':'relu'}
arch[28] = {'name':'fc', 'out_chs':1000}

"""
Generate dense VGG11_BN architecture
- Only input/output channel number change
"""
def _genDenseArchMobileNet(model, out_f_dir1, out_f_dir2, arch_name, dense_chs, chs_map=None):

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += '__all__ = [\'mobilenet\']\n'
  ctx += 'class MobileNet(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=1000):\n'
  ctx += '\t\tsuper(MobileNet, self).__init__()\n'

  lyr = layerUtil(model, dense_chs)

  # Layer definition
  for idx in sorted(arch):
    ctx += lyr.getLayerDef(arch[idx])

  # Architecture sequential
  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1')
  ctx += lyr.forward('bn1')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv2')
  ctx += lyr.forward('bn2')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv3')
  ctx += lyr.forward('bn3')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv4')
  ctx += lyr.forward('bn4')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv5')
  ctx += lyr.forward('bn5')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv6')
  ctx += lyr.forward('bn6')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv7')
  ctx += lyr.forward('bn7')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv8')
  ctx += lyr.forward('bn8')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv9')
  ctx += lyr.forward('bn9')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv10')
  ctx += lyr.forward('bn10')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv11')
  ctx += lyr.forward('bn11')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv12')
  ctx += lyr.forward('bn12')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv13')
  ctx += lyr.forward('bn13')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv14')
  ctx += lyr.forward('bn14')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv15')
  ctx += lyr.forward('bn15')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv16')
  ctx += lyr.forward('bn16')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv17')
  ctx += lyr.forward('bn17')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv18')
  ctx += lyr.forward('bn18')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv19')
  ctx += lyr.forward('bn19')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv20')
  ctx += lyr.forward('bn20')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv21')
  ctx += lyr.forward('bn21')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv22')
  ctx += lyr.forward('bn22')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv23')
  ctx += lyr.forward('bn23')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv24')
  ctx += lyr.forward('bn24')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv25')
  ctx += lyr.forward('bn25')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('conv26')
  ctx += lyr.forward('bn26')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv27')
  ctx += lyr.forward('bn27')
  ctx += lyr.forward('relu')

  ctx += lyr.forward('avgpool')
  ctx += '\t\tx = x.view(-1, x.size(1))\n'
  ctx += lyr.forward('fc')
  ctx += '\t\treturn x\n'

  # AlexNet definition
  ctx += 'def mobilenet(**kwargs):\n'
  ctx += '\tmodel = MobileNet(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir2):
      os.makedirs(out_f_dir2)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join(out_f_dir1, 'mobilenet.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir2, arch_name),'w')
  f_out2.write(ctx)

