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

class layerUtil():
  def __init__(self, model, dense_chs):
    self.setModel(model, dense_chs)

  @classmethod
  def setModel(cls, model, dense_chs):
    cls.model = model
    cls.dense_chs = dense_chs

  @classmethod
  def getLayerDef(cls, arch):
    if   'conv' in arch['name']:  return cls.convLayer(arch['name'], arch)
    elif 'bn' in arch['name']:    return cls.bnLayer(arch['name'])
    elif 'relu' == arch['name']:  return cls.reluLayer()
    elif 'pool' == arch['name']:  return cls.poolLayer(arch)
    elif 'fc'   in arch['name']:  return cls.fcLayer(arch['name'], arch)
    elif 'avgpool' == arch['name']: return cls.avgPool(arch)
    elif 'avgpool_adt' == arch['name']: return cls.avgPoolAdt(arch)
    elif 'maxpool' == arch['name']: return cls.maxPool(arch)
    elif 'dropout' == arch['name']:  return cls.dropoutLayer()

  @classmethod
  def convLayer(cls, name, lyr_info):
    param_name = 'module.'+name+'.weight'
    if param_name in cls.model.state_dict():
      param         = cls.model.state_dict()[param_name]
      dims          = list(param.shape)
      in_chs        = str(dims[1])
      out_chs       = str(dims[0])
      kernel_size   = str(lyr_info['kernel_size'])
      stride        = str(lyr_info['stride'])
      padding       = str(lyr_info['padding'])
      bias          = lyr_info['bias'] if 'bias' in lyr_info else True

      # Depth-wise convolution
      if 'dw' in lyr_info:
          if lyr_info['dw']:
            # - Depth-wise convolution has same number of input/output channels
            # - Depth-wise convolution require 'group' information
            return '\t\tself.{} = nn.Conv2d({}, {}, kernel_size={}, stride={}, padding={}, bias={}, groups={})\n'.format(
                    name, out_chs, out_chs, kernel_size, stride, padding, bias, out_chs)
      return '\t\tself.{} = nn.Conv2d({}, {}, kernel_size={}, stride={}, padding={}, bias={})\n'.format(
          name, in_chs, out_chs, kernel_size, stride, padding, bias)
    else:
      return ''

  @classmethod
  def fcLayer(cls, name, lyr_info):
    param = cls.model.state_dict()['module.'+name+'.weight']
    dims = list(param.shape)
    in_chs, out_chs = str(dims[1]), str(dims[0])
    if name in ['fc', 'fc3']:
      return '\t\tself.{} = nn.Linear({}, num_classes)\n'.format(name, in_chs)
    else:
      return '\t\tself.{} = nn.Linear({}, {})\n'.format(name, in_chs, out_chs)

  @classmethod
  def bnLayer(cls, name):
    param_name = 'module.'+name+'.weight'
    if param_name in cls.model.state_dict():
      param = cls.model.state_dict()[param_name]
      dims = list(param.shape)
      out_chs = str(dims[0])
      return '\t\tself.{} = nn.BatchNorm2d({})\n'.format(name, out_chs)
    else:
      return ''

  @classmethod
  def poolLayer(cls, lyr_info):
    kernel_size = lyr_info['kernel_size']
    stride      = lyr_info['stride']
    padding     = lyr_info['padding'] if 'padding' in lyr_info else None

    if padding == None:
      return '\t\tself.pool = nn.MaxPool2d(kernel_size={}, stride={})\n'.format(
              kernel_size, stride)
    else:
      return '\t\tself.pool = nn.MaxPool2d(kernel_size={}, stride={}, padding={})\n'.format(
              kernel_size, stride, padding)

  @classmethod
  def resnet_module(cls, chs_map, is_gating, lyr1, lyr2, lyr3=None):
    conv1, conv2 = 'conv'+str(lyr1), 'conv'+str(lyr2)
    bn1, bn2     = 'bn'+str(lyr1), 'bn'+str(lyr2)

    if lyr3 != None:
      conv3, bn3 = 'conv'+str(lyr3), 'bn'+str(lyr3)

    # Any one of convolution layers is removed => Remove the residual path
    params = ['module.'+conv1+'.weight', 'module.'+conv2+'.weight']
    if lyr3 != None:
      params.append('module.'+conv3+'.weight')

    if any(i for i in params if i not in cls.model.state_dict()):
      return ''
    else:
      if is_gating:
        ctx = cls.mask(conv1, chs_map, i='_x')
        ctx += cls.forward(conv1)
        ctx += cls.forward(bn1)
        ctx += cls.forward('relu')
        ctx += cls.forward(conv2)
        ctx += cls.forward(bn2)
        if lyr3 != None:
          ctx += cls.forward('relu')
          ctx += cls.forward(conv3)
          ctx += cls.forward(bn3)
          ctx += cls.merge(conv3, chs_map)
        else:
          ctx += cls.merge(conv2, chs_map)
        ctx += cls.sum()
        ctx += cls.forward('relu', i='_x', o='_x')

      else:
        ctx = cls.forward(conv1, i='_x')
        ctx += cls.forward(bn1)
        ctx += cls.forward('relu')
        ctx += cls.forward(conv2)
        ctx += cls.forward(bn2)
        if lyr3 != None:
          ctx += cls.forward('relu')
          ctx += cls.forward(conv3)
          ctx += cls.forward(bn3)
        ctx += cls.sum()
        ctx += cls.forward('relu', i='_x', o='_x')
      return ctx

  @classmethod
  def resnet_module_pool(cls, chs_map1, chs_map2, is_gating, lyr1, lyr2, lyr3, lyr4=None):
    conv1, conv2, conv3 = 'conv'+str(lyr1), 'conv'+str(lyr2), 'conv'+str(lyr3)
    bn1, bn2, bn3       = 'bn'+str(lyr1), 'bn'+str(lyr2), 'bn'+str(lyr3)

    if lyr4 != None:
      conv4, bn4 = 'conv'+str(lyr4), 'bn'+str(lyr4)

    params = ['module.'+conv1+'.weight', 'module.'+conv2+'.weight']
    if lyr4 != None:
      params.append('module.'+conv3+'.weight')
    no_res = any(i for i in params if i not in cls.model.state_dict())

    if no_res:
      ctx1 = ''
    else:
      if is_gating:
        ctx1 = cls.mask(conv1, chs_map1, i='_x')
        ctx1 += cls.forward(conv1)
        ctx1 += cls.forward(bn1)
        ctx1 += cls.forward('relu')
        ctx1 += cls.forward(conv2)
        ctx1 += cls.forward(bn2)
        if lyr4 != None:
          ctx1 += cls.forward('relu')
          ctx1 += cls.forward(conv3)
          ctx1 += cls.forward(bn3)
          ctx1 += cls.merge(conv3, chs_map2)
        else:
          ctx1 += cls.merge(conv2, chs_map2)

      else:
        ctx1 = cls.forward(conv1, i='_x')
        ctx1 += cls.forward(bn1)
        ctx1 += cls.forward('relu')
        ctx1 += cls.forward(conv2)
        ctx1 += cls.forward(bn2)
        if lyr4 != None:
          ctx1 += cls.forward('relu')
          ctx1 += cls.forward(conv3)
          ctx1 += cls.forward(bn3)

    if lyr4 != None:
      conv_short, bn_short = conv4, bn4
    else:
      conv_short, bn_short = conv3, bn3

    if is_gating:
      ctx2 = cls.mask(conv_short, chs_map1, i='_x', o='_x')
      ctx2 += cls.forward(conv_short, i='_x', o='_x')
      ctx2 += cls.forward(bn_short, i='_x', o='_x')
      ctx2 += cls.empty_ch(i='_x')
      ctx2 += cls.merge(conv_short, chs_map2, i='_x', o='_x')
    else:
      ctx2 = cls.forward(conv_short, i='_x', o='_x')
      ctx2 += cls.forward(bn_short, i='_x', o='_x')

    ctx3 = ctx1 + ctx2
    ctx3 += '' if no_res else cls.sum() 
    ctx3 += cls.forward('relu', i='_x', o='_x')
    return ctx3

  @classmethod
  def avgPool(cls, lyr_info):
    return '\t\tself.avgpool = nn.AvgPool2d({})\n'.format(lyr_info['num'])

  @classmethod
  def avgPoolAdt(cls, lyr_info):
    return '\t\tself.avgpool = nn.AdaptiveAvgPool2d({})\n'.format(lyr_info['num'])

  @classmethod
  def maxPool(cls, lyr_info):
    return '\t\tself.maxpool = nn.MaxPool2d(kernel_size={}, stride={}, padding={})\n'.format(
      lyr_info['kernel_size'], lyr_info['stride'], lyr_info['padding'])

    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

  @classmethod
  def reluLayer(cls):
    return '\t\tself.relu = nn.ReLU(inplace=True)\n'

  @classmethod                                                                            
  def dropoutLayer(cls):                                                                  
    return '\t\tself.dropout = nn.Dropout()\n'

  @classmethod
  def mask(cls, layer, chs_map, i='x', o='x'):
    indices = []
    # Get index to the dense channels
    for ich in cls.dense_chs[cls.n(layer)]['in_chs']:
      indices.append(chs_map[ich])

    ctx = "\t\t{} = torch.index_select({}, 1, torch.tensor({}).cuda())\n".format(
          o, i, indices)
    return ctx

  @classmethod
  def empty_ch(cls, i='x', o='__x'):
    return '\t\t{} = torch.full([{}.size()[0], {}.size()[2], {}.size()[3]], 0).cuda()\n'.format(
                      o, i, i, i)

  @classmethod
  def merge(cls, layer, chs_map, i='x', o='x'):
    stack = '\t\t{} =torch.stack(['.format(o)
    idx = 0
    for och in sorted(chs_map):
      if och in cls.dense_chs[cls.n(layer)]['out_chs']:
        stack += '{}[:,{},:,:], '.format(i, idx)
        idx +=1
      else:
        stack += '__x, '
    stack += '], dim=1)\n'
    return stack

  @classmethod
  def forward(cls, name, i='x', o='x'):
    return '\t\t{} = self.{}({})\n'.format(o, name, i)

  @classmethod
  def sum(cls, i1='_x', i2='x', o='_x'):
    return '\t\t{} = {} + {}\n'.format(o, i1, i2)

  @classmethod
  def n(cls, name):
    return 'module.'+name+'.weight'
