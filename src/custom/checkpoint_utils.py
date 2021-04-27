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

import os, sys
from collections import defaultdict
import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
import torch.nn as nn

import heapq

sys.path.append('..')
import models.cifar as models_cifar
import models.imagenet as models_imagenet
from .resnet_stages import *
from .rm_layers import getRmLayers

# Packages to calculate inference cost
from scripts.feature_size_cifar import cifar_feature_size, imagenet_feature_size

WORD_SIZE = 4
MFLOPS = 1000000/2

class Checkpoint():
  def __init__(self, arch, dataset, model_path, num_classes, depth=None):
    #print("{}, {}".format(models.__dict__, arch))
    self.arch = arch
    if dataset == 'imagnet':
      self.model = models_imagenet.__dict__[arch]()
    else:
      self.model = models_cifar.__dict__[arch](num_classes=num_classes)
    self.model = torch.nn.DataParallel(self.model)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    self.model.load_state_dict(checkpoint['state_dict'])
    self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.005)
    self.optimizer.load_state_dict(checkpoint['optimizer'])
    self.epoch = checkpoint['epoch']

  def getEpoch(self):
    return self.epoch

  def printParams(self):
    print ("[INFO] print learning parameters")
    for name, param in self.model.named_parameters():
      print("{}: {}".format(name, list(param.shape)))

  def getConvStructSparsity(self, threshold, file_name=None, arch=None, dataset='imagenet'):
    return _getConvStructSparsity(self.model, threshold, file_name, self.arch, dataset)

  def getFilterData(self, target_lyr):
    return _getFilterData(self.model, target_lyr)


def third_largest(numbers):
  return heapq.nlargest(3, numbers)[2]


""" Return 1D list of weights for the target layer
"""
def _getFilterData(model, target_lyr):
  fil_data = {}
  for name, param in model.named_parameters():
    if ('weight' in name) and ('conv' in name):
      if any(i for i in target_lyr if i in name):
        dims = list(param.shape)
        chs = []
        for out_ch in range(dims[0]):
          chs.append(param.data[out_ch,:,:,:].numpy().flatten())
        fil_data[name] = chs
  return fil_data


""" Return
1. All layers' sparsity heat-map of filters (input channels / output channels)
2. Output channel sparsity by epoch
"""
def _getConvStructSparsity(model, threshold, file_name, arch, dataset):
  conv_struct_density = {}
  conv_rand_density = {}
  sparse_bi_map = {}
  sparse_val_map = {}
  conv_id = 0
  model_size = 0
  acc_inf_cost = 0

  if dataset == 'imagenet':
    fmap = imagenet_feature_size[arch]
  else:
    fmap = cifar_feature_size[arch]
  
  tot_weights = 0

  for name, param in model.named_parameters():
    if ('weight' in name) and ('conv' in name or 'fc' in name):
      # Filter sparsity graph: Row(in_chs), Col(out_chs)
      # Tensor dims = [out_chs, in_chs, fil_height, fil_width]
      layer = []
      dims = list(param.shape)

      if len(dims) == 4:
        channel_map = np.zeros([dims[1], dims[0]])
        filter_size = dims[2] * dims[3]
        for in_ch in range(dims[1]):
          fil_row = []
          for out_ch in range(dims[0]):
            fil = param.data.numpy()[out_ch,in_ch,:,:]
            fil_max = np.absolute(fil).max()
            fil_row.append(fil_max)
            #if fil_max > threshold:
            if fil_max > 0.:
              channel_map[in_ch, out_ch] = 1
          layer.append(fil_row)

      elif len(dims) == 2:
        channel_map = np.zeros([dims[1], dims[0]])
        filter_size = 1
        for in_ch in range(dims[1]):
          fil_row = []
          for out_ch in range(dims[0]):
            fil = param.data.numpy()[out_ch,in_ch]
            fil_max = np.absolute(fil)
            fil_row.append(fil_max)
            #if fil_max > threshold:
            if fil_max > 0.:
              channel_map[in_ch, out_ch] = 1
          layer.append(fil_row)

      # ratio of non_zero weights
      weights = param.data.numpy()
      weight_density = float(weights[weights >= threshold].size) / weights.size

      tot_weights += weights[weights >= threshold].size

      sparse_val_map[conv_id] = np.array(layer)
      sparse_bi_map[conv_id] = channel_map

      rows = channel_map.max(axis=1) # in_channels
      cols = channel_map.max(axis=0) # out_channels

#      if 'conv' in name or 'fc' in name:
#        if file_name != None:
#          out_file.write("\n{}:{}, ".format(name, 'i_ch'))
#          for row in rows:
#            out_file.write("{},".format(row))
#          out_file.write("\n{}:{}, ".format(name, 'o_ch'))
#          for col in cols:
#            out_file.write("{},".format(col))
#        else:
#          print ("{}:{}, {}".format(name, 'i_ch', rows))
#          print ("{}:{}, {}".format(name, 'o_ch', cols))

      num_dense_out_ch = float(np.count_nonzero(cols))
      num_dense_in_ch = float(np.count_nonzero(rows))

      out_density =  num_dense_out_ch / len(cols)
      in_density = num_dense_in_ch / len(rows)

      conv_struct_density[conv_id] = {'in_ch':in_density, 'out_ch':out_density}
      conv_rand_density[conv_id] = weight_density
      #print("{}: {}".format(name, weight_density))

      model_size += num_dense_out_ch * num_dense_in_ch * filter_size # Add filters
      model_size += num_dense_out_ch  # Add bias

      # Calculate inference cost = (CRS)(K)(NPQ)
      fmap_name = name.split('module.')[1].split('.weight')[0]
      if len(dims) == 4:
        inf_cost = (num_dense_in_ch * dims[2] * dims[3]) * (num_dense_out_ch) * (fmap[fmap_name][1]**2)
      elif len(dims) == 2:
        inf_cost = (num_dense_in_ch * num_dense_out_ch)
      #print("{}, {}, {}".format(fmap_name, num_dense_in_ch, num_dense_out_ch))

      conv_id += 1
      acc_inf_cost += inf_cost

  print("tot_weights:{}".format(tot_weights))

  return sparse_bi_map, \
         sparse_val_map, \
         conv_id, \
         conv_struct_density, \
         conv_rand_density, \
         (model_size * WORD_SIZE), \
         acc_inf_cost/MFLOPS


"""
Make only the (conv, FC) layer parameters sparse 
- Match other layers' parameters when reconfiguring network
- Only work for the flattened networks
"""
def _makeSparse(model, threshold, arch, threshold_type, dataset, is_gating=False, reconf=True):

  print ("[INFO] Force the sparse filters to zero...")
  dense_chs, chs_temp, idx = {}, {}, 0

  for name, param in model.named_parameters():
    dims = list(param.shape)
    if (('conv' in name) or ('fc' in name)) and ('weight' in name):

      with torch.no_grad():
        param = torch.where(param < threshold, torch.tensor(0.).cuda(), param)

      dense_in_chs, dense_out_chs = [], []
      if param.dim() == 4:
        if 'conv' in name:
          conv_dw = int(name.split('.')[1].split('conv')[1]) %2 == 0
        else:
          conv_dw = False
        # Forcing sparse input channels to zero
        if ('mobilenet' not in arch) or ('mobilenet' in arch and not conv_dw):
          for c in range(dims[1]):
            if param[:,c,:,:].abs().max() > 0:
              dense_in_chs.append(c)

        # Forcing sparse output channels to zero
        for c in range(dims[0]):
          if param[c,:,:,:].abs().max() > 0:
            dense_out_chs.append(c)

      # Forcing input channels of FC layer to zero
      elif param.dim() == 2:
        # Last FC layers (fc, fc3): Remove only the input neurons
        for c in range(dims[1]):
          if param[:,c].abs().max() > 0:
            dense_in_chs.append(c)
        # FC layer in the middle remove their output neurons
        if any(i for i in ['fc1', 'fc2'] if i in name):
          for c in range(dims[0]):
            if param[c,:].abs().max() > 0:
              dense_out_chs.append(c)
        else:
          # [fc, fc3] output channels (class probabilities) are all dense
          dense_out_chs = [c for c in range(dims[0])]
      
      chs_temp[idx] = {'name':name, 'in_chs':dense_in_chs, 'out_chs':dense_out_chs}
      idx += 1
      dense_chs[name] = {'in_chs':dense_in_chs, 'out_chs':dense_out_chs, 'idx':idx}

      # print the inter-layer tensor dim [out_ch, in_ch, feature_h, feature_w]
      if not reconf:
          if 'fc' in name:
              print("[{}]: [{}, {}]".format(name, 
                                            len(dense_chs[name]['out_chs']),
                                            len(dense_chs[name]['in_chs']),
                                            ))
          else:
              print("[{}]: [{}, {}, {}, {}]".format(name, 
                                                    len(dense_chs[name]['out_chs']),
                                                    len(dense_chs[name]['in_chs']),
                                                    param.shape[2],
                                                    param.shape[3],
                                                    ))
  """
  Inter-layer channel is_gating
  - Union: Maintain all dense channels on the shared nodes (No indexing)
  - Individual: Add gating layers >> Layers at the shared node skip more computation
  """
  if 'resnet' in arch:
    if 'cifar' in dataset:
      stages, ch_maps = stages_cifar[arch], []
    else:
      stages, ch_maps = stages_imagenet[arch], []

    # Within a residual branch >> Union of adjacent pairs
    adj_lyrs = stages[10]
    for adj_lyr in adj_lyrs:
      if any(i for i in adj_lyr if i not in dense_chs):
        """ not doing anything """
      else:
        for idx in range(len(adj_lyr)-1):
          edge = list(set().union(dense_chs[adj_lyr[idx]]['out_chs'],
                                  dense_chs[adj_lyr[idx+1]]['in_chs']))
          dense_chs[adj_lyr[idx]]['out_chs'] = edge
          dense_chs[adj_lyr[idx+1]]['in_chs'] = edge

    # Shared nodes >> Leave union of all in/out channels 
    if is_gating:
      for idx in range(len(stages)-1):
        edges = [] # Container of dense edges indexes
        for lyr_name in stages[idx]['i']:
          if lyr_name in dense_chs:
            edges = list(set().union(edges, dense_chs[lyr_name]['in_chs']))
        for lyr_name in stages[idx]['o']:
          if lyr_name in dense_chs:
            edges = list(set().union(edges, dense_chs[lyr_name]['out_chs']))

        # Edit the dense channel indexes
        ch_map = {}
        for idx, edge in enumerate(sorted(edges)):
          ch_map[edge] = idx
        ch_maps.append(ch_map)
      return dense_chs, ch_maps

    else:
      for idx in range(len(stages)-1):
        edges = [] 
        # Find union of the channels sharing the same node
        for lyr_name in stages[idx]['i']:
          if lyr_name in dense_chs:
            edges = list(set().union(edges, dense_chs[lyr_name]['in_chs']))
        for lyr_name in stages[idx]['o']:
          if lyr_name in dense_chs:
            edges = list(set().union(edges, dense_chs[lyr_name]['out_chs']))

        # Maintain the dense channels at the shared node
        for lyr_name in stages[idx]['i']:
          if lyr_name in dense_chs:
            #print ("Input_ch [{}]: {} => {}".format(lyr_name, len(dense_chs[lyr_name]['in_chs']), len(edges)))
            dense_chs[lyr_name]['in_chs'] = edges

        for lyr_name in stages[idx]['o']:
          if lyr_name in dense_chs:
            #print ("Output_ch [{}]: {} => {}".format(lyr_name, len(dense_chs[lyr_name]['out_chs']), len(edges)))
            dense_chs[lyr_name]['out_chs'] = edges

      #for name in dense_chs:
      #  print ("[{}]: {}, {}".format(name, dense_chs[name]['in_chs'], dense_chs[name]['out_chs']))

      return dense_chs, None

  # Non-residual networks
  elif 'mobilenet' in arch:
    for idx in sorted(chs_temp):
      # From conv2 layer
      if idx != 0:
        # Depth-wise convolution layer: Matintain the union of adjacent layers' dense channels
        if ((idx+1) %2 == 0) and ('fc' not in chs_temp[idx]['name']):
          edge = list(set().union(chs_temp[idx-1]['out_chs'], chs_temp[idx+1]['in_chs']))
          dense_chs[ chs_temp[idx-1]['name'] ]['out_chs'] = edge
          dense_chs[ chs_temp[idx]['name'] ]['out_chs'] = edge
          dense_chs[ chs_temp[idx+1]['name'] ]['in_chs'] = edge

          ## Search the target DW-convolution layer and change group#
          conv_idx = 0
          for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
              # Found the target conv-layer
              if idx == conv_idx:
                layer.groups = len(edge)
                break
              else:
                conv_idx +=1

        elif 'fc' in chs_temp[idx]['name']:
          edge = list(set().union(chs_temp[idx]['in_chs'], chs_temp[idx-1]['out_chs']))
          dense_chs[ chs_temp[idx]['name'] ]['in_chs'] = edge
          dense_chs[ chs_temp[idx-1]['name'] ]['out_chs'] = edge

    return dense_chs, None

  # Non-residual networks
  else:
    for idx in sorted(chs_temp):
      if idx != 0:
        # Dense input channels <= previous layers's output channel granularity
        if 'fc1' in chs_temp[idx]['name']: 
          feature_size = 7*7
          edge = []
          for prev_dense_ch in dense_chs[ chs_temp[idx-1]['name'] ]['out_chs']:
            for i in range(feature_size):
              edge.append(prev_dense_ch * feature_size + i)
          dense_chs[ chs_temp[idx]['name'] ]['in_chs'] = edge
        else:
          if is_gating:
            edge = [x for x in chs_temp[idx-1]['out_chs'] if x in chs_temp[idx]['in_chs']]
          else:
            edge = list(set().union(chs_temp[idx-1]['out_chs'], chs_temp[idx]['in_chs']))
  
          dense_chs[ chs_temp[idx-1]['name'] ]['out_chs'] = edge
          dense_chs[ chs_temp[idx]['name'] ]['in_chs'] = edge
    return dense_chs, None


"""
Generate a new dense network model
- Rearrange/remove channels from filters
- Rearrange/remove the channels of non-convolution layers
- Remove the dead (all zero channels) layers
- Manage optimization/momentum/buffer parameters
"""
def _genDenseModel(model, dense_chs, optimizer, arch, dataset):
  print ("[INFO] Squeezing the sparse model to dense one...")

  rm_list = []
  new_mom_list = []
  
  for name, param in model.named_parameters():
    # Get Momentum parameters to adjust
    # mom_param = optimizer.state[param]['momentum_buffer']

    # Change parameters of neural computing layers (Conv, FC) 
    if (('conv' in name) or ('fc' in name)) and ('weight' in name):

      if 'conv' in name:
        conv_dw = int(name.split('.')[1].split('conv')[1]) %2 == 0
      else:
        conv_dw = False

      dims = list(param.shape)

      if 'mobilenet' in arch and conv_dw:
        dense_in_ch_idxs = [0]
      else:
        dense_in_ch_idxs = dense_chs[name]['in_chs']
        
      dense_out_ch_idxs = dense_chs[name]['out_chs']
      num_in_ch, num_out_ch = len(dense_in_ch_idxs), len(dense_out_ch_idxs)
       
      # Enlist layers with zero channels for removal
      if 'resnet' in arch and (num_in_ch == 0 or num_out_ch == 0):
        rm_list.append(name)

      else:
        # Generate a new dense tensor and replace (Convolution layer)
        if len(dims) == 4 and (num_in_ch != param.shape[1] or num_out_ch != param.shape[0]):

          # new conv param
          new_param = Parameter(torch.Tensor(num_out_ch, num_in_ch, dims[2], dims[3])).cuda()
          
          # new_mom_param = Parameter(torch.Tensor(num_out_ch, num_in_ch, dims[2], dims[3])).cuda()
          
          # populate new param
          for in_idx, in_ch in enumerate(sorted(dense_in_ch_idxs)):
            for out_idx, out_ch in enumerate(sorted(dense_out_ch_idxs)):
              with torch.no_grad():
                new_param[out_idx,in_idx,:,:] = param[out_ch,in_ch,:,:]
                # new_mom_param[out_idx,in_idx,:,:] = mom_param[out_ch,in_ch,:,:]

          current_layer = model._modules["module"]._modules[name.split('.')[1]]
          
          # initialize new pruned conv layer to replace the old one
          new_layer = nn.Conv2d(1, 1, kernel_size=current_layer.kernel_size, stride=current_layer.stride, padding=current_layer.padding, bias=False).cuda()
          
          #  set new layer weight
          with torch.no_grad():
            new_layer.weight = Parameter(new_param).cuda()
            
          # replace old conv layer with new one
          model._modules["module"]._modules[name.split('.')[1]] = new_layer

          # new_mom_list.append((new_layer.weight, new_mom_param))

          print("[{}]: {} >> {}".format(name, dims, list(new_param.shape)))

        # Generate a new dense tensor and replace (FC layer)
        elif len(dims) == 2 and (num_in_ch != param.shape[1] or num_out_ch != param.shape[0]):
          new_param = Parameter(torch.Tensor(num_out_ch, num_in_ch)).cuda()
          
          # new_mom_param = Parameter(torch.Tensor(num_out_ch, num_in_ch)).cuda()

          current_layer = model._modules["module"]._modules[name.split('.')[1]]
          
          # initialize pruned FC layer to replace the old one
          new_layer = nn.Linear(1, 1)
          
          # populate new param
          if ('fc1' in name) or ('fc2' in name):
            for in_idx, in_ch in enumerate(sorted(dense_in_ch_idxs)):
              for out_idx, out_ch in enumerate(sorted(dense_out_ch_idxs)):
                with torch.no_grad():
                  new_param[out_idx,in_idx] = param[out_ch,in_ch]
                  # new_mom_param[out_idx,in_idx] = mom_param[out_ch,in_ch]
          else:
            for in_idx, in_ch in enumerate(sorted(dense_in_ch_idxs)):
              with torch.no_grad():
                new_param[:,in_idx] = param[:,in_ch]
                # new_mom_param[:,in_idx] = mom_param[:,in_ch]
          
          #  set new layer weight and bias
          with torch.no_grad():
            new_layer.weight = Parameter(new_param).cuda()
            new_layer.bias = Parameter(current_layer.bias[sorted(dense_out_ch_idxs)]).cuda()
          
          # replace old FC layer with new one
          model._modules["module"]._modules[name.split('.')[1]] = new_layer
          # new_mom_list.append((new_layer.weight, new_mom_param))
          
          print("[{}]: {} >> {}".format(name, dims, list(new_param.shape)))
          
        # else:
        #   new_mom_list.append((param, mom_param))

        # param.data = new_param
        # optimizer.state[param]['momentum_buffer'].data = new_mom_param
        
       


  # Change moving_mean and moving_var of BN

  # iterate through modules
  for name, current_layer in model._modules["module"].named_modules():
    
    if 'bn' in name:
      w_name = name.replace('bn', 'module.conv')+'.weight'
      dense_out_ch_idxs = dense_chs[w_name]['out_chs']
      num_out_ch = len(dense_out_ch_idxs)

      if num_out_ch != current_layer.running_mean.shape[0]:
        print("[{}]: {} >> {}".format(name, current_layer.running_mean.shape[0], num_out_ch))
        # new weight and bias params
        new_weight = Parameter(torch.Tensor(num_out_ch)).cuda()
        new_bias = Parameter(torch.Tensor(num_out_ch)).cuda()

        # new BN layer to replace the old one
        new_layer = nn.BatchNorm2d(num_out_ch).cuda()

        # populate new params
        for out_idx, out_ch in enumerate(sorted(dense_out_ch_idxs)):
          with torch.no_grad():
            new_layer.running_mean[out_idx] = current_layer.running_mean[out_ch]
            new_layer.running_var[out_idx] = current_layer.running_var[out_ch]

            new_weight[out_idx] = current_layer.weight[out_ch]
            new_bias[out_idx] = current_layer.bias[out_ch]
        
        # set new BN layer's params to the new populated params
        with torch.no_grad():
          new_layer.weight = Parameter(new_weight).cuda()
          new_layer.bias = Parameter(new_bias).cuda()
        
        # replace the current BN layer with the new one
        model._modules["module"]._modules[name] = new_layer

      


  """
  Remove layers (Only applicable to ResNet-like networks)
  - Remove model parameters
  - Remove parameters/states in optimizer
  """
  def getLayerIdx(lyr_name):
    if 'conv' in lyr_name:
      conv_id = dense_chs[lyr_name+'.weight']['idx']
      return [3 * (conv_id - 1)], [lyr_name+'.weight']
    elif 'bn' in lyr_name:
      conv_name = lyr_name.replace('bn', 'conv')
      conv_id = dense_chs[conv_name+'.weight']['idx']
      return [3 * conv_id -1, 3 * conv_id -2], [lyr_name+'.bias', lyr_name+'.weight']

  if len(rm_list) > 0:
    rm_lyrs = []
    for name in rm_list:
      rm_lyr = getRmLayers(name, arch, dataset)
      if any(i for i in rm_lyr if i not in rm_lyrs):
        rm_lyrs.extend(rm_lyr)
    
    # Remove model parameters
    for rm_lyr in rm_lyrs:
      model.del_param_in_flat_arch(rm_lyr)

    idxs, rm_params = [], []
    for rm_lyr in rm_lyrs:
      idx, rm_param = getLayerIdx(rm_lyr)
      idxs.extend(idx)
      rm_params.extend(rm_param)

    # Remove optimizer states
    for name, param in model.named_parameters():
      for rm_param in rm_params:
        if name == rm_param:
          print('deleting param')
          del optimizer.state[param]

    # Sanity check: Print out optimizer parameters before change
    #print ("[INFO] ==== Size of parameter group (Before)")
    #for g in optimizer.param_groups:
    #  for idx, g2 in enumerate(g['params']):
    #    print("idx:{}, param_shape:{}".format(idx, list(g2.shape)))
    
    # Remove optimizer parameters
    # Adjuster: Absolute parameter location changes after each removal
    for idx_adjuster, idx in enumerate(sorted(idxs)):
      del optimizer.param_groups[0]['params'][idx - idx_adjuster]

  # return new_mom_list
