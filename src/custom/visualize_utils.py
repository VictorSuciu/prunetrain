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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors
import math as mt

""" Plot the filter sparsity pattern
# checkpoint_name: input-channel
# sparse_map: 2D data each for the max filter value (x: input channel index, y: output channel index)
# threshold: criteria to quantize value to zero
# out_dir: directory to store output file
# num_lyrs: the number of convolution layers
"""
def plotFilterSparsity(checkpoint_name, sparse_map, threshold, out_dir, num_lyrs):
  rows = 3
  cols = int(mt.ceil(float(num_lyrs) / 3))

  cmap = 'cool'
  vmin = threshold
  vmax = 1
  norm=LogNorm(vmin=vmin,vmax=vmax) # thresholding at 0.001

  fig, axs = plt.subplots(rows, cols)
  fig.suptitle(checkpoint_name)

  imgs = []

  for num_lyr in range(num_lyrs):
    r = num_lyr // cols
    c = num_lyr % cols
    
    # Skip the first conv layer
    heatmap = sparse_map[num_lyr]
    imgs.append(axs[r,c].imshow(heatmap,
                               interpolation='nearest', 
                               origin='lower',
                               cmap=cmap,
                               norm=norm))
    axs[r, c].label_outer()
    axs[r, c].set_title(str(num_lyr +1))

  cbar = fig.colorbar(imgs[0], ax=axs, orientation='horizontal', fraction=.1)
  cbar.ax.set_xlabel('MAX value of each filter')
  fig.set_size_inches(15, 11.25)
  epoch = checkpoint_name.split('checkpoint.')[1].split('.tar')[0]
  fig.savefig(out_dir+'/epoch_'+epoch, dpi=100)
  plt.close()


""" Plot the value of output channels over epochs
Goal: Show the sparsified channel value revive through back-propagation
symlog graph to visualize zero value
"""
def plotLayerSparsity(sparse_val_map, out_dir):
  # out_chs = {lyr:{ch:[values at each epoch]}}
  out_chs = {}

  for epoch_idx, sparse_map in sparse_val_map.items():
    for lyr_idx, lyr_sparse_map in sparse_map.items():

      if lyr_idx not in out_chs:
        out_chs[lyr_idx] = {}

      max_out_chs = lyr_sparse_map.max(axis=0)
      for ch_idx, max_out_ch in enumerate(max_out_chs):

        # Add max_out_ch value for each 
        if ch_idx not in out_chs[lyr_idx]:
          out_chs[lyr_idx][ch_idx] = []
        out_chs[lyr_idx][ch_idx].append(max_out_ch)

  for lyr in out_chs:
    out_chs_list = []

    for idx in out_chs[lyr]:
      out_chs_list.append(out_chs[lyr][idx])
    out_chs_list = np.array(out_chs_list)

    cmap = 'cool'
    vmin = 0.0001 # Thresholding boundary
    vmax = 10
    norm=LogNorm(vmin=vmin,vmax=vmax) # thresholding at 0.001

    fig, axs = plt.subplots(1,1)
    heatmap = np.array(out_chs_list)
    img = axs.imshow(heatmap,
               interpolation='nearest', 
               origin='lower',
               cmap=cmap,
               norm=norm)

    axs.set_xlabel('Epoch')
    axs.set_ylabel('MAX( ABS(weights of each channel) )')
    cbar = fig.colorbar(img, ax=axs, orientation='vertical', fraction=.1)
    cbar.ax.set_xlabel('Heatmap')
    fig.set_size_inches(10, 15)
    fig.savefig(os.path.join(out_dir, 'out_ch_conv'+str(lyr)+'.pdf'), format='pdf')

"""
Plot the value of output channels over epochs
Goal: Show the sparsified channel value revive through back-propagation
symlog graph to visualize zero value
"""
def plotFilterData(fil_data, out_dir):

  for name in fil_data:
    #for ch_idx, ch in enumerate(fil_data[name]):

    fig, axs = plt.subplots(1,1)
    for ch_idx, ch in enumerate(fil_data[name]):
      #print("=====>> {} : {}".format(len([i for i in range(len(fil_data[name]))]), len(fil_data[name])))
      axs.plot( [i for i in range(len(fil_data[name][ch_idx]))], sorted(fil_data[name][ch_idx]), 
                    #marker='.',
                    #linestyle = 'None'
      )

    axs.set_yscale('log')
    #axs.set_ylim(ymin=0.000001, ymax=10)
    axs.set_title(name)
    fig.set_size_inches(11, 11)
    fig.savefig(os.path.join(out_dir, 'filter_data'+str(name)+'_'+str(ch_idx)+'.pdf'), format='pdf')

