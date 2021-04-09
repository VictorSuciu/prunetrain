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
from os import listdir
from os.path import isfile, join
from statistics import mean

from ..src.custom.checkpoint_utils_fp32 import Checkpoint
from custom.visualize_utils import plotFilterSparsity, plotLayerSparsity, plotFilterData

MB = 1024*1024

out_dir = 'path/to/store/output'
model_dir = '/path/to/model'

check_point_names = [f for f in listdir(model_dir) if isfile(join(model_dir, f))  and 'checkpoint' in f]

temp = ['checkpoint90.tar']
check_point_names = temp

dataset = 'imagenet'
arch = "resnet50_flat"
threshold = 0.0001
depth = 20
num_classes = 1000

gen_figs = True
get_fil_data = True
target_lyr = ['conv40', 'conv41', 'conv42']

def calcConvSparsity(epochs, out_dir):
    avg_in_by_epoch  ={}
    avg_out_by_epoch  ={}
    max_epoch = 0

    for e in epochs:
        lyrs_density = epochs[e]

        list_in = []
        list_out = []

        for l in lyrs_density:
            list_in.append(lyrs_density[l]['in_ch'])
            list_out.append(lyrs_density[l]['out_ch'])

        avg_in_by_epoch[e] = mean(list_in)
        avg_out_by_epoch[e] = mean(list_out)
        max_epoch = max(max_epoch, e)

    print("========= input channel density ==========")
    for e in epochs:
        print ("{}, {}".format(e, str(avg_in_by_epoch[e])))

    print("========= output channel density ==========")
    for e in epochs:
        print ("{}, {}".format(e, str(avg_out_by_epoch[e])))

def main():
    conv_density_epochs = {}
    sparse_val_maps = {}

    for idx, check_point_name in enumerate(check_point_names):
        print ("Processing check_point: " +os.path.join(model_dir, check_point_name))
        model = Checkpoint(arch, 
                           dataset,
                           os.path.join(model_dir, check_point_name), 
                           num_classes)

        if idx == 0 : model.printParams()

        # Generate conv layer sparsity
        sparse_bi_map, sparse_val_map, num_lyrs, conv_density, model_size, inf_cost =\
                model.getConvStructSparsity(threshold, out_dir+"/out_txt")
        
        if get_fil_data:
            fil_data = model.getFilterData(target_lyr)

        sparse_val_maps[idx] = sparse_val_map                
        print ("==> Model_size: {}, inference_cost: {}".format(model_size / MB, inf_cost))
        conv_density_epochs[model.getEpoch()] = conv_density

        #if gen_figs:
        #    plotFilterSparsity(check_point_name, sparse_bi_map, threshold, out_dir, num_lyrs)

    calcConvSparsity(conv_density_epochs, out_dir)

    #if gen_figs:
    #    plotLayerSparsity(sparse_val_maps, out_dir)
    #    if get_fil_data:
    #        plotFilterData(fil_data, out_dir)

if __name__ == "__main__":
    main()
