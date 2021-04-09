import os, sys
from os import listdir
from os.path import isfile, join
from statistics import mean
from collections import OrderedDict

from custom.checkpoint_utils_fp32 import Checkpoint

MB = 1024*1024

out_dir = '/path/to/store/output'
model_dir = '/path/to/model'

check_point_names = [f for f in listdir(model_dir) if isfile(join(model_dir, f))  and 'checkpoint' in f]

temp = []
for check_point_name in check_point_names:
    if "checkpoint90.tar" in check_point_name:
        temp.append(check_point_name)

check_point_names = temp
print(check_point_names)

dataset = 'imagenet'
arch = "resnet50_flat_01"
target_lyr = 1
threshold = 0.0001
depth = 20
num_classes = 100
gen_figs = True
sparse_val_maps = OrderedDict()

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

    for idx, check_point_name in enumerate(check_point_names):
        print ("Processing check_point: " +os.path.join(model_dir, check_point_name))
        model = Checkpoint(arch, 
                           dataset,
                           os.path.join(model_dir, check_point_name), 
                           num_classes,
                           depth)

        if idx == 0 : model.printParams()

        # Generate conv layer sparsity
        sparse_bi_map, sparse_val_map, num_lyrs, conv_density, model_size, inf_cost =\
                model.getConvStructSparsity(threshold, out_dir+"/out_txt")

        sparse_val_maps[idx] = sparse_val_map
        conv_density_epochs[model.getEpoch()] = conv_density
        print ("==> Model_size: {}, inference_cost: {}".format(model_size / MB, inf_cost))

        #if gen_figs:
        #    plotFilterSparsity(check_point_name, sparse_bi_map, threshold, out_dir, num_lyrs)

    #if gen_figs:
    #    plotLayerSparsity(sparse_val_maps, target_lyr)

    calcConvSparsity(conv_density_epochs, out_dir)

if __name__ == "__main__":
    main()
