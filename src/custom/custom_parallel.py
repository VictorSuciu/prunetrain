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

import torch.nn as nn

class CustomDataParallel(nn.DataParallel):  
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(CustomDataParallel, self).__init__(module, device_ids, output_device, dim)

    """ Remove sparsified module parameter from the network model
    # rm_name: name of module to remove
    """
    def del_param_in_flat_arch(self, rm_name):
        # We remove an entire layer holding the delete target parameters
        rm_module = rm_name.split('.')
        module  = self._modules[rm_module[0]]
        if module._modules[rm_module[1]] != None:
            print("[INFO] Removing parameters/buffers in module [{}]".format(rm_module[0]+'.'+rm_module[1]))
            del module._modules[rm_module[1]]