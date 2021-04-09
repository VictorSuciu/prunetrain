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


""" Module name constructor
# Model naming for each layer. Layer is assumed to be flattened.
"""
def n(name):
    if isinstance(name, int):
        return 'module.conv'+str(name)+'.weight'
    else:
        return 'module.'+name+'.weight'


""" ResNet models for CIFAR
# Convolution layer indices that share the same node
# Convolution layer indices for each residual path
"""

# ResNet20
resnet20_cifar = {0:{}, 1:{}, 2:{}, 10:{}}
resnet20_cifar[0]['i'] = [n(2), n(4), n(6), n(8), n(10)]
resnet20_cifar[0]['o'] = [n(1), n(3), n(5), n(7)]
resnet20_cifar[1]['i'] = [n(11), n(13), n(15), n(17)]
resnet20_cifar[1]['o'] = [n(9), n(10), n(12), n(14)]
resnet20_cifar[2]['i'] = [n(18), n(20), n('fc')]
resnet20_cifar[2]['o'] = [n(16), n(17), n(19), n(21)]

resnet20_cifar[10] = [
    [n(2), n(3)], [n(4), n(5)], [n(6), n(7)], [n(8), n(9)],
    [n(11), n(12)], [n(13), n(14)], [n(15), n(16)], [n(18), n(19)],
    [n(20), n(21)]
]

# ResNet32
resnet32_cifar = {0:{}, 1:{}, 2:{}, 10:{}}
resnet32_cifar[0]['i'] = [n(2), n(4), n(6), n(8), n(10), n(12), n(14)]
resnet32_cifar[0]['o'] = [n(1), n(3), n(5), n(7), n(9), n(11)]
resnet32_cifar[1]['i'] = [n(15), n(17), n(19), n(21), n(23), n(25)]
resnet32_cifar[1]['o'] = [n(13), n(14), n(16), n(18), n(20), n(22)]
resnet32_cifar[2]['i'] = [n(26), n(28), n(30), n(32), n('fc')]
resnet32_cifar[2]['o'] = [n(24), n(25), n(27), n(29), n(31), n(33)]

resnet32_cifar[10] = [
    [n(2), n(3)],   [n(4), n(5)],   [n(6), n(7)],   [n(8), n(9)],
    [n(10), n(11)], [n(12), n(13)], [n(15), n(16)], [n(17), n(18)],
    [n(19), n(20)], [n(21), n(22)], [n(23), n(24)], [n(26), n(27)],
    [n(28), n(29)], [n(30), n(31)], [n(32), n(33)]
]

# ResNet50
resnet50_bt_cifar = {0:{}, 1:{}, 2:{}, 3:{}, 10:{}}
resnet50_bt_cifar[0]['i'] = [n(2), n(5)]
resnet50_bt_cifar[0]['o'] = [n(1)]
resnet50_bt_cifar[1]['i'] = [n(6), n(9), n(12), n(15), n(18), n(21), n(24), n(27), n(30)]
resnet50_bt_cifar[1]['o'] = [n(4), n(5), n(8), n(11), n(14), n(17), n(20), n(23), n(26)]
resnet50_bt_cifar[2]['i'] = [n(31), n(34), n(37), n(40), n(43), n(46), n(49), n(52), n(55)]
resnet50_bt_cifar[2]['o'] = [n(29), n(30), n(33), n(36), n(39), n(42), n(45), n(48), n(51)]
resnet50_bt_cifar[3]['i'] = [n(56), n(59), n(62), n(65), n(68), n(71), n(74), n('fc')]
resnet50_bt_cifar[3]['o'] = [n(54), n(55), n(58), n(61), n(64), n(67), n(70), n(73), n(76)]

resnet50_bt_cifar[10] = [
    [n(2),  n(3),  n(4)],  [n(6),  n(7),  n(8)],  [n(9),  n(10), n(11)], [n(12), n(13), n(14)],
    [n(15), n(16), n(17)], [n(18), n(19), n(20)], [n(21), n(22), n(23)], [n(24), n(25), n(26)],
    [n(27), n(28), n(29)], [n(31), n(32), n(33)], [n(34), n(35), n(36)], [n(37), n(38), n(39)],
    [n(40), n(41), n(42)], [n(43), n(44), n(45)], [n(46), n(47), n(48)], [n(49), n(50), n(51)],
    [n(52), n(53), n(54)], [n(56), n(57), n(58)], [n(59), n(60), n(61)], [n(62), n(63), n(64)],
    [n(65), n(66), n(67)], [n(68), n(69), n(70)], [n(71), n(72), n(73)], [n(74), n(75), n(76)]
]

stages_cifar = {
    'resnet20_flat':resnet20_cifar,
    'resnet32_flat':resnet32_cifar,
    'resnet50_bt_flat':resnet50_bt_cifar,
}

""" ResNet models for ImageNet
# Convolution layer indices that share the same node
# Convolution layer indices for each residual path
"""

# ResNet50
resnet50_imagenet = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 10:{}}
resnet50_imagenet[0]['i'] = [n(2), n(5)]
resnet50_imagenet[0]['o'] = [n(1)]
resnet50_imagenet[1]['i'] = [n(6),  n(9),  n(12), n(15)]
resnet50_imagenet[1]['o'] = [n(4),  n(5),  n(8),  n(11)]
resnet50_imagenet[2]['i'] = [n(16), n(19), n(22), n(25), n(28)]
resnet50_imagenet[2]['o'] = [n(14), n(15), n(18), n(21), n(24)]
resnet50_imagenet[3]['i'] = [n(29), n(32), n(35), n(38), n(41), n(44), n(47)]
resnet50_imagenet[3]['o'] = [n(27), n(28), n(31), n(34), n(37), n(40), n(43)]
resnet50_imagenet[4]['i'] = [n(48), n(51), n('fc')]
resnet50_imagenet[4]['o'] = [n(46), n(47), n(50), n(53)]

resnet50_imagenet[10] = [
    [n(2),  n(3),  n(4)],  [n(6),  n(7),  n(8)],  [n(9),  n(10), n(11)], [n(12), n(13), n(14)],
    [n(16), n(17), n(18)], [n(19), n(20), n(21)], [n(22), n(23), n(24)], [n(25), n(26), n(27)],
    [n(29), n(30), n(31)], [n(32), n(33), n(34)], [n(35), n(36), n(37)], [n(38), n(39), n(40)],
    [n(41), n(42), n(43)], [n(44), n(45), n(46)], [n(48), n(49), n(50)], [n(51), n(52), n(53)],
]

stages_imagenet = {
    'resnet50_flat':resnet50_imagenet,
    'resnet50_gn':resnet50_imagenet,
}

