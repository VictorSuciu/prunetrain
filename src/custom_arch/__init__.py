from .arch_utils import *

# CIFAR10/100
from .custom_alexnet import _genDenseArchAlexNet
from .custom_vgg8_bn import _genDenseArchVGG8BN
from .custom_vgg11_bn import _genDenseArchVGG11BN
from .custom_vgg13_bn import _genDenseArchVGG13BN
from .custom_resnet32 import _genDenseArchResNet32
from .custom_resnet50_bt import _genDenseArchResNet50BT

# ImageNet
from .custom_resnet50 import _genDenseArchResNet50
from .custom_mobilenet import _genDenseArchMobileNet
from .custom_vgg16_bn import _genDenseArchVGG16

custom_arch_cifar = {
    'alexnet_flat':_genDenseArchAlexNet,
    'vgg8_bn_flat':_genDenseArchVGG8BN,
    'vgg11_bn_flat':_genDenseArchVGG11BN,
    'vgg13_bn_flat':_genDenseArchVGG13BN,
    'resnet32_flat':_genDenseArchResNet32,
    'resnet50_bt_flat':_genDenseArchResNet50BT
}

custom_arch_imagenet = {
    'resnet50_flat':_genDenseArchResNet50,
    'mobilenet_flat':_genDenseArchMobileNet,
    'vgg16_flat':_genDenseArchVGG16,
}
