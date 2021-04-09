alexnet = {}
alexnet['conv1'] = (32, 32)
alexnet['conv2'] = (8, 8)
alexnet['conv3'], alexnet['conv4'], alexnet['conv5'] = (4, 4), (4, 4), (4, 4)

vgg8 = {}
vgg8['conv1'] = (32, 32)
vgg8['conv2'] = (16, 16)
vgg8['conv3'] = (8, 8)
vgg8['conv4'] = (4, 4)
vgg8['conv5'] = (2, 2)

vgg11 = {}
vgg11['conv1'] = (32, 32)
vgg11['conv2'] = (16, 16)
vgg11['conv3'], vgg11['conv4'] = (8, 8), (8, 8)
vgg11['conv5'], vgg11['conv6'] = (4, 4), (4, 4)
vgg11['conv7'], vgg11['conv8'] = (2, 2), (2, 2)

vgg13 = {}
vgg13['conv1'], vgg13['conv2'] = (32, 32), (32, 32)
vgg13['conv3'], vgg13['conv4'] = (16, 16), (16, 16)
vgg13['conv5'], vgg13['conv6'] = (8, 8), (8, 8)
vgg13['conv7'], vgg13['conv8'] = (4, 4), (4, 4)
vgg13['conv9'], vgg13['conv10'] = (2, 2), (2, 2)

# ResNet20
resnet20 = dict.fromkeys(['conv'+str(i) for i in range(1,22)])
resnet20.update(dict.fromkeys(['conv'+str(i) for i in range(1,8)],  (32,32)))
resnet20.update(dict.fromkeys(['conv8', 'conv10'],  (32,16)))
resnet20['conv9'] = (16,16)
resnet20.update(dict.fromkeys(['conv'+str(i) for i in range(11,15)], (16,16)))
resnet20.update(dict.fromkeys(['conv15', 'conv17'],  (16,8)))
resnet20['conv16'] = (8,8)
resnet20.update(dict.fromkeys(['conv'+str(i) for i in range(18,22)], (8,8)))

# ResNet32
resnet32 = dict.fromkeys(['conv'+str(i) for i in range(1,34)])
resnet32.update(dict.fromkeys(['conv'+str(i) for i in range(1,12)],  (32,32)))
resnet32.update(dict.fromkeys(['conv12', 'conv14'],  (32,16)))
resnet32['conv13'] = (16,16)
resnet32.update(dict.fromkeys(['conv'+str(i) for i in range(15,23)], (16,16)))
resnet32.update(dict.fromkeys(['conv23', 'conv25'],  (16,8)))
resnet32['conv24'] = (8,8)
resnet32.update(dict.fromkeys(['conv'+str(i) for i in range(26,34)], (8,8)))

# ResNet32_BT
resnet32_bt = dict.fromkeys(['conv'+str(i) for i in range(1,50)])
resnet32_bt.update(dict.fromkeys(['conv'+str(i) for i in range(1,19)],  (32,32)))
resnet32_bt.update(dict.fromkeys(['conv19', 'conv21'],  (32,16)))
resnet32_bt['conv20'] = (16,16)
resnet32_bt.update(dict.fromkeys(['conv'+str(i) for i in range(22,35)], (16,16)))
resnet32_bt.update(dict.fromkeys(['conv35', 'conv37'],  (16,8)))
resnet32_bt['conv36'] = (8,8)
resnet32_bt.update(dict.fromkeys(['conv'+str(i) for i in range(38,50)], (8,8)))

# ResNet50_BT
resnet50_bt = dict.fromkeys(['conv'+str(i) for i in range(1,77)])
resnet50_bt.update(dict.fromkeys(['conv'+str(i) for i in range(1,28)],  (32,32)))
resnet50_bt.update(dict.fromkeys(['conv28', 'conv30'],  (32,16)))
resnet50_bt['conv29'] = (16,16)
resnet50_bt.update(dict.fromkeys(['conv'+str(i) for i in range(31,53)], (16,16)))
resnet50_bt.update(dict.fromkeys(['conv53', 'conv55'],  (16,8)))
resnet50_bt['conv54'] = (8,8)
resnet50_bt.update(dict.fromkeys(['conv'+str(i) for i in range(56,77)], (8,8)))

# ResNet56_BT
resnet56_bt = dict.fromkeys(['conv'+str(i) for i in range(1,86)])
resnet56_bt.update(dict.fromkeys(['conv'+str(i) for i in range(1,31)],  (32,32)))
resnet56_bt.update(dict.fromkeys(['conv31', 'conv33'],  (32,16)))
resnet56_bt['conv32'] = (16,16)
resnet56_bt.update(dict.fromkeys(['conv'+str(i) for i in range(34,59)], (16,16)))
resnet56_bt.update(dict.fromkeys(['conv59', 'conv61'],  (16,8)))
resnet56_bt['conv60'] = (8,8)
resnet56_bt.update(dict.fromkeys(['conv'+str(i) for i in range(62,86)], (8,8)))

######### ImageNet data #########

# ResNet50
resnet50 = dict.fromkeys(['conv'+str(i) for i in range(1,54)])
resnet50['conv1'] = (224,112)
resnet50.update(dict.fromkeys(['conv'+str(i) for i in range(2,13)],  (56,56)))
resnet50.update(dict.fromkeys(['conv13', 'conv15'],  (56,28)))
resnet50['conv14'] = (28,28)
resnet50.update(dict.fromkeys(['conv'+str(i) for i in range(16,26)], (28,28)))
resnet50.update(dict.fromkeys(['conv26', 'conv28'],  (28,14)))
resnet50['conv27'] = (14,14)
resnet50.update(dict.fromkeys(['conv'+str(i) for i in range(29,45)], (14,14)))
resnet50.update(dict.fromkeys(['conv45', 'conv47'],  (14,7)))
resnet50['conv46'] = (7,7)
resnet50.update(dict.fromkeys(['conv'+str(i) for i in range(48,54)], (7,7)))

# MobileNet (224)
mobilenet = dict.fromkeys(['conv'+str(i) for i in range(1, 28)])
mobilenet['conv1'] = (224,112)
mobilenet.update(dict.fromkeys(['conv'+str(i) for i in range(2,4)],  (112,112)))
mobilenet['conv4'] = (112,56)
mobilenet.update(dict.fromkeys(['conv'+str(i) for i in range(5,8)],  (56,56)))
mobilenet['conv8'] = (56,28)
mobilenet.update(dict.fromkeys(['conv'+str(i) for i in range(9,12)],  (28,28)))
mobilenet['conv12'] = (28,14)
mobilenet.update(dict.fromkeys(['conv'+str(i) for i in range(13,24)],  (14,14)))
mobilenet['conv24'] = (14,7)
mobilenet.update(dict.fromkeys(['conv'+str(i) for i in range(25,28)],  (7,7)))

vgg16 = {}                                                                                
vgg16['conv1'],  vgg16['conv2'] = (224, 224), (224, 224)                                  
vgg16['conv3'],  vgg16['conv4'] = (112, 112), (112, 112)                                  
vgg16['conv5'],  vgg16['conv6'],  vgg16['conv7']  = (56, 56), (56, 56), (56, 56)          
vgg16['conv8'],  vgg16['conv9'],  vgg16['conv10'] = (28, 28), (28, 28), (28, 28)          
vgg16['conv11'], vgg16['conv12'], vgg16['conv13'] = (14, 14), (14, 14), (14, 14)

cifar_feature_size = {
    'alexnet'       :alexnet,
    'vgg8'          :vgg8,
    'vgg8_bn_flat'  :vgg8,
    'vgg11'         :vgg11,
    'vgg11_bn_flat' :vgg11,
    'vgg13'         :vgg13,
    'vgg13_bn_flat' :vgg13,
    'resnet20_flat'      :resnet20,
    'resnet32_flat'      :resnet32,
    'resnet32_bt_flat'   :resnet32_bt,
    'resnet32_bt_flat_temp'   :resnet32_bt,
    'resnet50_bt_flat'   :resnet50_bt,
    'resnet56_bt_flat'   :resnet56_bt,
}

imagenet_feature_size = {
    'resnet50'      :resnet50,
    'resnet50_flat'      :resnet50,
    'resnet50_flat_01'      :resnet50,
    'mobilenet'     :mobilenet,
    'vgg16_flat'    :vgg16,
}
