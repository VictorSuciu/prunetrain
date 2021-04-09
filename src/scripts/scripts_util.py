# Base architecture

alexnet = {}
alexnet['conv1'] = [64,3,11,11]
alexnet['conv2'] = [192,64,5,5]
alexnet['conv3'] = [384,192,3,3]
alexnet['conv4'] = [256,384,3,3]
alexnet['conv5'] = [256,256,3,3]
alexnet['fc']    = [100,256]

vgg8 = {}
vgg8['conv1'] = [64,3,3,3]
vgg8['conv2'] = [128,64,3,3]
vgg8['conv3'] = [256,128,3,3]
vgg8['conv4'] = [512,256,3,3]
vgg8['conv5'] = [512,512,3,3]
vgg8['fc']    = [100,512]

vgg11 = {}
vgg11['conv1'] = [64,3,3,3]
vgg11['conv2'] = [128,64,3,3]
vgg11['conv3'] = [256,128,3,3]
vgg11['conv4'] = [256,256,3,3]
vgg11['conv5'] = [512,256,3,3]
vgg11['conv6'] = [512,512,3,3]
vgg11['conv7'] = [512,512,3,3]
vgg11['conv8'] = [512,512,3,3]
vgg11['fc']    = [100,512]

vgg13 = {}
vgg13['conv1']  = [64,3,3,3]
vgg13['conv2']  = [64,64,3,3]
vgg13['conv3']  = [128,64,3,3]
vgg13['conv4']  = [128,128,3,3]
vgg13['conv5']  = [256,128,3,3]
vgg13['conv6']  = [256,256,3,3]
vgg13['conv7']  = [512,256,3,3]
vgg13['conv8']  = [512,512,3,3]
vgg13['conv9']  = [512,512,3,3]
vgg13['conv10'] = [512,512,3,3]
vgg13['fc']     = [100,512]

resnet20 = {
    'conv1':[16,3,3,3], 'conv2':[16,16,3,3], 'conv3':[16,16,3,3], 'conv4':[16,16,3,3],
    'conv5':[16,16,3,3], 'conv6':[16,16,3,3], 'conv7':[16,16,3,3], 'conv8':[32,16,3,3],
    'conv9':[32,32,3,3], 'conv10':[32,16,3,3], 'conv11':[32,32,3,3], 'conv12':[32,32,3,3],
    'conv13':[32,32,1,1], 'conv14':[32,32,3,3], 'conv15':[64,32,3,3], 'conv16':[64,64,3,3],
    'conv17':[64,32,1,1], 'conv18':[64,64,3,3], 'conv19':[64,64,3,3], 'conv20':[64,64,3,3],
    'conv21':[64,64,3,3], 'fc':[100,64]
}

resnet32 = {
    'conv1':[16,3,3,3], 'conv2':[16,16,3,3], 'conv3':[16,16,3,3], 'conv4':[16,16,3,3],
    'conv5':[16,16,3,3], 'conv6':[16,16,3,3], 'conv7':[16,16,3,3], 'conv8':[16,16,3,3],
    'conv9':[16,16,3,3], 'conv10':[16,16,3,3], 'conv11':[16,16,3,3], 'conv12':[32,16,3,3],
    'conv13':[32,32,3,3], 'conv14':[32,16,1,1], 'conv15':[32,32,3,3], 'conv16':[32,32,3,3],
    'conv17':[32,32,3,3], 'conv18':[32,32,3,3], 'conv19':[32,32,3,3], 'conv20':[32,32,3,3],
    'conv21':[32,32,3,3], 'conv22':[32,32,3,3], 'conv23':[64,32,3,3], 'conv24':[64,64,3,3],
    'conv25':[64,32,1,1], 'conv26':[64,64,3,3], 'conv27':[64,64,3,3], 'conv28':[64,64,3,3],
    'conv29':[64,64,3,3], 'conv30':[64,64,3,3], 'conv31':[64,64,3,3], 'conv32':[64,64,3,3],
    'conv33':[64,64,3,3], 'fc':[100,64]
}

resnet32_bt = {
    'conv1':[16,3,3,3],   
    'conv2':[16,16,1,1],   'conv3':[16,16,3,3],   'conv4':[64,16,1,1],  'conv5':[64,16,1,1],
    'conv6':[16,64,1,1],   'conv7':[16,16,3,3],   'conv8':[64,16,1,1],
    'conv9':[16,64,1,1],   'conv10':[16,16,3,3],  'conv11':[64,16,1,1],
    'conv12':[16,64,1,1],  'conv13':[16,16,3,3],  'conv14':[64,16,1,1],
    'conv15':[16,64,1,1],  'conv16':[16,16,3,3],  'conv17':[64,16,1,1],
    'conv18':[32,64,1,1],  'conv19':[32,32,3,3],  'conv20':[128,32,1,1], 'conv21':[128,64,1,1],
    'conv22':[32,128,1,1], 'conv23':[32,32,3,3],  'conv24':[128,32,1,1],
    'conv25':[32,128,1,1], 'conv26':[32,32,3,3],  'conv27':[128,32,1,1],
    'conv28':[32,128,1,1], 'conv29':[32,32,3,3],  'conv30':[128,32,1,1],
    'conv31':[32,128,1,1], 'conv32':[32,32,3,3],  'conv33':[128,32,1,1],
    'conv34':[64,128,1,1], 'conv35':[64,64,3,3],  'conv36':[256,64,1,1], 'conv37':[256,128,1,1],
    'conv38':[64,256,1,1], 'conv39':[64,64,3,3],  'conv40':[256,64,1,1],
    'conv41':[64,256,1,1], 'conv42':[64,64,3,3],  'conv43':[256,64,1,1],
    'conv44':[64,256,1,1], 'conv45':[64,64,3,3],  'conv46':[256,64,1,1],
    'conv47':[64,256,1,1], 'conv48':[64,64,3,3],  'conv49':[256,64,1,1], 'fc':[100,256]
}

resnet50_bt = {
    'conv1':[16,3,3,3],   
    'conv2':[16,16,1,1],   'conv3':[16,16,3,3],   'conv4':[64,16,1,1],  'conv5':[64,16,1,1],
    'conv6':[16,64,1,1],   'conv7':[16,16,3,3],   'conv8':[64,16,1,1],
    'conv9':[16,64,1,1],   'conv10':[16,16,3,3],  'conv11':[64,16,1,1],
    'conv12':[16,64,1,1],  'conv13':[16,16,3,3],  'conv14':[64,16,1,1],
    'conv15':[16,64,1,1],  'conv16':[16,16,3,3],  'conv17':[64,16,1,1],
    'conv18':[16,64,1,1],  'conv19':[16,16,3,3],  'conv20':[64,16,1,1],
    'conv21':[16,64,1,1],  'conv22':[16,16,3,3],  'conv23':[64,16,1,1],
    'conv24':[16,64,1,1],  'conv25':[16,16,3,3],  'conv26':[64,16,1,1],
    'conv27':[32,64,1,1],  'conv28':[32,32,3,3],  'conv29':[128,32,1,1], 'conv30':[128,64,1,1],
    'conv31':[32,128,1,1], 'conv32':[32,32,3,3],  'conv33':[128,32,1,1],
    'conv34':[32,128,1,1], 'conv35':[32,32,3,3],  'conv36':[128,32,1,1],
    'conv37':[32,128,1,1], 'conv38':[32,32,3,3],  'conv39':[128,32,1,1],
    'conv40':[32,128,1,1], 'conv41':[32,32,3,3],  'conv42':[128,32,1,1],
    'conv43':[32,128,1,1], 'conv44':[32,32,3,3],  'conv45':[128,32,1,1],
    'conv46':[32,128,1,1], 'conv47':[32,32,3,3],  'conv48':[128,32,1,1],
    'conv49':[32,128,1,1], 'conv50':[32,32,3,3],  'conv51':[128,32,1,1],
    'conv52':[64,128,1,1], 'conv53':[64,64,3,3],  'conv54':[256,64,1,1], 'conv55':[256,128,1,1],
    'conv56':[64,256,1,1], 'conv57':[64,64,3,3],  'conv58':[256,64,1,1],
    'conv59':[64,256,1,1], 'conv60':[64,64,3,3],  'conv61':[256,64,1,1],
    'conv62':[64,256,1,1], 'conv63':[64,64,3,3],  'conv64':[256,64,1,1],
    'conv65':[64,256,1,1], 'conv66':[64,64,3,3],  'conv67':[256,64,1,1],
    'conv68':[64,256,1,1], 'conv69':[64,64,3,3],  'conv70':[256,64,1,1],
    'conv71':[64,256,1,1], 'conv72':[64,64,3,3],  'conv73':[256,64,1,1],
    'conv74':[64,256,1,1], 'conv75':[64,64,3,3],  'conv76':[256,64,1,1], 'fc':[100,256]
}

############## ImageNet ###############

resnet50 = {
        'conv1':[64,3,7,7],
        'conv2':[64,64,1,1],     'conv3':[64,64,3,3],     'conv4':[256,64,1,1],   'conv5':[256,64,1,1],
        'conv6':[64,256,1,1],    'conv7':[64,64,3,3],     'conv8':[256,64,1,1],
        'conv9':[64,256,1,1],    'conv10':[64,64,3,3],    'conv11':[256,64,1,1],
        'conv12':[128,256,1,1],  'conv13':[128,128,3,3],  'conv14':[512,128,1,1], 'conv15':[512,256,1,1],
        'conv16':[128,512,1,1],  'conv17':[128,128,3,3],  'conv18':[512,128,1,1],
        'conv19':[128,512,1,1],  'conv20':[128,128,3,3],  'conv21':[512,128,1,1],
        'conv22':[128,512,1,1],  'conv23':[128,128,3,3],  'conv24':[512,128,1,1],
        'conv25':[256,512,1,1],  'conv26':[256,256,3,3],  'conv27':[1024,256,1,1], 'conv28':[1024,512,1,1],
        'conv29':[256,1024,1,1], 'conv30':[256,256,3,3],  'conv31':[1024,256,1,1],
        'conv32':[256,1024,1,1], 'conv33':[256,256,3,3],  'conv34':[1024,256,1,1],
        'conv35':[256,1024,1,1], 'conv36':[256,256,3,3],  'conv37':[1024,256,1,1],
        'conv38':[256,1024,1,1], 'conv39':[256,256,3,3],  'conv40':[1024,256,1,1],
        'conv41':[256,1024,1,1], 'conv42':[256,256,3,3],  'conv43':[1024,256,1,1],
        'conv44':[512,1024,1,1], 'conv45':[512,512,3,3],  'conv46':[2048,512,1,1], 'conv47':[2048,1024,1,1],
        'conv48':[512,2048,1,1], 'conv49':[512,512,3,3],  'conv50':[2048,512,1,1], 
        'conv51':[512,2048,1,1], 'conv52':[512,512,3,3],  'conv53':[2048,512,1,1], 
        'fc':[1000,2048]
        }

mobilenet = {
        'conv1':[32, 3, 3, 3],     'conv2':[32, 1, 3, 3],     'conv3':[64, 32, 1, 1],    
        'conv4':[64, 1, 3, 3],     'conv5':[128, 64, 1, 1],   'conv6':[128, 1, 3, 3],    
        'conv7':[128, 128, 1, 1],  'conv8':[128, 1, 3, 3],    'conv9':[256, 128, 1, 1],  
        'conv10':[256, 1, 3, 3],    'conv11':[256, 256, 1, 1],  'conv12':[256, 1, 3, 3],    
        'conv13':[512, 256, 1, 1],  'conv14':[512, 1, 3, 3],    'conv15':[512, 512, 1, 1],  
        'conv16':[512, 1, 3, 3],    'conv17':[512, 512, 1, 1],  'conv18':[512, 1, 3, 3],    
        'conv19':[512, 512, 1, 1],  'conv20':[512, 1, 3, 3],    'conv21':[512, 512, 1, 1],  
        'conv22':[512, 1, 3, 3],    'conv23':[512, 512, 1, 1],  'conv24':[512, 1, 3, 3],    
        'conv25':[1024, 512, 1, 1], 'conv26':[1024, 1, 3, 3],   'conv27':[1024, 1024, 1, 1],
        'fc':[1000, 1024],
        }

base_archs = {
        'alexnet'       :alexnet,
        'vgg8'          :vgg8,
        'vgg11'         :vgg11,
        'vgg13'         :vgg13,
        'resnet20_flat'      :resnet20,
        'resnet32_flat'      :resnet32,
        'resnet32_bt_flat'   :resnet32_bt,
        'resnet32_bt_flat_temp'   :resnet32_bt,
        'resnet50_bt_flat'   :resnet50_bt,
        'resnet50'           :resnet50,
        'mobilenet'          :mobilenet,
        }

