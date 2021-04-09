import os
import yaml
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, required=True, help='path to validation dataset')
parser.add_argument('--dataset', default='imagenet', type=str, 
                    choices=['imagenet', 'cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--model', default = 'resnet50', type=str, help='model name')
parser.add_argument('--num-gpus', default=1, type=int, help='number of GPUs used in training')
parser.add_argument('--penalty-ratio', default=0.2, type=float, help='group lasso regularization penalty ratio')
args = parser.parse_args()

# Load configuration
if args.dataset == 'imagenet':
    assert args.model in ['resnet50', 'vgg16', 'mobilenet']
    cfg_file = "imagenet_{}.yaml".format(args.model)
    runfile = 'python src/imagenet.py'
elif args.dataset in ['cifar10', 'cifar100']:
    assert args.model in ['resnet50', 'resnet32', 'resnet20', 'vgg13', 'vgg11', 'vgg8']
    cfg_file = "cifar_{}.yaml".format(args.model)
    runfile = 'python src/cifar.py'
else:
    raise ValueError("{} is a wrong dataset".format(args.dataset))

with open(os.path.join('configs/', cfg_file)) as f_config:
   cfg = yaml.safe_load(f_config)
   if args.dataset.startswith('cifar'):
       cfg['base']['model_dir'].replace('cifar', args.dataset)

cfg['base']['learning-rate'] = cfg['base']['learning-rate'] *args.num_gpus
cfg['base']['train_batch']   = int(cfg['base']['train_batch']*args.num_gpus)
if args.dataset == 'imagenet':
    cfg['base']['learning-rate'] /= 4
    cfg['base']['train_batch']   /= 4

if not os.path.exists(cfg['base']['model_dir']):
    os.makedirs(cfg['base']['model_dir'])

# Data parallelism setup
gpu_id = '0'
for i in range(1,args.num_gpus):
    gpu_id +=','+str(i)

# Reconfigured network file naming and directory to store
arch_name = cfg['base']['arch']+'_'+cfg['base']['description']
arch_out_dir = os.path.join(cfg['base']['model_dir'], 'arch', cfg['base']['description'])

# Build command line
# Iterate reconfiguration intervals
for cur_epoch in range(0, cfg['base']['epochs'], cfg['pt']['sparse_interval']):
    cmd_line = runfile
    cmd_line += ' --workers '               +str(cfg['base']['workers'])
    cmd_line += ' --data_path '             +args.data_path if args.dataset == 'imagenet' else ''
    cmd_line += ' --dataset '               +args.dataset if args.dataset.startswith('cifar') else ''
    cmd_line += ' --epochs '                +str(cur_epoch + cfg['pt']['sparse_interval'])
    cmd_line += ' --learning-rate '         +str(cfg['base']['learning-rate'])
    cmd_line += ' --schedule '              +str(cfg['base']['schedule'])
    cmd_line += ' --checkpoint '            +os.path.join(cfg['base']['model_dir'], cfg['base']['description'])
    cmd_line += ' --arch '                  +cfg['base']['arch']
    cmd_line += ' --gpu-id '                +gpu_id
    cmd_line += ' --train_batch '           +str(cfg['base']['train_batch'])
    cmd_line += ' --test_batch '            +str(cfg['base']['test_batch'])
    cmd_line += ' --save_checkpoin '        +str(cfg['base']['save_checkpoint'])
    cmd_line += ' --resume '                +cfg['base']['resume'] if cfg['base']['resume'] != '' else ''

    cmd_line += ' --sparse_interval '       +str(cfg['pt']['sparse_interval'])
    cmd_line += ' --threshold '             +str(cfg['pt']['threshold'])
    cmd_line += ' --var_group_lasso_coeff ' +str(args.penalty_ratio)
    cmd_line += ' --arch_name '             +arch_name+'_'+str(cur_epoch+cfg['pt']['sparse_interval'])+'.py'
    cmd_line += ' --en_group_lasso '        if cfg['pt']['en_group_lasso'] else ''
    cmd_line += ' --arch_out_dir1 '         +cfg['base']['arch_dir']
    cmd_line += ' --arch_out_dir2 '         +arch_out_dir if cfg['pt']['reconf_arch'] else ''
    cmd_line += ' >> '                      +os.path.join(cfg['base']['model_dir'], cfg['base']['description'])+'.log'

    print (cmd_line)
    os.system(cmd_line)

    # Checkpoint to resume
    checkpoint = 'checkpoint.pth.tar'
    cfg['base']['resume'] = os.path.join(cfg['base']['model_dir'], cfg['base']['description'], 'checkpoint.pth.tar')
