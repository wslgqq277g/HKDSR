import os
import sys
import time
import torch
device=2
# full=torch.cuda.get_device_name(device)
# list=['3090','3080','2080','A100','TITAN','V100']
# for i in list:
#     if i in full:
#         devicename=i


# model='vgg16'
# # cmd0='wandb offline'
# # cmd1='python train.py    --model '+model+'  --gpu_id '+str(device)+'  --wandb_notes  '+model+devicename
cmd1='CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4327 train.py -opt options/train/EDSR/train_EDSR_l1_6_Mx2.yml --launcher pytorch'


# cmd2='python train_hsokd.py  --kd_weight 0.1 --consistency_rampup 300 --aux 2  --dknw 20 --ensemw 10  --dataset CIFAR100       --num_branches 4 --gpu_id '+str(device)+'  --wandb_notes vgg '
# cmd3='python train_hsokd.py  --kd_weight 0.1 --consistency_rampup 300 --aux 2  --dknw 30 --ensemw 10  --dataset CIFAR100       --num_branches 4 --gpu_id '+str(device)+'  --wandb_notes vgg '
# cmd4='python train_hsokd.py  --kd_weight 0.1 --consistency_rampup 300 --aux 2  --dknw 40 --ensemw 10  --dataset CIFAR100       --num_branches 4 --gpu_id '+str(device)+'  --wandb_notes vgg '
# # print(device)
# cmd_list=[cmd1,cmd2,cmd3,cmd4]
# # cmd_list=[cmd1,cmd2,cmd3,cmd4]
cmd_list=[cmd1]
sta1=2+4*device
sta2=1+4*device

def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    # print(gpu_status)
    gpu_memory = int(gpu_status[sta1].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[sta2].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory


def narrow_setup(cmd_list=None,interval=2):
    for cmd in cmd_list:
        gpu_power, gpu_memory = gpu_info()
        i = 0
        while gpu_memory > 500 :  # set waiting condition
            gpu_power, gpu_memory = gpu_info()
            i = i % 5
            symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
            gpu_power_str = 'gpu power:%d W |' % gpu_power
            gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
            sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
            sys.stdout.flush()
            time.sleep(interval)
            i += 1
        print('\n' + cmd)

        os.system(cmd)


if __name__ == '__main__':
    print(device)
    narrow_setup(cmd_list)
    # a,b=gpu_info()
    # print(a)
    # print(b)


