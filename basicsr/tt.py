import datetime
import logging
import math
import time
import torch
from os import path as osp
import os
from torch import nn
import torch
import torch.nn.functional as F
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

dir = os.path.split(os.path.realpath(__file__))[0]

opt={'model_type': 'SR_ftModel',  'num_gpu': 1,'is_train':False,'dist':False,
'network_g':{  'type': 'RCAN_ft',

     'n_resgroups':10,
     'n_resblocks':20,
    'n_feats':64,
    'reduction':16,
    'scale':2,
    'rgb_range':255,
    'n_colors':3,
'path':dir
    # 'rgb_mean': [0.4488, 0.4371, 0.4040]
}}

from pytorch_wavelets import DWTForward

_reduction_modes = ['none', 'mean', 'sum']
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

def wavelet_trans(fea):   #already checked its ok
    xfm = DWTForward(J=1, mode='zero', wave='haar')
    for p in xfm.parameters():
        p.requires_grad = False
    Yl, Yh = xfm(fea)
    fea_list=[]
    fea_list.append(Yl)
    # print(Yh[0].shape)
    for j in range(len(Yh)):
        for i in range(Yh[j].shape[2]):
            fea_list.append(Yh[j][:,:,i,:,:])
            # print(Yh[0][:,:,i,:,:].shape,'yyy')
            # print(Yh[0].shape, 'yy1111111111111y')
    return fea_list
dir=osp.join(osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir)),'checkpoints')
# dict_path=osp.join(dir,'net_g_170000.pth')
# cc=torch.load(dict_path)
# cc=torch.load('/home/glinrui/basic/checkpoints/rcan_net_g_70000.pth')
#
dict = torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_90000.pth')

for name,a in dict['params_ema'].items():
    print(name)
# model = build_model(opt)

# for name,param in cc.named_paramters():
#     print(name)

# root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
# print(osp.join(osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir)),'datasets/DIV2K/DIV2K_train_HR_sub'))
# c=torch.cat([torch.randn((1,2,3)),torch.randn((1,2,3))],dim=0)
# print(c)
# print(nn.Sigmoid()(c))
# print(F.softmax(nn.Sigmoid()(c),dim=0))
# print(F.softmax(c,dim=0))
# print(5/4*8)
# gt=torch.randn((1,3,5,5))
# s_gt=torch.randn((1,3,5,5))
# t_gt=torch.randn((1,3,5,5))
# gt_list = wavelet_trans(gt)  # [(b,c,h1,w1)*waveorder]
# loss = 0
# s_gt = s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2)
# t_gt = t_gt.view(t_gt.size(0), t_gt.size(1), -1).transpose(1, 2)
#
# for gt_trans in gt_list:
#     gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
#
#     print(gt_trans.shape,'gt_trans')
#     print(s_gt.shape,'s_gt')
#
#     simi_s = torch.bmm(s_gt, gt_trans)
#     simi_t = torch.bmm(t_gt, gt_trans)
#     loss += l1_loss(simi_s, simi_t)
# print(loss)
# a=torch.randn((1,2,3,3))
# print(a)
# c=F.softmax(a,dim=1)
# print(c)
# print(b)
# print(b*c[:,0,:,:])
# root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
# print(root_path,'root_path')
# # print(os.path.realpath(__file__))
# print(os.path.split(os.path.realpath(__file__))[0])

#当前文件所在的目录，即父路径
# print(os.path.split(os.path.realpath(__file__))[0])
#找到父路径下的其他文件，即同级的其他文件
# print(os.path.join(proDir,"config.ini"))


# model = build_model(opt)
# for name,param in model.net_g.named_parameters():
#   print(name)

# dict=torch.load('./checkpoints/RCAN_BIX2.pt')
# for name,v in dict.items():
#   print(name)

#
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt options/train/RCAN/train_RCAN_kd_x2.yml --launcher pytorch
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
# python -m torch.distributed.launch --nproc_per_node=6 --master_port=4321 train.py -opt options/train/RCAN/train_RCAN_l1_x2.yml --launcher pytorch
#
# # # CUDA_VISIBLE_DEVICES=0 \
# # # python train.py -opt /home/isalab303/XXX/basic/options/train/RCAN/train_RCAN_ft_x2.yml
# cp ./models/sr_l1_model.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/models/
# cp ./models/sr_ft_model.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/models/
# cp ./models/base_model.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/models/
# cp ./models/sr_l1_model.py /home/isalab303/.conda/envs/xxx/lib/python3.9/site-packages/basicsr/models/
# cp ./utils/options.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/utils/
# cp ./utils/options.py /home/isalab303/.conda/envs/xxx/lib/python3.9/site-packages/basicsr/utils/
# cp ./utils/options.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/utils/
# #
# # # cp ./losses/basic_loss.py /home/isalab303/.conda/envs/xxx/lib/python3.9/site-packages/basicsr/losses/
# cp ./archs/rcan_l1_arch.py /home/isalab303/.conda/envs/xxx/lib/python3.9/site-packages/basicsr/archs/
# cp ./archs/rcan_ft_arch.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/archs/
#
# cp ./losses/basic_loss.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/losses/

#
# CUDA_VISIBLE_DEVICES=0,1,2 \
# python -m torch.distributed.launch --nproc_per_node=3 --master_port=4321 train.py -opt /home/isalab303/XXX/basic/options/train/RCAN/train_RCAN_x2.yml --launcher pytorch
# python  test.py -opt /home/glinrui/basic/options/test/RCAN/test__FT_RCAN.yml
# python  test.py -opt /home/glinrui/basic/options/test/RCAN/test_RCAN.yml
#
#
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt /home/glinrui/basic/options/train/RCAN/train_RCAN_kd_x2.yml --launcher pytorch
#
#
# ##3080ti
# cp ./losses/basic_loss.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/losses/
# cp ./models/sr_kd_model.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/models/
# cp ./models/base_model.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/models/
# # cp ./utils/options.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/utils/
# cp ./archs/rcan_ft_arch.py /home/glinrui/.conda/envs/glr_39/lib/python3.10/site-packages/basicsr/archs/
#
# ##3090
cp ./losses/basic_loss.py /home/isalab305/.conda/envs/xxx/lib/python3.9/site-packages/basicsr/losses/
cp ./models/sr_kd_model.py /home/isalab305/.conda/envs/xxx/lib/python3.9/site-packages/basicsr/models/
cp ./models/sr_ft_model.py /home/isalab305/.conda/envs/xxx/lib/python3.9/site-packages/basicsr/models/
cp ./models/base_model.py /home/isalab305/.conda/envs/xxx/lib/python3.9/site-packages/basicsr/models/
cp ./utils/options.py /home/isalab305/.conda/envs/xxx/lib/python3.9/site-packages/basicsr/utils/
cp ./archs/edsr_ft_arch.py /home/isalab305/.conda/envs/xxx/lib/python3.9/site-packages/basicsr/archs/
#


cd XXX/basic/basicsr
conda activate xxx

CUDA_VISIBLE_DEVICES=4 \
python  train.py -opt options/train/RCAN/train_RCAN_l1_x3.yml
python  train.py -opt options/train/RCAN/train_RCAN_x3.yml

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4335 train.py -opt options/train/RCAN/4.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=2,3,4 \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=4322 train.py -opt options/train/EDSR/train_EDSR_onlyadapt_Mx2.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=5,6,7 \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=4323 train.py -opt options/train/EDSR/train_EDSR_onlysim_Mx2.yml --launcher pytorch
#
CUDA_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4325 train.py -opt options/train/EDSR/train_EDSR_l1_Mx2.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4325 train.py -opt options/train/SwinIR/train_SwinIR_light_SRx2_scratch.yml --launcher pytorch

#
# ##a6000
cp ./losses/basic_loss.py /home/glr965/anaconda3/envs/fakd/lib/python3.9/site-packages/basicsr/losses/
cp ./models/sr_kd_model.py /home/glr965/anaconda3/envs/fakd/lib/python3.9/site-packages/basicsr/models/
cp ./models/sr_kd_model.py /home/glr965/anaconda3/envs/fakd/lib/python3.9/site-packages/basicsr/models/
cp ./models/base_model.py /home/glr965/anaconda3/envs/fakd/lib/python3.9/site-packages/basicsr/models/
cp ./utils/options.py /home/glr965/anaconda3/envs/fakd/lib/python3.9/site-packages/basicsr/utils/
cp ./archs/rcan_ft_arch.py /home/glr965/anaconda3/envs/fakd/lib/python3.9/site-packages/basicsr/archs/
cp ./archs/edsr_ft_arch.py /home/glr965/anaconda3/envs/fakd/lib/python3.9/site-packages/basicsr/archs/
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4921 train.py -opt options/train/RCAN/train_RCAN_x2.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4322 train.py -opt options/train/RCAN/train_RCAN_l1_x2.yml --launcher pytorch


CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4923 train.py -opt options/train/SwinIR/train_SwinIR_cnnkdx2.yml --launcher pytorch

