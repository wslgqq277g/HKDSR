#

import copy
import torch
import os
import math
# from basicsr.utils.registry import ARCH_REGISTRY

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


#
# def make_layer(basic_block, num_basic_block, **kwarg):
#     """Make layers by stacking the same blocks.
#
#     Args:
#         basic_block (nn.module): nn.module class for basic block.
#         num_basic_block (int): number of blocks.
#
#     Returns:
#         nn.Sequential: Stacked blocks in nn.Sequential.
#     """
#     layers = []
#     for _ in range(num_basic_block):
#         layers.append(basic_block(**kwarg))
#     return nn.Sequential(*layers)
#
#
# class ResidualBlockNoBN(nn.Module):
#     """Residual block without BN.
#
#     Args:
#         num_feat (int): Channel number of intermediate features.
#             Default: 64.
#         res_scale (float): Residual scale. Default: 1.
#         pytorch_init (bool): If set to True, use pytorch default init,
#             otherwise, use default_init_weights. Default: False.
#     """
#
#     def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
#         super(ResidualBlockNoBN, self).__init__()
#         self.res_scale = res_scale
#         self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
#         self.relu = nn.ReLU(inplace=True)
#
#         if not pytorch_init:
#             default_init_weights([self.conv1, self.conv2], 0.1)
#
#     def forward(self, x):
#         identity = x
#         out = self.conv2(self.relu(self.conv1(x)))
#         return identity + out * self.res_scale
#
#
# class Upsample(nn.Sequential):
#     """Upsample module.
#
#     Args:
#         scale (int): Scale factor. Supported scales: 2^n and 3.
#         num_feat (int): Channel number of intermediate features.
#     """
#
#     def __init__(self, scale, num_feat):
#         m = []
#         if (scale & (scale - 1)) == 0:  # scale = 2^n
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
#                 m.append(nn.PixelShuffle(2))
#         elif scale == 3:
#             m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
#             m.append(nn.PixelShuffle(3))
#         else:
#             raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
#         super(Upsample, self).__init__(*m)
#
# class EDSR_ftb(nn.Module):
#     """EDSR network structure.
#
#     Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
#     Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch
#
#     Args:
#         num_in_ch (int): Channel number of inputs.
#         num_out_ch (int): Channel number of outputs.
#         num_feat (int): Channel number of intermediate features.
#             Default: 64.
#         num_block (int): Block number in the trunk network. Default: 16.
#         upscale (int): Upsampling factor. Support 2^n and 3.
#             Default: 4.
#         res_scale (float): Used to scale the residual in residual block.
#             Default: 1.
#         img_range (float): Image range. Default: 255.
#         rgb_mean (tuple[float]): Image mean in RGB orders.
#             Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
#     """
#
#     def __init__(self,
#                  num_in_ch,
#                  num_out_ch,
#
#                  path=None,
#                  num_feat=64,
#                  num_block=16,
#                  upscale=4,
#                  res_scale=1,
#                  img_range=255.,
#                  pretrained=False,
#                  tea=False,
#                  group_id=[],
#
#                  kd=False,
#                  rgb_mean=(0.4488, 0.4371, 0.4040)):
#         super(EDSR_ftb, self).__init__()
#         if len(group_id)!=0:
#             self.group=[round(num_block*i/10) for i in group_id]
#         else:
#             self.group=[]
#
#         self.tea=tea
#         self.upscale=upscale
#         self.num_block=num_block
#         self.img_range = img_range
#         self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
#         self.path=path
#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#
#         for i in range(num_block):
#             setattr(self,f'body.{i}',make_layer(ResidualBlockNoBN, 1, num_feat=num_feat, res_scale=res_scale, pytorch_init=True))
#         if self.tea == False:
#         #     self.group = [round(num_block * i) for i in group_id]
#         #
#         #     if i in self.group:
#             setattr(self, 'up_conv_after_body',
#                     nn.Conv2d(num_feat, num_feat, 3, 1, 1))
#             setattr(self, 'up_upsample',
#                     Upsample(upscale, num_feat))
#             setattr(self, 'up_conv_last',
#                     nn.Conv2d(num_feat, num_out_ch, 3, 1, 1))
#
#         self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.upsample = Upsample(upscale, num_feat)
#         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#         self.kd=kd
#         self.pretrained=pretrained
#         if self.kd:
#             if pretrained:
#                 # for p in self.parameters():
#                 #     p.requires_grad = False
#                 if tea:
#                     self.load_pretrained()
#                     for name, p in self.named_parameters():
#                             p.requires_grad = False
#
#                 else:
#                     self.load_pretrained()
#                     for name,p in self.named_parameters():
#                         if 'up_' not in name:
#                             p.requires_grad = False
#                         else:
#                             p.requires_grad = True
#
#             else:
#                 for name, p in self.named_parameters():
#                     if 'up_' not in name:
#                         p.requires_grad = True
#                     else:
#                         p.requires_grad = False
#
#         else:
#             # print(pretrained,'pretrainedpretrained')
#             if pretrained:
#                 # for p in self.parameters():
#                 #     p.requires_grad = False
#                 if tea:
#                     self.load_pretrained()
#                     for name, p in self.named_parameters():
#                             p.requires_grad = False
#
#                 else:
#                     self.load_pretrained()
#                     for name,p in self.named_parameters():
#                         if 'up_' in name:
#                             p.requires_grad = False
#                         else:
#                             p.requires_grad = True
#             else:
#                 for name, p in self.named_parameters():
#                     print(name,'up_' not in name,'asdas')
#                     if 'up_' not in name:
#                         print(name)
#                         print(p.requires_grad)
#                         p.requires_grad = True
#                         print(p.requires_grad)
#
#                     else:
#                         p.requires_grad = False
#         for name, p in self.named_parameters():
#             # print(self.tea,'self.tea')
#             if self.tea==False:
#                 if p.requires_grad == False:
#                     print(name,'noupdate!!!!!!!!!!!')
#
#     def forward(self, x):
#         self.mean = self.mean.type_as(x)
#         output_list=[]
#         fea_list=[]
#         x = (x - self.mean) * self.img_range
#         x = self.conv_first(x)
#         res=x
#         # if self.tea:
#         #     print(len(fea_list),'llll')
#
#         for group_id in range(self.num_block):
#             res = getattr(self, 'body.{}'.format(str(group_id)))(res)
#             # if self.tea == False:
#             if self.tea == False:
#                 if group_id in self.group:
#
#                     output = getattr(self, 'up_conv_after_body')(res)
#                     output += x
#                     output = getattr(self, 'up_upsample')(output)
#                     output = getattr(self, 'up_conv_last')(output)
#                     fea_list.append(output)
#                     output = output / self.img_range + self.mean
#                     output_list.append(output)
#             else:
#                 output = self.conv_after_body(res)
#                 output += x
#                 output = self.upsample(output)
#                 output = self.conv_last(output)
#                 fea_list.append(output)
#                 output = output / self.img_range + self.mean
#                 output_list.append(output)
#
#         # if self.tea:
#         #     print(len(fea_list),'aaallll')
#
#         res = self.conv_after_body(res)
#
#         res += x
#
#         x = self.conv_last(self.upsample(res))
#         fea_list.append(x)
#
#         x = x / self.img_range + self.mean
#         output_list.append(x)
#
#         # if self.tea:
#         #     for i ,out in enumerate(output_list):
#         #         output_list[i]=(output_list[i]*0.8+output_list[-1]*0.2)
#         #     return output_list,fea_list
#         # else:
#         # if self.tea:
#         #     print(len(fea_list),'llll')
#         return output_list,fea_list
#
#
#     def dict_copy(self, model_dict):
#         state_dict = self.state_dict()
#         for k, v in model_dict.items():
#             # if self.tea==False:
#             #     print(k,'---name')
#             #     for name,p in state_dict.items():
#             #         if name==k:
#             #             print('!!!!yule')
#             if isinstance(v, nn.Parameter):
#                 v = v.data
#             state_dict[k].copy_(v)
#
#
#     def load_model(self, model_dict):
#         model_state_dict = self.state_dict()
#         pretrained_dict={}
#         # pretrained_dict = {
#         #     k: v
#         if self.tea:
#             for k, v in model_dict.items():
#                 if k in model_state_dict  and v.shape == model_state_dict[k].shape:
#                     pretrained_dict[k]=v
#                 elif k.replace('conv','0.conv',1) in model_state_dict  and v.shape == model_state_dict[k.replace('conv','0.conv',1)].shape:
#                     pretrained_dict[k.replace('conv','0.conv',1)] = v
#
#             # }
#             print(
#                 f"the prune number is {round((len(model_state_dict.keys()) - len(pretrained_dict.keys())) * 100 / len(model_state_dict.keys()), 3)}%"
#             )
#             print("missing keys:")
#             # print('ft'*50)
#             for key in model_state_dict.keys():
#                 if key not in pretrained_dict:
#                     print(key)
#             self.dict_copy(pretrained_dict)
#         else:
#             for k, _ in model_state_dict.items():
#                 if 'up_' in k :
#                     u=k.replace('up_','')
#                     pretrained_dict[k]=model_dict[u]
#             print(pretrained_dict.keys(),'keysss')
#             self.dict_copy(pretrained_dict)
#
#     def load_state_dict_teacher(self, state_dict):
#         self.load_model(state_dict)
#
#     def load_state_dict_student(self, state_dict):
#         self.load_model(state_dict)
#     def load_pretrained(self):
#         if int(self.upscale) == 2:
#             print("loading EDSRx2")
#             if self.tea:
#                 print('ft_KD!')
#                 dict=torch.load(self.path)
#                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/EDSR_BIX2.pt')
#                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/EDSRx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                 # dict=torch.load(self.path+'/checkpoints/EDSR_BIX2.pt')
#                 self.load_model(dict['params'])
#             else:
#                 print('ft_pretrain!')
#
#                 dict=torch.load(self.path)
#
#                 self.load_model(dict['params_ema'])
#         elif int(self.upscale) == 3:
#             print("loading EDSRx3")
#             # dict=torch.load(self.path+'/checkpoints/EDSR_BIX3.pt')
#             # self.load_model(dict)
#             if self.kd:
#                 if self.tea:
#                     print('ft_KD!')
#                     dict=torch.load(self.path)
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/EDSR_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/EDSRx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/EDSR_BIX2.pt')
#                     self.load_model(dict['params'])
#                 else:
#                     print('ft_pretrain!')
#
#                     dict=torch.load(self.path)
#                     self.load_model(dict['params_ema'])
#
#                     # self.load_model(dict)
#             else:
#                 if self.tea:
#                     print('ft_KD!')
#                     dict=torch.load(self.path)
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/EDSR_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/EDSRx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/EDSR_BIX2.pt')
#                     self.load_model(dict['params'])
#                 else:
#                     print('ft_pretrain!')
#
#                     dict=torch.load(self.path)
#
#                     self.load_model(dict['params_ema'])
#
#
#         elif int(self.upscale) == 4:
#             print("loading EDSRx4")
#             # dict=torch.load(self.path+'/checkpoints/EDSR_BIX4.pt')
#             # self.load_model(dict)
#             if self.kd:
#                 if self.tea:
#                     print('ft_KD!')
#                     dict=torch.load(self.path)
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/EDSR_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/EDSRx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/EDSR_BIX2.pt')
#                     self.load_model(dict['params'])
#                 else:
#                     print('ft_pretrain!')
#
#                     dict=torch.load(self.path)
#                     self.load_model(dict['params_ema'])
#
#                     # self.load_model(dict)
#             else:
#                 if self.tea:
#                     print('ft_KD!')
#                     dict=torch.load(self.path)
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/EDSR_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/EDSRx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/EDSR_BIX2.pt')
#                     self.load_model(dict['params_ema'])
#                 else:
#                     print('ft_pretrain!')
#
#                     dict=torch.load(self.path)
#
#                     self.load_model(dict['params'])
#
# #             if self.tea:
# #                 print('ft_KD!')
# #                 dict=torch.load(self.path)
# #                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/EDSR_BIX2.pt')
# #                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/EDSRx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
# #                 # dict=torch.load(self.path+'/checkpoints/EDSR_BIX2.pt')
# #                 self.load_model(dict['params_ema'])
# #             else:
# #                 print('ft_pretrain!')
#
# #                 dict=torch.load(self.path)
#
# #                 self.load_model(dict)
#
#         elif int(self.upscale) == 8:
#             print("loading EDSRx8")
#             # dict=torch.load(self.path+'/checkpoints/EDSR_BIX8.pt')
#             # self.load_model(dict)
#             if self.tea:
#                 print('ft_KD!')
#                 dict=torch.load(self.path)
#                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/EDSR_BIX2.pt')
#                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/EDSRx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                 # dict=torch.load(self.path+'/checkpoints/EDSR_BIX2.pt')
#                 self.load_model(dict['params_ema'])
#             else:
#                 print('ft_pretrain!')
#
#                 dict=torch.load(self.path)
#
#                 self.load_model(dict)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):

        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        # print(x.shape,'*'*30)
        # print(x.device,'*'*30)
        # for name,param in self.named_parameters():
        #     print(f'{name}:',param.shape,param.device)
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        # modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        residual = res
        res += x
        # return res, residual
        return res


## Residual Channel Attention Network (RCAN)
# @ARCH_REGISTRY.register()
class RCAN_ftb(nn.Module):
    def __init__(self,
                 n_resgroups,
                 n_resblocks,
                 n_feats,
                 reduction,
                 scale,
                 rgb_range,
                 n_colors,
                 path,
                 kd=False,
                 tea=False,
                 gau=False,
                 group_id=[2, 6],
                 pretrained=True, conv=default_conv):
        super(RCAN_ftb, self).__init__()
        self.gau = gau
        res_scale = 1
        self.path = path
        self.n_resgroups = n_resgroups
        kernel_size = 3
        self.scale = scale
        act = nn.ReLU()
        self.tea = tea
        # RGB mean for DIV2K
        self.sub_mean = MeanShift(rgb_range)
        self.img_range = rgb_range

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]
        rgb_mean = (0.4488, 0.4371, 0.4040)
        # define body module
        for group_i in range(self.n_resgroups):
            setattr(self, 'body.{}'.format(str(group_i)), \
                    ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, \
                                  res_scale=res_scale, n_resblocks=n_resblocks))
        # self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        setattr(self, f'body.{self.n_resgroups}', conv(n_feats, n_feats, kernel_size))
        # define tail module
        modules_tail = [
            Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.add_mean = MeanShift(rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
        self.group_id = group_id
        # if not pretrained:
        #     for group_id in range(self.n_resgroups):
        #         setattr(self, 'allevi_conv.{}'.format(str(group_id)), conv(n_feats, n_feats,1))

        # self.alleviate_conv=nn.Sequential(*[conv(n_feats, n_feats,1)])

        self.kd = kd
        self.pretrained = pretrained
        if self.kd:
            if pretrained:
                # for p in self.parameters():
                #     p.requires_grad = False
                if tea:
                    self.load_pretrained()
                    for name, p in self.named_parameters():
                        p.requires_grad = False

                else:
                    self.load_pretrained()
                    for name, p in self.named_parameters():
                        if 'upsampler_a' in name:
                            p.requires_grad = False
                        else:
                            p.requires_grad = True
                    for name, p in self.named_parameters():
                        if p.requires_grad == False:
                            print(name)

            else:
                for name, p in self.named_parameters():
                    p.requires_grad = True
        else:
            if pretrained:
                # for p in self.parameters():
                #     p.requires_grad = False
                if tea:
                    self.load_pretrained()
                    for name, p in self.named_parameters():
                        p.requires_grad = False

                else:
                    self.load_pretrained()
                    for name, p in self.named_parameters():
                        if 'ft' in name:
                            p.requires_grad = False
                        else:
                            p.requires_grad = True
            else:
                for name, p in self.named_parameters():
                    p.requires_grad = True

    def forward(self, x):
        output_list = []
        fea_list = []
        # print(x.shape)
        # x = self.sub_mean(x)
        # print(2)
        x = self.sub_mean(255 * x)
        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range

        x = self.head(x)
        # feature_maps.append(x)
        res = x
        for group_id in range(self.n_resgroups):
            # print(res.shape, '*' * 30)
            # print('groupid:',f'{group_id}','*'*30)
            res = getattr(self, 'body.{}'.format(str(group_id)))(res)
            if self.tea == False:
                if group_id in self.group_id:
                    # if not self.pretrained:
                    #     res_a = getattr(self, 'allevi_conv.{}'.format(str(group_id)))(res)
                    #     feature_maps.append(res_a)
                    # else:
                    kk = res
                    fea_list.append(kk)
                    # out =res+ x

                    # out = self.upsampler_a(out)
                    # # x = x / self.img_range + self.mean
                    # out = self.add_mean(out) / 255
                    # output_list.append(out)


            else:
                kk = res
                fea_list.append(kk)

                # fea_list.append(res)
                # out = res + x

                # out = self.tail(out)
                # # x = x / self.img_range + self.mean
                # out = self.add_mean(out) / 255
                # output_list.append(out)

        res = getattr(self, f'body.{self.n_resgroups}')(res)
        kk = res
        fea_list.append(kk)

        # fea_list.append(res)
        #
        #
        res += x

        x = self.tail(res)
        # x = x / self.img_range + self.mean
        x = self.add_mean(x) / 255
        # x = self.add_mean(x)

        if self.gau:
            output_list.append(self.gaussian_noise(x))
        else:
            output_list.append(x)
        # fea_list.append(x)
        # print(torch.max(x),'max')
        return output_list, fea_list

    def dict_copy(self, model_dict):
        state_dict = self.state_dict()
        for k, v in model_dict.items():
            if isinstance(v, nn.Parameter):
                v = v.data
            state_dict[k].copy_(v)

    def gaussian_noise(self, image, mean=0, sigma=0.01):
        # image=image/255
        device = image.device
        maxx = torch.max(image).cpu().item()
        image = image.cpu().numpy()
        noise = np.random.normal(mean, sigma, image.shape)
        gaussian_out = image + noise
        gaussian_out = np.clip(gaussian_out, 0, maxx)
        gaussian_out = torch.from_numpy(gaussian_out).to(device)
        # gaussian_out=np.uint8(gaussian_out*255)
        # noise=np.uint8(noise*255)
        return gaussian_out

    def load_model(self, model_dict):
        model_state_dict = self.state_dict()
        if self.tea:
            pretrained_dict = {
                k: v
                for k, v in model_dict.items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            # if self.tea==False:
            #     for name,_ in model_state_dict.items():
            #         if 'upsampler_a' in name:
            #             pretrained_dict[name]=model_dict[name.replace('upsampler_a','tail')]
            print(
                f"the prune number is {round((len(model_state_dict.keys()) - len(pretrained_dict.keys())) * 100 / len(model_state_dict.keys()), 3)}%"
            )
            print("missing keys:")
            # print('ft'*50)
            for key in model_state_dict.keys():
                if key not in pretrained_dict:
                    print(key)

        else:
            predict = {}
            pretrained_dict = {}
            for name, p in model_state_dict.items():
                if 'upsampler_a' in name:
                    pretrained_dict[name] = model_dict[name.replace('upsampler_a', 'tail')]
                    predict[name] = p
        # print("missing keys:")
        # # print('ft'*50)
        # for key in model_state_dict.keys():
        #     if key not in pretrained_dict:
        #         print(key)
        self.dict_copy(pretrained_dict)
        # if self.tea==False:
        #     for name,p in predict.items():
        #         if 'bias' in name:
        #             print(p)
        #             print(pretrained_dict[name])

    def load_state_dict_teacher(self, state_dict):
        self.load_model(state_dict)

    def load_state_dict_student(self, state_dict):
        self.load_model(state_dict)

    def load_pretrained(self):
        if int(self.scale) == 2:
            print("loading RCANx2")
            if self.tea:
                print('ft_KD!')
                dict = torch.load(self.path)
                # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
                # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
                # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
                self.load_model(dict['params_ema'])
            else:
                print('ft_pretrain!')

                dict = torch.load(self.path)

                self.load_model(dict)
        elif int(self.scale) == 3:
            print("loading RCANx3")
            # dict=torch.load(self.path+'/checkpoints/RCAN_BIX3.pt')
            # self.load_model(dict)
            if self.kd:
                if self.tea:
                    print('ft_KD!')
                    dict = torch.load(self.path)
                    # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
                    # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
                    # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
                    self.load_model(dict)
                else:
                    print('ft_pretrain!')

                    dict = torch.load(self.path)
                    self.load_model(dict)

                    # self.load_model(dict)
            else:
                if self.tea:
                    print('ft_KD!')
                    dict = torch.load(self.path)
                    # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
                    # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
                    # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
                    self.load_model(dict)
                else:
                    print('ft_pretrain!')

                    dict = torch.load(self.path)

                    self.load_model(dict)


        elif int(self.scale) == 4:
            print("loading RCANx4")
            # dict=torch.load(self.path+'/checkpoints/RCAN_BIX4.pt')
            # self.load_model(dict)
            if self.kd:
                if self.tea:
                    print('ft_KD!')
                    dict = torch.load(self.path)
                    # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
                    # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
                    # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
                    self.load_model(dict)
                else:
                    print('ft_pretrain!')

                    dict = torch.load(self.path)
                    self.load_model(dict)

                    # self.load_model(dict)
            else:
                if self.tea:
                    print('ft_KD!')
                    dict = torch.load(self.path)
                    # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
                    # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
                    # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
                    self.load_model(dict)
                else:
                    print('ft_pretrain!')

                    dict = torch.load(self.path)

                    self.load_model(dict)

        #             if self.tea:
        #                 print('ft_KD!')
        #                 dict=torch.load(self.path)
        #                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
        #                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
        #                 # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
        #                 self.load_model(dict['params_ema'])
        #             else:
        #                 print('ft_pretrain!')

        #                 dict=torch.load(self.path)

        #                 self.load_model(dict)

        elif int(self.scale) == 8:
            print("loading RCANx8")
            # dict=torch.load(self.path+'/checkpoints/RCAN_BIX8.pt')
            # self.load_model(dict)
            if self.tea:
                print('ft_KD!')
                dict = torch.load(self.path)
                # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
                # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
                # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
                self.load_model(dict['params_ema'])
            else:
                print('ft_pretrain!')

                dict = torch.load(self.path)

                self.load_model(dict)

if __name__ == '__main__':
    # from torchstat import stat
    from thop import profile
    # model = EDSR_ftb(
    #
    # 3,
    # 3,
    #
    # path = None,
    # num_feat = 16,
    # num_block = 16,
    # upscale = 3,
    # res_scale = 1,
    # img_range = 255.,
    # pretrained = False,
    # tea = False,
    # group_id = [],
    #
    # kd = False,)


    model = RCAN_ftb(
        10,
        20,
        64,
        16,
        3,
        255,
        3,
        None,
        kd=False,
        tea=False,
        gau=False,
        group_id=[2, 6],
        pretrained=False)
    input = torch.randn(1, 3, 84, 84)
    macs, params = profile(model, inputs=(input,))

    # flops, params = profile(model, inputs=(1, 3, 256,256))
    # from thop import clever_format

    # macs, params = clever_format([macs, params], "%.3f", verbose=False)
    print(
        "%.2f | %.2f" % ( params / (1000 ** 2), macs / (1000**3 ))
    )

    # print('FLOPs = ' + str(macs / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')

    # stat(model,(3,256,256))


