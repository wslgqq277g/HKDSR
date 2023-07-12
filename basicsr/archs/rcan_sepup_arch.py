#

import copy
import torch
import os
import math
from basicsr.utils.registry import ARCH_REGISTRY

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        modules_body = []
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
@ARCH_REGISTRY.register()
class RCAN_sepup(nn.Module):
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
                 pretrained=True, conv=default_conv):
        super(RCAN_sepup, self).__init__()
        res_scale=1
        self.path=path
        self.n_resgroups = n_resgroups
        kernel_size = 3
        self.scale = scale
        act = nn.ReLU()
        self.tea=tea
        # RGB mean for DIV2K
        self.sub_mean = MeanShift(rgb_range)
        self.img_range = rgb_range
        if self.tea:
            self.group_id=[0,4,9]
        else:
            self.group_id=[0,2,4]

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]
        rgb_mean = (0.4488, 0.4371, 0.4040)
        # define body module
        for group_i in range(self.n_resgroups):
            setattr(self, 'body.{}'.format(str(group_i)), \
                    ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, \
                                  res_scale=res_scale, n_resblocks=n_resblocks))
            if group_i in self.group_id:
                setattr(self, 'body_c.{}'.format(str(group_i)), \
                        conv(n_feats, n_feats, kernel_size))
                setattr(self,'upsample.{}'.format(str(group_i)),nn.Sequential(
                    Upsampler(conv, self.scale, n_feats, act=False),
                    conv(n_feats, n_colors, kernel_size)))

        # self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        setattr(self, f'body.{self.n_resgroups}', conv(n_feats, n_feats, kernel_size))
        # define tail module
        modules_tail = [
            Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.add_mean = MeanShift(rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
        # self.group_id=group_id
        # if not pretrained:
        #     for group_id in range(self.n_resgroups):
        #         setattr(self, 'allevi_conv.{}'.format(str(group_id)), conv(n_feats, n_feats,1))

            # self.alleviate_conv=nn.Sequential(*[conv(n_feats, n_feats,1)])

        self.kd=kd
        self.pretrained=pretrained
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
                    for name,p in self.named_parameters():
                        p.requires_grad = True
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
            else:
                for name, p in self.named_parameters():
                    p.requires_grad = True

        # if self.pretrained==True:
        #     if self.tea==False:
        #         modules_tail_a = [
        #             Upsampler(conv, self.scale, n_feats, act=False),
        #             conv(n_feats, n_colors, kernel_size)]
        #
        #         self.upsampler_a = nn.Sequential(*modules_tail_a)

        # if self.kd:
        #     if pretrained:
        #         # for p in self.parameters():
        #         #     p.requires_grad = False
        #         if tea:
        #             self.load_pretrained()
        #             for name, p in self.named_parameters():
        #                     p.requires_grad = False
        #
        #         else:
        #             self.load_pretrained()
        #             for name,p in self.named_parameters():
        #                 if 'upsampler_a' in name:
        #                     p.requires_grad = True
        #                 else:
        #                     p.requires_grad = True
        #             for name,p in self.named_parameters():
        #                 if p.requires_grad==False:
        #                     print(name)
        #
        #     else:
        #         for name, p in self.named_parameters():
        #             p.requires_grad = True
        # else:
        #     if pretrained:
        #         # for p in self.parameters():
        #         #     p.requires_grad = False
        #         if tea:
        #             self.load_pretrained()
        #             for name, p in self.named_parameters():
        #                     p.requires_grad = False
        #
        #         else:
        #             self.load_pretrained()
        #             for name,p in self.named_parameters():
        #                 if 'ft' in name:
        #                     p.requires_grad = True
        #                 else:
        #                     p.requires_grad = True
        #     else:
        #         for name, p in self.named_parameters():
        #             p.requires_grad = True


    def forward(self, x):
        output_list = []
        # fea_list = []
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
            # if self.tea==False:
            if group_id in self.group_id :
                # if not self.pretrained:
                #     res_a = getattr(self, 'allevi_conv.{}'.format(str(group_id)))(res)
                #     feature_maps.append(res_a)
                # else:
                out = getattr(self,'body_c.{}'.format(str(group_id)))(res)

                out=out+x
                out = getattr(self,'upsample.{}'.format(str(group_id)))(out)
                # x = x / self.img_range + self.mean
                out = self.add_mean(out) / 255
                output_list.append(out)


            # else:
            #     fea_list.append(res)
            #     out = res + x
            #
            #     out = self.tail(out)
            #     # x = x / self.img_range + self.mean
            #     out = self.add_mean(out) / 255
            #     output_list.append(out)

        res = getattr(self, f'body.{self.n_resgroups}')(res)
        # # feature_maps.append(res)
        #
        #
        res += x

        x = self.tail(res)
        # x = x / self.img_range + self.mean
        x = self.add_mean(x) / 255
        # x = self.add_mean(x)
        output_list.append(x)
        # fea_list.append(x)
        return output_list

    def dict_copy(self, model_dict):
        state_dict = self.state_dict()
        for k, v in model_dict.items():
            if isinstance(v, nn.Parameter):
                v = v.data
            state_dict[k].copy_(v)


    def load_model(self, model_dict):
        model_state_dict = self.state_dict()
        pretrained_dict = {
            k: v
            for k, v in model_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        print(
            f"the prune number is {round((len(model_state_dict.keys()) - len(pretrained_dict.keys())) * 100 / len(model_state_dict.keys()), 3)}%"
        )
        print("missing keys:")
        # print('ft'*50)
        for key in model_state_dict.keys():
            if key not in pretrained_dict:
                print(key)
        self.dict_copy(pretrained_dict)

    def load_state_dict_teacher(self, state_dict):
        self.load_model(state_dict)

    def load_state_dict_student(self, state_dict):
        self.load_model(state_dict)
    def load_pretrained(self):
        if int(self.scale) == 2:
            print("loading RCANx2")

            print('finetuning start!')
            dict=torch.load(self.path)
            # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
            # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
            # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
            self.load_model(dict)
        elif int(self.scale) == 3:
            print("loading RCANx3")
            # dict=torch.load(self.path+'/checkpoints/RCAN_BIX3.pt')
            # self.load_model(dict)
            if self.kd:
                if self.tea:
                    print('ft_KD!')
                    dict=torch.load(self.path)
                    # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
                    # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
                    # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
                    self.load_model(dict)
                else:
                    print('ft_pretrain!')

                    dict=torch.load(self.path)
                    self.load_model(dict['params_ema'])

                    # self.load_model(dict)
            else:
                if self.tea:
                    print('ft_KD!')
                    dict=torch.load(self.path)
                    # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
                    # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
                    # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
                    self.load_model(dict)
                else:
                    print('ft_pretrain!')

                    dict=torch.load(self.path)

                    self.load_model(dict)


        elif int(self.scale) == 4:
            print("loading RCANx4")
            # dict=torch.load(self.path+'/checkpoints/RCAN_BIX4.pt')
            # self.load_model(dict)
            if self.kd:
                if self.tea:
                    print('ft_KD!')
                    dict=torch.load(self.path)
                    # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
                    # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
                    # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
                    self.load_model(dict)
                else:
                    print('ft_pretrain!')

                    dict=torch.load(self.path)
                    self.load_model(dict['params_ema'])

                    # self.load_model(dict)
            else:
                if self.tea:
                    print('ft_KD!')
                    dict=torch.load(self.path)
                    # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
                    # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
                    # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
                    self.load_model(dict)
                else:
                    print('ft_pretrain!')

                    dict=torch.load(self.path)

                    self.load_model(dict)

#
#     def dict_copy(self, model_dict):
#         state_dict = self.state_dict()
#         for k, v in model_dict.items():
#             if isinstance(v, nn.Parameter):
#                 v = v.data
#             state_dict[k].copy_(v)
#
#
#     def load_model(self, model_dict):
#         model_state_dict = self.state_dict()
#         if self.tea:
#             pretrained_dict = {
#                 k: v
#                 for k, v in model_dict.items()
#                 if k in model_state_dict and v.shape == model_state_dict[k].shape
#             }
#             # if self.tea==False:
#             #     for name,_ in model_state_dict.items():
#             #         if 'upsampler_a' in name:
#             #             pretrained_dict[name]=model_dict[name.replace('upsampler_a','tail')]
#             print(
#                 f"the prune number is {round((len(model_state_dict.keys()) - len(pretrained_dict.keys())) * 100 / len(model_state_dict.keys()), 3)}%"
#             )
#             print("missing keys:")
#             # print('ft'*50)
#             for key in model_state_dict.keys():
#                 if key not in pretrained_dict:
#                     print(key)
#
#         else:
#             predict={}
#             pretrained_dict={}
#             for name,p in model_state_dict.items():
#                 if 'upsampler_a' in name:
#                     pretrained_dict[name]=model_dict[name.replace('upsampler_a','tail')]
#                     predict[name]=p
#         # print("missing keys:")
#         # # print('ft'*50)
#         # for key in model_state_dict.keys():
#         #     if key not in pretrained_dict:
#         #         print(key)
#         self.dict_copy(pretrained_dict)
#         # if self.tea==False:
#         #     for name,p in predict.items():
#         #         if 'bias' in name:
#         #             print(p)
#         #             print(pretrained_dict[name])
#     def load_state_dict_teacher(self, state_dict):
#         self.load_model(state_dict)
#
#     def load_state_dict_student(self, state_dict):
#         self.load_model(state_dict)
#     def load_pretrained(self):
#         if int(self.scale) == 2:
#             print("loading RCANx2")
#             if self.tea:
#                 print('ft_KD!')
#                 dict=torch.load(self.path)
#                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                 # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                 self.load_model(dict)
#             else:
#                 print('ft_pretrain!')
#
#                 dict=torch.load(self.path)
#
#                 self.load_model(dict)
#         elif int(self.scale) == 3:
#             print("loading RCANx3")
#             # dict=torch.load(self.path+'/checkpoints/RCAN_BIX3.pt')
#             # self.load_model(dict)
#             if self.kd:
#                 if self.tea:
#                     print('ft_KD!')
#                     dict=torch.load(self.path)
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                     self.load_model(dict)
#                 else:
#                     print('ft_pretrain!')
#
#                     dict=torch.load(self.path)
#                     self.load_model(dict)
#
#                     # self.load_model(dict)
#             else:
#                 if self.tea:
#                     print('ft_KD!')
#                     dict=torch.load(self.path)
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                     self.load_model(dict)
#                 else:
#                     print('ft_pretrain!')
#
#                     dict=torch.load(self.path)
#
#                     self.load_model(dict)
#
#
#         elif int(self.scale) == 4:
#             print("loading RCANx4")
#             # dict=torch.load(self.path+'/checkpoints/RCAN_BIX4.pt')
#             # self.load_model(dict)
#             if self.kd:
#                 if self.tea:
#                     print('ft_KD!')
#                     dict=torch.load(self.path)
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                     self.load_model(dict)
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
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                     self.load_model(dict)
#                 else:
#                     print('ft_pretrain!')
#
#                     dict=torch.load(self.path)
#
#                     self.load_model(dict)
#
# #             if self.tea:
# #                 print('ft_KD!')
# #                 dict=torch.load(self.path)
# #                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
# #                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
# #                 # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
# #                 self.load_model(dict['params_ema'])
# #             else:
# #                 print('ft_pretrain!')
#
# #                 dict=torch.load(self.path)
#
# #                 self.load_model(dict)
#
#         elif int(self.scale) == 8:
#             print("loading RCANx8")
#             # dict=torch.load(self.path+'/checkpoints/RCAN_BIX8.pt')
#             # self.load_model(dict)
#             if self.tea:
#                 print('ft_KD!')
#                 dict=torch.load(self.path)
#                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                 # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                 self.load_model(dict['params_ema'])
#             else:
#                 print('ft_pretrain!')
#
#                 dict=torch.load(self.path)
#
#                 self.load_model(dict)
#


#
#
# import copy
# import torch
# import os
# import math
# from basicsr.utils.registry import ARCH_REGISTRY
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def default_conv(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size // 2), bias=bias)
#
#
# class MeanShift(nn.Conv2d):
#     def __init__(
#             self, rgb_range,
#             rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
#         for p in self.parameters():
#             p.requires_grad = False
#
#
# class BasicBlock(nn.Sequential):
#     def __init__(
#             self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
#             bn=True, act=nn.ReLU(True)):
#
#         m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
#         if bn:
#             m.append(nn.BatchNorm2d(out_channels))
#         if act is not None:
#             m.append(act)
#
#         super(BasicBlock, self).__init__(*m)
#
#
# class ResBlock(nn.Module):
#     def __init__(
#             self, conv, n_feats, kernel_size,
#             bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
#             if bn:
#                 m.append(nn.BatchNorm2d(n_feats))
#             if i == 0:
#                 m.append(act)
#
#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#
#         res = self.body(x).mul(self.res_scale)
#         res += x
#
#         return res
#
#
# class Upsampler(nn.Sequential):
#     def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
#
#         m = []
#         if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(conv(n_feats, 4 * n_feats, 3, bias))
#                 m.append(nn.PixelShuffle(2))
#                 if bn:
#                     m.append(nn.BatchNorm2d(n_feats))
#                 if act == 'relu':
#                     m.append(nn.ReLU(True))
#                 elif act == 'prelu':
#                     m.append(nn.PReLU(n_feats))
#
#         elif scale == 3:
#             m.append(conv(n_feats, 9 * n_feats, 3, bias))
#             m.append(nn.PixelShuffle(3))
#             if bn:
#                 m.append(nn.BatchNorm2d(n_feats))
#             if act == 'relu':
#                 m.append(nn.ReLU(True))
#             elif act == 'prelu':
#                 m.append(nn.PReLU(n_feats))
#         else:
#             raise NotImplementedError
#
#         super(Upsampler, self).__init__(*m)
#
#
# ## Channel Attention (CA) Layer
# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
#
#
# ## Residual Channel Attention Block (RCAB)
# class RCAB(nn.Module):
#     def __init__(
#             self, conv, n_feat, kernel_size, reduction,
#             bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(RCAB, self).__init__()
#         modules_body = []
#         for i in range(2):
#             modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
#             if i == 0: modules_body.append(act)
#         modules_body.append(CALayer(n_feat, reduction))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         # print(x.shape,'*'*30)
#         # print(x.device,'*'*30)
#         # for name,param in self.named_parameters():
#         #     print(f'{name}:',param.shape,param.device)
#         res = self.body(x)
#         # res = self.body(x).mul(self.res_scale)
#         res += x
#         return res
#
#
# ## Residual Group (RG)
# class ResidualGroup(nn.Module):
#     def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
#         super(ResidualGroup, self).__init__()
#         modules_body = []
#         modules_body = [
#             RCAB(
#                 conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=1) \
#             for _ in range(n_resblocks)]
#         modules_body.append(conv(n_feat, n_feat, kernel_size))
#         self.body = nn.Sequential(*modules_body)
#
#     def forward(self, x):
#         res = self.body(x)
#         residual = res
#         res += x
#         # return res, residual
#         return res
#
#
# ## Residual Channel Attention Network (RCAN)
# @ARCH_REGISTRY.register()
# class RCAN_ftb(nn.Module):
#     def __init__(self,
#                  n_resgroups,
#                  n_resblocks,
#                  n_feats,
#                  reduction,
#                  scale,
#                  rgb_range,
#                  n_colors,
#                  path,
#                  kd=False,
#                  tea=False,
#                  pretrained=True, conv=default_conv):
#         super(RCAN_ftb, self).__init__()
#         res_scale=1
#         self.path=path
#
#         self.n_resgroups = n_resgroups
#         kernel_size = 3
#         self.scale = scale
#         act = nn.ReLU()
#         self.tea=tea
#         # RGB mean for DIV2K
#         self.sub_mean = MeanShift(rgb_range)
#         self.img_range = rgb_range
#
#         # define head module
#         modules_head = [conv(n_colors, n_feats, kernel_size)]
#         rgb_mean = (0.4488, 0.4371, 0.4040)
#         # define body module
#
#         for group_id in range(self.n_resgroups):
#             setattr(self, 'body.{}'.format(str(group_id)), \
#                     ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, \
#                                   res_scale=res_scale, n_resblocks=n_resblocks))
#         if self.tea==False:
#             # setattr(self, 'ft_body.10', conv(n_feats, n_feats, kernel_size))
#
#             setattr(self, 'ft_tail', \
#                     nn.Sequential(Upsampler(conv, self.scale, n_feats, act=False),
#                                   conv(n_feats, n_colors, kernel_size)))
#         # self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
#
#         setattr(self, f'body.{self.n_resgroups}', conv(n_feats, n_feats, kernel_size))
#         # define tail module
#         modules_tail = [
#             Upsampler(conv, self.scale, n_feats, act=False),
#             conv(n_feats, n_colors, kernel_size)]
#
#         self.add_mean = MeanShift(rgb_range, sign=1)
#
#         self.head = nn.Sequential(*modules_head)
#         self.tail = nn.Sequential(*modules_tail)
#         # self.group_id=[2,6]
#         # if not pretrained:
#         #     for group_id in range(self.n_resgroups):
#         #         setattr(self, 'allevi_conv.{}'.format(str(group_id)), conv(n_feats, n_feats,1))
#
#             # self.alleviate_conv=nn.Sequential(*[conv(n_feats, n_feats,1)])
#
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
#                         if 'ft_' not in name:
#                             p.requires_grad = True
#                         else:
#                             print(name,'11111111111'*50)
#                             p.requires_grad = False
#
#             else:
#                 for name, p in self.named_parameters():
#                     p.requires_grad = True
#         else:
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
#                         if 'ft' in name:
#                             p.requires_grad = True
#                         else:
#                             p.requires_grad = False
#             else:
#                 for name, p in self.named_parameters():
#                     p.requires_grad = True
#
#
#     def forward(self, x):
#         output_list = []
#         fea_list = []
#         # print(x.shape)
#         # x = self.sub_mean(x)
#         # print(2)
#         x = self.sub_mean(255 * x)
#         # self.mean = self.mean.type_as(x)
#         # x = (x - self.mean) * self.img_range
#
#         x = self.head(x)
#         # feature_maps.append(x)
#         res = x
#         for group_id in range(self.n_resgroups):
#             # print(res.shape, '*' * 30)
#             # print('groupid:',f'{group_id}','*'*30)
#             res = getattr(self, 'body.{}'.format(str(group_id)))(res)
#             if self.tea == False:
#
#                 if group_id==2 or group_id==6 :
#                     # if not self.pretrained:
#                     #     res_a = getattr(self, 'allevi_conv.{}'.format(str(group_id)))(res)
#                     #     feature_maps.append(res_a)
#                     # else:123
#
#                         fea_list.append(res)
#
#                         # output=getattr(self, 'ft_body.10')(res)
#                         output = res+x
#                         output=getattr(self, 'ft_tail')(output)
#                         # fea_list.append(output)
#                         output = self.add_mean(output) / 255
#                         output_list.append(output)
#             else:
#
#                 fea_list.append(res)
#
#                 # output = getattr(self, f'body.{self.n_resgroups}')(res)
#                 # # feature_maps.append(res)
#                 #
#                 #
#                 output =res+ x
#
#                 output = self.tail(output)
#                 # x = x / self.img_range + self.mean
#                 # fea_list.append(output)
#
#                 output = self.add_mean(output) / 255
#                 # x = self.add_mean(x)
#                 output_list.append(output)
#
#         res = getattr(self, f'body.{self.n_resgroups}')(res)
#         # # feature_maps.append(res)
#         #
#         # fea_list.append(res)
#
#         #
#         res =res+ x
#
#         x = self.tail(res)
#         # x = x / self.img_range + self.mean
#         # fea_list.append(x)
#
#         x = self.add_mean(x) / 255
#         fea_list.append(x)
#
#         # x = self.add_mean(x)
#         output_list.append(x)
#         return output_list,fea_list
#
#
#     def dict_copy(self, model_dict):
#         state_dict = self.state_dict()
#         for k, v in model_dict.items():
#             if isinstance(v, nn.Parameter):
#                 v = v.data
#             state_dict[k].copy_(v)
#
#
#     def load_model(self, model_dict):
#         model_state_dict = self.state_dict()
#         # for name,params in model_state_dict.items():
#
#         print('tail.0.0.weight' in model_state_dict,'*'*50)
#         pretrained_dict={}
#         if self.tea!=False:
#             pretrained_dict = {
#                 k: v
#                 for k, v in model_dict.items()
#                 if k in model_state_dict and v.shape == model_state_dict[k].shape
#             }
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
#             # for name,p in model_dict.items():
#             #     print(name,'111111111')
#             for k, _ in model_state_dict.items():
#                 if 'ft_' in k :
#                     c=k
#                     u=k.replace('ft_','')
#                     # print(u)
#                     # print(model_dict[u].shape,'model_dict[u]')
#                     pretrained_dict[k]=model_dict[u]
#                     # print(pretrained_dict[k].shape,'pretrained_dict[k]')
#                     if '1.bias' in k:
#                         print(model_state_dict[k],'*'*50)
#                         print(pretrained_dict[k], '99' * 50)
#             print(pretrained_dict.keys(),'keysss')
#             self.dict_copy(pretrained_dict)
#
#         model_state_dict1 = self.state_dict()
#         for name,params in model_state_dict1.items():
#             if 'ft' in name:
#                 if '1.bias' in name:
#                     print(model_state_dict1[name], '666' * 50)
#
#             # p1=model_state_dict1[name].cpu()
#             # p2=params.cpu()
#             # a=torch.sum(p1)
#             # b = torch.sum(p2)
#             # if a!=b:
#             #     print(name,'*'*50)
#             # # print(a==b)
#
#         # # for k, p in model_dict.items():
#         # #     print(k)
#         # for k, p in model_state_dict.items():
#         #
#         #     if 'ft_' in k:
#         #         u = k.replace('ft_', '')
#         #     else:
#         #         u=k
#         #     if 'ft_' in k:
#         #         if u in model_dict:
#         #             a1=model_state_dict[k].cpu()
#         #             a2=model_dict[u].cpu()
#         #             print(a1==a2)
#     def load_state_dict_teacher(self, state_dict):
#         self.load_model(state_dict)
#
#     def load_state_dict_student(self, state_dict):
#         self.load_model(state_dict)
#     def load_pretrained(self):
#         if int(self.scale) == 2:
#             print("loading RCANx2")
#             if self.tea:
#                 print('ft_KD!')
#                 dict=torch.load(self.path)
#                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                 # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                 self.load_model(dict['params_ema'])
#             else:
#                 print('ft_pretrain!')
#                 print(self.path,'self.path')
#                 dict=torch.load(self.path)
#
#                 self.load_model(dict['params_ema'])
#         elif int(self.scale) == 3:
#             print("loading RCANx3")
#             # dict=torch.load(self.path+'/checkpoints/RCAN_BIX3.pt')
#             # self.load_model(dict)
#             if self.kd:
#                 if self.tea:
#                     print('ft_KD!')
#                     dict=torch.load(self.path)
#                     self.load_model(dict)
#                 else:
#                     print('ft_pretrain!')
#
#                     dict=torch.load(self.path)
#                     self.load_model(dict)
#
#                     # self.load_model(dict)
#             else:
#                 if self.tea:
#                     print('ft_KD!')
#                     dict=torch.load(self.path)
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                     self.load_model(dict['params_ema'])
#                 else:
#                     print('ft_pretrain!')
#
#                     dict=torch.load(self.path)
#
#                     self.load_model(dict['params_ema'])
#
#
#         elif int(self.scale) == 4:
#             print("loading RCANx4")
#             # dict=torch.load(self.path+'/checkpoints/RCAN_BIX4.pt')
#             # self.load_model(dict)
#             if self.kd:
#                 if self.tea:
#                     print('ft_KD!')
#                     dict=torch.load(self.path)
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                     self.load_model(dict['params_ema'])
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
#                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                     self.load_model(dict['params_ema'])
#                 else:
#                     print('ft_pretrain!')
#
#                     dict=torch.load(self.path)
#
#                     self.load_model(dict)
#
# #             if self.tea:
# #                 print('ft_KD!')
# #                 dict=torch.load(self.path)
# #                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
# #                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
# #                 # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
# #                 self.load_model(dict['params_ema'])
# #             else:
# #                 print('ft_pretrain!')
#
# #                 dict=torch.load(self.path)
#
# #                 self.load_model(dict)
#
#         elif int(self.scale) == 8:
#             print("loading RCANx8")
#             # dict=torch.load(self.path+'/checkpoints/RCAN_BIX8.pt')
#             # self.load_model(dict)
#             if self.tea:
#                 print('ft_KD!')
#                 dict=torch.load(self.path)
#                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
#                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
#                 # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
#                 self.load_model(dict['params_ema'])
#             else:
#                 print('ft_pretrain!')
#
#                 dict=torch.load(self.path)
#
#                 self.load_model(dict)
#
#
# # import copy
# # import torch
# # import os
# # import math
# # from basicsr.utils.registry import ARCH_REGISTRY
# #
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# #
# # def default_conv(in_channels, out_channels, kernel_size, bias=True):
# #     return nn.Conv2d(
# #         in_channels, out_channels, kernel_size,
# #         padding=(kernel_size // 2), bias=bias)
# #
# #
# # class MeanShift(nn.Conv2d):
# #     def __init__(
# #             self, rgb_range,
# #             rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
# #         super(MeanShift, self).__init__(3, 3, kernel_size=1)
# #         std = torch.Tensor(rgb_std)
# #         self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
# #         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
# #         for p in self.parameters():
# #             p.requires_grad = False
# #
# #
# # class BasicBlock(nn.Sequential):
# #     def __init__(
# #             self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
# #             bn=True, act=nn.ReLU(True)):
# #
# #         m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
# #         if bn:
# #             m.append(nn.BatchNorm2d(out_channels))
# #         if act is not None:
# #             m.append(act)
# #
# #         super(BasicBlock, self).__init__(*m)
# #
# #
# # class ResBlock(nn.Module):
# #     def __init__(
# #             self, conv, n_feats, kernel_size,
# #             bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
# #
# #         super(ResBlock, self).__init__()
# #         m = []
# #         for i in range(2):
# #             m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
# #             if bn:
# #                 m.append(nn.BatchNorm2d(n_feats))
# #             if i == 0:
# #                 m.append(act)
# #
# #         self.body = nn.Sequential(*m)
# #         self.res_scale = res_scale
# #
# #     def forward(self, x):
# #
# #         res = self.body(x).mul(self.res_scale)
# #         res += x
# #
# #         return res
# #
# #
# # class Upsampler(nn.Sequential):
# #     def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
# #
# #         m = []
# #         if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
# #             for _ in range(int(math.log(scale, 2))):
# #                 m.append(conv(n_feats, 4 * n_feats, 3, bias))
# #                 m.append(nn.PixelShuffle(2))
# #                 if bn:
# #                     m.append(nn.BatchNorm2d(n_feats))
# #                 if act == 'relu':
# #                     m.append(nn.ReLU(True))
# #                 elif act == 'prelu':
# #                     m.append(nn.PReLU(n_feats))
# #
# #         elif scale == 3:
# #             m.append(conv(n_feats, 9 * n_feats, 3, bias))
# #             m.append(nn.PixelShuffle(3))
# #             if bn:
# #                 m.append(nn.BatchNorm2d(n_feats))
# #             if act == 'relu':
# #                 m.append(nn.ReLU(True))
# #             elif act == 'prelu':
# #                 m.append(nn.PReLU(n_feats))
# #         else:
# #             raise NotImplementedError
# #
# #         super(Upsampler, self).__init__(*m)
# #
# #
# # ## Channel Attention (CA) Layer
# # class CALayer(nn.Module):
# #     def __init__(self, channel, reduction=16):
# #         super(CALayer, self).__init__()
# #         # global average pooling: feature --> point
# #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
# #         # feature channel downscale and upscale --> channel weight
# #         self.conv_du = nn.Sequential(
# #             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
# #             nn.Sigmoid()
# #         )
# #
# #     def forward(self, x):
# #         y = self.avg_pool(x)
# #         y = self.conv_du(y)
# #         return x * y
# #
# #
# # ## Residual Channel Attention Block (RCAB)
# # class RCAB(nn.Module):
# #     def __init__(
# #             self, conv, n_feat, kernel_size, reduction,
# #             bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
# #
# #         super(RCAB, self).__init__()
# #         modules_body = []
# #         for i in range(2):
# #             modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
# #             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
# #             if i == 0: modules_body.append(act)
# #         modules_body.append(CALayer(n_feat, reduction))
# #         self.body = nn.Sequential(*modules_body)
# #         self.res_scale = res_scale
# #
# #     def forward(self, x):
# #         # print(x.shape,'*'*30)
# #         # print(x.device,'*'*30)
# #         # for name,param in self.named_parameters():
# #         #     print(f'{name}:',param.shape,param.device)
# #         res = self.body(x)
# #         # res = self.body(x).mul(self.res_scale)
# #         res += x
# #         return res
# #
# #
# # ## Residual Group (RG)
# # class ResidualGroup(nn.Module):
# #     def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
# #         super(ResidualGroup, self).__init__()
# #         modules_body = []
# #         modules_body = [
# #             RCAB(
# #                 conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=1) \
# #             for _ in range(n_resblocks)]
# #         modules_body.append(conv(n_feat, n_feat, kernel_size))
# #         self.body = nn.Sequential(*modules_body)
# #
# #     def forward(self, x):
# #         res = self.body(x)
# #         residual = res
# #         res += x
# #         # return res, residual
# #         return res
# #
# #
# # ## Residual Channel Attention Network (RCAN)
# # @ARCH_REGISTRY.register()
# # class RCAN_ftb(nn.Module):
# #     def __init__(self,
# #                  n_resgroups,
# #                  n_resblocks,
# #                  n_feats,
# #                  reduction,
# #                  scale,
# #                  rgb_range,
# #                  n_colors,
# #                  path,
# #                  kd=False,
# #                  tea=False,
# #                  pretrained=True, conv=default_conv):
# #         super(RCAN_ftb, self).__init__()
# #         res_scale=1
# #         self.path=path
# #
# #         self.n_resgroups = n_resgroups
# #         kernel_size = 3
# #         self.scale = scale
# #         act = nn.ReLU()
# #         self.tea=tea
# #         # RGB mean for DIV2K
# #         self.sub_mean = MeanShift(rgb_range)
# #         self.img_range = rgb_range
# #
# #         # define head module
# #         modules_head = [conv(n_colors, n_feats, kernel_size)]
# #         rgb_mean = (0.4488, 0.4371, 0.4040)
# #         # define body module
# #
# #         for group_id in range(self.n_resgroups):
# #             setattr(self, 'body.{}'.format(str(group_id)), \
# #                     ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, \
# #                                   res_scale=res_scale, n_resblocks=n_resblocks))
# #         if self.tea==False:
# #             setattr(self, 'ft_body.10', conv(n_feats, n_feats, kernel_size))
# #
# #             setattr(self, 'ft_tail', \
# #                     nn.Sequential(Upsampler(conv, self.scale, n_feats, act=False),
# #                                   conv(n_feats, n_colors, kernel_size)))
# #         # self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
# #
# #         setattr(self, f'body.{self.n_resgroups}', conv(n_feats, n_feats, kernel_size))
# #         # define tail module
# #         modules_tail = [
# #             Upsampler(conv, self.scale, n_feats, act=False),
# #             conv(n_feats, n_colors, kernel_size)]
# #
# #         self.add_mean = MeanShift(rgb_range, sign=1)
# #
# #         self.head = nn.Sequential(*modules_head)
# #         self.tail = nn.Sequential(*modules_tail)
# #         # self.group_id=[2,6]
# #         # if not pretrained:
# #         #     for group_id in range(self.n_resgroups):
# #         #         setattr(self, 'allevi_conv.{}'.format(str(group_id)), conv(n_feats, n_feats,1))
# #
# #             # self.alleviate_conv=nn.Sequential(*[conv(n_feats, n_feats,1)])
# #
# #         self.kd=kd
# #         self.pretrained=pretrained
# #         if self.kd:
# #             if pretrained:
# #                 # for p in self.parameters():
# #                 #     p.requires_grad = False
# #                 if tea:
# #                     self.load_pretrained()
# #                     for name, p in self.named_parameters():
# #                             p.requires_grad = False
# #
# #                 else:
# #                     self.load_pretrained()
# #                     for name,p in self.named_parameters():
# #                         if 'ft_' not in name:
# #                             p.requires_grad = True
# #                         else:
# #                             p.requires_grad = False
# #
# #             else:
# #                 for name, p in self.named_parameters():
# #                     p.requires_grad = True
# #         else:
# #             if pretrained:
# #                 # for p in self.parameters():
# #                 #     p.requires_grad = False
# #                 if tea:
# #                     self.load_pretrained()
# #                     for name, p in self.named_parameters():
# #                             p.requires_grad = False
# #
# #                 else:
# #                     self.load_pretrained()
# #                     for name,p in self.named_parameters():
# #                         if 'ft' in name:
# #                             p.requires_grad = True
# #                         else:
# #                             p.requires_grad = False
# #             else:
# #                 for name, p in self.named_parameters():
# #                     p.requires_grad = True
# #
# #
# #     def forward(self, x):
# #         output_list = []
# #         fea_list = []
# #         # print(x.shape)
# #         # x = self.sub_mean(x)
# #         # print(2)
# #         x = self.sub_mean(255 * x)
# #         # self.mean = self.mean.type_as(x)
# #         # x = (x - self.mean) * self.img_range
# #
# #         x = self.head(x)
# #         # feature_maps.append(x)
# #         res = x
# #         for group_id in range(self.n_resgroups):
# #             # print(res.shape, '*' * 30)
# #             # print('groupid:',f'{group_id}','*'*30)
# #             res = getattr(self, 'body.{}'.format(str(group_id)))(res)
# #             if group_id==4 or group_id==6 :
# #                 # if not self.pretrained:
# #                 #     res_a = getattr(self, 'allevi_conv.{}'.format(str(group_id)))(res)
# #                 #     feature_maps.append(res_a)
# #                 # else:123
# #                 if self.tea==False:
# #                     output=getattr(self, 'ft_body.10')(res)
# #                     output += x
# #                     output=getattr(self, 'ft_tail')(output)
# #                     fea_list.append(output)
# #                     output = self.add_mean(output) / 255
# #                     output_list.append(output)
# #                 else:
# #
# #                     output = getattr(self, f'body.{self.n_resgroups}')(res)
# #                     # # feature_maps.append(res)
# #                     #
# #                     #
# #                     output += x
# #
# #                     output = self.tail(output)
# #                     # x = x / self.img_range + self.mean
# #                     fea_list.append(output)
# #
# #                     output = self.add_mean(output) / 255
# #                     # x = self.add_mean(x)
# #                     output_list.append(output)
# #
# #         res = getattr(self, f'body.{self.n_resgroups}')(res)
# #         # # feature_maps.append(res)
# #         #
# #         #
# #         res += x
# #
# #         x = self.tail(res)
# #         # x = x / self.img_range + self.mean
# #         fea_list.append(x)
# #
# #         x = self.add_mean(x) / 255
# #         # x = self.add_mean(x)
# #         output_list.append(x)
# #         return output_list,fea_list
# #
# #
# #     def dict_copy(self, model_dict):
# #         state_dict = self.state_dict()
# #         for k, v in model_dict.items():
# #             if isinstance(v, nn.Parameter):
# #                 v = v.data
# #             state_dict[k].copy_(v)
# #
# #
# #     def load_model(self, model_dict):
# #         model_state_dict = self.state_dict()
# #         pretrained_dict={}
# #         if self.tea!=False:
# #             pretrained_dict = {
# #                 k: v
# #                 for k, v in model_dict.items()
# #                 if k in model_state_dict and v.shape == model_state_dict[k].shape
# #             }
# #             print(
# #                 f"the prune number is {round((len(model_state_dict.keys()) - len(pretrained_dict.keys())) * 100 / len(model_state_dict.keys()), 3)}%"
# #             )
# #             print("missing keys:")
# #             # print('ft'*50)
# #             for key in model_state_dict.keys():
# #                 if key not in pretrained_dict:
# #                     print(key)
# #             self.dict_copy(pretrained_dict)
# #         else:
# #             # for name,p in model_dict.items():
# #             #     print(name,'111111111')
# #             for k, _ in model_state_dict.items():
# #                 if 'ft_' in k :
# #                     u=k.replace('ft_','')
# #                     pretrained_dict[k]=model_dict[u]
# #             print(pretrained_dict.keys(),'keysss')
# #             self.dict_copy(pretrained_dict)
# #
# #     def load_state_dict_teacher(self, state_dict):
# #         self.load_model(state_dict)
# #
# #     def load_state_dict_student(self, state_dict):
# #         self.load_model(state_dict)
# #     def load_pretrained(self):
# #         if int(self.scale) == 2:
# #             print("loading RCANx2")
# #             if self.tea:
# #                 print('ft_KD!')
# #                 dict=torch.load(self.path)
# #                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
# #                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
# #                 # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
# #                 self.load_model(dict['params_ema'])
# #             else:
# #                 print('ft_pretrain!')
# #
# #                 dict=torch.load(self.path)
# #
# #                 self.load_model(dict)
# #         elif int(self.scale) == 3:
# #             print("loading RCANx3")
# #             # dict=torch.load(self.path+'/checkpoints/RCAN_BIX3.pt')
# #             # self.load_model(dict)
# #             if self.kd:
# #                 if self.tea:
# #                     print('ft_KD!')
# #                     dict=torch.load(self.path)
# #                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
# #                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
# #                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
# #                     self.load_model(dict['params_ema'])
# #                 else:
# #                     print('ft_pretrain!')
# #
# #                     dict=torch.load(self.path)
# #                     self.load_model(dict['params_ema'])
# #
# #                     # self.load_model(dict)
# #             else:
# #                 if self.tea:
# #                     print('ft_KD!')
# #                     dict=torch.load(self.path)
# #                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
# #                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
# #                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
# #                     self.load_model(dict['params_ema'])
# #                 else:
# #                     print('ft_pretrain!')
# #
# #                     dict=torch.load(self.path)
# #
# #                     self.load_model(dict)
# #
# #
# #         elif int(self.scale) == 4:
# #             print("loading RCANx4")
# #             # dict=torch.load(self.path+'/checkpoints/RCAN_BIX4.pt')
# #             # self.load_model(dict)
# #             if self.kd:
# #                 if self.tea:
# #                     print('ft_KD!')
# #                     dict=torch.load(self.path)
# #                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
# #                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
# #                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
# #                     self.load_model(dict['params_ema'])
# #                 else:
# #                     print('ft_pretrain!')
# #
# #                     dict=torch.load(self.path)
# #                     self.load_model(dict['params_ema'])
# #
# #                     # self.load_model(dict)
# #             else:
# #                 if self.tea:
# #                     print('ft_KD!')
# #                     dict=torch.load(self.path)
# #                     # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
# #                     # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
# #                     # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
# #                     self.load_model(dict['params_ema'])
# #                 else:
# #                     print('ft_pretrain!')
# #
# #                     dict=torch.load(self.path)
# #
# #                     self.load_model(dict)
# #
# # #             if self.tea:
# # #                 print('ft_KD!')
# # #                 dict=torch.load(self.path)
# # #                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
# # #                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
# # #                 # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
# # #                 self.load_model(dict['params_ema'])
# # #             else:
# # #                 print('ft_pretrain!')
# #
# # #                 dict=torch.load(self.path)
# #
# # #                 self.load_model(dict)
# #
# #         elif int(self.scale) == 8:
# #             print("loading RCANx8")
# #             # dict=torch.load(self.path+'/checkpoints/RCAN_BIX8.pt')
# #             # self.load_model(dict)
# #             if self.tea:
# #                 print('ft_KD!')
# #                 dict=torch.load(self.path)
# #                 # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
# #                 # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
# #                 # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
# #                 self.load_model(dict['params_ema'])
# #             else:
# #                 print('ft_pretrain!')
# #
# #                 dict=torch.load(self.path)
# #
# #                 self.load_model(dict)
# #
