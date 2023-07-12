import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from pytorch_wavelets import DWTForward
import torch.nn.functional as F

_reduction_modes = ['none', 'mean', 'sum']


import numpy as np

@LOSS_REGISTRY.register()
class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""
    def __init__(self,loss_weight=1.0, reduction='mean',
                 num_input_channels=64,
                 num_mid_channel=64,
                 num_target_channels=64,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps
        self.loss_weight=loss_weight

    def forward(self, input, target):
        # pool for dimentsion match
        # s_H, t_H = input.shape[2], target.shape[2]
        # if s_H > t_H:
        #     input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        # elif s_H < t_H:
        #     target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        # else:
        #     pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (
                (pred_mean - target) ** 2 / pred_var + torch.log(pred_var)
        )
        loss = torch.mean(neg_log_prob)

        return loss*self.loss_weight


@LOSS_REGISTRY.register()
class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()
        self.loss_weight=loss_weight

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        # print(type(loss),'loss')
        # print(type(self.loss_weight),'self.loss_weight')
        return loss*self.loss_weight

@LOSS_REGISTRY.register()
class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, loss_weight=1.0, reduction='mean',p=2):
        super(Attention, self).__init__()
        self.p = p
        self.loss_weight=loss_weight
    def forward(self, g_s, g_t):
        # only calculate min(len(g_s), len(g_t))-pair at_loss with the help of zip function
        return self.at_loss(g_s, g_t)

    def at_loss(self, f_s, f_t):
        loss = (self.at(f_s) - self.at(f_t)).pow(2).mean()*self.loss_weight

        return loss

    def at(self, f):
        # mean(1) function reduce feature map BxCxHxW into BxHxW by averaging the channel response
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
