import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from pytorch_wavelets import DWTForward
import torch.nn.functional as F

_reduction_modes = ['none', 'mean', 'sum']

from torchvision import models

import math

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=False).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

@LOSS_REGISTRY.register()
class ContrastLoss(nn.Module):
    def __init__(self, loss_weight, reduction=0,d_func='L1', t_detach = False, is_one=False):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = loss_weight
        self.d_func = d_func
        self.is_one = is_one
        self.t_detach = t_detach
        self.upsampler = nn.Upsample(scale_factor=2, mode='bicubic')

    def forward(self, teacher, student, neg, blur_neg=None):
        teacher_vgg, student_vgg, neg_vgg, = self.vgg(teacher), self.vgg(student), self.vgg(neg)
        blur_neg_vgg = None
        if blur_neg is not None:
            blur_neg_vgg = self.vgg(blur_neg)
        if self.d_func == "L1":
            self.forward_func = self.L1_forward
        elif self.d_func == 'cos':
            self.forward_func = self.cos_forward

        return self.forward_func(teacher_vgg, student_vgg, neg_vgg, blur_neg_vgg)

    def L1_forward(self, teacher, student, neg, blur_neg=None):
        """
        :param teacher: 5*batchsize*color*patchsize*patchsize
        :param student: 5*batchsize*color*patchsize*patchsize
        :param neg: 5*negnum*color*patchsize*patchsize
        :return:
        """
        loss = 0

        for i in range(len(teacher)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(student[i].shape[0], 1, 1, 1, 1)
            neg_i = neg_i.permute(1, 0, 2, 3, 4)### batchsize*negnum*color*patchsize*patchsize
            if blur_neg is not None:
                blur_neg_i = blur_neg[i].unsqueeze(0)
                neg_i = torch.cat((neg_i, blur_neg_i))


            if self.t_detach:
                d_ts = self.l1(teacher[i].detach(), student[i])
            else:
                d_ts = self.l1(teacher[i], student[i])
            d_sn = torch.mean(torch.abs(neg_i.detach() - student[i]).sum(0))

            contrastive = d_ts / (d_sn + 1e-7)
            loss += self.weights * contrastive
        return loss


    def cos_forward(self, teacher, student, neg, blur_neg=None):
        loss = 0
        for i in range(len(teacher)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(student[i].shape[0], 1, 1, 1, 1)
            neg_i = neg_i.permute(1, 0, 2, 3, 4)
            if blur_neg is not None:
                blur_neg_i = blur_neg[i].unsqueeze(0)
                neg_i = torch.cat((neg_i, blur_neg_i))

            if self.t_detach:
                d_ts = torch.cosine_similarity(teacher[i].detach(), student[i], dim=0).mean()
            else:
                d_ts = torch.cosine_similarity(teacher[i], student[i], dim=0).mean()
            d_sn = self.calc_cos_stu_neg(student[i], neg_i.detach())

            contrastive = -torch.log(torch.exp(d_ts)/(torch.exp(d_sn)+1e-7))
            loss += self.weights[i] * contrastive
        return loss

    def calc_cos_stu_neg(self, stu, neg):
        n = stu.shape[0]
        m = neg.shape[0]

        stu = stu.view(n, -1)
        neg = neg.view(m, n, -1)
        # normalize
        stu = F.normalize(stu, p=2, dim=1)
        neg = F.normalize(neg, p=2, dim=2)
        # multiply
        d_sn = torch.mean((stu * neg).sum(0))
        return
def wavelet_trans1(fea, device):  # already checked its ok
    xfm = DWTForward(J=1, mode='zero', wave='haar').to(device)
    for p in xfm.parameters():
        p.requires_grad = False
    Yl, Yh = xfm(fea)
    fea_list = []
    # fea_list.append(Yl)           #低频就算了不去匹配
    # print(Yh[0].shape)
    for j in range(len(Yh)):
        for i in range(Yh[j].shape[2]):
            fea_list.append(Yh[j][:, :, i, :, :])
            # print(Yh[0][:,:,i,:,:].shape,'yyy')
            # print(Yh[0].shape, 'yy1111111111111y')
    return fea_list


def wavelet_trans2(fea, device):  # already checked its ok
    xfm = DWTForward(J=2, mode='zero', wave='haar').to(device)
    for p in xfm.parameters():
        p.requires_grad = False
    Yl, Yh = xfm(fea)
    fea_list = []
    fea_list.append(Yl)
    # print(Yh[0].shape)
    for j in range(len(Yh)):
        for i in range(Yh[j].shape[2]):
            fea_list.append(Yh[j][:, :, i, :, :])
            # print(Yh[0][:,:,i,:,:].shape,'yyy')
            # print(Yh[0].shape, 'yy1111111111111y')
    return fea_list


def wavelet_trans3(fea, device):  # already checked its ok
    xfm = DWTForward(J=3, mode='zero', wave='haar').to(device)
    for p in xfm.parameters():
        p.requires_grad = False
    Yl, Yh = xfm(fea)
    fea_list = []
    fea_list.append(Yl)
    # print(Yh[0].shape)
    for j in range(len(Yh)):
        for i in range(Yh[j].shape[2]):
            fea_list.append(Yh[j][:, :, i, :, :])
            # print(Yh[0][:,:,i,:,:].shape,'yyy')
            # print(Yh[0].shape, 'yy1111111111111y')
    return fea_list


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class SimilarityLoss(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(SimilarityLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, g_s, g_t):
        return sum([self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.reshape(bsz, -1)
        f_t = f_t.reshape(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s, dim=1)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t, dim=1)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

@LOSS_REGISTRY.register()
class SimLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SimLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, gt, s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        # gt_list=wavelet_trans(gt)    #[(b,c,h1,w1)*waveorder]
        def spatial_similarity(fm):
            fm = fm.view(fm.size(0), fm.size(1), -1)
            norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
            s = norm_fm.transpose(1, 2).bmm(norm_fm)
            s = s.unsqueeze(1)
            return s

        def nor(fm):
            fm = fm.view(fm.size(0), fm.size(1), -1)
            norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
            return norm_fm

        loss = 0
        # s_gt = s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2)
        # t_gt = t_gt.view(t_gt.size(0), t_gt.size(1), -1).transpose(1, 2)

        # s_gt=nor(s_gt).transpose(1, 2)     #b,c,hw
        # t_gt=nor(t_gt).transpose(1, 2)
        # if index==0:
        #     gtt=wavelet_trans1(gt,gt.device)
        #     s_gtt=wavelet_trans1(s_gt,gt.device)
        #     t_gtt=wavelet_trans1(t_gt,gt.device)
        #     s_gt = [s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2) for s_gt in s_gtt]
        #     t_gt = [t_gt.view(t_gt.size(0), t_gt.size(1), -1).transpose(1, 2) for t_gt in t_gtt]
        #     gtt = [gt.view(gt.size(0), gt.size(1), -1) for gt in gtt]
        # elif index==1:
        #     gtt=wavelet_trans2(gt,gt.device)
        s_gtt = wavelet_trans2(s_gt, gt[0].device)
        t_gtt = wavelet_trans2(t_gt, gt[0].device)
        s_gt = [s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2) for s_gt in s_gtt]
        t_gt = [t_gt.view(t_gt.size(0), t_gt.size(1), -1).transpose(1, 2) for t_gt in t_gtt]
        gtt = [gttt.view(gttt.size(0), gttt.size(1), -1) for gttt in gt]
        #
        #
        # else:
        #     gtt=wavelet_trans3(gt,gt.device)
        #     s_gtt=wavelet_trans3(s_gt,gt.device)
        #     t_gtt=wavelet_trans3(t_gt,gt.device)
        #     s_gt = [s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2) for s_gt in s_gtt]
        #     t_gt = [t_gt.view(t_gt.size(0), t_gt.size(1), -1).transpose(1, 2) for t_gt in t_gtt]
        #     gtt = [gt.view(gt.size(0), gt.size(1), -1) for gt in gtt]
        # print(len(gtt))
        # print(len(s_gt))
        # print(len(s_gt))
        for i, gt_trans in enumerate(gtt):
            if i == 0:
                continue
            #     gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
            simi_s = torch.bmm(s_gt[i], gt_trans)
            simi_t = torch.bmm(t_gt[i], gt_trans)
            loss += l1_loss(simi_s, simi_t, weight, reduction=self.reduction)
        return loss
        # for i,gt_trans in enumerate(gt):   #t     itan setting
        #     if i==0:
        #         continue
        #     # gt_trans=nor(gt_trans)
        #     gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
        #     # if i==0:
        #     #     w=0.02
        #     # else:
        #     #     w=1
        #     # print(s_gt.shape,'gt_transgt_trans')
        #     # print(gt_trans.shape,'gt_transgt_transgt_trans')
        #     simi_s=torch.bmm(s_gt,gt_trans)
        #     simi_t=torch.bmm(t_gt,gt_trans)
        #     loss +=l1_loss(simi_s, simi_t, weight, reduction=self.reduction)
        # return loss


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w))
        # output
        if x.shape[-1] != out_shape:
            print(x.shape[-1], 'x.shape[-1]')
            print(out_shape, 'out_shape')
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x  # 这个y是放到一个outfeature的list里面不参与后续的计算  x会用来进行后续的计算
        # 这个x是用来传递的 不涉及loss计算


def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = fs
            tmpft = ft
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


@LOSS_REGISTRY.register()
class reviewkd(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(reviewkd, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        # for i in range(3):
        #     setattr(self, 'adaptconv.{}'.format(str(i)),
        #             nn.Sequential(
        #                 nn.Conv2d(2, 8, 1, padding=0, bias=True),
        #                 nn.ReLU(inplace=True),
        #                 nn.Conv2d(8, 2, 1, padding=0, bias=True),
        #                 # nn.Sigmoid()
        #             ))
        in_channels = [64, 64, 64, 64, 64]
        out_channels = 64
        abfs = nn.ModuleList()

        mid_channel = 512
        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels, idx < len(in_channels) - 1))
        self.abfs = abfs[::-1]

        for name, p in self.named_parameters():
            p.requires_grad = True

    def forward(self, s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        results = []
        s_gt = s_gt[::-1]

        out_features, res_features = self.abfs[0](s_gt[0], out_shape=48)
        results.append(out_features)
        for features, abf in zip(s_gt, self.abfs[1:]):
            out_features, res_features = abf(features, res_features, 48, 48)
            results.insert(0, out_features)

            loss_all = 0.0
        for fs, ft in zip(results, t_gt):
            n, c, h, w = fs.shape
            loss = F.mse_loss(fs, ft, reduction='mean')
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                # if l >=h:
                #     continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                tmpft = F.adaptive_avg_pool2d(ft, (l, l))
                tmpfs = fs
                tmpft = ft
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all


#         return self.loss_weight * l1_loss(s_gt, fusion_gt, weight, reduction=self.reduction)

#         return loss

@LOSS_REGISTRY.register()
class LowSimLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(LowSimLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, gt, s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        # gt_list=wavelet_trans(gt)    #[(b,c,h1,w1)*waveorder]
        def spatial_similarity(fm):
            fm = fm.view(fm.size(0), fm.size(1), -1)
            norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
            s = norm_fm.transpose(1, 2).bmm(norm_fm)
            s = s.unsqueeze(1)
            return s

        def nor(fm):
            fm = fm.view(fm.size(0), fm.size(1), -1)
            norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
            return norm_fm

        loss = 0
        # s_gt = s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2)
        # t_gt = t_gt.view(t_gt.size(0), t_gt.size(1), -1).transpose(1, 2)

        # s_gt=nor(s_gt).transpose(1, 2)     #b,c,hw
        # t_gt=nor(t_gt).transpose(1, 2)
        # if index==0:
        #     gtt=wavelet_trans1(gt,gt.device)
        #     s_gtt=wavelet_trans1(s_gt,gt.device)
        #     t_gtt=wavelet_trans1(t_gt,gt.device)
        #     s_gt = [s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2) for s_gt in s_gtt]
        #     t_gt = [t_gt.view(t_gt.size(0), t_gt.size(1), -1).transpose(1, 2) for t_gt in t_gtt]
        #     gtt = [gt.view(gt.size(0), gt.size(1), -1) for gt in gtt]
        # elif index==1:
        #     gtt=wavelet_trans2(gt,gt.device)
        s_gtt = wavelet_trans2(s_gt, gt[0].device)
        t_gtt = wavelet_trans2(t_gt, gt[0].device)
        s_gt = [s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2) for s_gt in s_gtt]
        t_gt = [t_gt.view(t_gt.size(0), t_gt.size(1), -1).transpose(1, 2) for t_gt in t_gtt]
        gtt = [gttt.view(gttt.size(0), gttt.size(1), -1) for gttt in gt]
        #
        #
        # else:
        #     gtt=wavelet_trans3(gt,gt.device)
        #     s_gtt=wavelet_trans3(s_gt,gt.device)
        #     t_gtt=wavelet_trans3(t_gt,gt.device)
        #     s_gt = [s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2) for s_gt in s_gtt]
        #     t_gt = [t_gt.view(t_gt.size(0), t_gt.size(1), -1).transpose(1, 2) for t_gt in t_gtt]
        #     gtt = [gt.view(gt.size(0), gt.size(1), -1) for gt in gtt]

        for i, gt_trans in enumerate(gtt[:1]):
            #     gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
            simi_s = torch.bmm(s_gt[i], gt_trans)
            simi_t = torch.bmm(t_gt[i], gt_trans)
            loss += l1_loss(simi_s, simi_t, weight, reduction=self.reduction)
        return loss
        # for i,gt_trans in enumerate(gt):   #t     itan setting
        #     if i==0:
        #         continue
        #     # gt_trans=nor(gt_trans)
        #     gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
        #     # if i==0:
        #     #     w=0.02
        #     # else:
        #     #     w=1
        #     # print(s_gt.shape,'gt_transgt_trans')
        #     # print(gt_trans.shape,'gt_transgt_transgt_trans')
        #     simi_s=torch.bmm(s_gt,gt_trans)
        #     simi_t=torch.bmm(t_gt,gt_trans)
        #     loss +=l1_loss(simi_s, simi_t, weight, reduction=self.reduction)
        # return loss


@LOSS_REGISTRY.register()
class DirectSimLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(DirectSimLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, gt_list, s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # gt_list=wavelet_trans(gt)    #[(b,c,h1,w1)*waveorder]
        loss = 0

        def spatial_similarity(fm):
            fm = fm.view(fm.size(0), fm.size(1), -1)
            norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
            s = norm_fm.transpose(1, 2).bmm(norm_fm)
            s = s.unsqueeze(1)
            return s

        s_gt = s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2)
        t_gt = t_gt.view(t_gt.size(0), t_gt.size(1), -1).transpose(1, 2)
        # s_gt = spatial_similarity(s_gt)
        # t_gt = spatial_similarity(t_gt)

        # for gt_trans in gt_list[1:]:
        #     gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
        #     simi_s=torch.bmm(s_gt,gt_trans)
        #     simi_t=torch.bmm(t_gt,gt_trans)
        #     loss +=l1_loss(simi_s, simi_t, weight, reduction=self.reduction)
        # return loss
        simi_s = torch.bmm(s_gt, s_gt.transpose(1, 2))
        simi_t = torch.bmm(t_gt, t_gt.transpose(1, 2))
        loss += l1_loss(simi_s, simi_t, weight, reduction=self.reduction)
        # loss +=l1_loss(s_gt, t_gt, weight, reduction=self.reduction)
        return loss


@LOSS_REGISTRY.register()
class WaveLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(WaveLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # gt_list=wavelet_trans(gt)    #[(b,c,h1,w1)*waveorder]
        loss = 0
        device = s_gt.device
        # for gt_trans in gt_list[1:]:
        #     gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
        #     simi_s=torch.bmm(s_gt,gt_trans)
        #     simi_t=torch.bmm(t_gt,gt_trans)
        #     loss +=l1_loss(simi_s, simi_t, weight, reduction=self.reduction)
        # return loss
        t_gt = wavelet_trans1(t_gt, device)
        s_gt = wavelet_trans1(s_gt, device)
        for i, gt_trans1 in enumerate(t_gt):  # titan setting

            # if i == 0:
            #     w = 0.1  #######3  w=1的时候是5e+1   =0.02是 4.82e00
            #     # w=1
            # else:
            #     w = 1

            loss += l1_loss(s_gt[i], gt_trans1, weight, reduction=self.reduction)
        return loss*self.loss_weight


@LOSS_REGISTRY.register()
class AdaptLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        for i in range(3):
            setattr(self, 'adaptconv.{}'.format(str(i)),
                    nn.Sequential(
                        nn.Conv2d(2, 32, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 2, 1, padding=0, bias=True),
                        nn.Sigmoid()
                    ))
        for name, p in self.named_parameters():
            p.requires_grad = True

    def forward(self, s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        channel_list = []
        for i in range(3):
            s_gt_split = s_gt[:, i, :, :].unsqueeze(1)
            t_gt_split = t_gt[:, i, :, :].unsqueeze(1)
            fusion_split = torch.cat([s_gt_split, t_gt_split], dim=1)

            score = F.softmax(getattr(self, 'adaptconv.{}'.format(str(i)))(fusion_split), dim=1)

            channel_list.append(
                s_gt_split * (score[:, 0, :, :].unsqueeze(1)) + t_gt_split * (score[:, 1, :, :].unsqueeze(1)))

        fusion_gt = torch.cat(channel_list, dim=1)
        # f_gt=wavelet_trans1(fusion_gt,fusion_gt.device)
        # s_gt=wavelet_trans1(s_gt,fusion_gt.device)
        # loss=0
        # for i ,fea in enumerate(f_gt):
        #     if i==0:
        #         continue
        #     else:
        #         loss+=l1_loss(s_gt[i], fea, weight, reduction=self.reduction)
        # fusion_gt = (fusion_gt + gt) / 2
        return self.loss_weight * l1_loss(s_gt, fusion_gt, weight, reduction=self.reduction)

        # return loss

@LOSS_REGISTRY.register()
class AdaptLoss_gt(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptLoss_gt, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        for i in range(3):
            setattr(self, 'adaptconv.{}'.format(str(i)),
                    nn.Sequential(
                        nn.Conv2d(2, 32, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 2, 1, padding=0, bias=True),
                        nn.Sigmoid()
                    ))
        for name, p in self.named_parameters():
            p.requires_grad = True

    def forward(self, gt,s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        channel_list = []
        for i in range(3):
            gt_split = gt[:, i, :, :].unsqueeze(1)
            t_gt_split = t_gt[:, i, :, :].unsqueeze(1)
            fusion_split = torch.cat([gt_split, t_gt_split], dim=1)

            score = F.softmax(getattr(self, 'adaptconv.{}'.format(str(i)))(fusion_split), dim=1)

            channel_list.append(
                gt_split * (score[:, 0, :, :].unsqueeze(1)) + t_gt_split * (score[:, 1, :, :].unsqueeze(1)))

        fusion_gt = torch.cat(channel_list, dim=1)
        # f_gt=wavelet_trans1(fusion_gt,fusion_gt.device)
        # s_gt=wavelet_trans1(s_gt,fusion_gt.device)
        # loss=0
        # for i ,fea in enumerate(f_gt):
        #     if i==0:
        #         continue
        #     else:
        #         loss+=l1_loss(s_gt[i], fea, weight, reduction=self.reduction)
        # fusion_gt = (fusion_gt + gt) / 2
        return self.loss_weight * l1_loss(s_gt, fusion_gt, weight, reduction=self.reduction)

        # return loss
@LOSS_REGISTRY.register()
class AdaptLoss_three(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptLoss_three, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        for i in range(3):
            setattr(self, 'adaptconv.{}'.format(str(i)),
                    nn.Sequential(
                        nn.Conv2d(3, 48, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(48, 3, 1, padding=0, bias=True),
                        nn.Sigmoid()
                    ))
        for name, p in self.named_parameters():
            p.requires_grad = True

    def forward(self, gt,s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        channel_list = []
        for i in range(3):
            gt_split = gt[:, i, :, :].unsqueeze(1)
            s_split = s_gt[:, i, :, :].unsqueeze(1)
            t_gt_split = t_gt[:, i, :, :].unsqueeze(1)
            fusion_split = torch.cat([gt_split, s_split,t_gt_split], dim=1)

            score = F.softmax(getattr(self, 'adaptconv.{}'.format(str(i)))(fusion_split), dim=1)

            channel_list.append(
                gt_split * (score[:, 0, :, :].unsqueeze(1)) +s_split * (score[:, 1, :, :].unsqueeze(1))+\
                t_gt_split * (score[:, 2, :, :].unsqueeze(1)))

        fusion_gt = torch.cat(channel_list, dim=1)
        # f_gt=wavelet_trans1(fusion_gt,fusion_gt.device)
        # s_gt=wavelet_trans1(s_gt,fusion_gt.device)
        # loss=0
        # for i ,fea in enumerate(f_gt):
        #     if i==0:
        #         continue
        #     else:
        #         loss+=l1_loss(s_gt[i], fea, weight, reduction=self.reduction)
        # fusion_gt = (fusion_gt + gt) / 2
        return self.loss_weight * l1_loss(s_gt, fusion_gt, weight, reduction=self.reduction)

        # return loss
@LOSS_REGISTRY.register()
class AdaptLoss_nochannel_three(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptLoss_nochannel_three, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # for i in range(3):
        setattr(self, 'adaptconv',
                nn.Sequential(
                    nn.Conv2d(9, 72, 1, padding=0, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(72, 3, 1, padding=0, bias=True),
                    nn.Sigmoid()
                ))
        for name, p in self.named_parameters():
            p.requires_grad = True

    def forward(self, gt,s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # channel_list = []
        # for i in range(3):
        #     gt_split = gt[:, i, :, :].unsqueeze(1)
        #     s_split = s_gt[:, i, :, :].unsqueeze(1)
        #     t_gt_split = t_gt[:, i, :, :].unsqueeze(1)
        fusion_split = torch.cat([gt,s_gt, t_gt], dim=1)

        score = F.softmax(getattr(self, 'adaptconv')(self.avg_pool(fusion_split)), dim=1)

            # channel_list.append(
            #     gt_split * (score[:, 0:3, :, :].unsqueeze(1)) +s_split * (score[:, 3:6, :, :].unsqueeze(1))+\
            #     t_gt_split * (score[:, 6:9, :, :].unsqueeze(1)))

        fusion_gt = gt * (score[:, 0, :, :].unsqueeze(1)) +s_gt * (score[:, 1, :, :].unsqueeze(1))+\
                t_gt * (score[:, 2, :, :].unsqueeze(1))
        # f_gt=wavelet_trans1(fusion_gt,fusion_gt.device)
        # s_gt=wavelet_trans1(s_gt,fusion_gt.device)
        # loss=0
        # for i ,fea in enumerate(f_gt):
        #     if i==0:
        #         continue
        #     else:
        #         loss+=l1_loss(s_gt[i], fea, weight, reduction=self.reduction)
        # fusion_gt = (fusion_gt + gt) / 2
        return self.loss_weight * l1_loss(s_gt, fusion_gt, weight, reduction=self.reduction)

        # return loss


def spatial_similarity(fm):
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
    s = norm_fm.transpose(1,2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s

def channel_similarity(fm):
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 2)).unsqueeze(2).expand(fm.shape) + 1e-8)
    s = norm_fm.bmm(norm_fm.transpose(1,2))
    s = s.unsqueeze(1)     #迷惑操作
    return s


@LOSS_REGISTRY.register()
class fakd(nn.Module):


    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(fakd, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # for i in range(3):
        # setattr(self, 'adaptconv',
        #         nn.Sequential(
        #             nn.Conv2d(6, 96, 1, padding=0, bias=True),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(96, 2, 1, padding=0, bias=True),
        #             nn.Sigmoid()
        #         ))
        # for name, p in self.named_parameters():
        #     p.requires_grad = True

    def forward(self, s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # channel_list = []
        loss=0
        # for i in range(len(s_gt)):
        loss+=l1_loss(channel_similarity(s_gt),channel_similarity(t_gt))
        loss += l1_loss(spatial_similarity(s_gt), spatial_similarity(t_gt))
        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class AdaptLoss_nochannel(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptLoss_nochannel, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # for i in range(3):
        setattr(self, 'adaptconv',
                nn.Sequential(
                    nn.Conv2d(6, 96, 1, padding=0, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96, 2, 1, padding=0, bias=True),
                    nn.Sigmoid()
                ))
        for name, p in self.named_parameters():
            p.requires_grad = True

    def forward(self, gt,s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # channel_list = []
        # for i in range(3):
        #     gt_split = gt[:, i, :, :].unsqueeze(1)
        #     s_split = s_gt[:, i, :, :].unsqueeze(1)
        #     t_gt_split = t_gt[:, i, :, :].unsqueeze(1)
        fusion_split = torch.cat([gt, t_gt], dim=1)

        score = F.softmax(getattr(self, 'adaptconv')(self.avg_pool(fusion_split)), dim=1)

            # channel_list.append(
            #     gt_split * (score[:, 0:3, :, :].unsqueeze(1)) +s_split * (score[:, 3:6, :, :].unsqueeze(1))+\
            #     t_gt_split * (score[:, 6:9, :, :].unsqueeze(1)))

        fusion_gt = gt * (score[:, 0, :, :].unsqueeze(1))+\
                t_gt * (score[:, 1, :, :].unsqueeze(1))
        # f_gt=wavelet_trans1(fusion_gt,fusion_gt.device)
        # s_gt=wavelet_trans1(s_gt,fusion_gt.device)
        # loss=0
        # for i ,fea in enumerate(f_gt):
        #     if i==0:
        #         continue
        #     else:
        #         loss+=l1_loss(s_gt[i], fea, weight, reduction=self.reduction)
        # fusion_gt = (fusion_gt + gt) / 2
        return self.loss_weight * l1_loss(s_gt, fusion_gt, weight, reduction=self.reduction)

        # return loss

@LOSS_REGISTRY.register()
class AdaptLoss_nochannel_test(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptLoss_nochannel_test, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # for i in range(3):
        setattr(self, 'adaptconv',
                nn.Sequential(
                    nn.Conv2d(6, 96, 1, padding=0, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96, 2, 1, padding=0, bias=True),
                    nn.Sigmoid()
                ))
        for name, p in self.named_parameters():
            p.requires_grad = True

    def forward(self, gt,s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # channel_list = []
        # for i in range(3):
        #     gt_split = gt[:, i, :, :].unsqueeze(1)
        #     s_split = s_gt[:, i, :, :].unsqueeze(1)
        #     t_gt_split = t_gt[:, i, :, :].unsqueeze(1)
        fusion_split = torch.cat([gt, t_gt], dim=1)

        score = F.softmax(getattr(self, 'adaptconv')(self.avg_pool(fusion_split)), dim=1)

            # channel_list.append(
            #     gt_split * (score[:, 0:3, :, :].unsqueeze(1)) +s_split * (score[:, 3:6, :, :].unsqueeze(1))+\
            #     t_gt_split * (score[:, 6:9, :, :].unsqueeze(1)))

        fusion_gt = gt * (score[:, 0, :, :].unsqueeze(1))+\
                t_gt * (score[:, 1, :, :].unsqueeze(1))
        # f_gt=wavelet_trans1(fusion_gt,fusion_gt.device)
        # s_gt=wavelet_trans1(s_gt,fusion_gt.device)
        # loss=0
        # for i ,fea in enumerate(f_gt):
        #     if i==0:
        #         continue
        #     else:
        #         loss+=l1_loss(s_gt[i], fea, weight, reduction=self.reduction)
        # fusion_gt = (fusion_gt + gt) / 2
        return fusion_gt

        # return loss

@LOSS_REGISTRY.register()
class AdaptLoss_onlygt(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptLoss_onlygt, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # for i in range(3):
        # setattr(self, 'adaptconv',
        #         nn.Sequential(
        #             nn.Conv2d(6, 96, 1, padding=0, bias=True),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(96, 2, 1, padding=0, bias=True),
        #             nn.Sigmoid()
        #         ))
        # for name, p in self.named_parameters():
        #     p.requires_grad = True

    def forward(self, gt,s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # channel_list = []
        # for i in range(3):
        #     gt_split = gt[:, i, :, :].unsqueeze(1)
        #     s_split = s_gt[:, i, :, :].unsqueeze(1)
        #     t_gt_split = t_gt[:, i, :, :].unsqueeze(1)
        # fusion_split = torch.cat([gt, t_gt], dim=1)
        #
        # score = F.softmax(getattr(self, 'adaptconv')(self.avg_pool(fusion_split)), dim=1)
        #
        #     # channel_list.append(
        #     #     gt_split * (score[:, 0:3, :, :].unsqueeze(1)) +s_split * (score[:, 3:6, :, :].unsqueeze(1))+\
        #     #     t_gt_split * (score[:, 6:9, :, :].unsqueeze(1)))
        #
        # fusion_gt = gt * (score[:, 0, :, :].unsqueeze(1))+\
        #         t_gt * (score[:, 1, :, :].unsqueeze(1))
        # f_gt=wavelet_trans1(fusion_gt,fusion_gt.device)
        # s_gt=wavelet_trans1(s_gt,fusion_gt.device)
        # loss=0
        # for i ,fea in enumerate(f_gt):
        #     if i==0:
        #         continue
        #     else:
        #         loss+=l1_loss(s_gt[i], fea, weight, reduction=self.reduction)
        # fusion_gt = (fusion_gt + gt) / 2
        return self.loss_weight * l1_loss(s_gt, gt, weight, reduction=self.reduction)

        # return loss

@LOSS_REGISTRY.register()
class AdaptLoss_directfuse(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptLoss_directfuse, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        for i in range(3):
            setattr(self, 'adaptconv.{}'.format(str(i)),
                    nn.Sequential(
                        nn.Conv2d(2, 32, 1, padding=0, bias=True),
                        nn.AdaptiveAvgPool2d(output_size=1),
                        nn.Flatten(),
                        nn.Linear(32, 2),
                        # nn.ReLU(inplace=True),
                        # nn.Conv2d(8, 1, 1, padding=0, bias=True),

                        # nn.Sigmoid()
                    ))

    def forward(self, gt, s_gt, t_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        channel_list = []
        for i in range(3):
            gt_split = gt[:, i, :, :].unsqueeze(1)
            t_gt_split = t_gt[:, i, :, :].unsqueeze(1)
            fusion_split = torch.cat([gt_split, t_gt_split], dim=1)
            score = F.softmax(getattr(self, 'adaptconv.{}'.format(str(i)))(fusion_split), dim=1)
            # print(score.shape,'score')
            # print(score[:, 0].unsqueeze(1).shape,'asdas')
            # print(gt_split.shape,'asdasdasd')
            # print((gt_split*(score[:,0].unsqueeze(1)).shape),'asdasd')
            a = score[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            b = score[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # print(a.shape,'aaaaaa')
            # print(gt_split.shape,'gt_splitgt_splitgt_split')
            # print((gt_split*a,'asdasd'))
            # print(a,'a',b,'b')
            channel_list.append(gt_split * a + t_gt_split * b)
        fusion_gt = torch.cat([fea for fea in channel_list], dim=1)
        return self.loss_weight * l1_loss(s_gt, fusion_gt, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class Adapt_SIMLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Adapt_SIMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        setattr(self, 'adaptconv.{}'.format(str(0)),
                nn.Sequential(
                    nn.Conv2d(6, 12, 1, padding=0, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(12, 2, 1, padding=0, bias=True),

                    # nn.Sigmoid()
                ))
        # self.conv1=nn.Conv2d(64, 3, 1, padding=0, bias=True),
        # self.conv2=nn.Conv2d(64, 3, 1, padding=0, bias=True),

        for name, p in self.named_parameters():
            p.requires_grad = True

    def forward(self, gt_list, s_gt, t_gt, t_gtout, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        fusion_split = torch.cat([t_gt, t_gtout], dim=1)

        score = F.softmax(getattr(self, 'adaptconv.{}'.format(str(0)))(fusion_split), dim=1)

        fuse_fea = t_gt * (score[:, 0, :, :].unsqueeze(1)) + t_gtout * (score[:, 1, :, :].unsqueeze(1))

        loss = 0

        s_gt = s_gt.view(s_gt.size(0), s_gt.size(1), -1).transpose(1, 2)
        fuse_fea = fuse_fea.view(fuse_fea.size(0), fuse_fea.size(1), -1).transpose(1, 2)

        # s_gt=self.conv1(s_gt)
        # fuse_fea=self.conv2(fuse_fea)
        # for gt_trans in gt_list:   #titan setting
        # print(gt_trans.shape,'gt_trans')
        # print(fuse_fea.shape,'fuse_fea')
        # print(s_gt.shape,'s_gt')

        for i, gt_trans in enumerate(gt_list):  # titan setting

            gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
            if i == 0:
                w = 0.02
            else:
                w = 1

            # gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
            simi_s = torch.bmm(s_gt, gt_trans)
            simi_t = torch.bmm(fuse_fea, gt_trans)
            loss += l1_loss(simi_s, simi_t, weight, reduction=self.reduction) * w

        return loss
        # for gt_trans in gt_list[1:]:
        #     # print(gt_trans.shape,'gt_trans')
        #     # print(fuse_fea.shape,'fuse_fea')
        #     # print(s_gt.shape,'s_gt')
        #     gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
        #     simi_s=torch.bmm(s_gt,gt_trans)
        #     simi_t=torch.bmm(fuse_fea,gt_trans)
        #     loss +=l1_loss(simi_s, simi_t, weight, reduction=self.reduction)
        #
        #
        # return loss

        # return self.loss_weight * l1_loss(s_gt, fuse_fea, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class Adapt_WaveLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Adapt_WaveLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        setattr(self, 'adaptconv.{}'.format(str(0)),
                nn.Sequential(
                    nn.Conv2d(6, 12, 1, padding=0, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(12, 2, 1, padding=0, bias=True),

                    # nn.Sigmoid()
                ))
        # self.conv1=nn.Conv2d(64, 3, 1, padding=0, bias=True),
        # self.conv2=nn.Conv2d(64, 3, 1, padding=0, bias=True),

        for name, p in self.named_parameters():
            p.requires_grad = True

    def forward(self, s_gt, t_gt, t_gtout, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        fusion_split = torch.cat([t_gt, t_gtout], dim=1)

        score = F.softmax(getattr(self, 'adaptconv.{}'.format(str(0)))(fusion_split), dim=1)

        fuse_fea = t_gt * (score[:, 0, :, :].unsqueeze(1)) + t_gtout * (score[:, 1, :, :].unsqueeze(1))

        loss = 0

        s_gt = wavelet_trans(s_gt, s_gt[0].device)
        fuse_fea = wavelet_trans(fuse_fea, s_gt[0].device)

        # s_gt=self.conv1(s_gt)
        # fuse_fea=self.conv2(fuse_fea)
        # for gt_trans in gt_list:   #titan setting
        # print(gt_trans.shape,'gt_trans')
        # print(fuse_fea.shape,'fuse_fea')
        # print(s_gt.shape,'s_gt')

        for i, fuse_trans in enumerate(fuse_fea):  # titan setting

            if i == 0:
                w = 0.02
            else:
                w = 1

            # gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
            loss += l1_loss(s_gt[i], fuse_trans, weight, reduction=self.reduction) * w

        return loss
        # for gt_trans in gt_list[1:]:
        #     # print(gt_trans.shape,'gt_trans')
        #     # print(fuse_fea.shape,'fuse_fea')
        #     # print(s_gt.shape,'s_gt')
        #     gt_trans = gt_trans.view(gt_trans.size(0), gt_trans.size(1), -1)
        #     simi_s=torch.bmm(s_gt,gt_trans)
        #     simi_t=torch.bmm(fuse_fea,gt_trans)
        #     loss +=l1_loss(simi_s, simi_t, weight, reduction=self.reduction)
        #
        #
        # return loss

        # return self.loss_weight * l1_loss(s_gt, fuse_fea, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
