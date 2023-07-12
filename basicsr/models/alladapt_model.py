import torch
from torchstat import stat
# from thop import profile

from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch import nn as nn

# from torch.cuda.amp import autocast as autocast, GradScaler
from skimage.util import random_noise
import numpy as np

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from pytorch_wavelets import DWTForward
import random
import torch.nn.functional as F


def _gen_cutout_coord(height, width, size):
    if height > size:
        m_size = int(size // 2)
        height_loc = random.randint(m_size + 1, height - m_size - 1)
        width_loc = random.randint(m_size + 1, width - m_size - 1)

        upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
        lower_coord = (min(height, height_loc + size // 2), min(width, width_loc + size // 2))

        return upper_coord, lower_coord
    else:
        return None, None


def cutout(image, upper_coord, lower_coord):
    if image.size(2) > 96:
        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0
        # print(upper_coord,'up')
        # print(lower_coord,'low')
        return image[..., upper_coord[0]: lower_coord[0], upper_coord[1]: lower_coord[1]].clone().contiguous()
    else:
        return image


def wavelet_trans(fea):  # already checked its ok
    xfm = DWTForward(J=2, mode='zero', wave='haar').cuda()
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


@MODEL_REGISTRY.register()
class SR_kd2Model(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SR_kd2Model, self).__init__(opt)

        # define network
        # print(type(opt['network_g']))
        # opt['dist']=False
        self.scale = opt['scale']
        self.net_g = build_network(opt['network_g'])
        self.net_t = build_network(opt['network_t'])
        self.net_g = self.model_to_device(self.net_g)
        self.net_t = self.net_t.to(self.device)
        self.sim_w = opt['sim_w']
        self.sim_wt = opt['sim_wt']
        self.ada_w = opt['ada_w']
        self.ada_wt = opt['ada_wt']
        self.gau = opt['gau']
        self.numberlist = opt['number_list']
        # if opt['network_t'].get('group_id',None):
        #     self.ft = len(opt['network_t']['group_id'])
        # else:
        #     self.ft=2
        # self.net_t = self.model_to_device(self.net_t)
        for name, param in self.net_g.named_parameters():
            if param.requires_grad == False:
                print(name, 'not need update')
        for name, param in self.net_t.named_parameters():
            if param.requires_grad == True:
                assert False
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_t', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            # assert False,'realy working'
            self.load_network(self.net_t, load_path, self.opt['path'].get('strict_load_g', True), param_key)
            # i=0
            # for name,param in self.net_t.named_parameters():
            #     i+=1
            #     if i ==10:
            #         print(param,'check')
            #         break
            #
            # assert False,'realy working'

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if train_opt.get('sim_opt'):

            self.cri_highsim = build_loss(train_opt['sim_opt']).to(self.device)
            # self.cri_lowsim = build_loss(train_opt['lowsim_opt']).to(self.device)
        else:
            self.cri_highsim = None
            self.cri_lowsim = None
        if train_opt.get('wave_opt'):

            self.cri_wave = build_loss(train_opt['wave_opt']).to(self.device)
        else:
            self.cri_wave = None
        if train_opt.get('kd_opt'):
            self.cri_kd = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_kd = None

        if train_opt.get('ada_sim_opt'):
            self.cri_adasim = nn.ModuleList([])
            for i in range(2):
                self.cri_adasim.append(build_loss(train_opt['ada_sim_opt']).to(self.device))

        else:
            self.cri_adasim = None
        if train_opt.get('ada_wave_opt'):
            self.cri_adawave = nn.ModuleList([])
            for i in range(2):
                self.cri_adawave.append(build_loss(train_opt['ada_wave_opt']).to(self.device))

        else:
            self.cri_adawave = None

        if train_opt.get('adpat_opt'):
            self.cri_adapt = nn.ModuleList([])
            for i in range(len(self.numberlist) + 1):
                self.cri_adapt.append(build_loss(train_opt['adpat_opt']).to(self.device))
        else:
            self.cri_adapt = None

        # if train_opt.get('review_opt'):
        #     self.cri_review = nn.ModuleList([])
        #     for i in range(len(self.numberlist)+1):
        #         self.cri_review.append(build_loss(train_opt['review_opt']).to(self.device))
        # else:
        #     self.cri_review = None
        if train_opt.get('review_opt'):
            self.cri_review = build_loss(train_opt['review_opt']).to(self.device)
        else:
            self.cri_review = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        # self.scaler = GradScaler()  # 训练前实例化一个GradScaler对象

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        if self.cri_adapt:
            for k, v in self.cri_adapt.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')
        if self.cri_review:
            for k, v in self.cri_review.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')

        if self.cri_adasim:
            for k, v in self.cri_adasim.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')
        if self.cri_adawave:
            for k, v in self.cri_adawave.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # pritn(self.lq.shape,'self.lqshape')
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        index_list = []
        for i in range(len(self.numberlist)):
            proba = random.random()
            split = 2
            if proba > split:
                index = round(random.uniform(self.numberlist[i] + 1, 9))
            else:
                index = self.numberlist[i]
            index_list.append(index)
        index_list = self.numberlist
        # self.optimizer_g.zero_grad()
        # self.output_s,self.fea_s = self.net_g(self.lq)
        # self.output_t,self.fea_t = self.net_t(self.lq)
        # with autocast():
        # ＃
        # 前后开启autocast
        self.optimizer_g.zero_grad()

        a=torch.randn((1,3,256,256))
        self.output_s, self.fea_s = self.net_g(self.lq)
        self.output_t, self.fea_t = self.net_t(self.lq)
        stat(self.net_g, (3, 256, 256))
        assert False,'test'
        l_total = 0
        loss_dict = OrderedDict()
        # loss_dict['l_total'] = 0

        # pixel loss
        if self.cri_kd:
            # loss_dict['l_kd_s'] = 0
            loss_dict['l_kd_t'] = 0
            l_kd = 0
            # for fea in self.output_s:
            #     print(fea.shape,'sssshape')
            # for fea in self.output_t:
            #     print(fea.shape,'tttttttttttshape')

            # for i in range(len(self.fea_s)):
            #     if i != len(self.output_s) - 1:
            #
            #         # for i in range(len(self.fea_s)):
            # #     if i != len(self.fea_s)-1:
            # #         if i != len(self.fea_s) - 1:
            #         # index = index_list[i]
            #         index = self.numberlist[i]
            #     else:
            #         index=-1
            #     # print(index)
            #     # kd_loss = self.cri_kd(self.fea_s[i], self.fea_t[index])/100
            #     kd_loss = self.cri_kd(self.output_s[i], self.output_t[index])/100
            #     l_kd += kd_loss
            #     loss_dict['l_kd_s'] += kd_loss

            # else:
            kd_loss = self.cri_kd(self.output_s[-1], self.output_t[-1])
            l_kd += kd_loss
            loss_dict['l_kd_t'] += kd_loss

            l_total += l_kd

        if self.cri_pix:
            loss_dict['l_pix'] = 0
            l_pix = self.cri_pix(self.output_s[-1], self.gt)
            l_total += l_pix
            loss_dict['l_pix'] += l_pix

        # perceptual loss
        if self.cri_highsim or self.cri_adasim:
            # loss_dict['l_sim_s0'] = 0
            # loss_dict['l_sim_s1'] = 0
            loss_dict['l_sim_t'] = 0
            l_sim = 0

            upper_coord, lower_coord = _gen_cutout_coord(self.gt.size(2), self.gt.size(3), 96)
            # print(self.gt.shape,'self.gt_simshape')

            self.gt_sim = cutout(self.gt, upper_coord, lower_coord)
            # self.gt_sim = self.gt
            # print(self.gt_sim.shape,'self.gt_simshapeaaa')
            # print(self.gt_sim.shape,'self.gt_simself.gt_simself.gt_simself.gt_sim')
            # print(self.gt_sim.size(3),'self.gt_simself.gt_simself.gt_simself.gt_sim')
            gt_list = wavelet_trans(self.gt_sim)  # [(b,c,h1,w1)*waveorder]

            # for i, num in enumerate(self.numberlist):
            #     index = index_list[i]
            #     sim_loss = self.cri_highsim(gt_list,
            #                             cutout(self.output_s[i], upper_coord, lower_coord),
            #                             cutout(self.output_t[index], upper_coord, lower_coord)) * self.sim_w  # scale2 /2
            #     # cutout(self.output_s[i], upper_coord, lower_coord),
            #     # cutout(self.output_t[i], upper_coord, lower_coord))*self.sim_w  # scale2 /2
            #     l_sim += sim_loss
            #     loss_dict['l_sim_s'+str(i)] += sim_loss
            #
            #     # if i!=len(self.output_s)-1:
            #     #     sim_loss = self.cri_adasim[i](gt_list,
            #     #                             # cutout(self.fea_s[i],upper_coord, lower_coord),
            #     #                             # cutout(self.fea_t[i],upper_coord, lower_coord))*self.sim_w  #scale2 /2
            #     #                             cutout(self.output_s[i], upper_coord, lower_coord),
            #     #                             cutout(self.output_t[i], upper_coord, lower_coord),
            #     #                               cutout(self.output_t[-1], upper_coord, lower_coord))*self.sim_w  # scale2 /2
            #     #     l_sim+=sim_loss
            #     #     loss_dict['l_sim_s'] += sim_loss

            sim_loss = self.cri_highsim(gt_list,
                                        cutout(self.output_s[-1], upper_coord, lower_coord),
                                        cutout(self.output_t[-1], upper_coord, lower_coord)) * self.sim_wt  # scale2 /2
            # # cutout(self.output_s[i], upper_coord, lower_coord),
            # # cutout(self.output_t[i], upper_coord, lower_coord))*self.sim_wt  # scale2 /2
            l_sim += sim_loss
            loss_dict['l_sim_t'] += sim_loss

            #
            # gt_list=wavelet_trans(self.gt)    #[(b,c,h1,w1)*waveorder]
            # for i in range(len(self.output_s)):
            #     if i!=len(self.output_s)-1:
            #         sim_loss = self.cri_sim(gt_list,self.fea_s[i],self.fea_t[i])*self.sim_w  #scale2 /2
            #         l_sim+=sim_loss
            #         loss_dict['l_sim_s'] += sim_loss
            #     else:
            #         sim_loss = self.cri_sim(gt_list,self.fea_s[i],self.fea_t[i])*self.sim_wt  #scale2 /2
            #         l_sim+=sim_loss
            #         loss_dict['l_sim_t'] += sim_loss

            l_total += l_sim

        # print(current_iter,'asdasdasd')
        # print(type(current_iter),'asdasd')asd
        # loss_dict['l_afterpixel'] = 0
        # l_afterpixel=0
        # # for fea in self.output_s:
        # #     print(fea.shape,'sssshape')
        # # for fea in self.output_t:
        # #     print(fea.shape,'tttttttttttshape')
        # # for i in range(len(self.output_t)):
        #
        # # for i in range(len(self.output_t)):
        # for i in range(len(self.output_s)):
        #     after_loss = self.cri_pix(self.output_s[i],self.output_t[i])
        #     l_afterpixel+=after_loss
        #     loss_dict['l_afterpixel'] += after_loss
        # l_total += l_afterpixel

        if self.cri_adapt:
            # loss_dict['l_adapt_s'] = 0
            loss_dict['l_adapt_t'] = 0
            l_adapt = 0

            # for i, num in enumerate(self.numberlist):
            #     index = index_list[i]
            #     # for i in range(len(self.output_s)):
            #     #
            #     adapt_loss = self.cri_adapt[i](self.gt, self.output_s[i],self.output_t[index]) * self.ada_w  # scale2 /6
            #     l_adapt += adapt_loss
            #     loss_dict['l_adapt_s'] += adapt_loss

            # adapt_loss = self.cri_adapt[-1](self.gt, self.output_s[-1], self.output_t[-1]) * self.ada_wt  # scale2 /6

            adapt_loss = self.cri_adapt[-1](self.gt, self.output_s[-1], self.output_t[-1]) * self.ada_wt  # scale2 /3
            l_adapt += adapt_loss
            loss_dict['l_adapt_t'] += adapt_loss

            l_total += l_adapt

        if self.cri_review:
            loss_dict['cri_review'] = 0
            l_review = 0

            review_loss = self.cri_review(self.fea_s, self.fea_t) / 10000
            l_review += review_loss
            loss_dict['cri_review'] += review_loss

            l_total += l_review

        if self.cri_wave:
            loss_dict['l_wave_s'] = 0
            loss_dict['l_wave_t'] = 0
            l_wave = 0
            # gt_list=wavelet_trans(self.gt)    #[(b,c,h1,w1)*waveorder]
            for i in range(len(self.output_s)):
                if i != len(self.output_s) - 1:
                    # wave_loss = self.cri_adawave[i](self.fea_s[i],self.fea_s[-1].detach(),self.fea_t[0])*self.sim_w  #scale2 /2
                    wave_loss = self.cri_wave(self.fea_s[i], self.fea_t[i]) * self.sim_w  # scale2 *2

                    l_wave += wave_loss
                    loss_dict['l_wave_s'] += wave_loss
                else:
                    # print(type(self.fea_t[0]),'ttttppp')
                    # print(len(self.fea_t[0]),'lll')

                    wave_loss = self.cri_wave(self.fea_s[i], self.fea_t[i]) * self.sim_wt  # scale2 *2
                    l_wave += wave_loss
                    loss_dict['l_wave_t'] += wave_loss

            l_total += l_wave

        loss_dict['l_total'] = l_total
        # self.scaler.scale(l_total).backward()
        # self.scaler.step(self.optimizer_g)
        # self.scaler.update()

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, self.fea_s = self.net_g_ema(self.lq)
                self.output_t, self.fea_t = self.net_t(self.lq)

        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, self.fea_s = self.net_g(self.lq)
                self.output_t, self.fea_t = self.net_t(self.lq)

            self.net_g.train()

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

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            # output_list = self.output_s

            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                for i in range(1):
                    for metric in self.opt['val']['metrics'].keys():
                        self.metric_results[metric + f'_s_{i}'] = 0
                        self.metric_results[metric + f'_t_{i}'] = 0
                        # self.metric_results =
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name, 3)
        # zero self.metric_results
        if with_metrics:
            for i in range(1):
                for metric in self.opt['val']['metrics'].keys():
                    self.metric_results[metric + f'_s_{i}'] = 0
                    self.metric_results[metric + f'_t_{i}'] = 0

            # self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            # print(self.gt,'self.gt')
            output_list = self.output[-1]
            for i in range(len(output_list)):
                print('valmodule!!!')
                break
                self.output = output_list[i]
                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['result']])
                metric_data['img'] = sr_img
                if 'gt' in visuals:
                    if i == 0:
                        gt_img = tensor2img([visuals['gt']])
                        metric_data['img2'] = gt_img
                    # if i==len(output_list)-1:
                    #     del self.gt

                # tentative for out of GPU memory
                # if i == len(output_list) - 1:
                #     del self.lq
                del self.output
                torch.cuda.empty_cache()

                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}.png')
                    imwrite(sr_img, save_img_path)

                if with_metrics:
                    # calculate metrics

                    for name, opt_ in self.opt['val']['metrics'].items():
                        if name + f'_s_{i}' not in list(self.metric_results.keys()):
                            self.metric_results[name + f'_s_{i}'] = 0
                        self.metric_results[name + f'_s_{i}'] += calculate_metric(metric_data, opt_)
                if use_pbar:
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
            # output_list = [self.output_t[i] for i in self.numberlist]
            output_list = []
            # output_list.append(self.gaussian_noise(self.output_t[-1]))
            output_list.append(self.output_t[-1])
            # print(torch.max()
            for i in range(len(output_list)):
                self.output = output_list[i]
                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['result']])
                metric_data['img'] = sr_img
                if 'gt' in visuals:
                    if i == 0:
                        gt_img = tensor2img([visuals['gt']])
                        metric_data['img2'] = gt_img
                    if i == len(output_list) - 1:
                        del self.gt

                # tentative for out of GPU memory
                if i == len(output_list) - 1:
                    del self.lq
                del self.output
                torch.cuda.empty_cache()

                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}.png')
                    imwrite(sr_img, save_img_path)

                if with_metrics:
                    # calculate metrics

                    for name, opt_ in self.opt['val']['metrics'].items():
                        # if name+f'_s_{i}' not in list(self.metric_results.keys())  :
                        #     self.metric_results[name + f'_s_{i}']=0
                        if name + f'_t_{i}' not in list(self.metric_results.keys()):
                            self.metric_results[name + f'_t_{i}'] = 0

                        self.metric_results[name + f'_t_{i}'] += calculate_metric(metric_data, opt_)
                if use_pbar:
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')

            if use_pbar:
                pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
