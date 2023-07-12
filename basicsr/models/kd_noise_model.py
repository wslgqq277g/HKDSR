import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import random
import numpy as np
import math
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels,random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F



# from thop import profile


@MODEL_REGISTRY.register()
class NKD_SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(NKD_SRModel, self).__init__(opt)

        # define network
        # print(type(opt['network_g']))
        # opt['dist']=False
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.net_t = build_network(opt['network_t'])
        self.net_t = self.model_to_device(self.net_t)



        # blur settings for the first degradation
        # self.blur_kernel_size = opt['blur_kernel_size']
        # self.kernel_list = opt['kernel_list']
        # self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        # self.blur_sigma = opt['blur_sigma']
        # self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        # self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        # self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # a final sinc filter
        # self.final_sinc_prob = opt['final_sinc_prob']

        # self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        # self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        # self.pulse_tensor[10, 10] = 1

        # self.print_network(self.net_g)

        # load pretrained models
        # load_path = self.opt['path'].get('pretrain_network_g', None)
        # if load_path is not None:
        #     param_key = self.opt['path'].get('param_key_g', 'params')
        #     self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        # self.kernel1 = self.kernel_g()

        if self.is_train:
            self.init_training_settings()

    # def kernel_g(self):
    #     kernel_size = random.choice(self.kernel_range)
    #     if np.random.uniform() < self.opt['sinc_prob']:
    #         # this sinc filter setting is for kernels ranging from [7, 21]
    #         if kernel_size < 13:
    #             omega_c = np.random.uniform(np.pi / 3, np.pi)
    #         else:
    #             omega_c = np.random.uniform(np.pi / 5, np.pi)
    #         kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    #     else:
    #         kernel = random_mixed_kernels(
    #             self.kernel_list,
    #             self.kernel_prob,
    #             kernel_size,
    #             self.blur_sigma,
    #             self.blur_sigma, [-math.pi, math.pi],
    #             self.betag_range,
    #             self.betap_range,
    #             noise_range=None)
    #     # pad kernel
    #     pad_size = (21 - kernel_size) // 2
    #     kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    #     print(kernel.shape,'111111111')
    #     return kernel

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
            # load_path = self.opt['path'].get('pretrain_network_g', None)
            # if load_path is not None:
            #     self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            # else:
            #     self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get('kd_opt'):
            self.kd = build_loss(train_opt['kd_opt']).to(self.device)
        else:
            self.kd = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
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
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        # print(self.lq.shape)
        # print(self.gt.shape)


        # assert False


    def degradation_feed_data(self,data):
        self.lq = data['lq'].to(self.device)
        self.kernel1 = data['kernel1'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.gt_usm = self.usm_sharpener(self.gt)
        # print(self.kernel1.shape,'***************111')
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 0.5)
        else:
            scale = 0.5
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        self.gt_usm = F.interpolate(self.gt_usm, scale_factor=scale, mode=mode)

        # ----------------------- The first degradation process ----------------------- #

        # blur
        # assert False
        batch_num=len(self.gt_usm)
        for i in range(batch_num):
            out=self.gt_usm[i].unsqueeze(0)
            degradation_type = random.choices(['blur', 'noise', 'compression'], self.opt['deg_prob'])[0]
            if degradation_type == 'blur':

                out = filter2D(out, self.kernel1[i])

            # add noise
            # degradation_type = random.choices(['blur', 'noise', 'compression'], self.opt['deg_prob'])[0]
            elif degradation_type == 'noise':

                gray_noise_prob = self.opt['gray_noise_prob']
                if np.random.uniform() < self.opt['gaussian_noise_prob']:
                    out = random_add_gaussian_noise_pt(
                        out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
            # JPEG compression
            # degradation_type = random.choices(['blur', 'noise', 'compression'], self.opt['deg_prob'])[0]
            elif degradation_type == 'compression':

                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
                out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out = self.jpeger(out, quality=jpeg_p)
            if i==0:
                self.lq_noise = out.to(self.device)
            else:
                self.lq_noise = torch.cat([self.lq_noise,out]).to(self.device)


    def kd_optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()




        self.output = self.net_g(self.lq_noise)
        self.output_t = self.net_t(self.lq_noise)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.kd:
            l_kd=self.kd(self.output,self.output_t)
            l_total += l_kd
            loss_dict['l_kd'] = l_kd

        self.output = self.net_g(self.lq)[-1]

        # l_total = 0
        # loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()




        self.output = self.net_g(self.lq)[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)[-1]
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)[-1]
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
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
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
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

