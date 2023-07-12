import torch
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import Upsample, make_layer
# from basicsr.utils.registry import ARCH_REGISTRY
# from arch_util import Upsample, make_layer


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


@ARCH_REGISTRY.register()
class RCAN(nn.Module):
    """Residual Channel Attention Networks.

    ``Paper: Image Super-Resolution Using Very Deep Residual Channel Attention Networks``

    Reference: https://github.com/yulunzhang/RCAN

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 path=None,
                 pretrained=False,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(RCAN, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.upscale=upscale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.path = path

        if pretrained:
            # for p in self.parameters():
            #     p.requires_grad = False
            self.load_pretrained()
            # for name, p in self.named_parameters():
            #     p.requires_grad = False

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean
        # fealist=[]
        # outputlist=[]
        # outputlist.append(x)
        # return outputlist,fealist
        return x

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

    def load_model(self, model_dict_p):
        model_state_dict = self.state_dict()
        # print(model_dict.keys())
        # print(model_state_dict.keys())

        # for k in model_state_dict.keys():
        #     if 'sample' in k:
        #         print(k)
        #
        # assert False
        # assert  False
        # model_dict=model_dict_p['params']
        pretrained_dict={}
        for k, v in model_dict.items():
            if 'head.0' in k:
                k=k.replace('head.0','conv_first')
                pretrained_dict[k]=v
            if 'body_group' in k:
                k=k.replace('body_group','body.')
                if 'weight' in k :
                    k=k[:-7]
                    m='.weight'
                elif 'bias' in k:
                    k=k[:-5]
                    m='.bias'
                if len(k)==20:      #body.9.body.9.body.0
                    k=k[:6]+'.residual_group'+k[11:14]+'rcab'+k[18:]

                elif len(k) == 21:  # body.9.body.10.body.0
                    k=k[:6]+'.residual_group'+k[11:15]+'rcab'+k[19:]

                elif len(k) == 30:  # body.9.body.9.body.3.conv_du.0
                    k=k[:6]+'.residual_group'+k[11:14]+'rcab'+k[18:21]+'attention.'+str(int(k[-1])+1)

                elif len(k) == 31:  # body.9.body.10.body.3.conv_du.0
                    k=k[:6]+'.residual_group'+k[11:15]+'rcab'+k[19:22]+'attention.'+str(int(k[-1])+1)

                elif len(k) == 13:  # body.7.body.6
                    k=k[:6]+'.conv'
                elif len(k) == 14:  # body.7.body.20
                    k=k[:6]+'.conv'
                elif len(k)==7:  #body.10
                    k='conv_after_body'
                elif len(k)==9:  #    body_tail
                    k='conv_after_body'

                k=k+m
                pretrained_dict[k]=v

            if 'tail.0.0' in k:
                k=k.replace('tail.0.0','upsample.0')
                pretrained_dict[k]=v
            if 'tail.0.2' in k:
                k=k.replace('tail.0.2','upsample.2')
                pretrained_dict[k]=v

            if 'tail.1' in k:
                k=k.replace('tail.1','conv_last')
                pretrained_dict[k]=v

            elif 'body' in k:
                if 'weight' in k :
                    k=k[:-7]
                    m='.weight'
                elif 'bias' in k:
                    k=k[:-5]
                    m='.bias'
                if len(k)==20:      #body.9.body.9.body.0
                    k=k[:6]+'.residual_group'+k[11:14]+'rcab'+k[18:]

                elif len(k) == 21:  # body.9.body.10.body.0
                    k=k[:6]+'.residual_group'+k[11:15]+'rcab'+k[19:]

                elif len(k) == 30:  # body.9.body.9.body.3.conv_du.0
                    k=k[:6]+'.residual_group'+k[11:14]+'rcab'+k[18:21]+'attention.'+str(int(k[-1])+1)

                elif len(k) == 31:  # body.9.body.10.body.3.conv_du.0
                    k=k[:6]+'.residual_group'+k[11:15]+'rcab'+k[19:22]+'attention.'+str(int(k[-1])+1)

                elif len(k) == 13:  # body.7.body.6
                    k=k[:6]+'.conv'
                elif len(k) == 14:  # body.7.body.20
                    k=k[:6]+'.conv'
                elif len(k)==7:  #body.10
                    k='conv_after_body'
                elif len(k)==9:  #    body_tail
                    k='conv_after_body'

                k=k+m
                pretrained_dict[k]=v

            if 'tail.0.0' in k:
                k=k.replace('tail.0.0','upsample.0')
                pretrained_dict[k]=v
            if 'tail.0.2' in k:
                k=k.replace('tail.0.2','upsample.2')
                pretrained_dict[k]=v

            if 'tail.1' in k:
                k=k.replace('tail.1','conv_last')
                pretrained_dict[k]=v
            # pretrained_dict[k]=v

        # print(pretrained_dict)
        true_pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        # if self.tea==False:
        #     for name,_ in model_state_dict.items():
        #         if 'upsampler_a' in name:
        #             pretrained_dict[name]=model_dict[name.replace('upsampler_a','tail')]
        print(
            f"the prune number is {round((len(model_state_dict.keys()) - len(true_pretrained_dict.keys())) * 100 / len(model_state_dict.keys()), 3)}%"
        )
        print("missing keys:")
        # print('ft'*50)
        for key in model_state_dict.keys():
            if key not in true_pretrained_dict:
                print(key,'*')
        # assert False

        # print("missing keys:")
        # # print('ft'*50)
        # for key in model_state_dict.keys():
        #     if key not in pretrained_dict:
        #         print(key)
        self.dict_copy(true_pretrained_dict)
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
        if int(self.upscale) == 2:
            print("loading RCANx2")
            # print('ft_KD!')
            dict = torch.load(self.path)
            # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
            # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
            # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
            self.load_model(dict)
        elif int(self.upscale) == 3:
            print("loading RCANx3")
            # print('ft_KD!')
            dict = torch.load(self.path)
            # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
            # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
            # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
            self.load_model(dict)
        elif int(self.upscale) == 4:
            print("loading RCANx4")
            # print('ft_KD!')
            dict = torch.load(self.path)
            # dict=torch.load('/home/isalab305/XXX/teacher_checkpoint/RCAN_BIX2.pt')
            # dict=torch.load('/home/isalab305/XXX/basic/experiments/RCANx2_ft_DIV2K_rand0_1021/models/net_g_100000.pth')
            # dict=torch.load(self.path+'/checkpoints/RCAN_BIX2.pt')
            self.load_model(dict)



if __name__ == '__main__':
    model = RCAN(
        3,
        3,
        num_feat=64,
        num_group=10,
        num_block=20,
        squeeze_factor=16,
        upscale=2)

    stat(model,(3,256,256))


