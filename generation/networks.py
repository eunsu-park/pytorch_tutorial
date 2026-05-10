import torch
import torch.nn as nn
import torch.nn.init as init


# ──────────────────────────────────────────────────────────────
# 본 모듈은 Pix2Pix(Conditional GAN) 구조를 따른다.
# - Generator : 입력 영상(inp) → 변환 영상(gen) (U-Net)
# - Discriminator : (입력 영상, 출력 영상) 쌍을 받아 진짜/가짜 판별
#   → 첫 Conv 의 입력 채널이 in_channels + out_channels 인 이유
#
# 따라서 train.py 에서는 다음과 같이 호출되어야 함.
#   pred_real = D( torch.cat([inp, tar], dim=1) )    # 진짜 쌍
#   pred_fake = D( torch.cat([inp, gen], dim=1) )    # 가짜 쌍
# ──────────────────────────────────────────────────────────────


class PatchDiscriminator(nn.Module):
    """
    Pix2Pix 의 PatchGAN Discriminator
    각 패치 단위로 진짜/가짜를 판별. 출력은 logit 맵(use_sigmoid=False) 또는 확률 맵(=True).
    학습 시 BCEWithLogitsLoss 와 함께 쓰면 안전 (use_sigmoid=False 권장).
    """
    def __init__(self, in_channels, out_channels, nb_layers, nb_feat_init, nb_feat_max, use_sigmoid):
        """
        Args:
            in_channels  : Generator 입력 영상의 채널 수 (Discriminator 입력의 일부)
            out_channels : Generator 출력 영상의 채널 수 (Discriminator 입력의 일부)
            nb_layers    : 다운샘플링 Conv 블록 수
            nb_feat_init : 첫 블록의 출력 feature 수
            nb_feat_max  : feature 수 상한
            use_sigmoid  : True 이면 출력에 sigmoid 적용 (BCELoss 호환)
                           False 이면 logit 그대로 (BCEWithLogitsLoss 와 함께 사용 — 권장)
        """
        super(PatchDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_layers = nb_layers
        self.nb_feat_init = nb_feat_init
        self.nb_feat_max = nb_feat_max
        self.use_sigmoid = use_sigmoid
        self.build()

    def build(self):
        nb_feat_in = self.in_channels + self.out_channels
        nb_feat_out = self.nb_feat_init
        block = [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=True),
                 nn.LeakyReLU(0.2)]

        for _ in range(1, self.nb_layers):
            nb_feat_in = nb_feat_out
            nb_feat_out = min(nb_feat_out*2, self.nb_feat_max)
            block += [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(nb_feat_out), nn.LeakyReLU(0.2)]

        nb_feat_in = nb_feat_out
        nb_feat_out = min(nb_feat_out*2, self.nb_feat_max)
        block += [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(nb_feat_out), nn.LeakyReLU(0.2)]

        nb_feat_in = nb_feat_out
        nb_feat_out = 1
        block += [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=1, padding=1, bias=True)]
        if self.use_sigmoid :
            block += [nn.Sigmoid()]

        self.model = nn.Sequential(*block)
        
    def forward(self, x):
        return self.model(x)


class PixelDiscriminator(nn.Module):
    """
    Pix2Pix 의 1x1 PixelGAN Discriminator
    각 픽셀 단위로 진짜/가짜를 판별 (PatchGAN 의 극단적 케이스, patch=1).
    """
    def __init__(self, in_channels, out_channels, nb_feat_init, use_sigmoid):
        """
        Args:
            in_channels  : Generator 입력 영상의 채널 수
            out_channels : Generator 출력 영상의 채널 수
            nb_feat_init : 첫 블록의 출력 feature 수
            use_sigmoid  : True/False 의미는 PatchDiscriminator 와 동일
        """
        super(PixelDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_feat_init = nb_feat_init
        self.use_sigmoid = use_sigmoid
        self.build()

    def build(self):
        nb_feat_in = self.in_channels + self.out_channels
        nb_feat_out = self.nb_feat_init
        block = [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=1, stride=1, padding=0, bias=True),
                 nn.LeakyReLU(0.2)]
        
        nb_feat_in = nb_feat_out
        nb_feat_out = nb_feat_in * 2
        block += [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=1, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(nb_feat_out), nn.LeakyReLU(0.2)]
        
        nb_feat_in = nb_feat_out
        nb_feat_out = 1
        block += [nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=1, stride=1, padding=0, bias=True)]
        if self.use_sigmoid :
            block += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*block)
        
    def forward(self, x):
        return self.model(x)


class UnetDown(nn.Module):
    """U-Net 의 다운샘플링 블록 : LeakyReLU → Conv(stride=2) → BN"""
    def __init__(self, nb_feat_in, nb_feat_out):
        super(UnetDown, self).__init__()
        self.build(nb_feat_in, nb_feat_out)

    def build(self, nb_feat_in, nb_feat_out):
        block = [nn.LeakyReLU(0.2),
                 nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nb_feat_out)]
        self.model = nn.Sequential(*block)

    def forward(self, inp):
        return self.model(inp)
    

class UnetCenter(nn.Module):
    """U-Net 의 가장 안쪽(bottleneck) 블록 : down → ReLU → up → BN → Dropout"""
    def __init__(self, nb_feat_in, nb_feat_out):
        super(UnetCenter, self).__init__()
        self.build(nb_feat_in, nb_feat_out)

    def build(self, nb_feat_in, nb_feat_out):
        block = [nn.LeakyReLU(0.2),
                 nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=True),
                 nn.ReLU(),
                 nn.ConvTranspose2d(nb_feat_out, nb_feat_in, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nb_feat_in),
                 nn.Dropout2d(0.5)]
        self.model = nn.Sequential(*block)

    def forward(self, inp):
        return self.model(inp)

class UnetUp(nn.Module):
    """U-Net 의 업샘플링 블록 : ReLU → ConvTranspose(stride=2) → BN (+선택적 Dropout)"""
    def __init__(self, nb_feat_in, nb_feat_out, use_dropout):
        super(UnetUp, self).__init__()
        self.build(nb_feat_in, nb_feat_out, use_dropout)

    def build(self, nb_feat_in, nb_feat_out, use_dropout):
        block = [nn.ReLU(),
                 nn.ConvTranspose2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nb_feat_out)]
        if use_dropout == True :
                block += [nn.Dropout2d(0.5)]
        self.model = nn.Sequential(*block)

    def forward(self, inp):
        return self.model(inp)


class UnetGenerator(nn.Module):
    """
    U-Net 기반 Generator (Pix2Pix)
    nb_down_G 단계로 다운샘플링했다가 동일 단계로 업샘플링하면서 skip connection 으로 합침.
    """
    def __init__(self, in_channels, out_channels, nb_down_G, nb_feat_init_G, use_dropout, use_tanh):
        super(UnetGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_down_G = nb_down_G
        self.nb_feat_init_G = nb_feat_init_G
        self.use_dropout = use_dropout
        self.use_tanh = use_tanh
        self.build()

    def build(self):
        nb_feat_after = self.nb_feat_init_G
        self.block_down_0 = nn.Conv2d(self.in_channels, nb_feat_after, kernel_size=4, stride=2, padding=1, bias=True)
        block_up_0 = [nn.ConvTranspose2d(nb_feat_after*2, self.out_channels, kernel_size=4, stride=2, padding=1, bias=True)]
        if self.use_tanh == True :
            block_up_0 += [nn.Tanh()]
        self.block_up_0 = nn.Sequential(*block_up_0)
        for i in range(self.nb_down_G - 2):
            nb_feat_before = nb_feat_after
            nb_feat_after = min(nb_feat_after*2, 512)
            setattr(self, 'block_down_%d'%(i+1), UnetDown(nb_feat_before, nb_feat_after))
            use_dropout = True if i < 2 else False
            setattr(self, 'block_up_%d'%(i+1), UnetUp(nb_feat_after*2, nb_feat_before, use_dropout))
        nb_feat_before = nb_feat_after
        nb_feat_after = min(nb_feat_after*2, 512)
        self.block_center = UnetCenter(nb_feat_before, nb_feat_after)

    def forward(self, inp):
        # 명시적 형태(이해를 돕기 위한 4-스텝 예시)
        #   down_0 = self.block_down_0(inp)
        #   down_1 = self.block_down_1(down_0)
        #   down_2 = self.block_down_2(down_1)
        #   down_3 = self.block_down_3(down_2)
        #   center = self.block_center(down_3)
        #   up_3   = self.block_up_3(torch.cat([center, down_3], 1))
        #   up_2   = self.block_up_2(torch.cat([up_3, down_2], 1))
        #   up_1   = self.block_up_1(torch.cat([up_2, down_1], 1))
        #   up_0   = self.block_up_0(torch.cat([up_1, down_0], 1))
        #   return up_0
        # 아래는 nb_down_G 가 가변일 때를 위한 일반화 구현 — 동작은 위와 동일.
        layers = [inp]
        for i in range(self.nb_down_G-1):
            layers.append(getattr(self, 'block_down_%d'%(i)) (layers[-1]))

        last = self.block_center(layers[-1])

        for j in range(self.nb_down_G -1):
            tmp = torch.cat([last, layers[-j-1]], 1)
            layer = getattr(self, 'block_up_%d'%(self.nb_down_G - j - 2))
            last = layer(tmp)

        return last


def define_discriminator(opt, state_dict=None, device=None):
    """
    opt 에 따라 Patch / Pixel Discriminator 를 생성한다.
    nb_layers_D == 0 이면 PixelDiscriminator, 그 외에는 PatchDiscriminator 사용.
    """
    in_channels = opt.in_channels
    out_channels = opt.out_channels
    nb_layers = opt.nb_layers_D
    nb_feat_init = opt.nb_feat_init_D
    nb_feat_max = opt.nb_feat_max_D
    use_sigmoid = opt.use_sigmoid_D

    if nb_layers < 0 :
        raise ValueError("nb_layers must be greater than 0")
    elif nb_layers == 0 :
        discriminator = PixelDiscriminator(in_channels, out_channels, nb_feat_init, use_sigmoid)
    else :
        discriminator = PatchDiscriminator(in_channels, out_channels, nb_layers, nb_feat_init, nb_feat_max, use_sigmoid)

    if state_dict is not None :
        discriminator.load_state_dict(state_dict)
    else :
        discriminator = init_network(discriminator, init_type='normal', init_gain=0.02)

    if device is not None :
        discriminator.to(device)

    return discriminator


def define_generator(opt, state_dict=None, device=None):
    """opt 에 따라 U-Net Generator 를 생성한다."""
    in_channels = opt.in_channels
    out_channels = opt.out_channels
    nb_down_G = opt.nb_down_G
    nb_feat_init_G = opt.nb_feat_init_G
    use_dropout = opt.use_dropout_G
    use_tanh = opt.use_tanh_G

    generator = UnetGenerator(in_channels, out_channels, nb_down_G, nb_feat_init_G, use_dropout, use_tanh)

    if state_dict is not None :
        generator.load_state_dict(state_dict)
    else :
        generator = init_network(generator, init_type='normal', init_gain=0.02)

    if device is not None :
        generator.to(device)

    return generator


def init_network(network, init_type='normal', init_gain=0.02):
    """
    네트워크의 가중치를 초기화하는 함수
    Args:
        net : nn.Module
            네트워크
        init_type : str, default='normal'
            초기화 방법
        init_gain : float, default=0.02
            초기화 gain
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1: 
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    network.apply(init_func)
    return network
