import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torchvision.ops import RoIAlign
from torchvision import models
from torchvision.models import resnet50
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock


from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class Upsample(nn.Module): 
    # block1 제외 나머지 block들 첫번째 styleconv의 upsample factor = 2, 나머지는 factor = 1
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, 4, 16))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style):
        out = self.conv(input, style)
        # out = out + self.bias
        out = self.activate(out)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        # output channel=3, kernel size=1
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class MaskConv(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        # output channel=1, kernel size=1
        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, up_factor=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)

class Origin_ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            torch.nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias# and not activate,
            )
        )
        "변경"
        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], encoder=False):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        if encoder:
            self.conv2 = ConvLayer(in_channel, out_channel, 3)
            self.skip = ConvLayer(
                in_channel, out_channel, 1, activate=False, bias=False
            )
        else:
            self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=True, activate=False, bias=False
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Origin_ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], encoder=False):
        super().__init__()

        self.conv1 = Origin_ConvLayer(in_channel, in_channel, 3)
        if encoder:
            self.conv2 = Origin_ConvLayer(in_channel, out_channel, 3)
            self.skip = Origin_ConvLayer(
                in_channel, out_channel, 1, activate=False, bias=False
            )
        else:
            self.conv2 = Origin_ConvLayer(in_channel, out_channel, 3, downsample=True)
            self.skip = Origin_ConvLayer(
                in_channel, out_channel, 1, downsample=True, activate=False, bias=False
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512,
            128: 512,
            256: 512,
        }

        size = 64
        convs = [ConvLayer(3, 512, 1)]

        log_size = int(math.log(size, 2))

        in_channel = 512

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel+1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(32768, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

class Encoder(nn.Module):
    def __init__(self, w_dim=512):
        super().__init__()
        
        size = 256
        channels = {
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64
        }        
        
        self.w_dim = w_dim
        log_size = int(math.log(size, 2))
        
        self.n_latents = log_size*2 - 2
        
        convs = []
        convs.append(ConvLayer(3, 32, 3))
        convs.append(ConvLayer(32, 64, 3))
        convs.append(nn.MaxPool2d(kernel_size=2, stride=2))

        res_num = [1,2,5,3]
        res = 0
        
        in_channel = 64
        for i in range(8, 4, -1):
            out_channel = channels[2 ** (i-1)]
            for j in range(0,res_num[res]):
                if j==0:
                    convs.append(ResBlock(in_channel, out_channel,encoder=True))
                else:
                    convs.append(ResBlock(out_channel, out_channel,encoder=True))

            if i == 5: # Conv4-1 (마지막 layer)
                convs.append(ConvLayer(out_channel, out_channel, 3))
            else: 
                convs.append(ConvLayer(out_channel, out_channel, 3))
                convs.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channel=out_channel
            res+=1
            
        convs.append(nn.AvgPool2d(kernel_size=16,stride=1))
        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        out = self.convs(input)
        return out.view(len(input), 512)


class ImageToLatent(torch.nn.Module):
    def __init__(self, image_size=256):
        super().__init__()
        
        self.image_size = image_size
        self.activation = torch.nn.ELU()
        
        self.resnet = list(resnet50(pretrained=False).children())[:-2]
        self.resnet = torch.nn.Sequential(*self.resnet)
        self.conv2d = torch.nn.Conv2d(2048, 512, kernel_size=1)
        # self.flatten = torch.nn.Flatten()
        # self.dense1 = torch.nn.Linear(16384, 256)
        # self.dense2 = torch.nn.Linear(256, (18 * 512))
        self.maxpool = nn.MaxPool2d(kernel_size=8, stride=1)


    def forward(self, image):
        x = self.resnet(image)
        x = self.conv2d(x)
        x = self.maxpool(x)
        x= x.view(len(image), 512)

        return x

class Style_Encoder(nn.Module):
    def __init__(self, w_dim=512):
        super().__init__()
        
        size = 256
        channels = {
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64
        }        
        
        self.w_dim = w_dim
        log_size = int(math.log(size, 2))
        
        self.n_latents = log_size*2 - 2
        
        convs = []
        convs.append(Origin_ConvLayer(3, 32, 3))
        convs.append(Origin_ConvLayer(32, 64, 3))
        convs.append(nn.MaxPool2d(kernel_size=2, stride=2))

        res_num = [1,2,5,3]
        res = 0
        
        in_channel = 64
        for i in range(8, 4, -1):
            out_channel = channels[2 ** (i-1)]
            for j in range(0,res_num[res]):
                if j==0:
                    convs.append(Origin_ResBlock(in_channel, out_channel,encoder=True))
                else:
                    convs.append(Origin_ResBlock(out_channel, out_channel,encoder=True))

            if i == 5: # Conv4-1 (마지막 layer)
                convs.append(Origin_ConvLayer(out_channel, out_channel, 3))
            else: 
                convs.append(Origin_ConvLayer(out_channel, out_channel, 3))
                convs.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channel=out_channel
            res+=1

        final = nn.AvgPool2d(kernel_size=16,stride=1)
        self.final = final
        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        out = self.convs(input)
        out = self.final(out)
        return out.view(len(input), 512)

class Content_Encoder(nn.Module):
    def __init__(self, w_dim=512, input_channel=1):
        super().__init__()
        
        size = 256
        channels = {
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64
        }        
        
        self.w_dim = w_dim
        log_size = int(math.log(size, 2))
        
        self.n_latents = log_size*2 - 2
        
        convs = []
        convs.append(Origin_ConvLayer(input_channel, 32, 3)) #Gray Scale Image
        convs.append(Origin_ConvLayer(32, 64, 3))
        convs.append(nn.MaxPool2d(kernel_size=2, stride=2))

        res_num = [1,2,5,3]
        res = 0
        
        in_channel = 64
        for i in range(8, 4, -1):
            out_channel = channels[2 ** (i-1)]
            for j in range(0,res_num[res]):
                if j==0:
                    convs.append(Origin_ResBlock(in_channel, out_channel,encoder=True))
                else:
                    convs.append(Origin_ResBlock(out_channel, out_channel,encoder=True))

            if i == 5: # Conv4-1 (마지막 layer)
                convs.append(Origin_ConvLayer(out_channel, out_channel, 3))
            else: 
                convs.append(Origin_ConvLayer(out_channel, out_channel, 3))
                convs.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channel=out_channel
            res+=1

        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        out = self.convs(input) #out shape:[1,512,4,16]
        return out

#---------------------------------------------------------------
# tsb 구현 레포의 content encoder, style encoder, mapping network

class ContentResnet(models.ResNet):
    def __init__(self):
        # resnet18 init
        super().__init__(BasicBlock, [2, 2, 2, 2])

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return x

class StyleResnet(models.ResNet):
    def __init__(self):
        # resnet18 init
        super().__init__(BasicBlock, [2, 2, 2, 2])
        self.fc = torch.nn.Identity()

class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''
    def __init__(self, name):
        self.name = name
    
    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        
        return weight * math.sqrt(2 / fan_in)
    
    @staticmethod
    def apply(module, name):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)
    
    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)

# Quick apply for scaled weight
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module

class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        
        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)


class Intermediate_Generator(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    '''
    def __init__(self, dim_latent):
        super().__init__()
        layers = [PixelNorm()]
        layers.append(SLinear(dim_latent, dim_latent))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(SLinear(dim_latent, dim_latent))
        layers.append(nn.LeakyReLU(0.2))
            
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, latent_z):
        latent_w = self.mapping(latent_z.squeeze())
        return latent_w    
#---------------------------------------------------------------
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class Generator(nn.Module):
    def __init__(
        self,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01
    ):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512
        }
        
        # style mapping network 2 layers
        layers = [PixelNorm()]
        for i in range(0, 2):
            layers.append(
                EqualLinear(
                    512, 512, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )
        self.input = ConstantInput(512)
        self.style = nn.Sequential(*layers)
        style_dim = 512

        self.style_encoder = StyleResnet() #style encoder
        self.content_encoder = ContentResnet() # content encoder

        # block 1
        self.conv1 = StyledConv(
            512, 512, 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(512, style_dim, upsample=False)

        size = 256

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.n_latent = 14 #mask conV 없을 때는 14, 있을 때는 15

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.masks = nn.ModuleList()

        # styleconv channel=512, kernel=3
        for i in range(0,4): #block 2-5
            self.convs.append(
                StyledConv(
                    512,
                    512,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel
                )
            )

            self.convs.append(
                StyledConv(
                    512, 512, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(512, style_dim))

    def set_style_input(self, batch_size, latent_dim):

        setattr(self, "constant_style",
                nn.Parameter(torch.randn(batch_size, latent_dim)))

    def forward(
        self,
        content,
        styles,
        random_out=False,
        target_styles=None,
        start_index=None,
        finish_index=None,
        return_latents=False,
        return_styles=False,
        style_mix=False,
        input_is_latent=False,
        random_style=False,
        inject_index=None,
        latent_recon=False,
        return_var=False,
        # optimize_style_latent=False,
    ):  
        content = self.content_encoder(content)

        if random_style==False:
            styles = [self.style_encoder(styles)]
        else:
            styles = styles
        re_latent = styles[0]

        if return_var:
            var_tmp = torch.var(styles[0],axis=1)
        # style mapping network
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if style_mix:
            t_styles = [self.style(t) for t in target_styles]
            styles +=t_styles

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        
        """
        if len(styles) < 2:
            start_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, start_index, 1)

            else:
                latent = styles[0]

        else:
            if start_index is None:
                start_index = random.randint(1, self.n_latent - 1) 
            # self.n_latent=14
            if not finish_index: 
                latent = styles[0].unsqueeze(1).repeat(1, start_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - start_index, 1)

                latent = torch.cat([latent, latent2], 1)
            else:
                latent = styles[0].unsqueeze(1).repeat(1, start_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, finish_index - start_index + 1, 1)
                latent3 = styles[0].unsqueeze(1).repeat(1, self.n_latent - finish_index - 1, 1)

                latent = torch.cat([latent, latent2], 1)
                latent = torch.cat([latent, latent3], 1)

        if random_content==True: # content encoder 사용x, random content로 train
            out = self.input(content)
        else: #content encoder 사용
        """
        # block1
        out = self.conv1(content, latent[:, 0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i])
            out = conv2(out, latent[:, i + 1])
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        elif style_mix:
            return image, None
        elif latent_recon:
            return re_latent
        elif return_var:
            return image, var_tmp
        else:
            return image, None

#run current code
if __name__ == "__main__":
    # test_encoder = Content_Encoder(input_channel=3)

    # test_img = torch.randn(1, 3, 256, 64)

    # output = test_encoder(test_img)

    # test_generator = Generator()
    # print(output.shape)


    # test_encoder = Content_Encoder(input_channel=3)
    test_generator = Generator()

    style_img = torch.randn(8,3,64,256)
    import torchvision
    content_img = torchvision.transforms.functional.resize(style_img, (256,256))

    content_img = torch.randn(8,3,64,256)

    output= test_generator(content_img, style_img)

    # content_latent = test_encoder(content_img)


    # style = torch.ones(1,1,512)

    # output = test_generator(content_latent, style)

    # print(output.shape)
    print('done')

    