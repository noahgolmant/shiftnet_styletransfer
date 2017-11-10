import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from shiftnet_cuda_v2 import Shift3x3_cuda, GenericShift_cuda

class TransformerNet(torch.nn.Module):
    def __init__(self, shiftnet=True):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=15, stride=1) # k size was 9
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        #if shiftnet:
        #    print('using shiftnet!')
        #    self.conv2 = ShiftConv(32, 64, kernel_size=7, stride=2) # k size was 3
        #else:
        if shiftnet:
		print('using shiftnet!')
	self.conv2 = ConvLayer(32, 64, kernel_size=5, stride=2) # was 3
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        if shiftnet:
        	self.conv3 = ShiftConv(64, 128, stride=2)
        else:
        	self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128, shiftnet)
        self.res2 = ResidualBlock(128, shiftnet)
        self.res3 = ResidualBlock(128, shiftnet)
        self.res4 = ResidualBlock(128, shiftnet)
        self.res5 = ResidualBlock(128, shiftnet)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, shiftnet, kernel_size=5, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, shiftnet, kernel_size=7, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=15, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class Shift3x3(torch.nn.Module):
    def __init__(self, planes):
        super(Shift3x3, self).__init__()

        self.planes = planes
        kernel = np.zeros((planes, 1, 3, 3), dtype=np.float32)
        ch_idx = 0
        for i in range(3):
            for j in range(3):
                num_ch = planes//9+planes%9 if i == 1 and j == 1 else planes//9
                kernel[ch_idx:ch_idx+num_ch, 0, i, j] = 1
                ch_idx += num_ch

        self.register_parameter('bias', None)
        # self.register_buffer('kernel', torch.from_numpy(kernel))
        self.kernel = Variable(torch.from_numpy(kernel), requires_grad=False).cuda()

    def forward(self, input):
        return F.conv2d(input,
                        self.kernel, #Variable(self.kernel, requires_grad=False),
                        self.bias,
                        (1, 1), # stride
                        (1, 1), # padding
                        1, # dilation
                        self.planes, #groups
                       )

class ShiftConv(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
		super(ShiftConv, self).__init__()
		self.shift = GenericShift_cuda(kernel_size, dilation)
		self.conv  = torch.nn.Conv2d(in_channels, out_channels, 1, stride)

	def forward(self, x):
		return self.conv(self.shift(x))


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, shiftnet=True):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, shiftnet):
        super(ResidualBlock, self).__init__()
        if shiftnet:
        	self.conv1 = ShiftConv(channels, channels, stride=1)
        else: 
        	self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        if shiftnet:
        	self.conv2 = ShiftConv(channels, channels, stride=1)
        else:
        	self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, shiftnet, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
	self.shiftnet = shiftnet
        if upsample:
            self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        if shiftnet:
        	self.conv2d = ShiftConv(in_channels, out_channels, kernel_size, stride=stride)
        else:
        	self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = x_in if self.shiftnet else self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
