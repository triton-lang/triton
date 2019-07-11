import torch
import math
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from torch.distributions import categorical

torch.ops.load_library("/home/philippe/development/triton/build/examples/python/pytorch/libtorch_triton.so")

#################################
#######   Convolutions ##########
#################################

class ConvFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        if bias is None:
          bias = torch.empty(0)
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        output = torch.ops.triton.conv_fprop(input, weight, bias, stride[0], stride[1], padding[0], padding[1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.ops.triton.conv_bprop(grad_output, weight, bias, input.shape[2], input.shape[3], stride[0], stride[1], padding[0], padding[1])
        if ctx.needs_input_grad[1]:
            grad_weight = torch.ops.triton.conv_wgrad(input, grad_output, bias, weight.shape[1], weight.shape[2], stride[0], stride[1], padding[0], padding[1])
        if ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, (0, 2, 3))
        return grad_input, grad_weight, grad_bias, None, None
        

class _ConvNd(torch.nn.Module):
  
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        # not everything is supported by Triton
        assert all(x==1 or x==2 for x in stride)
        assert all(x==1 for x in dilation)
        assert transposed == False
        assert all(x==0 for x in output_padding)
        assert groups == 1
        # initialize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = torch.nn.Parameter(torch.Tensor(
            in_channels, kernel_size[0], kernel_size[1], out_channels))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        return ConvFunction.apply(input, self.weight, self.bias, self.stride, self.padding)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
            
  
class Conv2d(_ConvNd):
  
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

#################################
####   Shift-Convolutions #######
#################################

class ShiftConvFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, width, shift_h, shift_w):
        if bias is None:
          bias = torch.empty(0)
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.width = width
        ctx.shift_h = shift_h
        ctx.shift_w = shift_w
        output = torch.ops.triton.shift_conv_y(input, weight, bias,
                                               width[0], width[1],
                                               stride[0], stride[1],
                                               shift_h, shift_w)
        return output

    @staticmethod
    def backward(ctx, dy):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        width = ctx.width
        shift_h = ctx.shift_h
        shift_w = ctx.shift_w
        dx = dw = dbias = None
        if ctx.needs_input_grad[0]:
            dx = torch.ops.triton.shift_conv_dx(dy.contiguous(), weight, bias, width[0], width[1], stride[0], stride[1], shift_h, shift_w)
        if ctx.needs_input_grad[1]:
            dw = torch.ops.triton.shift_conv_dw(dy.contiguous(), input, bias, width[0], width[1], stride[0], stride[1], shift_h, shift_w)
        if ctx.needs_input_grad[2]:
            dbias = torch.sum(dy, (1, 2, 3))
        #print('dx', ctx.needs_input_grad[0], np.isnan(dx.cpu().numpy()).any())
        #print('dw', ctx.needs_input_grad[1], np.isnan(dw.cpu().numpy()).any())
        return dx, dw, dbias, None, None, None, None


class _ShiftConvNd(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super(_ShiftConvNd, self).__init__()
        # initialize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.shift_h = self.make_shift(kernel_size[0])
        self.shift_w = self.make_shift(kernel_size[1])
        self.reset_parameters()

    def forward(self, input):
        return ShiftConvFunction.apply(input, self.weight, self.bias, self.stride,
                                       self.kernel_size, self.shift_h, self.shift_w)

    def make_shift(self, kernel_size):
        if kernel_size == 3:
            p = torch.Tensor([0.3, 0.4, 0.3])
        elif kernel_size == 5:
            p = torch.Tensor([0.1, 0.25, 0.3, 0.25, 0.1])
        elif kernel_size == 7:
            p = torch.Tensor([0.075, 0.1, 0.175, 0.3, 0.175, 0.1, 0.075])
        elif kernel_size == 9:
            p = torch.Tensor([0.05, 0.075, 0.1, 0.175, 0.2, 0.175, 0.1, 0.075, 0.05])
        else:
            raise RuntimeError('Unsupported kernel size')
        return categorical.Categorical(p).sample((self.in_channels,)) - (kernel_size // 2)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

class ShiftConv2d(_ShiftConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        super(ShiftConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, bias)

#################################
#########   BatchNorm ###########
#################################

class BatchNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        ctx.eps = eps
        y, mean, var = torch.ops.triton.batchnorm_ymv(x, gamma, beta, eps)
        ctx.save_for_backward(x, gamma, beta, mean, var)
        return y

    @staticmethod
    def backward(ctx, dy):
        eps = ctx.eps
        x, gamma, beta, mean, var  = ctx.saved_tensors
        dx, dg, db = torch.ops.triton.batchnorm_dxdgdb(dy.contiguous(), x, gamma, mean, var, eps)
        return dx, dg, db, None


class _BatchNorm(torch.nn.Module):

    def __init__(self, num_features, eps=1e-5):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.Tensor(num_features))
        self.bias = torch.nn.Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return BatchNormFunction.apply(input, self.weight, self.bias, self.eps)

class BatchNorm2d(_BatchNorm):

    pass
