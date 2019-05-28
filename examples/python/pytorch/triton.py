import torch
from torch.nn.modules.utils import _single, _pair, _triple
import math

torch.ops.load_library("/home/philippe/development/triton/build/examples/python/pytorch/libtorch_triton.so")

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
