import torch
import math

torch.ops.load_library("/home/philippe/Development/triton/build/examples/python/pytorch/libtorch_triton.so")

class ConvFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, padding):
        ctx.save_for_backward(input, weight)
        ctx.padding = padding
        output = torch.ops.triton.conv_fprop(input, weight, padding, padding)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        padding = ctx.padding
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.ops.triton.conv_bprop(grad_output, weight, padding, padding)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.ops.triton.conv_wgrad(input, grad_output, padding, padding)
        return grad_input, grad_weight, None
        

class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = torch.nn.Parameter(torch.Tensor(
            in_channels, kernel_size[0], kernel_size[1], out_channels))
        self.reset_parameters()

    def forward(self, input):
        return ConvFunction.apply(input, self.weight, self.padding)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
