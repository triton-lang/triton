import torch
import torch.nn as nn
import isaac.pytorch as sc

class VggBlock(nn.Module):

    def __init__(self, in_num, out_num, bias, activation, alpha, dim = 3):
        super(VggBlock, self).__init__()
        self.conv1 = sc.ConvType[dim](in_num, out_num, 3, bias = bias, activation = activation, alpha = alpha)
        self.conv2 = sc.ConvType[dim](out_num, out_num, 3, bias = bias, activation = activation, alpha = alpha)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_num, out_num, bias, activation, alpha, dim = 3):
        super(ResidualBlock, self).__init__()
        self.conv1 = sc.ConvType[dim](in_num, out_num, (1, 3, 3), padding = (0, 1, 1), bias = bias, activation = activation, alpha = alpha)
        self.conv2 = sc.ConvType[dim](out_num, out_num, (3, 3, 3), padding = (1, 1, 1), bias = bias, activation = activation, alpha = alpha)
        self.conv3 = sc.ConvType[dim](out_num, out_num, (3, 3, 3), padding = (1, 1, 1), bias = bias, activation = activation, alpha = alpha, residual='add')

    def forward(self, x):
        residual = out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out, residual)
        return out



class UNet(nn.Module):

    def __init__(self, in_num=1, out_num=3, filters=[1,28,36,48,64,80], relu_slope=0., relu_type='relu', residual = 'add', BasicBlock = ResidualBlock):
        super(UNet, self).__init__()
        dim = 3

        # Attributes
        self.relu_type = relu_type
        self.relu_slope = relu_slope
        self.filters = filters
        self.depth = len(filters) - 1
        self.in_num = in_num
        self.out_num = out_num

        # Downward convolutions
        first = sc.ConvType[dim](filters[0], filters[1], kernel_size = (1, 5, 5), padding = (0, 2, 2), bias = True, activation = relu_type, alpha = relu_slope)
        DownBlock = lambda in_num, out_num: BasicBlock(in_num, out_num, True, relu_type, relu_slope)
        self.down_conv = nn.ModuleList([first] + [DownBlock(filters[x], filters[x+1]) for x in range(1, self.depth)])

        # Downsample
        Downsample = lambda window_size: sc.MaxPoolType[dim](window_size, window_size)
        self.down_sample = nn.ModuleList([Downsample((1, 2, 2)) for x in range(self.depth - 1)])

        # Upsample
        Upsample = lambda in_num, out_num: sc.ConvType[dim](in_num, out_num, (1,1,1), upsample = (1,2,2), activation = 'linear', bias = True, residual = residual)
        self.up_sample = nn.ModuleList([Upsample(filters[x], filters[x-1]) for x in range(self.depth, 1, -1)])

        # Upward convolution
        expand = 2 if residual=='cat' else 1
        UpBlock = lambda in_num, out_num: BasicBlock(in_num, out_num, True, relu_type, relu_slope)
        self.up_conv = [UpBlock(expand*filters[x-1], filters[x-1]) for x in range(self.depth, 2, -1)]
        self.up_conv += [sc.ConvType[dim](filters[1], out_num, (1, 5, 5), padding = (0, 2, 2), bias=True, activation='sigmoid', alpha=0)]
        self.up_conv = nn.ModuleList(self.up_conv)

        # Use bias initialized to zero instead of no-bias because batch-norm will be folded
        for x in self.modules():
            if isinstance(x, sc.ConvNd):
                x.bias.data.zero_()

    def forward(self, x):
        z = [None]*self.depth
        # Down branch
        for i in range(self.depth - 1):
            z[i] = self.down_conv[i](x)
            x = self.down_sample[i](z[i])
        # Center
        x = self.down_conv[-1](x)
        # Up branch
        for i in range(self.depth - 1):
            x = self.up_sample[i](x, z[self.depth - 2 - i])
            x = self.up_conv[i](x)
        return x
