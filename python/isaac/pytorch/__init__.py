import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple

from torch.autograd import Function
from ctypes import *
from .c_lib import *
import cffi
import struct
import numpy as np

def pad_left(dim, x, value):
    return (value,)*(dim-len(x)) + x

def PackNd(input, alpha, beta):
    output = torch.Tensor().type(torch.IntTensor).cuda()
    isaac_pack_nd(input, output, alpha, beta)
    return output


class ConvNdFunction(Function):
    def __init__(self, activation, alpha, scale, pad = (0, 0, 0), strides = (1, 1, 1), upsample = (1, 1, 1), crop = (0, 0, 0, 0, 0, 0), quantized_in = False, quantized_out = False, residual = '', optimization_level = 1):
        self.activation = activation.encode('utf-8')
        self.residual = '' if residual is None else residual
        self.residual = self.residual.encode('utf-8')
        self.alpha = float(alpha)
        self.scale = scale
        self.pad = pad_left(3, pad, 0)
        self.strides = pad_left(3, strides, 1)
        self.upsample = pad_left(3, upsample, 1)
        self.crop = crop
        self.ffi = c_lib._ffi
        self.quantized_in = quantized_in
        self.quantized_out = quantized_out
        self.function = {(False, False): isaac_conv_nd_float_float,
                          (True, False): isaac_conv_nd_int_float,
                          (False, True): isaac_conv_nd_float_int,
                          (True, True): isaac_conv_nd_int_int}[quantized_in, quantized_out]
        self.optimization_level = optimization_level

    def forward(self, input, weight, bias, z):
        z = z if z.nelement() else self.ffi.NULL
        bias = bias if bias.size() else self.ffi.NULL
        output = input.new().type(torch.cuda.IntTensor if self.quantized_out else torch.cuda.FloatTensor)
        T = torch.utils.ffi._torch_to_cffi.get(output.type())
        outputs = self.ffi.new(T + '*[]', [self.ffi.cast(T + '*', x._cdata) for x in [output]])
        output_scales = self.ffi.new('float[]', [self.scale[2]])
        self.function(input, weight, outputs, 1,
                      self.upsample[0], self.upsample[1], self.upsample[2], # Upsample
                      self.pad[0], self.pad[1], self.pad[2], # Pad
                      self.strides[0], self.strides[1], self.strides[2], # Strides
                      bias, # Bias
                      self.activation, self.alpha, # Activation
                      self.scale[0], self.scale[1], output_scales, self.scale[3], # Quantization
                      self.residual, z, self.crop[0], self.crop[1], self.crop[2], self.crop[3], self.crop[4], self.crop[5], # Crop-cat
                      self.optimization_level
                      )
        return output

class PoolNdFunction(Function):
    def __init__(self, type, kernel_size, scale, pad = (0, 0, 0), strides = (1, 1, 1), quantized_in = False, quantized_out = False, optimization_level = 1):
        self.kernel_size = pad_left(3, kernel_size, 1)
        self.pad = pad_left(3, pad, 0)
        self.strides = pad_left(3, strides, 1)
        self.scale = scale
        self.ffi = cffi.FFI()
        self.quantized_in = quantized_in
        self.quantized_out = quantized_out
        self.type = type.encode('utf-8')
        self.function = {(False, False): isaac_pool_nd_float_float,
                          (True, False): isaac_pool_nd_int_float,
                          (True, True): isaac_pool_nd_int_int}[quantized_in, quantized_out]
        self.optimization_level = optimization_level

    def forward(self, input):
        output = input.new().type(torch.cuda.IntTensor if self.quantized_out else torch.cuda.FloatTensor)
        self.function(input, output,
                      self.type,
                      self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                      self.pad[0], self.pad[1], self.pad[2],
                      float(self.scale[0]), float(self.scale[1]),
                      self.strides[0], self.strides[1], self.strides[2],
                      self.optimization_level)
        return output

class LinearFunction(Function):

    def __init__(self, scale, quantized_in = False, quantized_out = False, optimization_level = 1):
        self.alpha = 1.
        self.beta = 0.
        self.scale = scale
        self.quantized_in = quantized_in
        self.quantized_out = quantized_out
        self.function = {(False, False): isaac_linear_float_float,
                          (True, False): isaac_linear_int_float}[self.quantized_in, self.quantized_out]
        self.optimization_level = 1


    def forward(self, input, weight, bias):
        output = input.new().type(torch.cuda.IntTensor if self.quantized_out else torch.cuda.FloatTensor)
        self.function(input, weight, output, bias,
                      self.alpha, self.beta,
                      self.scale[0], self.scale[1], self.scale[2],
                      self.optimization_level)
        return output

#############################
##       Quantization      ##
#############################
class Quantizer:

    def scale(self, x, activations, bins = None):
        data = x

        if activations:
            mids = bins[:-1] + np.diff(bins)/2
            # Compute CDF
            cdf = np.cumsum(x.numpy())
            cdf = cdf / cdf[-1]
            # Generate data
            n_samples = int(1e6)
            values = np.random.rand(n_samples)
            value_bins = np.searchsorted(cdf, values)
            data = torch.Tensor(mids[value_bins]).cuda()


        def loss(threshold):
            scale = 127. / threshold
            q = torch.clamp(data * scale, -128, 127)
            q = torch.round(q) / scale
            return torch.mean((data - q)**2)

        # Truncation indices
        abs_data = torch.abs(data)
        a, b = float(torch.min(abs_data)), float(torch.max(abs_data))
        epsilon = (b - a)*1e-2
        for i in range(16):
            c = (a + b) / 2
            (a, b) = (c, b) if loss(c) > loss(c + epsilon) else (a, c)
            previous = c


#        # Debug
#        if activations:
#            import matplotlib.pyplot as plt
#            f, axs = plt.subplots(3, 1, sharex=True, sharey=True)

#            bins = np.linspace(data.min(), data.max(), 192)
#            bins[0] = 0
#            bins[1] = 1e-6
#            width = np.diff(bins)
#            center = (bins[:-1] + bins[1:]) / 2
#            # Original distribution
#            original, _ = np.histogram(data.cpu().numpy(), bins=bins, density=True)
#            original[0] = 0

#            # Quantized distributions
#            handles = [None]
#            for ax, (threshold, color, alpha) in  zip(axs, [(1.5, 'blue', 0.3),
#                                                                   (c, 'indianred', 0.3),
#                                                                   (data.max(), 'green', 0.3)]):
#                scale = 127. / threshold
#                quantized = torch.clamp(data * scale, -128, 127)
#                quantized = torch.round(quantized) / scale
#                hist, _ = np.histogram(quantized.cpu().numpy(), bins=bins, density=True)
#                hist[0] = 0
#                bars = ax.bar(center, hist, align='center', width=width, alpha = alpha, color=color, label=r'Quantized ($\tau={:.2f}$)'.format(threshold))
#                ax.axvline(x=threshold, linestyle='--', color=color)
#                ax.text(threshold + (bins[0] - bins[-1])*0.02, 2., 'Saturate', rotation=90, fontsize=18)
#                line = ax.plot(center, original, color='black', linewidth=1., label='Original')
#                ax.set_xlim(left=np.min(bins), right=np.max(bins))
#                handles[0] = line[0]
#                handles += [bars]
#                for tick in ax.xaxis.get_major_ticks():
#                    tick.label.set_fontsize(18)
#                for tick in ax.yaxis.get_major_ticks():
#                    tick.label.set_fontsize(18)

#            # Show
#            f.text(0.5, 0.04, 'Activation value', ha='center', fontsize=18)
#            f.text(0.07, 0.5, 'Probability density', va='center', rotation='vertical', fontsize=18)
#            plt.legend(handles=handles,loc="upper center", bbox_to_anchor=[0.5, 3.8],
#                       ncol=4, shadow=True, fancybox=True, fontsize=18)
#            plt.show()

        return 127. / c

    def __init__(self, stages = 3):
        self.history = dict()
        self.stages = stages



class Quantizable:

    def __init__(self, weight = None):
        self.quantized_in = False
        self.quantized_out = False
        self.is_last = False
        self.is_first = False
        self.weight = weight
        self.state = None
        self.quantizer = None
        self.scale = [1., 1., 1.] + ([1.] if self.weight is not None else [])


    def prepare_quantization(self, quantizer):
        can_quantize_in = True if self.weight is None else self.weight.data.size()[0] % 4 == 0
        can_quantize_out = True if self.weight is None else self.weight.data.size()[-1] % 4 == 0
        self.state = {'quantized_in' : can_quantize_in and not self.is_first,
                      'quantized_out': can_quantize_out and not self.is_last,
                      'min': float('inf'), 'max': float('-inf'), 'scale': [],
                      'stage': 0}
        self.quantizer = quantizer

    def increment_stage(self):
        if self.state['stage'] == 0:
            self.state['bins'] = np.linspace(self.state['min'], self.state['max'], 2049)
            self.state['histogram'] = torch.zeros(2048).cpu()
        self.state['stage'] += 1

    def update(self, x, y, z):
        if self.state == None:
            return

        if self.state and self.state['stage'] == 0:
            y_abs = torch.abs(y.data)
            self.state['min'] = min(self.state['min'], torch.min(y_abs))
            self.state['max'] = max(self.state['max'], torch.max(y_abs))

        if self.state and self.state['stage'] == 1:
            # Update histogram
            if self.state['quantized_out']:
                y_abs = torch.abs(y.data)
                self.state['histogram'] += torch.histc(y_abs.cpu(), bins=2048, min=self.state['min'], max=self.state['max'])


        if self.state and self.state['stage'] == 2:
            # Activate or not quantization
            self.quantized_in = self.state['quantized_in']
            self.quantized_out = self.state['quantized_out']

            # Compute scales
            self.scale[0] = self.quantizer.history[id(x)] if self.quantized_in else 1.
            self.scale[-1] = self.quantizer.history[id(z)] if z is not None else 1.
            if self.quantized_out:
                scale = self.quantizer.scale(self.state['histogram'], True, bins = self.state['bins'])
                self.scale[-2] = self.quantizer.history[id(y)] = scale
            else:
                self.scale[-2] = 1.

            # Scale weights
            if self.quantized_in and self.weight is not None:
                self.scale[1] = self.quantizer.scale(self.weight.data, False)
                transpose_pack(self.weight, self.scale[1])

            # Reset
            self.state = None
            self.quantizer = None



def quantize(model, loader, num_iter, idx = 0):
    def do_on_quantizable(fn):
        for module in model.modules():
            if isinstance(module, Quantizable):
                fn(module)

    # Prepare quantization
    quantizer = Quantizer()
    do_on_quantizable(lambda x: x.prepare_quantization(quantizer))

    # Compute quantization parameters
    for i in range(quantizer.stages):
        iterator = iter(loader)
        for j in range(num_iter):
            with torch.no_grad():
                input = torch.autograd.Variable(next(iterator)[idx]).cuda()
                model.forward(input)
        if i < quantizer.stages - 1:
            do_on_quantizable(lambda x: x.increment_stage())

#############################
###      Convolutions     ###
#############################

def to_chwn_idx(dim):
    return list(range(1, dim)) + [0]

def from_chwn_idx(dim):
    return [dim-1] + list(range(0, dim-1))

def transpose_pack(x, scale):
    dim = len(x.size())
    x.data = PackNd(x.data.permute(*from_chwn_idx(dim)).clone(), float(scale), 0.0)
    x.data = x.data.permute(*to_chwn_idx(dim)).clone()

class ConvNd(nn.modules.conv._ConvNd, Quantizable):

    def __init__(self, dim, in_channels, out_channels, kernel_size, stride, padding, dilation, upsample, groups, bias, activation, alpha, residual):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, False, _single(0), groups, bias)
        Quantizable.__init__(self, self.weight)
        self.activation = activation
        self.alpha = alpha
        self.upsample = upsample
        self.dim = dim
        self.weight.data = self.weight.data.permute(*to_chwn_idx(self.dim))
        self.residual = residual

    def forward(self, x, z = None):
        # Cropping
        if z is None:
            crop = [1, 1, 1, 1, 1, 1]
        else:
            offset = tuple([(z.size()[i]-x.size()[i]*self.upsample[i - 2])//2 for i in range(2,z.dim())])
            offset = pad_left(3, offset, 0)
            crop = (offset[0], offset[0], offset[1], offset[1], offset[2], offset[2])
        # Bias
        bias = self.bias if self.bias is not None else torch.autograd.Variable()
        # Computation
        y = ConvNdFunction(self.activation, self.alpha, self.scale, pad=self.padding, strides=self.stride, upsample=self.upsample, crop=crop, quantized_in=self.quantized_in, quantized_out = self.quantized_out, residual = self.residual)\
                          (x, self.weight, bias, torch.autograd.Variable().cuda() if z is None else z)
        # Quantize if requested
        Quantizable.update(self, x, y, z)
        return y


# 1D Conv
class Conv1d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., residual = None):
        super(Conv1d, self).__init__(4, in_channels, out_channels, _single(kernel_size), _single(stride), _single(padding), _single(dilation), _single(upsample), groups, bias, activation, alpha, residual)

# 2D Conv
class Conv2d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., residual = None):
        super(Conv2d, self).__init__(4, in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation), _pair(upsample), groups, bias, activation, alpha, residual)

# 3D Conv
class Conv3d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, upsample=1, groups=1, bias=True, activation = 'linear', alpha = 0., residual = None):
        super(Conv3d, self).__init__(5, in_channels, out_channels, _triple(kernel_size), _triple(stride), _triple(padding), _triple(dilation), _triple(upsample), groups, bias, activation, alpha, residual)

#############################
###      Pooling          ###
#############################

class PoolNd(nn.Module, Quantizable):

    def __init__(self, kernel_size, type, stride, padding):
        nn.Module.__init__(self)
        Quantizable.__init__(self)
        self.kernel_size = kernel_size
        self.type = type
        self.stride = stride
        self.padding = padding
        # Quantization
        self.quantization_parameters = None
        self.quantized_in = False
        self.quantized_out = False
        self.is_last_conv = False
        self.is_first_conv = False


    def forward(self, x):
        # Computations
        y = PoolNdFunction(self.type, self.kernel_size, self.scale, pad=self.padding, strides=self.stride, quantized_in=self.quantized_in, quantized_out=self.quantized_out)(x)
        # Quantization if requested
        Quantizable.update(self, x, y, None)
        return y


# Max
class MaxPool1d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPool1d, self).__init__(_single(kernel_size), 'max', _single(stride), _single(padding))

class MaxPool2d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPool2d, self).__init__(_pair(kernel_size), 'max', _pair(stride), _pair(padding))

class MaxPool3d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPool3d, self).__init__(_triple(kernel_size), 'max', _triple(stride), _triple(padding))


# Average
class AvgPool1d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(AvgPool1d, self).__init__(_single(kernel_size), 'avg', _single(stride), _single(padding))

class AvgPool2d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(AvgPool2d, self).__init__(_pair(kernel_size), 'avg', _pair(stride), _pair(padding))

class AvgPool3d(PoolNd):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(AvgPool3d, self).__init__(_triple(kernel_size), 'avg', _triple(stride), _triple(padding))


#############################
###      Linear           ###
#############################

class Linear(nn.Linear, Quantizable):

    def __init__(self, in_features, out_features, bias=True):
        nn.Linear.__init__(self, in_features, out_features, bias)
        Quantizable.__init__(self, self.weight)
        self.weight.data = self.weight.data.permute(1, 0)

    def forward(self, x):
        y = LinearFunction(self.scale, self.quantized_in, self.quantized_out)(x, self.weight, self.bias)
        # Quantize if requested
        Quantizable.update(self, x, y, None)
        return y


#############################
###     Helpers           ###
#############################

ConvType = {1:Conv1d, 2:Conv2d, 3:Conv3d}
MaxPoolType = {1:MaxPool1d, 2:MaxPool2d, 3:MaxPool3d}
AvgPoolType = {1:AvgPool1d, 2:AvgPool2d, 3:AvgPool3d}

def convert(model, reference, filter = lambda x: True):
    # Helpers
    extract = lambda x: x.data if isinstance(x, torch.autograd.Variable) else x

    # Copy all non-BatchNorm related parameters
    reference_dict = reference
    reference_keys = [x for x in reference_dict.keys() if filter(x)]
    batch_norm_layers = ['.'.join(x.split('.')[:-1]) for x in reference_keys if '.running_mean' in x]
    batch_norm_keys = [x + '.running_mean' for x in batch_norm_layers]
    batch_norm_keys += [x + '.running_var' for x in batch_norm_layers]
    batch_norm_keys += [x + '.weight' for x in batch_norm_layers]
    batch_norm_keys += [x + '.bias' for x in batch_norm_layers]
    target_keys = [x for x in reference_keys if x not in batch_norm_keys]


    result_dict = model.state_dict()
    result_keys = list(result_dict.keys())
    # Handle biases added in result to deal with BatchNorm
    batch_norm_idx = [next(i for i, x in enumerate(reference_keys) if bn in x) for bn in batch_norm_layers]
    batch_norm_idx = np.array([target_keys.index(reference_keys[i - 1]) for i in batch_norm_idx])
    batch_norm_idx += np.arange(len(batch_norm_idx))
    conv_keys = ['.'.join(result_keys[i].split('.')[:-1]) for i in batch_norm_idx]
    to_ignore = set([x + '.bias' for x in conv_keys])
    result_keys = [x for x in result_keys if x not in to_ignore]


    for i_key, j_key in zip(result_keys, target_keys):
        weights = extract(reference_dict[j_key]).clone()
        # Transpose weights if necessary
        if(len(weights.size()) == 2):
            weights = weights.permute(1, 0)
        if(len(weights.size()) == 4):
            weights = weights.permute(1, 2, 3, 0)
        if(len(weights.size()) == 5):
            weights = weights.permute(1, 2, 3, 4, 0)
        result_dict[i_key] = weights

    # Fold Batch Normalization
    batch_norm_idx = [next(i for i, x in enumerate(reference_keys) if bn in x) for bn in batch_norm_layers]
    batch_norm_idx = np.array([target_keys.index(reference_keys[i - 1]) for i in batch_norm_idx])
    batch_norm_idx += np.arange(len(batch_norm_idx))
    conv_keys = ['.'.join(list(result_dict.keys())[i].split('.')[:-1]) for i in batch_norm_idx]

    for x, y in zip(conv_keys, batch_norm_layers):
        eps = 1e-5
        # Extract scales, mean, variance
        beta = extract(reference_dict['{}.bias'.format(y)]).cuda()
        gamma = extract(reference_dict['{}.weight'.format(y)]).cuda()
        mean = extract(reference_dict['{}.running_mean'.format(y)]).cuda()
        var = extract(reference_dict['{}.running_var'.format(y)]).cuda()
        alpha = gamma / torch.sqrt(var + eps)
        # Adjust conv weights/bias
        conv_bias = result_dict['{}.bias'.format(x)].cuda()
        conv_weight = result_dict['{}.weight'.format(x)].cuda()
        conv_bias = conv_bias*alpha + (beta - mean*alpha)
        for i in range(len(alpha)):
            if(len(conv_weight.size())==4):
                conv_weight[:,:,:,i] *= alpha[i]
            if(len(conv_weight.size())==5):
                conv_weight[:,:,:,:,i] *= alpha[i]
        # Write back to dictionnary
        result_dict['{}.bias'.format(x)] = conv_bias
        result_dict['{}.weight'.format(x)] = conv_weight

    # Write back state dict
    model.load_state_dict(result_dict)

