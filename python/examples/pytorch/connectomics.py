from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from em.model.deploy import unet3D_m1, unet3D_m2
from em.model.io import convert_state_dict

import numpy as np
import copy
import isaac.pytorch.models
from time import time
import timeit
import h5py
from builtins import object
import argparse
import re
import collections

def convert(legacy):
    result = isaac.pytorch.models.UNet().cuda()

    # Copy in proper order
    legacy_keys = list(legacy.state_dict().keys())
    result_keys = list(result.state_dict().keys())
    legacy_dict = legacy.state_dict()
    result_dict = result.state_dict()

    print(legacy_keys)
    print(result_keys)

    # Don't copy up-sampling weight
    pattern = re.compile("upS\.[0-9]\.0.weight")
    legacy_keys = [x for x in legacy_keys if not pattern.match(x)]

    for i, j in zip(result_keys, legacy_keys):
        weights = legacy_dict[j].clone()
        # Transpose weights if necessary
        if(len(weights.size()) > 1):
            weights = weights.permute(1, 2, 3, 4, 0)
        # Copy weights
        result_dict[i] = weights
    result.load_state_dict(result_dict)

    return result


class DataIterator(object):

    def __init__(self, batch_size, tile, data):
        self.current = [0, 0, 0]
        self.tile = tile
        self.sizes = data.shape
        self.batch_size = batch_size
        self.data = torch.Tensor(data.reshape(1, 1, *self.sizes)).cuda()

    def __iter__(self):
        return self

    def __next__(self):
        results = []
        for batch in range(args.batch_size):
            i = np.random.randint(0, self.sizes[0] - self.tile[0])
            j = np.random.randint(0, self.sizes[1] - self.tile[1])
            k = np.random.randint(0, self.sizes[2] - self.tile[2])
            results += [self.data[:, :, i:i+self.tile[0], j:j+self.tile[1], k:k+self.tile[2]].clone()]
        results = torch.cat(results, dim=0)
        return results,


if __name__ == '__main__':
    # Program options
    parser = argparse.ArgumentParser(description='ISAAC Electron Microscopy Inference')
    parser.add_argument('data', help='path to dataset')
    parser.add_argument('weights', help='path to model weights')
    parser.add_argument('--arch', '-a', default='unet', choices=['unet'])
    parser.add_argument('--batch-size', '-b', default=16, type=int, metavar='N', help='mini-batch size [default: 16]')
    parser.add_argument('--calibration-batches', '-c', default=4, type=int, metavar='N', help='number of batches for calibration [default: 16]')
    args = parser.parse_args()

    # Fix random seeds (for reproducibility)
    np.random.seed(0)

    # Load data
    T = np.array(h5py.File(args.data, 'r')['main']).astype(np.float32)/255
    dataset = DataIterator(args.batch_size, (18, 160, 160), T)
    iterator = iter(dataset)

    # Build models
    unet_ref = unet3D_m2().cuda()
    state_dict = torch.load(args.weights)['state_dict']
    state_dict = collections.OrderedDict([(x.replace('module.', ''), y) for x, y in state_dict.items()])
    unet_ref.load_state_dict(state_dict)
    unet_ref.eval()


    # Quantize
    print('Quantizing... ', end='', flush=True)
    pattern = re.compile("upS\.[0-9]\.0.weight")
    filter = lambda x: not pattern.match(x)
    unet_sc_int8 = isaac.pytorch.models.UNet(relu_type='relu', relu_slope=0.).cuda()
    unet_sc_fp32 = isaac.pytorch.models.UNet(relu_type='relu', relu_slope=0.).cuda()
    isaac.pytorch.convert(unet_sc_fp32, state_dict, filter)
    isaac.pytorch.convert(unet_sc_int8, state_dict, filter)
    isaac.pytorch.quantize(unet_sc_int8, iterator, args.calibration_batches)
    print('')

    # Benchmark
    print('Performance: ', end='', flush=True)
    X = Variable(next(iterator)[0], volatile=True).cuda()
    y_sc = unet_sc_int8(X)
    Nvoxels = np.prod(y_sc.size()[2:])
    t_sc = [x for x in timeit.repeat(lambda: (unet_sc_int8(X), torch.cuda.synchronize()), repeat=10, number=1)]
    t_ref = [x for x in timeit.repeat(lambda: (unet_sc_fp32(X), torch.cuda.synchronize()), repeat=10, number=1)]
    print('{:.2f} Mvox/s (Isaac) ; {:.2f} Mvox/s (PyTorch)'.format(Nvoxels/min(t_sc)*args.batch_size*1e-6, Nvoxels/min(t_ref)*args.batch_size*1e-6))

    # Evaluate
    print('Error: ', end='', flush=True)
    errors = np.zeros(10)
    for n in range(errors.size):
        X = Variable(next(iterator)[0], volatile=True).cuda()
        y_ref = unet_ref(X)
        y_sc = unet_sc_int8(X)
        errors[n] = torch.norm(y_ref - y_sc).data[0]/torch.norm(y_ref).data[0]
    print('{:.4f} [+- {:.4f}]'.format(np.mean(errors), np.std(errors)))


