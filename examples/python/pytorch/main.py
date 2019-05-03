import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.cpp_extension import load
from torch.distributions import categorical
from itertools import product

conv_triton = load( 'conv_triton', ['conv.cpp', 'conv.cu'], extra_cflags=['-O3'])
