import struct
import os
import keras as kr
from tools import mkdir

template = \
'''#include <cstddef>
#include <cstdint>

namespace isaac{{
namespace runtime{{
namespace {arch}{{

static const uint8_t {op}[] = {{ 
{data} }};

}}
}}
}}
'''


Activations = {'relu': 0, 'linear': 1}
def encode(L, data):
    if isinstance(L, kr.layers.Activation):
        data.extend(struct.pack('<2I', 0, Activations[L.activation.__name__]))
    if isinstance(L, kr.layers.Dense):
        W, bias = L.get_weights()
        data.extend(struct.pack('<{}I'.format(1 + W.ndim), 1, *W.shape))
        data.extend(W.tobytes(order='C'))
        data.extend(bias.tobytes(order='C'))

def cpp_file(arch, op, data):
    split = lambda x, n: [x[i:i+n] for i in range(0, len(x), n)]
    data_len = len(data)
    data = ',\n'.join([', '.join(map(hex, x)) for x in split(data, 10)])
    src = template.format(arch=arch, op=op, data=data)
    return src


def export(database_path, kernels, model, op, init_cuda):
    device, ctx, stream = init_cuda()
    data = bytearray()
    #Kernels
    data.extend(struct.pack('<{}I'.format(kernels.ndim), *kernels.shape))
    data.extend(kernels.tobytes(order='C'))
    #Models
    data.extend(struct.pack('I', len(model.layers)))
    for L in model.layers: 
        encode(L, data)
    #Write file
    arch = 'sm_' + '_'.join(map(str, device.compute_capability))
    prefix = os.path.join(database_path, arch)
    mkdir(prefix)
    with open(os.path.join(prefix, op + '.hpp'), 'w') as fp:
        fp.write(cpp_file(arch, op, data))


