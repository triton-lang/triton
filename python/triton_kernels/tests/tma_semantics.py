import triton
import torch
from triton.tools.tensor_descriptor import TensorDescriptor
import ctypes
from ctypes import byref, c_size_t, c_uint64

cuda = ctypes.CDLL('libcuda.so')


def is_valid_cuda_pointer(ptr):
    base = c_uint64(0)
    size = c_size_t(0)
    res = cuda.cuMemGetAddressRange_v2(byref(base), byref(size), c_uint64(ptr))
    return res == 0


@triton.jit
def _init(X, Y, off):
    Y.store([0], X.load([off]))


off = 768
# IMA disappears when commenting this out
# tmp = torch.empty(1024, device="cuda")
x = torch.full((1024, ), fill_value=1, dtype=torch.float32, device="cuda")
y = torch.full((1024, ), fill_value=42., dtype=torch.float32, device="cuda")
dx = TensorDescriptor(x.data_ptr() - off * x.dtype.itemsize, x.dtype, [x.shape[0]], x.stride(), [1024])
dy = TensorDescriptor(y.data_ptr(), y.dtype, y.shape, y.stride(), [1024])
_init[(1, )](dx, dy, off)
print("valid pointer:", is_valid_cuda_pointer(dx.base_ptr))
assert (y[:x.shape[0] - off] == 1).all()
assert (y[x.shape[0] - off:] == 0).all()
