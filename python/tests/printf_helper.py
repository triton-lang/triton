import torch
from torch.testing import assert_close

import triton
import triton.language as tl

torch_type = {
    "bool": torch.bool,
    'int8': torch.int8,
    'uint8': torch.uint8,
    'int16': torch.int16,
    "int32": torch.int32,
    'int64': torch.long,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64
}


def get_tensor(shape, data_type, b_positive=False):
    x = None
    if data_type.startswith('int'):
        x = torch.arange(0, shape[0], dtype=torch_type[data_type], device='cuda')
    else:
        x = torch.arange(0, shape[0], dtype=torch_type[data_type], device='cuda')

    return x

# @pytest.mark.parametrize('data_type',
#                          [("int8"),
#                           ('int16'),
#                           ('int32'),
#                           ("int64"),
#                           ('float16'),
#                           ("float32"),
#                           ("float64")])


def printf(data_type):
    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        tl.printf("", x)
        tl.store(Y + tl.arange(0, BLOCK), x)

    shape = (128, )
    # limit the range of integers so that the sum does not overflow
    x = get_tensor(shape, data_type)
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    kernel[(1,)](x, y, BLOCK=shape[0])
    assert_close(y, x)


printf("float16")
printf("int8")
