import triton._C.libtriton.triton as _triton
import triton


class float32:
    @staticmethod
    def make_ir(context):
        return _triton.ir.type.get_fp32(context)


class float16:
    @staticmethod
    def make_ir(context):
        return _triton.ir.type.get_fp16(context)


@triton.jit
def minimum(x, y):
    return triton.where(x < y, x, y)


@triton.jit
def maximum(x, y):
    return triton.where(x > y, x, y)


@triton.jit
def softmax(x, axis=0):
    assert axis == 0
    z = x - triton.max(x)
    num = triton.exp(z)
    den = triton.sum(num)
    return num / den


def cdiv(x, y):
    return (x + y - 1) // y
