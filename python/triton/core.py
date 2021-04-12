import triton._C.libtriton.triton as _triton
import triton

########################
# Built-in Functions   #
########################


class float32:
    @staticmethod
    def make_ir(context):
        return _triton.ir.type.get_fp32(context)


class float16:
    @staticmethod
    def make_ir(context):
        return _triton.ir.type.get_fp16(context)


@triton.jit()
def minimum(x, y):
    return triton.where(x < y, x, y)


def cdiv(x, y):
    return (x + y - 1) // y
