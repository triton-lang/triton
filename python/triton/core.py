import triton


@triton.jit
def minimum(x, y):
    return triton.where(x < y, x, y)


@triton.jit
def maximum(x, y):
    return triton.where(x > y, x, y)


@triton.jit
def softmax(x):
    z = x - triton.max(x, 0)
    num = triton.exp(z)
    den = triton.sum(num, 0)
    return num / den


def cdiv(x, y):
    return (x + y - 1) // y
