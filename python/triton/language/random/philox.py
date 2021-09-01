import triton
import triton.language as tl


# Notes
# 1. triton doesn't support uint32, so we use int32 instead and benefit from the fact that two's complement operations are equivalent to uint operations.
# 2. multiply_low_high is currently inefficient.
# 3. Even though technically philox sampling outputs int, in many places we pretends they were actualy uints e.g. uint_to_uniform_float


@triton.jit
def PHILOX_KEY_A():
    # 0x9E3779B9
    return -1640531527


@triton.jit
def PHILOX_KEY_B():
    # 0xBB67AE85
    return -1150833019


@triton.jit
def PHILOX_ROUND_A():
    # 0xD2511F53
    return -766435501


@triton.jit
def PHILOX_ROUND_B():
    # 0xCD9E8D57
    return -845247145


@triton.jit
def hacky_to_uint64(x):
    return ((x >> 1).to(tl.int64) << 1) + (x & 1).to(tl.int64)


@triton.jit
def multiply_low_high(a, b):
    return (
        a * b,
        ((hacky_to_uint64(a) * hacky_to_uint64(b)) >> 32).to(tl.int32)
    )


@triton.jit
def single_round(c0, c1, c2, c3, k0, k1):
    A = PHILOX_ROUND_A()
    B = PHILOX_ROUND_B()
    lo0, hi0 = multiply_low_high(A, c0)
    lo1, hi1 = multiply_low_high(B, c2)

    return (
        hi1 ^ c1 ^ k0,
        lo1,
        hi0 ^ c3 ^ k1,
        lo0,
    )


@triton.jit
def raise_key(k0, k1):
    return (
        k0 + PHILOX_KEY_A(),
        k1 + PHILOX_KEY_B(),
    )


@triton.jit
def philox_f(c0, c1, c2, c3, k0, k1):
    c0, c1, c2, c3 = single_round(c0, c1, c2, c3, k0, k1)
    k0, k1 = raise_key(k0, k1)
    c0, c1, c2, c3 = single_round(c0, c1, c2, c3, k0, k1)
    k0, k1 = raise_key(k0, k1)
    c0, c1, c2, c3 = single_round(c0, c1, c2, c3, k0, k1)
    k0, k1 = raise_key(k0, k1)
    c0, c1, c2, c3 = single_round(c0, c1, c2, c3, k0, k1)
    k0, k1 = raise_key(k0, k1)
    c0, c1, c2, c3 = single_round(c0, c1, c2, c3, k0, k1)
    k0, k1 = raise_key(k0, k1)
    c0, c1, c2, c3 = single_round(c0, c1, c2, c3, k0, k1)
    k0, k1 = raise_key(k0, k1)
    c0, c1, c2, c3 = single_round(c0, c1, c2, c3, k0, k1)
    k0, k1 = raise_key(k0, k1)
    c0, c1, c2, c3 = single_round(c0, c1, c2, c3, k0, k1)
    k0, k1 = raise_key(k0, k1)
    c0, c1, c2, c3 = single_round(c0, c1, c2, c3, k0, k1)
    k0, k1 = raise_key(k0, k1)
    c0, c1, c2, c3 = single_round(c0, c1, c2, c3, k0, k1)
    return c0, c1, c2, c3


@triton.jit
def random_4x(seed, offset):
    """Given seed and offset returns four random uint32"""
    z = 0
    return philox_f(offset, z, z, z, seed, z)


@triton.jit
def uint32_to_uniform_float(x):
    """Converts random uint32 to a float in range [0, 1)"""
    mantissa = x & 0x7fffff
    exp = 127
    res = mantissa | (exp << 23)
    return res.to(tl.float32, bitcast=True) - 1.0


@triton.jit
def pair_uniform_to_normal(u1, u2):
    """Box-Muller transform"""
    u1 = tl.maximum(1.0e-7, u1)
    th = 6.283185307179586 * u2
    r = tl.sqrt(-2.0 * tl.log(u1))
    return r * tl.cos(th), r * tl.sin(th)
