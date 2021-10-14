import triton
from . import core as tl


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
def uint32_to_uniform_float(x):
    """
    Numerically stable function to convert a random integer into a random float uniformly sampled in [0, 1). 
    This is originally designed from uint32, but it works with int32 too as long as the int32 uniformly 
    covers all the possible values it can take.
    """
    max = 4.656613e-10 # = 1/MAX_INT = 1/2147483647.
    x = tl.where(x < 0, -x - 1, x)
    return x * max

@triton.jit
def pair_uniform_to_normal(u1, u2):
    """Box-Muller transform"""
    u1 = tl.maximum(1.0e-7, u1)
    th = 6.283185307179586 * u2
    r = tl.sqrt(-2.0 * tl.log(u1))
    return r * tl.cos(th), r * tl.sin(th)


@triton.jit
def randint4x(seed, offset):
    """
    Given a :code:`seed` scalar and an :code:`offset` block, returns four 
    blocks of random :code:`int32`. 
    
    This is the maximally efficient entry point 
    to Triton's Philox pseudo-random number generator.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    z = 0
    seed = hacky_to_uint64(seed) # uint will solve this
    seed_hi = ((seed >> 32) & 0xffffffff).to(tl.int32)
    seed_lo = (seed & 0xffffffff).to(tl.int32)
    return philox_f(offset, z, z, z, seed_lo, seed_hi)


@triton.jit
def randint(seed, offset):
    """
    Given a :code:`seed` scalar and an :code:`offset` block, returns a single 
    block of random :code:`int32`. 
    
    If you need multiple streams of random numbers,
    using `randint4x` is likely to be faster than calling `randint` 4 times.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    ret, _, _, _ = randint4x(seed, offset)
    return ret


@triton.jit
def rand(seed, offset):
    """
    Given a :code:`seed` scalar and an :code:`offset` block, 
    returns a block of random :code:`float32` in :math:`U(0, 1)`

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    source = randint(seed, offset)
    return uint32_to_uniform_float(source)


@triton.jit
def randn(seed, offset):
    """
    Given a :code:`seed` scalar and an :code:`offset` block, 
    returns a block of random :code:`float32` in :math:`\mathcal{N}(0, 1)`

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    i1, i2, _, _ = randint4x(seed, offset)
    u1 = uint32_to_uniform_float(i1)
    u2 = uint32_to_uniform_float(i2)
    n1, _ = pair_uniform_to_normal(u1, u2)
    return n1


@triton.jit
def rand4x(seed, offsets):
    """
    Given a :code:`seed` scalar and an :code:`offsets` block,
    returns a 4 blocks of random :code:`float32` in :math:`U(0, 1)`

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    i1, i2, i3, i4 = randint4x(seed, offsets)
    u1 = uint32_to_uniform_float(i1)
    u2 = uint32_to_uniform_float(i2)
    u3 = uint32_to_uniform_float(i3)
    u4 = uint32_to_uniform_float(i4)
    return u1, u2, u3, u4


@triton.jit
def randn4x(seed, offset):
    """
    Given a :code:`seed` scalar and an :code:`offset` block,
    returns a 4 blocks of random :code:`float32` in :math:`\mathcal{N}(0, 1)`

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    u1, u2, u3, u4 = rand4x(seed, offset)
    n1, n2 = pair_uniform_to_normal(u1, u2)
    n3, n4 = pair_uniform_to_normal(u3, u4)
    return n1, n2, n3, n4
