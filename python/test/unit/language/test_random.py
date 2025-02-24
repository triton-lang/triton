import numpy as np
import pytest
import scipy.stats
import torch

import triton
import triton.language as tl

#####################################
# Reference Philox Implementation
#####################################


class PhiloxConfig:

    def __init__(self, PHILOX_ROUND_A, PHILOX_ROUND_B, PHILOX_KEY_A, PHILOX_KEY_B, DTYPE):
        self.PHILOX_ROUND_A = np.array(PHILOX_ROUND_A, dtype=DTYPE)
        self.PHILOX_ROUND_B = np.array(PHILOX_ROUND_B, dtype=DTYPE)
        self.PHILOX_KEY_A = np.array(PHILOX_KEY_A, dtype=DTYPE)
        self.PHILOX_KEY_B = np.array(PHILOX_KEY_B, dtype=DTYPE)
        self.DTYPE = DTYPE


# This is better for GPU
PHILOX_32 = PhiloxConfig(
    PHILOX_KEY_A=0x9E3779B9,
    PHILOX_KEY_B=0xBB67AE85,
    PHILOX_ROUND_A=0xD2511F53,
    PHILOX_ROUND_B=0xCD9E8D57,
    DTYPE=np.uint32,
)

# This is what numpy implements
PHILOX_64 = PhiloxConfig(
    PHILOX_KEY_A=0x9E3779B97F4A7C15,
    PHILOX_KEY_B=0xBB67AE8584CAA73B,
    PHILOX_ROUND_A=0xD2E7470EE14C6C93,
    PHILOX_ROUND_B=0xCA5A826395121157,
    DTYPE=np.uint64,
)


class CustomPhilox4x:

    def __init__(self, seed, config):
        self._config = config
        seed = self._into_pieces(seed)
        self._key = np.array(seed[:2], dtype=self._dtype)
        self._counter = np.array((0, 0) + seed[2:], dtype=self._dtype)

    @property
    def _dtype(self):
        return self._config.DTYPE

    def _into_pieces(self, n, pad=4):
        res = []
        bits = np.dtype(self._dtype).itemsize * 8
        while len(res) < pad:
            res.append(np.array((n & ((1 << bits) - 1)), dtype=self._dtype))
            n >>= bits
        assert n == 0
        return tuple(res)

    def _multiply_low_high(self, a, b):
        low = a * b
        high = int(a) * int(b)
        high = np.array(high >> (np.dtype(self._dtype).itemsize * 8), dtype=self._dtype)
        return low, high

    def _single_round(self, counter, key):
        lo0, hi0 = self._multiply_low_high(self._config.PHILOX_ROUND_A, counter[0])
        lo1, hi1 = self._multiply_low_high(self._config.PHILOX_ROUND_B, counter[2])
        ret0 = hi1 ^ counter[1] ^ key[0]
        ret1 = lo1
        ret2 = hi0 ^ counter[3] ^ key[1]
        ret3 = lo0
        return np.array([ret0, ret1, ret2, ret3], dtype=self._dtype)

    def _raise_key(self, key):
        pk = [self._config.PHILOX_KEY_A, self._config.PHILOX_KEY_B]
        return key + np.array(pk, dtype=self._dtype)

    def random_raw(self):
        counter = self._counter
        key = self._key
        for _ in range(10):
            counter = self._single_round(counter, key)
            key = self._raise_key(key)
        self.advance(1)
        return counter

    def advance(self, n_steps):
        self._counter[0] += n_steps
        assert self._counter[0] < 2**32, "FIXME: doesn't work for large offsets"


class CustomPhilox(CustomPhilox4x):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = []

    def random_raw(self):
        if len(self.buffer) == 0:
            self.buffer = list(super().random_raw())[::-1]
        return int(self.buffer.pop())


#####################################
# Unit Tests
#####################################

BLOCK = tl.constexpr(1024)

# test generation of random uint32


@pytest.mark.interpreter
@pytest.mark.parametrize('size, seed, dtype, const_seed', [(size, seed, dtype, const_seed)
                                                           for size in ['10', '4,53', '400']
                                                           for seed in [0, 42, 124, 54, 0xffffffff, 0x0000000fcafeb0ba]
                                                           for dtype in ['int32', 'int64']
                                                           for const_seed in [True, False]])
def test_randint(size, seed, device, dtype, const_seed):
    size = list(map(int, size.split(',')))
    torch_dtype = getattr(torch, dtype)
    numpy_dtype = getattr(np, f"u{dtype}")
    config = {'int32': PHILOX_32, 'int64': PHILOX_64}[dtype]

    @triton.jit
    def kernel(X, N, seed):
        pid = tl.program_id(0).to(X.dtype.element_ty)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        rand = tl.randint(seed, offset)
        tl.store(X + offset, rand, mask=offset < N)

    @triton.jit
    def const_kernel(X, N, seed: tl.constexpr):
        pid = tl.program_id(0).to(X.dtype.element_ty)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        rand = tl.randint(seed, offset)
        tl.store(X + offset, rand, mask=offset < N)

    # triton result
    x = torch.empty(size, dtype=torch_dtype, device=device)
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK.value), )
    if const_seed:
        const_kernel[grid](x, N, seed=seed)
    else:
        kernel[grid](x, N, seed)
    out_tri = x.cpu().numpy().astype(numpy_dtype).flatten().tolist()
    # reference result
    gen = CustomPhilox4x(seed, config=config)
    out_ref = [gen.random_raw()[0] for _ in out_tri]
    assert out_tri == out_ref


# test uniform PRNG


@pytest.mark.interpreter
@pytest.mark.parametrize('size, seed, dtype, const_seed', [(size, seed, dtype, const_seed)
                                                           for size in [100000]
                                                           for seed in [0, 42, 124, 54]
                                                           for dtype in ['int32', 'int64']
                                                           for const_seed in [True, False]])
def test_rand(size, seed, dtype, device, const_seed):

    @triton.jit
    def kernel(X, N, seed, dtype: tl.constexpr):
        pid = tl.program_id(0).to(dtype)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        rand = tl.rand(seed, offset)
        tl.store(X + offset, rand, mask=offset < N)

    @triton.jit
    def const_kernel(X, N, seed: tl.constexpr, dtype: tl.constexpr):
        pid = tl.program_id(0).to(dtype)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        rand = tl.rand(seed, offset)
        tl.store(X + offset, rand, mask=offset < N)

    # triton result
    x = torch.empty(size, dtype=torch.float32, device=device)
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK.value), )
    if const_seed:
        const_kernel[grid](x, N, seed=seed, dtype=getattr(tl, dtype))
    else:
        kernel[grid](x, N, seed, dtype=getattr(tl, dtype))
    assert all((x >= 0) & (x <= 1))
    assert scipy.stats.kstest(x.tolist(), 'uniform', args=(0, 1)).statistic < 0.01


def test_seed_is_int(device):

    @triton.jit
    def kernel(X, seed):
        offset = tl.arange(0, 1)
        rand = tl.rand(seed, offset)
        tl.store(X + offset, rand)

    x = torch.empty(1, dtype=torch.float32, device=device)
    with pytest.raises(triton.compiler.errors.CompilationError):
        seed0 = torch.zeros(1, dtype=torch.int32, device=device)
        kernel[(1, )](x, seed0)
    with pytest.raises(triton.compiler.errors.CompilationError):
        seed1 = 2.3
        kernel[(1, )](x, seed1)


# test normal PRNG


@pytest.mark.interpreter
@pytest.mark.parametrize('size, seed, dtype, const_seed', [(size, seed, dtype, const_seed)
                                                           for size in [100000]
                                                           for seed in [0, 42, 124, 54]
                                                           for dtype in ['int32', 'int64']
                                                           for const_seed in [True, False]])
def test_randn(size, seed, dtype, device, const_seed):

    @triton.jit
    def kernel(X, N, seed, dtype: tl.constexpr):
        pid = tl.program_id(0).to(dtype)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        rand = tl.randn(seed, offset)
        tl.store(X + offset, rand, mask=offset < N)

    @triton.jit
    def const_kernel(X, N, seed: tl.constexpr, dtype: tl.constexpr):
        pid = tl.program_id(0).to(dtype)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        rand = tl.randn(seed, offset)
        tl.store(X + offset, rand, mask=offset < N)

    # triton result
    x = torch.empty(size, dtype=torch.float32, device=device)
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK.value), )
    if const_seed:
        const_kernel[grid](x, N, seed=seed, dtype=getattr(tl, dtype))
    else:
        kernel[grid](x, N, seed, dtype=getattr(tl, dtype))
    assert abs(x.mean()) < 1e-2
    assert abs(x.std() - 1) < 1e-2


# tl.rand() should never produce >=1.0


@pytest.mark.interpreter
@pytest.mark.parametrize('dtype', ['int32', 'int64'])
def test_rand_limits(dtype, device):

    @triton.jit
    def kernel(input, output, n: tl.constexpr):
        idx = tl.arange(0, n)
        x = tl.load(input + idx)
        y = tl.random.uint_to_uniform_float(x)
        tl.store(output + idx, y)

    torch_dtype = getattr(torch, dtype)
    min_max_int = torch.tensor([
        torch.iinfo(torch_dtype).min,
        torch.iinfo(torch_dtype).max,
    ], dtype=torch_dtype, device=device)
    output = torch.empty(2, dtype=torch.float32, device=device)
    kernel[(1, )](min_max_int, output, 2)

    assert output[0] == output[1]
    assert 1.0 - torch.finfo(torch.float32).eps <= output[0].item() < 1.0
