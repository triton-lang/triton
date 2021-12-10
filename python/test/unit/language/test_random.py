import torch
import triton
import triton.language as tl
import pytest
import scipy.stats
import numpy as np

from numpy.random import Philox

#####################################
## Reference Philox Implementation
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
        while len(res) < pad:
            res.append(np.array(n, dtype=self._dtype))
            n >>= (np.dtype(self._dtype).itemsize * 8)
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
## Unit Tests
#####################################

BLOCK = 1024

# test generation of random uint32
@pytest.mark.parametrize('size, seed',
    [(size, seed) for size in ['10', '4,53', '10000']\
                  for seed in [0, 42, 124, 54, 0xffffffff, 0xdeadbeefcafeb0ba]]
)
def test_randint(size, seed, device='cuda'):
    size = list(map(int, size.split(',')))
    @triton.jit
    def kernel(X, N, seed):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        rand = tl.randint(seed, offset)
        tl.store(X + offset, rand, mask=offset < N)
    # triton result
    x = torch.empty(size, dtype=torch.int32, device=device)
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK),)
    kernel[grid](x, N, seed)
    out_tri = x.cpu().numpy().astype(np.uint32).flatten().tolist()
    # reference result
    gen = CustomPhilox4x(seed, config=PHILOX_32)
    out_ref = [gen.random_raw()[0] for _ in out_tri]
    assert out_tri == out_ref

# test uniform PRNG
@pytest.mark.parametrize('size, seed',
    [(size, seed) for size in [1000000]\
                  for seed in [0, 42, 124, 54]]
)
def test_rand(size, seed, device='cuda'):
    @triton.jit
    def kernel(X, N, seed):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        rand = tl.rand(seed, offset)
        tl.store(X + offset, rand, mask=offset < N)
    # triton result
    x = torch.empty(size, dtype=torch.float32, device=device)
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK),)
    kernel[grid](x, N, seed)
    assert scipy.stats.kstest(x.tolist(), 'uniform', args=(0, 1)).statistic < 0.01

# test normal PRNG
@pytest.mark.parametrize('size, seed',
    [(size, seed) for size in [1000000]\
                  for seed in [0, 42, 124, 54]]
)
def test_randn(size, seed, device='cuda'):
    @triton.jit
    def kernel(X, N, seed):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        rand = tl.randn(seed, offset)
        tl.store(X + offset, rand, mask=offset < N)
    # triton result
    x = torch.empty(size, dtype=torch.float32, device=device)
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK),)
    kernel[grid](x, N, seed)
    assert abs(x.mean()) < 1e-2
    assert abs(x.std() - 1) < 1e-2
