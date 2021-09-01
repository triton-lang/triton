import torch
import triton
import triton.language as tl
import scipy.stats
import numpy as np

from numpy.random import Philox
from triton.language.random.philox_reference import CustomPhilox, CustomPhilox4x, PHILOX_32, PHILOX_64
# TODO: change the import * below to this once importing is fixed.
# from triton.language.random import random_4x, uint32_to_uniform_float, pair_uniform_to_normal
from triton.language.random.philox import *
from triton.code_gen import elementwise_heuristics


@elementwise_heuristics(arg_idx=1)
@triton.jit
def _randints(T, N, seed, **meta):
    pid = tl.program_id(0)
    offset = pid * meta["BLOCK"] + tl.arange(0, meta["BLOCK"])
    mask = offset < N
    r, _, _, _ = random_4x(seed, offset)
    tl.store(T + offset, r, mask=mask)


def randints(size, seed):
    t = torch.empty(size, dtype=torch.int32).cuda()
    N = t.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)

    _randints[grid](t, N, seed)

    return t.cpu().numpy().astype(np.uint32)


@elementwise_heuristics(arg_idx=1)
@triton.jit
def _randnormals(T, N, seed, **meta):
    pid = tl.program_id(0)
    offset = pid * meta["BLOCK"] + tl.arange(0, meta["BLOCK"])
    mask = offset < N
    i1, i2, _, _ = random_4x(seed, offset)

    u1 = uint32_to_uniform_float(i1)
    u2 = uint32_to_uniform_float(i2)
    n1, n2 = pair_uniform_to_normal(u1, u2)
    r = tl.where(offset % 2 == 0, n1, n2)
    tl.store(T + offset, r, mask=mask)


def randnormals(size, seed):
    t = torch.empty(size, dtype=torch.float32).cuda()
    N = t.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _randnormals[grid](t, N, seed)
    return t


@elementwise_heuristics(arg_idx=2)
@triton.jit
def _uint32_to_uniform_tester(source, target, N, **meta):
    pid = tl.program_id(0)
    offset = pid * meta["BLOCK"] + tl.arange(0, meta["BLOCK"])
    mask = offset < N
    s = tl.load(source + offset)
    t = uint32_to_uniform_float(s)
    tl.store(target + offset, t, mask=mask)


def uint32_to_uniform_tester(source):
    target = -torch.ones(source.shape, dtype=torch.float32, device=source.device)
    N = source.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _uint32_to_uniform_tester[grid](source, target, N)
    return target


def test_philox_refrence():
    """Test reference Philox matches numpy implementation."""
    x = Philox(key=[0, 0], counter=[2**(np.dtype(np.uint64).itemsize * 8) - 1] * 4)
    y = CustomPhilox(seed=0, config=PHILOX_64)

    assert y.random_raw() == x.random_raw()
    assert y.random_raw() == x.random_raw()
    assert y.random_raw() == x.random_raw()
    assert y.random_raw() == x.random_raw()

    y = CustomPhilox(seed=0, config=PHILOX_64)
    y.advance(1)

    for _ in range(10000):
        assert y.random_raw() == x.random_raw()


def test_philox():
    """Testing if we implemented philox spec correctly."""
    for size in [(10,), (4, 53), (20000,)]:
        for seed in [0, 42, 124, 54]:
            triton = randints(size, seed).flatten().tolist()
            gen = CustomPhilox4x(seed, config=PHILOX_32)
            expected = [gen.random_raw()[0] for _ in triton]
            assert triton == expected


def test_randnormals():
    """Testing if we implemented philox spec correctly."""

    for seed in [0, 42, 124, 54]:
        t = randnormals(size=(1000000,), seed=seed)
        print(t.mean(), t.std())
        assert abs(t.mean()) < 1e-2
        assert abs(t.std() - 1) < 1e-2


def test_uint32_to_uniform_float():
    n = 100

    # check range of edge values
    source = torch.tensor(list(range(n)) + list(range(-n, 0)), dtype=torch.int32).cuda()
    target = uint32_to_uniform_tester(source).tolist()
    assert target == sorted(target)
    assert all(0.0 <= num < 1.0 for num in target)

    # check distribution is uniform
    source = torch.randint(-2**31, 2**31 - 1, dtype=torch.int32, size=(100000,)).cuda()
    target = uint32_to_uniform_tester(source).tolist()
    assert scipy.stats.kstest(target, 'uniform', args=(0, 1)).statistic < 0.01
