import numpy as np
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import default_alloc_fn, is_hip_cdna5
from triton.tools.tensor_descriptor import TensorDescriptor


@pytest.mark.parametrize('descriptor', ['host', 'device'])
# 1D has no outer stride; 5D spans every packed group.
@pytest.mark.parametrize('rank', [1, 5])
def test_tensor_descriptor_reduce_two_tiles(descriptor, rank, device):
    if not is_hip_cdna5():
        pytest.skip('CDNA5 TDM reduction emulation')

    triton.set_allocator(default_alloc_fn)

    @triton.jit
    def kernel(src, out, out_desc, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
        desc = tl.make_tensor_descriptor(out, SHAPE, STRIDES, SHAPE) if out_desc is None else out_desc
        elements: tl.constexpr = 32 * 2**(len(SHAPE) - 1)
        tile = tl.load(src + tl.program_id(0) * elements + tl.arange(0, elements)).reshape(SHAPE)
        offsets = (0, ) * len(SHAPE)
        desc.atomic_add(offsets, tile)

    shape = (2, ) * (rank - 1) + (32, )
    elements = int(np.prod(shape))
    source = (torch.arange(2 * elements, dtype=torch.int32, device=device) % 97 - 48).reshape((2, ) + shape)
    output = torch.zeros(shape, dtype=torch.int32, device=device)
    desc = TensorDescriptor.from_tensor(output, shape) if descriptor == 'host' else None
    kernel[(2, )](source, output, desc, shape, output.stride())
    torch.testing.assert_close(output, source[0] + source[1])


@pytest.mark.parametrize('descriptor', ['host', 'device'])
def test_tensor_descriptor_reduce_tdm_neighbors(descriptor, device):
    if not is_hip_cdna5():
        pytest.skip('CDNA5 TDM reduction emulation')

    triton.set_allocator(default_alloc_fn)

    @triton.jit
    def kernel(src_desc, dst_desc, src, dst, result, COUNT):
        if src_desc is None:
            inp = tl.make_tensor_descriptor(src, [8, 32], [32, 1], [8, 32])
            out = tl.make_tensor_descriptor(dst, [8, 32], [32, 1], [8, 32])
        else:
            inp = src_desc
            out = dst_desc
        tile = inp.load([0, 0])
        out.store([0, 0], tile)
        for i in range(COUNT):
            out.atomic_add([0, 0], tile)
        actual = out.load([0, 0])
        offsets = tl.arange(0, 8)[:, None] * 32 + tl.arange(0, 32)[None, :]
        tl.store(result + offsets, actual)

    source = (torch.arange(256, dtype=torch.int32, device=device) % 7).reshape(8, 32).to(torch.float32)
    output = torch.empty_like(source)
    result = torch.empty_like(output)
    src_desc = TensorDescriptor.from_tensor(source, [8, 32]) if descriptor == 'host' else None
    dst_desc = TensorDescriptor.from_tensor(output, [8, 32]) if descriptor == 'host' else None
    compiled = kernel[(1, )](src_desc, dst_desc, source, output, result, 2, num_stages=1)
    torch.testing.assert_close(result, source * 3)
    assert 'tensor_load_to_lds' in compiled.asm['amdgcn']
    assert 'tensor_store_from_lds' in compiled.asm['amdgcn']
    assert 'atomic' in compiled.asm['amdgcn']


@pytest.mark.parametrize('descriptor', ['host', 'device'])
@pytest.mark.parametrize('offset', [-4, 16])
def test_tensor_descriptor_reduce_strided_bounds(descriptor, offset, device):
    if not is_hip_cdna5():
        pytest.skip('CDNA5 TDM reduction emulation')

    triton.set_allocator(default_alloc_fn)

    @triton.jit
    def kernel(desc, ptr, OFFSET: tl.constexpr):
        out = tl.make_tensor_descriptor(ptr, [19, 29], [48, 1], [8, 32]) if desc is None else desc
        tile = tl.full((8, 32), 1, tl.int32)
        out.atomic_add([OFFSET, 0], tile)

    storage = torch.zeros((19, 48), dtype=torch.int32, device=device)
    desc = TensorDescriptor(storage, [19, 29], [48, 1], [8, 32]) if descriptor == 'host' else None
    kernel[(1, )](desc, storage, offset)
    expected = torch.zeros_like(storage)
    if offset >= 0:
        expected[offset:min(offset + 8, 19), :29] += 1
    torch.testing.assert_close(storage, expected)
