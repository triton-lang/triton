import pytest
import torch
from triton_kernels.tensor_details.layout import (
    BlackwellActMXScaleLayout,
    BlackwellMX4ValueShuffledLayout,
    BlackwellMXScaleLayout,
    BlackwellMXValueLayout,
    StridedLayout,
)
from triton_kernels.tensor_details.dtype import FP4
from triton_kernels.tensor import make_ragged_tensor_metadata, make_ragged_tensor_metadata_torch, wrap_torch_tensor, convert_layout, empty

# ------------------------------------------------------------
# Torch tests
# ------------------------------------------------------------

ZERO_SIZED_SHAPES = [(0, 64), (64, 0), (2, 0), (0, 2), (0, 64, 64)]


def test_act_scale_storage_preservation():
    slice_sizes = torch.tensor([2, 3], dtype=torch.int32)
    metadata = make_ragged_tensor_metadata_torch(slice_sizes, 5)
    equivalent = BlackwellActMXScaleLayout(metadata)
    reconstructed = BlackwellActMXScaleLayout(make_ragged_tensor_metadata_torch(slice_sizes, 5))

    assert equivalent.can_preserve_storage_as(BlackwellActMXScaleLayout(metadata), 2)
    assert not equivalent.can_preserve_storage_as(reconstructed, 2)


@pytest.mark.parametrize("shape", ZERO_SIZED_SHAPES)
@pytest.mark.parametrize("layout", [BlackwellMXScaleLayout(), BlackwellActMXScaleLayout(None)])
@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_scale_zero_sized_roundtrip(shape, layout, device):
    x = torch.empty(shape, dtype=torch.uint8, device=device)
    src = wrap_torch_tensor(x)

    swizzled = convert_layout(src, layout)
    roundtrip = convert_layout(swizzled, src.storage.layout)

    assert roundtrip.storage.data.shape == x.shape


@pytest.mark.parametrize("shape", ZERO_SIZED_SHAPES + [(0, 0), (2, 0, 6, 8)])
@pytest.mark.parametrize("major_dim", [-2, -1])
@pytest.mark.parametrize("layouts", [
    (BlackwellMXValueLayout(), ),
    (BlackwellMX4ValueShuffledLayout(), BlackwellMX4ValueShuffledLayout(block_k=256, block_n=128)),
])
@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_value_zero_sized_roundtrip(shape, major_dim, layouts, device):
    src_layout = StridedLayout(major_dim)
    src = empty(shape, dtype=FP4, device=device, layout=src_layout)

    converted = src
    for layout in layouts:
        converted = convert_layout(converted, layout)
        transformation = layout.make_transformation(list(shape), True)
        canonical = transformation.unswizzle_data(converted.data)
        assert converted.shape == list(shape)
        assert list(converted.data.shape) == transformation.storage_shape
        assert canonical.shape == (*shape[:-1], shape[-1] // 2)
    roundtrip = convert_layout(converted, src_layout)

    assert src.shape == roundtrip.shape == list(shape)
    assert roundtrip.data.shape == src.data.shape
    assert roundtrip.data.stride() == src.data.stride()


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("n", [0, 64])
def test_mxfp4_value_shuffled_rejects_odd_k(k, n):
    src = empty((k, n), dtype=FP4, device="cuda")
    with pytest.raises(ValueError, match="packing dimension -2 must have an even size"):
        convert_layout(src, BlackwellMX4ValueShuffledLayout())


@pytest.mark.parametrize("k", [0, 2])
@pytest.mark.parametrize("destination", [BlackwellMX4ValueShuffledLayout(), StridedLayout(-2), StridedLayout(-1)])
def test_mxfp4_value_shuffled_rejects_odd_n(k, destination):
    shape = [k, 3]
    layout = BlackwellMX4ValueShuffledLayout(block_n=128)
    data = torch.full(layout.storage_shape(shape, True), 0x11, dtype=torch.uint8, device="cuda")
    src = wrap_torch_tensor(data, dtype=FP4, shape=shape, layout=layout)
    with pytest.raises(ValueError, match="packing dimension -1 must have an even size"):
        convert_layout(src, destination)


@pytest.mark.parametrize("shape", [(256, 256), (258, 129), (2, 66, 33), (2, 3, 130, 65)])
@pytest.mark.parametrize("block_k", [128, 256])
@pytest.mark.parametrize("block_n", [128, 256])
@pytest.mark.parametrize("step", [1, 2])
def test_mxfp4_value_shuffled_matches_torch(shape, block_k, block_n, step):
    input_shape = tuple(step * size for size in shape[:-1]) + shape[-1:]
    data_cpu = torch.randint(0, 256, input_shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    data_cuda = data_cpu.cuda()
    index = (slice(step - 1, None, step), ) * (len(shape) - 1) + (slice(None), )
    data_cpu, data_cuda = data_cpu[index], data_cuda[index]
    logical_shape = list(data_cpu.shape)
    logical_shape[-1] *= 2
    layout = BlackwellMX4ValueShuffledLayout(block_k, block_n)
    transformation = layout.make_transformation(logical_shape, is_fp4=True)

    swizzled_cpu = transformation.swizzle_data(data_cpu)
    swizzled_cuda = transformation.swizzle_data(data_cuda)

    assert swizzled_cuda.is_contiguous()
    assert torch.equal(swizzled_cuda.cpu(), swizzled_cpu)
    if step == 2:
        swizzled_cpu = torch.stack((swizzled_cpu, swizzled_cpu), dim=-1)[..., 1]
        swizzled_cuda = torch.stack((swizzled_cuda, swizzled_cuda), dim=-1)[..., 1]
    restored_cpu = transformation.unswizzle_data(swizzled_cpu)
    restored_cuda = transformation.unswizzle_data(swizzled_cuda)
    assert restored_cuda.is_contiguous()
    assert torch.equal(restored_cuda.cpu(), restored_cpu)
    assert torch.equal(restored_cpu, data_cpu)


@pytest.mark.parametrize("shape", [(256, 128), (2, 258, 65)])
def test_mxfp4_value_shuffled_convert_layout_matches_torch(shape):
    data_cpu = torch.randint(0, 256, shape, dtype=torch.uint8)
    data_cuda = data_cpu.cuda()
    data_cpu, data_cuda = data_cpu.mT, data_cuda.mT
    src_cpu = wrap_torch_tensor(data_cpu, dtype=FP4)
    src_cuda = wrap_torch_tensor(data_cuda, dtype=FP4)
    layout = BlackwellMX4ValueShuffledLayout()

    expected = convert_layout(src_cpu, layout)
    actual = convert_layout(src_cuda, layout)
    roundtrip = convert_layout(actual, src_cuda.storage.layout)

    assert actual.storage.data.is_contiguous()
    assert torch.equal(actual.storage.data.cpu(), expected.storage.data)
    assert torch.equal(roundtrip.storage.data, data_cuda)


@pytest.mark.parametrize("shape", [(256, 128), (2, 258, 257), (2, 3, 130, 65)])
@pytest.mark.parametrize("layout", [
    BlackwellMXValueLayout(),
    BlackwellMX4ValueShuffledLayout(),
    BlackwellMX4ValueShuffledLayout(block_k=256, block_n=128)
])
@pytest.mark.parametrize("major_dim", [-2, -1])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_mxfp4_value_convert_layout_to_strided_matches_torch(shape, layout, major_dim, step, dtype):
    data = torch.randint(0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    source = convert_layout(wrap_torch_tensor(data, dtype=FP4), layout)
    source = wrap_torch_tensor(source.data.to(dtype), dtype=FP4, shape=source.shape, layout=layout)
    destination = StridedLayout(major_dim)
    expected = convert_layout(source, destination)

    data_cuda = source.data.cuda()
    if step == 2:
        contiguous_dim = data_cuda.stride().index(1)
        data_cuda = data_cuda.movedim(contiguous_dim, -1)
        data_cuda = torch.stack((torch.zeros_like(data_cuda), data_cuda), dim=-2)[..., 1, :]
        data_cuda = data_cuda.movedim(-1, contiguous_dim)
    source_cuda = wrap_torch_tensor(data_cuda, dtype=FP4, shape=source.shape, layout=layout)
    actual = convert_layout(source_cuda, destination)

    assert actual.shape == expected.shape
    assert actual.data.stride() == expected.data.stride()
    assert actual.data.dtype == expected.data.dtype
    assert torch.equal(actual.data.cpu(), expected.data)


@pytest.mark.parametrize(("shape", "major_dim"), [((64, 129), -2), ((129, 64), -1)])
def test_mxfp4_value_convert_layout_odd_source_packing(shape, major_dim):
    layout = BlackwellMXValueLayout()
    data = torch.full(layout.storage_shape(list(shape), True), 0x11, dtype=torch.uint8).mT.contiguous().mT
    source_cpu = wrap_torch_tensor(data, dtype=FP4, shape=shape, layout=layout)
    destination = StridedLayout(major_dim)

    with pytest.raises(RuntimeError) as expected:
        convert_layout(source_cpu, destination)

    source_cuda = wrap_torch_tensor(data.cuda(), dtype=FP4, shape=shape, layout=layout)
    with pytest.raises(RuntimeError) as actual:
        convert_layout(source_cuda, destination)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("inverse", [False, True])
def test_mxfp4_value_shuffled_peak_allocation(inverse):
    data = torch.empty((2048, 2048), dtype=torch.uint8, device="cuda")
    layout = BlackwellMX4ValueShuffledLayout()
    transformation = layout.make_transformation([2048, 4096], is_fp4=True)
    if inverse:
        data = transformation.swizzle_data(data)
        convert = transformation.unswizzle_data
    else:
        convert = transformation.swizzle_data
    warm = convert(data)
    torch.cuda.synchronize(data.device)
    del warm
    baseline = torch.cuda.memory_allocated(data.device)
    torch.cuda.reset_peak_memory_stats(data.device)

    actual = convert(data)
    torch.cuda.synchronize(data.device)
    peak = torch.cuda.max_memory_allocated(data.device) - baseline

    # Shuffling should allocate its output, not another whole weight tensor.
    assert peak <= actual.nbytes + 1024**2


@pytest.mark.parametrize(("slice_sizes", "shape"), [([0], (0, 64)), ([2, 0], (2, 0))])
@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_act_scale_zero_sized_ragged_roundtrip(slice_sizes, shape, device):
    metadata = make_ragged_tensor_metadata_torch(torch.tensor(slice_sizes, dtype=torch.int32), shape[0])
    x = torch.empty(shape, device=device)
    src = wrap_torch_tensor(x)

    swizzled = convert_layout(src, BlackwellActMXScaleLayout(metadata))
    roundtrip = convert_layout(swizzled, src.storage.layout)

    assert roundtrip.storage.data.shape == x.shape


@pytest.mark.parametrize(
    "shape",
    [
        (3, 4096, 1024),
        (10, 254, 60),
        (1, 320, 160),
        (2, 16, 512),
        (3, 2, 36),
    ],
)
def test_mxfp4_scale_roundtrip(shape):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    layout = BlackwellMXScaleLayout()
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    assert (res == x).all()


def test_mxfp4_scale_roundtrip_noncontiguous():
    x = torch.randint(0, 256, (2, 16, 1024), dtype=torch.uint8, device="cuda")[..., ::2]
    assert not x.is_contiguous()
    layout = BlackwellMXScaleLayout()
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    assert torch.equal(res, x)


def test_mxfp4_scale_swizzle_meta():
    x = torch.empty((2, 16, 32), dtype=torch.uint8, device="meta")
    layout = BlackwellMXScaleLayout()
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    swizzled = transformation.swizzle_data(x)
    assert swizzled.device.type == "meta"
    assert swizzled.shape == (1, 2, 4, 2, 256)


@pytest.mark.parametrize("shape", [(2, 256, 192), (1, 128, 64)])
def test_act_scale_roundtrip_batched(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    layout = BlackwellActMXScaleLayout(ragged_metadata=None)
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    torch.testing.assert_close(res, x)


@pytest.mark.parametrize("shape", [(256, 192), (128, 64), (130, 65)])
def test_act_scale_roundtrip_2d_without_ragged_metadata(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    layout = BlackwellActMXScaleLayout(ragged_metadata=None)
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    assert transformation.mode == "batched"
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    assert res.shape == shape
    assert torch.equal(res, x)


@pytest.mark.parametrize("shape", [(256, 192), (128, 64), (130, 65)])
def test_act_scale_convert_layout_roundtrip_2d_without_ragged_metadata(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    x_tri = wrap_torch_tensor(x)
    scale_layout = BlackwellActMXScaleLayout(ragged_metadata=None)
    x_tri_scale = convert_layout(x_tri, scale_layout)
    x_tri_roundtrip = convert_layout(x_tri_scale, StridedLayout(-1))
    assert torch.equal(x_tri_roundtrip.data, x)


@pytest.mark.parametrize(
    "slice_sizes, m, k, align_m",
    [
        ([17, 0, 33, 5], 100, 94, 8),
        ([1, 2, 3, 4, 5], 50, 15, 16),
    ],
)
def test_act_scale_roundtrip_ragged(slice_sizes, m, k, align_m):
    slice_sizes = torch.tensor(slice_sizes, device="cuda", dtype=torch.int32)
    m = max(m, slice_sizes.sum().item())  # there can be padded tokens in the input
    ragged_metadata = make_ragged_tensor_metadata(slice_sizes, m)
    x = torch.randn((m, k), device="cuda", dtype=torch.float32)
    layout = BlackwellActMXScaleLayout(ragged_metadata=ragged_metadata)
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))

    x_useful_rows = x[ragged_metadata.slice_offs[:-1], :]
    res_useful_rows = res[ragged_metadata.slice_offs[:-1], :]
    torch.testing.assert_close(res_useful_rows, x_useful_rows)
