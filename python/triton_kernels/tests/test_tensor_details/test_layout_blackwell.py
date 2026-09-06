import math
import pytest
import torch
import triton
from torch._subclasses.fake_tensor import FakeTensorMode
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
    (BlackwellMXValueLayout(), BlackwellMX4ValueShuffledLayout(), BlackwellMXValueLayout()),
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
    input_shape = tuple(step * size for size in shape)
    data_cpu = torch.randint(0, 256, input_shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    data_cuda = data_cpu.cuda()
    index = (slice(step - 1, None, step), ) * len(shape)
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


@pytest.mark.parametrize("shape", [(256, 128), (2, 258, 65), (2, 3, 130, 66)])
@pytest.mark.parametrize("layout", [
    BlackwellMXValueLayout(),
    BlackwellMX4ValueShuffledLayout(),
    BlackwellMX4ValueShuffledLayout(block_k=256, block_n=128),
])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.enable_warmup
def test_mxfp4_value_convert_layout_matches_torch(shape, layout, trans, step):
    input_shape = tuple(step * size for size in shape[:-1]) + shape[-1:]
    data_cpu = torch.randint(0, 256, (math.prod(input_shape) + 1, ), dtype=torch.uint8)
    data_cuda = data_cpu.cuda()
    data_cpu, data_cuda = data_cpu[1:].view(input_shape), data_cuda[1:].view(input_shape)
    index = (slice(step - 1, None, step), ) * (len(shape) - 1) + (slice(None), )
    data_cpu, data_cuda = data_cpu[index], data_cuda[index]
    if trans:
        data_cpu, data_cuda = data_cpu.mT, data_cuda.mT
    src_cpu = wrap_torch_tensor(data_cpu, dtype=FP4)
    src_cuda = wrap_torch_tensor(data_cuda, dtype=FP4)

    expected = convert_layout(src_cpu, layout)
    actual = convert_layout(src_cuda, layout)
    roundtrip = convert_layout(actual, src_cuda.storage.layout)

    assert actual.storage.data.shape == expected.storage.data.shape
    if isinstance(layout, BlackwellMXValueLayout):
        # Default Blackwell padding is unspecified; compare its logical bytes.
        assert actual.storage.data.stride() == expected.storage.data.stride()
        assert torch.equal(actual.data[..., :src_cpu.shape[-2] // 2, :].cpu(),
                           expected.data[..., :src_cpu.shape[-2] // 2, :])
    else:
        assert actual.storage.data.is_contiguous()
        assert torch.equal(actual.storage.data.cpu(), expected.storage.data)
    assert torch.equal(roundtrip.storage.data, data_cuda)


@pytest.mark.parametrize(("shape", "device"), [
    ((256, 512), "cuda"),
    ((130, 66), "cuda"),
    ((2, 130, 66), "cuda"),
    ((2, 3, 130, 66), "cuda"),
    ((2, 1, 3, 130, 66), "cuda"),
    ((0, 130, 66), "cuda"),
    ((2, 130, 66), "cpu"),
    ((2, 130, 66), "meta"),
])
@pytest.mark.parametrize("tiles", [(128, 256), (192, 48)])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("with_out", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_blackwell_fp4_shuffle_conversion(shape, tiles, step, with_out, device, inverse):
    data = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), dtype=torch.uint8,
                         generator=torch.Generator().manual_seed(0))
    canonical = wrap_torch_tensor(data, dtype=FP4)
    plain, shuffled = BlackwellMXValueLayout(), BlackwellMX4ValueShuffledLayout(*tiles)
    source_layout, destination = (shuffled, plain) if inverse else (plain, shuffled)
    source = convert_layout(canonical, source_layout)
    expected = convert_layout(canonical, destination)
    # Padding bytes must not leak into logical values or shuffled destination padding.
    if inverse:
        _, tiles_k, tiles_n, tile_n, tile_k = source.data.shape
        k = torch.arange(tiles_k * tile_k).reshape(1, tiles_k, 1, 1, tile_k)
        n = torch.arange(tiles_n * tile_n).reshape(1, 1, tiles_n, tile_n, 1)
        source.data.masked_fill_((k >= shape[-2] // 2) | (n >= shape[-1]), 0xAB)
    else:
        source.data[..., shape[-2] // 2:, :].fill_(0xAB)
    source_data = source.data.to(device)
    if step == 2:
        storage = torch.empty(tuple(2 * size for size in source_data.shape), dtype=torch.uint8, device=device)
        source_view = storage[tuple(slice(None, None, 2) for _ in range(source_data.ndim))]
        source_view.copy_(source_data)
        source_data = source_view
    source = wrap_torch_tensor(source_data, dtype=FP4, shape=shape, layout=source.storage.layout)
    out = None
    if with_out:
        storage = torch.full((expected.data.numel() * step + 1, ), 0xCD, dtype=torch.uint8, device=device)
        out_data = storage[1:].as_strided(expected.data.shape, tuple(step * s for s in expected.data.stride()))
        out = wrap_torch_tensor(out_data, dtype=FP4, shape=shape, layout=destination)

    actual = convert_layout(source, destination, out=out)

    assert actual.shape == list(shape)
    assert actual.data.shape == expected.data.shape
    assert actual.data.dtype == expected.data.dtype
    assert actual.device == source.device
    if with_out:
        assert actual is out
    elif inverse:
        assert actual.data.stride() == expected.data.stride()
    else:
        assert actual.data.is_contiguous()
    if device != "meta":
        actual_data, expected_data = actual.data.cpu(), expected.data
        if inverse:
            actual_data = actual_data[..., :shape[-2] // 2, :]
            expected_data = expected_data[..., :shape[-2] // 2, :]
            if device == "cuda" and with_out:
                assert torch.all(out.data[..., shape[-2] // 2:, :] == 0xCD)
        assert torch.equal(actual_data, expected_data)
        assert torch.equal(convert_layout(actual, StridedLayout()).data.cpu(), data)
        if with_out:
            assert storage[0].item() == 0xCD
            if step == 2:
                assert torch.all(storage[2::2] == 0xCD)


@pytest.mark.parametrize("with_out", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_blackwell_fp4_shuffle_peak_allocation(with_out, inverse):
    canonical = empty((8, 1024, 2048), dtype=FP4, device="cuda")
    plain, shuffled = BlackwellMXValueLayout(), BlackwellMX4ValueShuffledLayout()
    source_layout, destination = (shuffled, plain) if inverse else (plain, shuffled)
    source = convert_layout(canonical, source_layout)
    warm = convert_layout(source, destination)
    out = warm if with_out else None
    torch.cuda.synchronize()
    del warm
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    actual = convert_layout(source, destination, out=out)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline

    assert peak <= (0 if with_out else actual.data.nbytes) + 1024**2


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


@pytest.mark.parametrize("shape", [(258, 130), (2, 3, 2, 66, 34)])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("inverse", [False, True])
def test_mxfp4_value_conversion_strided_storage(shape, step, inverse):
    data = torch.randint(0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    logical_shape = [*shape[:-1], 2 * shape[-1]]
    transformation = BlackwellMXValueLayout().make_transformation(logical_shape, True)
    source = transformation.swizzle_data(data) if inverse else data
    expected = data if inverse else transformation.swizzle_data(data)
    # Slice every physical axis, including the packed axis and the batch axes.
    storage = torch.full(tuple(step * size + 1 for size in source.shape), 0xAB, dtype=torch.uint8)
    index = (slice(1, None, step), ) * source.ndim
    storage[index] = source
    source_cuda = storage.cuda()[index]

    actual = transformation.unswizzle_data(source_cuda) if inverse else transformation.swizzle_data(source_cuda)

    assert actual.stride() == expected.stride()
    if not inverse:
        actual = actual[..., :logical_shape[-2] // 2, :]
        expected = expected[..., :logical_shape[-2] // 2, :]
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(source_cuda.cpu(), source)


@pytest.mark.parametrize("shape,major", [(shape, major)
                                         for shape in [(256, 256), (2, 130, 66), (2, 4, 130, 66)]
                                         for major in range(len(shape))])
@pytest.mark.parametrize("layout", [
    BlackwellMXValueLayout(),
] + [
    BlackwellMX4ValueShuffledLayout(block_k, block_n)
    for block_k in [64, 128, 192, 256]
    for block_n in [32, 48, 128, 256]
])
@pytest.mark.parametrize("inverse", [False, True])
def test_mxfp4_value_convert_layout_out(shape, layout, major, inverse):
    data = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), dtype=torch.uint8,
                         generator=torch.Generator().manual_seed(0))
    strided = StridedLayout(major)
    source = convert_layout(wrap_torch_tensor(data, dtype=FP4), layout if inverse else strided)
    destination = strided if inverse else layout
    expected = convert_layout(source, destination)
    source_gpu = wrap_torch_tensor(source.data.cuda(), dtype=FP4, shape=shape, layout=source.storage.layout)
    storage = torch.full((expected.data.numel() * 2 + 1, ), 0xAB, dtype=torch.uint8, device="cuda")
    out_data = storage[1:].as_strided(expected.data.shape, tuple(s * 2 for s in expected.data.stride()))
    out = wrap_torch_tensor(out_data, dtype=FP4, shape=shape, layout=destination)

    actual = convert_layout(source_gpu, destination, out=out)

    assert actual is out
    actual_data, expected_data = actual.data.cpu(), expected.data
    if isinstance(destination, BlackwellMXValueLayout):
        # Plain Blackwell padding is unspecified; only logical bytes must match.
        actual_data = actual_data[..., :shape[-2] // 2, :]
        expected_data = expected_data[..., :shape[-2] // 2, :]
    assert torch.equal(actual_data, expected_data)
    assert torch.all(storage[::2] == 0xAB)
    assert torch.equal(source_gpu.data.cpu(), source.data)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("major", [-2, -1])
@pytest.mark.parametrize("with_out", [False, True])
def test_mxfp4_value_convert_layout_fake(inverse, major, with_out, monkeypatch):

    def reject_launch(*args, **kwargs):
        pytest.fail("ordinary FakeTensor conversion must not launch a Triton kernel")

    monkeypatch.setattr(triton.KernelInterface, "__getitem__", reject_launch)
    shape = (256, 512)
    layout, strided = BlackwellMXValueLayout(), StridedLayout(major)
    source = empty(shape, dtype=FP4, device="cpu", layout=strided)
    if inverse:
        source = convert_layout(source, layout)
    destination = strided if inverse else layout
    expected = convert_layout(source, destination)
    with FakeTensorMode():
        data = torch.empty_strided(source.data.shape, source.data.stride(), dtype=torch.uint8, device="cuda")
        source_fake = wrap_torch_tensor(data, dtype=FP4, shape=shape, layout=source.storage.layout)
        out = wrap_torch_tensor(
            torch.empty_strided(expected.data.shape, expected.data.stride(), dtype=torch.uint8, device="cuda"),
            dtype=FP4, shape=shape, layout=destination) if with_out else None
        actual = convert_layout(source_fake, destination, out=out)
        assert actual.shape == list(shape)
        assert actual.data.shape == expected.data.shape
        assert actual.data.stride() == expected.data.stride()
        if with_out:
            assert actual is out


@pytest.mark.parametrize("inverse", [False, True])
def test_mxfp4_value_transform_fake(inverse, monkeypatch):

    def reject_launch(*args, **kwargs):
        pytest.fail("ordinary FakeTensor transformation must not launch a Triton kernel")

    monkeypatch.setattr(triton.KernelInterface, "__getitem__", reject_launch)
    transformation = BlackwellMXValueLayout().make_transformation([256, 512], True)
    with FakeTensorMode():
        data = torch.empty(transformation.storage_shape if inverse else (256, 256), dtype=torch.uint8, device="cuda")
        actual = transformation.unswizzle_data(data) if inverse else transformation.swizzle_data(data)
        assert list(actual.shape) == ([256, 256] if inverse else transformation.storage_shape)


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


@pytest.mark.parametrize("layout,shape,major", [(layout, shape, major) for layout, shape in [
    (BlackwellActMXScaleLayout(None), (130, 9)),
    (BlackwellActMXScaleLayout(None), (2, 130, 9)),
    (BlackwellActMXScaleLayout(None), (130, 259)),
    (BlackwellActMXScaleLayout(None), (2, 130, 259)),
    (BlackwellActMXScaleLayout(None), (524416, 8)),
    (BlackwellMXScaleLayout(), (9, 130)),
    (BlackwellMXScaleLayout(), (2, 9, 130)),
    (BlackwellMXScaleLayout(), (259, 130)),
    (BlackwellMXScaleLayout(), (2, 259, 130)),
    (BlackwellMXScaleLayout(), (8, 524416)),
    (BlackwellMXScaleLayout(), (2, 3, 9, 130)),
] for major in range(len(shape))])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("with_out", [False, True])
def test_scale_convert_layout_strided_storage(layout, shape, inverse, major, step, with_out):
    strided = StridedLayout(major)
    data = torch.randint(0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    source = convert_layout(wrap_torch_tensor(data), layout if inverse else strided)
    destination = strided if inverse else layout
    expected = convert_layout(source, destination)
    storage = torch.empty(tuple(step * size + 1 for size in source.data.shape), dtype=torch.uint8, device="cuda")
    if not inverse:
        storage = storage.movedim(major, -1).contiguous().movedim(-1, major)
    source_data = storage[(slice(1, None, step), ) * storage.ndim]
    source_data.copy_(source.data)
    if not inverse:
        assert source_data.stride(major) == step
    source_gpu = wrap_torch_tensor(source_data, shape=shape, layout=source.storage.layout)
    out_storage = torch.full((expected.data.numel() * step + 1, ), 0xAB, dtype=torch.uint8, device="cuda")
    out_data = out_storage[1:].as_strided(expected.data.shape, tuple(s * step for s in expected.data.stride()))
    out = wrap_torch_tensor(out_data, shape=shape, layout=destination) if with_out else None

    actual = convert_layout(source_gpu, destination, out=out)

    assert actual.shape == list(shape)
    assert torch.equal(actual.data.cpu(), expected.data)
    assert torch.equal(source_gpu.data.cpu(), source.data)
    if with_out:
        assert actual is out
        assert out_storage[0].item() == 0xAB
        if step == 2:
            assert torch.all(out_storage[2::2] == 0xAB)
    else:
        assert actual.data.stride() == expected.data.stride()


@pytest.mark.parametrize("sizes,shape", [
    ([17, 0, 33, 5], (100, 94)),
    ([1, 127, 128, 129], (512, 17)),
    ([0, 0], (256, 9)),
    ([], (0, 9)),
    ([], (100, 9)),
    ([], (129, 9)),
    ([], (257, 9)),
])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("with_out", [False, True])
@pytest.mark.parametrize("major", [-2, -1])
def test_act_scale_convert_layout_ragged_padding(sizes, shape, step, with_out, major):
    metadata = make_ragged_tensor_metadata(torch.tensor(sizes, dtype=torch.int32, device="cuda"), shape[0])
    layout = BlackwellActMXScaleLayout(metadata)
    transformation = layout.make_transformation(list(shape), False)
    storage = torch.randint(0, 256, tuple(step * size + 1 for size in shape), dtype=torch.uint8,
                            generator=torch.Generator().manual_seed(0))
    data = storage[1::step, 1::step]
    padded = torch.zeros((transformation.M_pad, transformation.K_pad), dtype=torch.uint8)
    source_start, padded_start = 0, 0
    for size in sizes:
        padded[padded_start:padded_start + size, :shape[1]] = data[source_start:source_start + size]
        source_start += size
        padded_start += (size + 127) // 128 * 128
    expected = BlackwellActMXScaleLayout(None).make_transformation(list(padded.shape), False).swizzle_data(padded)
    source_storage = storage.cuda().movedim(major, -1).contiguous().movedim(-1, major)
    source = wrap_torch_tensor(source_storage[1::step, 1::step], layout=StridedLayout(major))
    out_storage = torch.full((expected.numel() * step + 1, ), 0xAB, dtype=torch.uint8, device="cuda")
    out_data = out_storage[1:].as_strided(expected.shape, tuple(s * step for s in expected.stride()))
    out = wrap_torch_tensor(out_data, shape=shape, layout=layout) if with_out else None

    actual = convert_layout(source, layout, out=out)

    assert torch.equal(actual.data.cpu(), expected)
    if with_out:
        assert actual is out
        assert out_storage[0].item() == 0xAB
        if step == 2:
            assert torch.all(out_storage[2::2] == 0xAB)

    restored_expected = data.clone()
    restored_expected[sum(sizes):] = 0
    for destination_major in [-2, -1]:
        destination = StridedLayout(destination_major)
        strides = destination.make_transformation(list(shape), False).storage_strides
        restored_storage = torch.full((data.numel() * step + 1, ), 0xAB, dtype=torch.uint8, device="cuda")
        restored_data = restored_storage[1:].as_strided(shape, tuple(s * step for s in strides))
        restored_out = wrap_torch_tensor(restored_data, layout=destination) if with_out else None
        restored = convert_layout(actual, destination, out=restored_out)
        assert torch.equal(restored.data.cpu(), restored_expected)
        if with_out:
            assert restored is restored_out
            assert restored_storage[0].item() == 0xAB
            if step == 2:
                assert torch.all(restored_storage[2::2] == 0xAB)


@pytest.mark.parametrize("layout", [BlackwellActMXScaleLayout(None), BlackwellMXScaleLayout()])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float32])
def test_scale_convert_layout_dtype(layout, dtype):
    bits = torch.randint(0, 256, (2, 130, 9), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    data = bits.view(dtype) if dtype.itemsize == 1 else bits.to(dtype)
    expected = convert_layout(wrap_torch_tensor(bits if dtype.itemsize == 1 else data), layout).data
    source = wrap_torch_tensor(data.cuda())

    actual = convert_layout(source, layout)

    assert actual.data.dtype == dtype
    assert torch.equal(actual.data.cpu().view(torch.uint8), expected.view(torch.uint8))
    restored = convert_layout(actual, StridedLayout())
    assert torch.equal(restored.data.cpu().view(torch.uint8), data.view(torch.uint8))


@pytest.mark.parametrize("kind", ["act", "mx", "ragged_act"])
@pytest.mark.parametrize("with_out", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_scale_convert_layout_peak_allocation(kind, with_out, inverse):
    shape = (32768, 128) if kind == "ragged_act" else (16, 2048, 128)
    data = torch.empty(shape, dtype=torch.uint8, device="cuda")
    if kind == "ragged_act":
        sizes = torch.tensor([8191, 8193, 8191, 8192], dtype=torch.int32, device="cuda")
        layout = BlackwellActMXScaleLayout(make_ragged_tensor_metadata(sizes, shape[0]))
    else:
        layout = BlackwellMXScaleLayout() if kind == "mx" else BlackwellActMXScaleLayout(None)
    major = -2 if kind == "mx" else -1
    if major == -2:
        data = data.mT
    source = wrap_torch_tensor(data, layout=StridedLayout(major))
    assert source.data.stride(major) == 1
    if inverse:
        source = convert_layout(source, layout)
        layout = StridedLayout(major)
    out = convert_layout(source, layout) if with_out else None
    warm = convert_layout(source, layout, out=out)
    torch.cuda.synchronize()
    del warm
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    actual = convert_layout(source, layout, out=out)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline

    assert peak <= (0 if with_out else actual.data.nbytes) + 1024**2
    if with_out:
        assert actual is out


@pytest.mark.parametrize("layout", [BlackwellActMXScaleLayout(None), BlackwellMXScaleLayout()])
@pytest.mark.parametrize("with_out", [False, True])
def test_scale_convert_layout_uses_input_device(layout, with_out):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    data = torch.randint(0, 256, (2, 130, 9), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    expected = convert_layout(wrap_torch_tensor(data), layout)
    stream = torch.cuda.Stream(device=1)
    with torch.cuda.stream(stream), torch.cuda.device(0):
        source = wrap_torch_tensor(data.to("cuda:1"))
        out = wrap_torch_tensor(torch.empty_like(expected.data, device="cuda:1"), shape=source.shape,
                                layout=layout) if with_out else None
        actual = convert_layout(source, layout, out=out)
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream(1) == stream
        assert actual.device == torch.device("cuda:1")
        stream.synchronize()
        assert torch.equal(actual.data.cpu(), expected.data)
        if with_out:
            assert actual is out
        restored = convert_layout(actual, StridedLayout())
        stream.synchronize()
        assert torch.equal(restored.data.cpu(), data)
        assert torch.cuda.current_device() == 0


@pytest.mark.parametrize("layout", [BlackwellActMXScaleLayout(None), BlackwellMXScaleLayout()])
def test_scale_convert_layout_fake(layout, monkeypatch):

    def reject_launch(*args, **kwargs):
        pytest.fail("ordinary FakeTensor conversion must not launch a Triton kernel")

    monkeypatch.setattr(triton.KernelInterface, "__getitem__", reject_launch)
    shape = (2, 130, 9)
    with FakeTensorMode():
        source = wrap_torch_tensor(torch.empty(shape, dtype=torch.uint8, device="cuda"))
        out = convert_layout(source, layout)
        assert convert_layout(source, layout, out=out) is out
        assert list(out.data.shape) == layout.storage_shape(shape, False)
        assert out.data.dtype == torch.uint8
        restored = convert_layout(out, StridedLayout())
        assert list(restored.data.shape) == list(shape)
        assert convert_layout(out, StridedLayout(), out=restored) is restored
