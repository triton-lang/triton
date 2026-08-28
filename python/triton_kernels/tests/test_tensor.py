import math

import pytest
import torch
import triton
import triton.language as tl
from triton_kernels.fpsan import embed, unembed
from triton_kernels.tensor_details.bitmatrix import _keyed_add
from triton_kernels.tensor_details.dtype import BIT, FP4, UINT8
from triton_kernels.tensor import (
    convert_layout,
    empty,
    make_ragged_tensor_metadata,
    make_ragged_tensor_metadata_torch,
    remap_ragged_tensor_metadata,
    remap_ragged_tensor_metadata_torch,
    make_bitmatrix_metadata,
    make_bitmatrix_metadata_torch,
    wrap_torch_tensor,
)
from triton_kernels.testing import assert_equal
from triton_kernels.tensor_details.layout import (
    BlackwellActMXScaleLayout,
    BlackwellMX4ValueShuffledLayout,
    BlackwellMXScaleLayout,
    BlackwellMXValueLayout,
    CDNA4MXScaleLayout,
    GFX1250MXScaleLayout,
    HopperMXScaleLayout,
    HopperMXValueLayout,
    StridedLayout,
)

_FP4_VALUE_LAYOUTS = [
    HopperMXValueLayout(-2, 3),
    HopperMXValueLayout(-1, 3),
    BlackwellMXValueLayout(),
    BlackwellMX4ValueShuffledLayout(),
]


@pytest.mark.parametrize("dtype", [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
])
@pytest.mark.parametrize("layout",
                         ["contiguous", "transposed", "sliced", "expanded", "channels_last", "scalar", "empty"])
def test_fpsan_embed_unembed_torch_tensor(dtype, layout, fresh_knobs):
    fresh_knobs.compilation.instrumentation_mode = ""
    integer_dtype = getattr(torch, f"int{dtype.itemsize * 8}")
    storage = (torch.arange(17 * 19, device="cuda", dtype=torch.int64) * 37).to(integer_dtype)
    one_bits = torch.tensor(1.0, dtype=dtype).view(integer_dtype).item()
    storage[1] = one_bits
    values = storage.reshape(17, 19)

    if layout == "transposed":
        values = values.T
    elif layout == "sliced":
        values = values[1::2, 1::3]
    elif layout == "expanded":
        values = values[:1, :].expand(7, -1)
    elif layout == "channels_last":
        values = storage[:210].reshape(2, 3, 5, 7).contiguous(memory_format=torch.channels_last)
    elif layout == "scalar":
        values = storage[1]
    elif layout == "empty":
        values = values[:0]

    x = values.view(dtype)
    payload = embed(x)
    assert payload.dtype == integer_dtype
    assert payload.shape == x.shape
    if layout in ("transposed", "channels_last"):
        assert payload.stride() == x.stride()

    one_positions = values == one_bits
    assert torch.equal(payload[one_positions], torch.ones_like(payload[one_positions]))

    roundtrip = unembed(payload, dtype)
    assert torch.equal(roundtrip.view(integer_dtype), values)

    unsigned_dtype = getattr(torch, f"uint{dtype.itemsize * 8}")
    unsigned_roundtrip = unembed(payload.view(unsigned_dtype), dtype)
    assert torch.equal(unsigned_roundtrip.view(integer_dtype), values)
    assert fresh_knobs.compilation.instrumentation_mode == ""


@pytest.mark.parametrize(
    ("logical_shape", "is_fp4", "layout", "storage_shape"),
    [
        ((3, 258, 514), True, StridedLayout(-2), [3, 129, 514]),
        ((0, 64), True, StridedLayout(-2), [0, 64]),
        ((3, 258, 514), True, HopperMXValueLayout(-2, 3), [3, 768, 192]),
        ((3, 258, 514), True, HopperMXValueLayout(-1, 3), [3, 128, 1280]),
        ((0, 64), True, HopperMXValueLayout(-2, 3), [0, 64]),
        ((3, 70, 65), False, HopperMXScaleLayout(-2, 4), [3, 2240, 4]),
        ((3, 70, 65), False, HopperMXScaleLayout(-1, 4), [3, 4, 2112]),
        ((0, 64), False, HopperMXScaleLayout(-2, 4), [0, 4]),
        ((3, 258, 514), True, BlackwellMXValueLayout(), [3, 256, 514]),
        ((0, 64), True, BlackwellMXValueLayout(), [0, 64]),
        ((3, 258, 514), True, BlackwellMX4ValueShuffledLayout(), [3, 4, 3, 256, 64]),
        ((0, 64), True, BlackwellMX4ValueShuffledLayout(), [1, 0, 1, 256, 64]),
        ((3, 254, 60), False, BlackwellMXScaleLayout(), [1, 3, 64, 2, 256]),
        ((0, 64), False, BlackwellMXScaleLayout(), [1, 1, 0, 2, 256]),
        ((130, 65), False, BlackwellActMXScaleLayout(None), [1, 2, 18, 2, 256]),
        ((3, 130, 65), False, BlackwellActMXScaleLayout(None), [1, 6, 18, 2, 256]),
        ((0, 64), False, BlackwellActMXScaleLayout(None), [1, 0, 16, 2, 256]),
        ((3, 254, 60), False, CDNA4MXScaleLayout(), [3, 8192, 2]),
        ((0, 64), False, CDNA4MXScaleLayout(), [1, 0, 2]),
        ((3, 254, 60), False, GFX1250MXScaleLayout(), [3, 32768, 1]),
        ((0, 64), False, GFX1250MXScaleLayout(), [1, 0, 1]),
    ],
)
def test_layout_storage_shape_matches_conversion(logical_shape, is_fp4, layout, storage_shape):
    dtype = FP4 if is_fp4 else UINT8
    tensor = empty(logical_shape, dtype=dtype, device="meta")

    converted = convert_layout(tensor, layout)

    assert layout.storage_shape(list(logical_shape), is_fp4) == storage_shape
    assert list(converted.storage.data.shape) == storage_shape


def test_ragged_layout_storage_shape():
    slice_sizes = torch.tensor([17, 0, 33, 5], dtype=torch.int32)
    metadata = make_ragged_tensor_metadata_torch(slice_sizes, 100)

    assert BlackwellActMXScaleLayout(metadata).storage_shape([100, 94], False) == [1, 4, 24, 2, 256]


@pytest.mark.parametrize("major_dim", [-1, -2])
@pytest.mark.parametrize("other_size", [0, 2])
def test_strided_layout_rejects_odd_fp4_packing_dim(major_dim, other_size):
    shape = [other_size, other_size]
    shape[major_dim] = 1

    with pytest.raises(ValueError):
        StridedLayout(major_dim).storage_shape(shape, True)


@pytest.mark.parametrize(
    ("transpose", "layout"),
    [
        (False, StridedLayout(-1)),
        (False, StridedLayout(1)),
        (True, StridedLayout(-2)),
        (True, StridedLayout(0)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_convert_layout_noop(transpose, layout, device):
    data = torch.randn((7, 11), device=device)
    if transpose:
        data = data.T
    tensor = wrap_torch_tensor(data)

    assert convert_layout(tensor, layout) is tensor
    alias = wrap_torch_tensor(data.view_as(data), layout=layout)
    assert convert_layout(tensor, layout, out=alias) is alias
    out = wrap_torch_tensor(torch.empty_like(data), layout=layout)
    assert convert_layout(tensor, layout, out=out) is out
    if device != "meta":
        assert torch.equal(out.data, data)


def test_convert_layout_noop_preserves_strided_view():
    tensor = wrap_torch_tensor(torch.randn((14, 11))[::2])

    assert convert_layout(tensor, StridedLayout(-1)) is tensor
    assert tensor.storage.data.stride() == (22, 1)


def test_convert_layout_rejects_strided_view_without_contiguous_dimension():
    tensor = wrap_torch_tensor(torch.randn((14, 22))[::2, ::2])

    with pytest.raises(ValueError):
        convert_layout(tensor, tensor.storage.layout)


def test_convert_layout_noop_does_not_ignore_transformation_kwargs():
    tensor = wrap_torch_tensor(torch.randn((7, 11)))

    with pytest.raises(TypeError):
        convert_layout(tensor, tensor.storage.layout, unsupported=True)


@pytest.mark.parametrize(
    ("storage_shape", "logical_shape", "dtype", "layout", "equivalent_layout"),
    [
        ((10, 254, 60), None, UINT8, BlackwellMXScaleLayout(), BlackwellMXScaleLayout()),
        ((130, 65), None, UINT8, BlackwellActMXScaleLayout(None), BlackwellActMXScaleLayout(None)),
        ((256, 64), (256, 128), FP4, BlackwellMXValueLayout(), BlackwellMXValueLayout()),
        ((128, 256), (128, 512), FP4, BlackwellMX4ValueShuffledLayout(), BlackwellMX4ValueShuffledLayout()),
        ((70, 65), None, UINT8, HopperMXScaleLayout(-2, 4), HopperMXScaleLayout(-2, 4)),
        ((64, 64), (64, 128), FP4, HopperMXValueLayout(-2, 3), HopperMXValueLayout(-2, 3)),
        ((10, 254, 60), None, UINT8, CDNA4MXScaleLayout(), CDNA4MXScaleLayout()),
        ((10, 254, 60), None, UINT8, GFX1250MXScaleLayout(), GFX1250MXScaleLayout()),
    ],
)
def test_convert_layout_noop_for_equivalent_layout(storage_shape, logical_shape, dtype, layout, equivalent_layout):
    tensor = wrap_torch_tensor(torch.randint(0, 256, storage_shape, dtype=torch.uint8), dtype=dtype,
                               shape=logical_shape)
    converted = convert_layout(tensor, layout)

    assert converted is not tensor
    assert convert_layout(converted, equivalent_layout) is converted
    out = wrap_torch_tensor(torch.empty_like(converted.data), dtype=dtype, shape=converted.shape,
                            layout=equivalent_layout)
    assert convert_layout(tensor, layout, out=out) is out
    assert torch.equal(out.data, converted.data)


@pytest.mark.parametrize("shape,layout,dtype", [
    ([2, 16], StridedLayout(-2), FP4),
    ([4, 8], StridedLayout(-1), FP4),
    ([4, 8], StridedLayout(-2), UINT8),
])
def test_convert_layout_out_rejects_incompatible_metadata(shape, layout, dtype):
    source = wrap_torch_tensor(torch.zeros((4, 4), dtype=torch.uint8), dtype=FP4)
    data = torch.full((2, 8), 0xAB, dtype=torch.uint8)
    out = wrap_torch_tensor(data, dtype=dtype, shape=shape, layout=layout)

    with pytest.raises(ValueError, match="out must have"):
        convert_layout(source, StridedLayout(-2), out=out)
    assert torch.all(data == 0xAB)


@pytest.mark.parametrize("shape,strides,dtype,device", [
    ((2, 8), (1, 2), torch.int16, "cpu"),
    ((2, 7), (1, 2), torch.uint8, "cpu"),
    ((2, 8), (0, 1), torch.uint8, "cpu"),
    ((2, 8), (1, 2), torch.uint8, "meta"),
])
def test_convert_layout_out_rejects_incompatible_storage(shape, strides, dtype, device):
    source = wrap_torch_tensor(torch.zeros((4, 4), dtype=torch.uint8), dtype=FP4)
    layout = StridedLayout(-2)
    data = torch.empty_strided(shape, strides, dtype=dtype, device=device).fill_(0xAB)
    out = wrap_torch_tensor(data, dtype=FP4, shape=source.shape, layout=layout)

    with pytest.raises(ValueError, match="out"):
        convert_layout(source, layout, out=out)
    if device != "meta":
        assert torch.all(data == 0xAB)


def test_convert_layout_out_requires_byte_fp4_storage():
    source = wrap_torch_tensor(torch.zeros((4, 4), dtype=torch.int32), dtype=FP4, shape=[4, 8])
    out = wrap_torch_tensor(torch.empty_like(source.data), dtype=FP4, shape=source.shape)

    with pytest.raises(ValueError, match="requires uint8 storage"):
        convert_layout(source, source.storage.layout, out=out)


@pytest.mark.parametrize("offset", [0, 1, 32])
@pytest.mark.parametrize("major_dim", [-2, -1])
def test_convert_layout_out_shared_storage(offset, major_dim):
    data = torch.arange(64, dtype=torch.uint8)
    source = wrap_torch_tensor(data[:32].view(4, 8), dtype=FP4)
    layout = StridedLayout(major_dim)
    shape = layout.storage_shape(source.shape, True)
    out = wrap_torch_tensor(data[offset:offset + 32].view(shape), dtype=FP4, shape=source.shape, layout=layout)
    expected = convert_layout(source, layout)

    if offset == 1 or (offset == 0 and major_dim == -2):
        with pytest.raises(ValueError, match="must not overlap"):
            convert_layout(source, layout, out=out)
        assert torch.equal(data, torch.arange(64, dtype=torch.uint8))
    else:
        assert convert_layout(source, layout, out=out) is out
        assert torch.equal(out.data, expected.data)


@pytest.mark.parametrize(
    ("storage_shape", "logical_shape", "dtype", "layout", "different_layout"),
    [
        ((70, 65), None, UINT8, HopperMXScaleLayout(-2, 4), HopperMXScaleLayout(-2, 8)),
        ((64, 64), (64, 128), FP4, HopperMXValueLayout(-2, 3), HopperMXValueLayout(-2, 2)),
        ((128, 256), (128, 512), FP4, BlackwellMX4ValueShuffledLayout(), BlackwellMX4ValueShuffledLayout(block_n=128)),
    ],
)
def test_convert_layout_converts_different_parameterized_layout(storage_shape, logical_shape, dtype, layout,
                                                                different_layout):
    tensor = wrap_torch_tensor(torch.randint(0, 256, storage_shape, dtype=torch.uint8), dtype=dtype,
                               shape=logical_shape)
    converted = convert_layout(tensor, layout)

    assert convert_layout(converted, different_layout) is not converted


@pytest.mark.parametrize("layout", _FP4_VALUE_LAYOUTS)
@pytest.mark.parametrize("major_dim", [-2, -1])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("with_out", [False, True])
def test_mxfp4_value_convert_layout_peak_allocation(layout, major_dim, step, inverse, with_out):
    data = torch.empty((2048, 2048), dtype=torch.uint8, device="cuda")
    strided = convert_layout(wrap_torch_tensor(data, dtype=FP4), StridedLayout(major_dim))
    source = convert_layout(strided, layout) if inverse else strided
    destination = strided.storage.layout if inverse else layout
    if step == 2:
        contiguous_dim = source.data.stride().index(1)
        storage = source.data.movedim(contiguous_dim, -1)
        storage = torch.stack((storage, storage), dim=-2)[..., 1, :].movedim(-1, contiguous_dim)
        source = wrap_torch_tensor(storage, dtype=FP4, shape=source.shape, layout=source.storage.layout)
    out = convert_layout(source, destination) if with_out else None
    warm = convert_layout(source, destination, out=out)
    torch.cuda.synchronize(source.device)
    del warm
    baseline = torch.cuda.memory_allocated(source.device)
    torch.cuda.reset_peak_memory_stats(source.device)

    actual = convert_layout(source, destination, out=out)
    torch.cuda.synchronize(source.device)
    peak = torch.cuda.max_memory_allocated(source.device) - baseline

    # Supplied output storage needs no additional weight-sized allocation.
    if with_out:
        assert actual is out
    assert peak <= (0 if with_out else actual.data.nbytes) + 1024**2


@pytest.mark.parametrize("layout", _FP4_VALUE_LAYOUTS + [HopperMXValueLayout(-2, 2), HopperMXValueLayout(-1, 2)])
@pytest.mark.parametrize("shape", [(2, 130, 66), (0, 130, 66)])
@pytest.mark.parametrize("major_dim", [-2, -1])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_mxfp4_value_convert_layout_out(layout, shape, major_dim, inverse, step, device):
    strided = StridedLayout(major_dim)
    storage_shape = strided.storage_shape(shape, True)
    data = torch.randint(0, 256, (math.prod(storage_shape) + 1, ), dtype=torch.uint8,
                         generator=torch.Generator().manual_seed(0))
    # Packing is explicit: C-order physical bytes may pack logical dimension -2.
    source = wrap_torch_tensor(data[1:].view(storage_shape), dtype=FP4, shape=shape, layout=strided)
    source = convert_layout(source, layout) if inverse else source
    destination = strided if inverse else layout
    expected = convert_layout(source, destination)
    source_data = source.data.to(device) if inverse else data.to(device)[1:].view(storage_shape)
    source = wrap_torch_tensor(source_data, dtype=FP4, shape=shape, layout=source.storage.layout)

    storage = torch.full((expected.data.numel() * step + 1, ), 0xAB, dtype=torch.uint8, device=device)
    out_data = storage[1:].as_strided(expected.data.shape, tuple(stride * step for stride in expected.data.stride()))
    out = wrap_torch_tensor(out_data, dtype=FP4, shape=shape, layout=destination)

    assert convert_layout(source, destination, out=out) is out
    assert out.data is out_data
    if device != "meta":
        actual = out.data
        reference = expected.data
        if not inverse and isinstance(layout, BlackwellMXValueLayout):
            actual = actual[..., :shape[-2] // 2, :]
            reference = reference[..., :shape[-2] // 2, :]
        assert torch.equal(actual.cpu(), reference)
        assert storage[0].item() == 0xAB
        if step == 2:
            assert torch.all(storage[2::2] == 0xAB)
        if not inverse:
            restored = convert_layout(out, strided)
            assert torch.equal(restored.data.cpu(), data[1:].view(storage_shape))


@pytest.mark.parametrize("layout", _FP4_VALUE_LAYOUTS)
@pytest.mark.parametrize("major_dim", [-3, -2, -1])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32])
def test_mxfp4_value_convert_layout_forward_fallback(layout, major_dim, dtype):
    data = torch.randint(0, 256, (4, 66, 66), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    source = convert_layout(wrap_torch_tensor(data, dtype=FP4), StridedLayout(major_dim))
    source = wrap_torch_tensor(source.data.to(dtype), dtype=FP4, shape=source.shape, layout=source.storage.layout)
    expected = convert_layout(source, layout)
    source_cuda = wrap_torch_tensor(source.data.cuda(), dtype=FP4, shape=source.shape, layout=source.storage.layout)

    actual = convert_layout(source_cuda, layout)

    assert actual.shape == expected.shape
    assert actual.data.shape == expected.data.shape
    assert actual.data.stride() == expected.data.stride()
    assert actual.data.dtype == expected.data.dtype
    if isinstance(layout, BlackwellMXValueLayout):
        assert torch.equal(actual.data[..., :source.shape[-2] // 2, :].cpu(),
                           expected.data[..., :source.shape[-2] // 2, :])
    else:
        assert torch.equal(actual.data.cpu(), expected.data)


@pytest.mark.parametrize("layout", _FP4_VALUE_LAYOUTS)
@pytest.mark.parametrize("major_dim", [-2, -1])
@pytest.mark.parametrize("inverse", [False, True])
def test_mxfp4_value_convert_layout_meta(layout, major_dim, inverse):
    shape = [2, 130, 66]
    source = empty(shape, dtype=FP4, device="cpu", layout=StridedLayout(major_dim))
    source = convert_layout(source, layout) if inverse else source
    destination = StridedLayout(major_dim) if inverse else layout
    expected = convert_layout(source, destination)

    source_meta = wrap_torch_tensor(source.data.to("meta"), dtype=FP4, shape=source.shape, layout=source.storage.layout)
    actual = convert_layout(source_meta, destination)

    assert actual.shape == expected.shape
    assert actual.data.shape == expected.data.shape
    assert actual.data.stride() == expected.data.stride()
    assert actual.data.dtype == expected.data.dtype
    assert actual.device.type == "meta"


@pytest.mark.parametrize("layout", _FP4_VALUE_LAYOUTS)
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("major_dim", [-2, -1])
@pytest.mark.parametrize("with_out", [False, True])
def test_convert_layout_uses_input_device(layout, inverse, major_dim, with_out):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")

    data = torch.arange(2 * 256 * 257, dtype=torch.int32, device="cpu").to(torch.uint8).reshape(2, 256, 257)
    canonical = wrap_torch_tensor(data, dtype=FP4, shape=[2, 256, 514])
    canonical = convert_layout(canonical, StridedLayout(major_dim))
    source = convert_layout(canonical, layout) if inverse else canonical
    destination = canonical.storage.layout if inverse else layout
    expected = convert_layout(source, destination)

    stream = torch.cuda.Stream(device=1)
    with torch.cuda.stream(stream), torch.cuda.device(0):
        source_cuda = wrap_torch_tensor(source.storage.data.to("cuda:1"), dtype=FP4, shape=source.shape,
                                        layout=source.storage.layout)
        out = wrap_torch_tensor(torch.empty_like(expected.data, device="cuda:1"), dtype=FP4, shape=source.shape,
                                layout=destination) if with_out else None
        actual = convert_layout(source_cuda, destination, out=out)

        if with_out:
            assert actual is out
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream(1) == stream
        assert actual.device == torch.device("cuda:1")
        assert actual.shape == expected.shape
        assert actual.storage.data.stride() == expected.storage.data.stride()
        stream.synchronize()
        assert torch.equal(actual.storage.data.cpu(), expected.storage.data)


@pytest.mark.parametrize("n_slices", [1, 7, 33, 911, 1025])
def test_make_ragged_tensor_metadata(n_slices):
    torch.manual_seed(0)
    device = "cuda"
    max_slice_size = 200
    n_total_rows = max_slice_size * n_slices
    slice_sizes = torch.randint(0, max_slice_size, (n_slices, ), dtype=torch.int32, device=device)
    slice_sizes[torch.randint(0, n_slices, (1, ))] = 0
    meta = make_ragged_tensor_metadata(slice_sizes, n_total_rows)
    ref = make_ragged_tensor_metadata_torch(slice_sizes, n_total_rows)
    assert_equal(meta.slice_sizes, ref.slice_sizes)
    assert_equal(meta.slice_offs, ref.slice_offs)
    assert_equal(meta.block_offs_data, ref.block_offs_data)
    assert_equal(meta.block_schedule_data, ref.block_schedule_data)


@pytest.mark.parametrize("n_slices", [9, 32, 911, 1025])
def test_remap_ragged_tensor_metadata(n_slices):
    device = "cuda"
    max_slice_size = 200
    n_total_rows = max_slice_size * n_slices
    slice_sizes = torch.randint(0, max_slice_size, (n_slices, ), dtype=torch.int32, device=device)
    slice_sizes[torch.randint(0, n_slices, (1, ))] = 0
    # randomly permute slices
    slice_map = torch.randperm(n_slices, device=device, dtype=torch.int32)
    # discard random slices
    slice_map[torch.randint(0, len(slice_map), (5, ))] = -1
    tri_metadata = make_ragged_tensor_metadata(slice_sizes, n_total_rows)
    ref_metadata = make_ragged_tensor_metadata_torch(slice_sizes, n_total_rows)
    tri_metadata = remap_ragged_tensor_metadata(tri_metadata, slice_map)
    ref_metadata = remap_ragged_tensor_metadata_torch(ref_metadata, slice_map)
    assert_equal(tri_metadata.slice_sizes, ref_metadata.slice_sizes)
    assert_equal(tri_metadata.slice_offs, ref_metadata.slice_offs)
    assert_equal(tri_metadata.block_offs_data, ref_metadata.block_offs_data)
    assert_equal(tri_metadata.block_schedule_data, ref_metadata.block_schedule_data)


@pytest.mark.parametrize("n_rows", [0, 7, 256, 17111])
@pytest.mark.parametrize("n_cols", [13, 32, 128, 811])
@pytest.mark.parametrize("k", [1, 3, 4, 7, 8, 12, 18, 33, 63])
def test_make_bitmatrix_metadata(n_rows, n_cols, k):
    if k > n_cols:
        pytest.skip("k must be <= n_cols")
    device = "cuda"
    torch.manual_seed(0)
    # random permutation of column indices
    # NOTE: `indx` *must* be sorted
    indx = torch.rand(n_rows, n_cols, device=device).argsort(dim=1).int()[:, :k]
    indx = torch.sort(indx, dim=1)[0]
    # create bitmask
    rows = torch.arange(n_rows, device=device).unsqueeze(1).expand_as(indx)
    bitmask_data = torch.zeros((n_rows, (n_cols + 31) // 32), dtype=torch.int32, device=device)
    bitmask_data.index_put_((rows, indx // 32), 1 << (indx % 32), accumulate=True)
    bitmask = wrap_torch_tensor(bitmask_data.view(torch.uint32), dtype=BIT, shape=(n_rows, n_cols))
    # make metadata and compare
    metadata_tri = make_bitmatrix_metadata(indx, bitmask)
    metadata_ref = make_bitmatrix_metadata_torch(indx, bitmask)
    assert_equal(metadata_tri.col_sum, metadata_ref.col_sum)
    assert_equal(metadata_tri.row_sorted_indx, metadata_ref.row_sorted_indx)
    assert_equal(metadata_tri.col_sorted_indx, metadata_ref.col_sorted_indx)


@triton.jit(debug=True)
def _keyed_add_scan_kernel(In, Out, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(In + offs)
    y = tl.associative_scan(x, 0, _keyed_add)
    tl.store(Out + offs, y)


def test_keyed_add_large_key_no_int_overflow():
    # Regression test for https://github.com/triton-lang/triton/issues/7945
    # `_keyed_add` accumulates a count in the lower 16 bits of a uint32 under a
    # key stored in the upper 16 bits. With overflow checks live (debug=True),
    # a large key -- e.g. the 0xffff0000 padding sentinel produced for masked
    # lanes -- used to overflow uint32 in `x + y` and abort the kernel with
    # "int32 overflow detected for operation add".
    device = "cuda"
    BLOCK = 16
    key = 0xffff  # the padding sentinel key used by the metadata kernels
    # All elements share `key`, so the inclusive scan accumulates the counts.
    x = torch.full((BLOCK, ), (key << 16) | 1, dtype=torch.uint32, device=device)
    out = torch.empty_like(x)
    # Would raise a device-side assertion before the fix.
    _keyed_add_scan_kernel[(1, )](x, out, BLOCK=BLOCK)
    # Compare in int64 (CUDA has no uint32 `arange`); values are well within int64.
    expected = (key << 16) | torch.arange(1, BLOCK + 1, dtype=torch.int64, device=device)
    assert torch.equal(out.to(torch.int64), expected)
