import enum
import functools
import os
import subprocess
import sys
import torch
import triton
import triton.language as tl
from triton.language.random import philox
from triton._internal_testing import is_compile_warmup
from triton_kernels.numerics import MAX_FINITE_FLOAT8E4B8, MAX_FINITE_FLOAT8E4NV, MAX_FINITE_FLOAT8E5
from triton_kernels.tensor import convert_layout as _convert_layout, wrap_torch_tensor, FP4, make_ragged_tensor_metadata
from triton_kernels.tensor_details.layout import BlackwellMXScaleLayout
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, MXFP_BLOCK_SIZE, NVFP_BLOCK_SIZE
import itertools
from dataclasses import replace


def assert_equal(ref, tri):
    if is_compile_warmup():
        return
    if isinstance(ref, torch.Tensor):
        assert torch.all(ref == tri)
    else:
        assert ref == tri


def assert_close(ref, tri, maxtol=None, rmstol=None, description="--", verbose=True):
    if is_compile_warmup():
        return
    if tri.dtype.itemsize == 1:
        ref_as_type = ref.to(tri.dtype)
        if ref.dtype == tri.dtype:
            assert torch.all(ref_as_type == tri)
            return
        ref = ref_as_type

    if ref.numel() == 0:
        return

    if maxtol is None:
        maxtol = 2e-2
    if rmstol is None:
        rmstol = 4e-3
    """
    Compare reference values against obtained values.
    """

    # cast to float32:
    ref = ref.to(torch.float32).detach()
    tri = tri.to(torch.float32).detach()
    assert ref.shape == tri.shape, f"Tensors must have same size {ref.shape=} {tri.shape=}"

    # deal with infinite elements:
    inf_mask_ref = torch.isinf(ref)
    inf_mask_tri = torch.isinf(tri)
    assert torch.equal(inf_mask_ref, inf_mask_tri), "Tensor must have same infinite elements"
    refn = torch.where(inf_mask_ref, 0, ref)
    trin = torch.where(inf_mask_tri, 0, tri)

    # normalise so that RMS calculation doesn't overflow:
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn *= multiplier
    trin *= multiplier

    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps

    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    if verbose:
        print("%s maximum relative error = %s (threshold = %s)" % (description, max_err, maxtol))
        print("%s RMS relative error = %s (threshold = %s)" % (description, rms_err, rmstol))

    if max_err > maxtol:
        bad_idxs = torch.nonzero(rel_err > maxtol)
        num_nonzero = bad_idxs.size(0)
        bad_idxs = bad_idxs[:1000]
        print("%d / %d mismatched elements (shape = %s) at coords %s" %
              (num_nonzero, rel_err.numel(), tuple(rel_err.shape), bad_idxs.tolist()))

        bad_idxs = bad_idxs.unbind(-1)
        print("ref values: ", ref[tuple(bad_idxs)].cpu())
        print("tri values: ", tri[tuple(bad_idxs)].cpu())

    assert max_err <= maxtol
    assert rms_err <= rmstol


class ComputeSanitizerTool(enum.Enum):
    MEMCHECK = "memcheck"
    RACECHECK = "racecheck"
    SYNCCHECK = "synccheck"
    INITCHECK = "initcheck"


def compute_sanitizer(**target_kwargs):
    """
    Decorator to run a test with compute sanitizer enabled and pytorch caching allocator disabled,
    to expose potential memory access errors.
    This decorator requires the `request` fixture to be present.
    If `run_sanitizer` argument is present and set to False, the sanitizer is not run.
    Running tests under compute sanitizer requires launching subprocess and is slow,
    so use sparingly
    """

    def decorator(test_fn):

        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            if os.environ.get("SKIP_COMPUTE_SANITIZER") == "1":
                test_fn(*args, **kwargs)
                return

            import psutil

            if target_kwargs.pop("clear_torch_cache", False):
                # If we don't pop clear_torch_cache, it won't pass
                # target_kwargs.items() <= kwargs.items() condition below.
                torch.cuda.empty_cache()
            tools_to_check = target_kwargs.pop("tools_to_check", [ComputeSanitizerTool.MEMCHECK])
            assert isinstance(tools_to_check, list), f"{tools_to_check=}"
            assert all(tool in ComputeSanitizerTool for tool in tools_to_check), (
                f"{(tool for tool in tools_to_check if tool not in ComputeSanitizerTool)=}")

            ppid_name = psutil.Process(os.getppid()).exe()
            run_compute_sanitizer = target_kwargs.items() <= kwargs.items()
            if "run_sanitizer" in kwargs:
                run_compute_sanitizer &= kwargs["run_sanitizer"]
            if run_compute_sanitizer and "compute-sanitizer" not in ppid_name:
                for tool in tools_to_check:
                    path = os.path.realpath(test_fn.__globals__["__file__"])
                    # get path of current file
                    env = {
                        "PATH": os.environ["PATH"],
                        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
                        "TORCH_SHOW_CPP_STACKTRACES": "1",
                        "CUDA_LAUNCH_BLOCKING": "1",
                    }
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
                    assert "request_fixture" in kwargs, (
                        "memcheck'ed test must have a (possibly unused) `request` fixture")
                    test_id = kwargs["request_fixture"].node.callspec.id
                    cmd = f"{path}::{test_fn.__name__}[{test_id}]"
                    cmd = [
                        "compute-sanitizer",
                        "--target-processes=application-only",
                        "--destroy-on-device-error=context",
                        f"--tool={tool.value}",
                        sys.executable,
                        "-m",
                        "pytest",
                        "-vsx",
                        cmd,
                    ]
                    for opt in ["--update_checksum", "--ignore_checksum_error"]:
                        if opt in sys.argv:
                            cmd.append(opt)
                    out = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                    sanitizer_ok = "ERROR SUMMARY: 0 errors" in str(
                        out.stdout) or "RACECHECK SUMMARY: 0 hazards displayed" in str(out.stdout)
                    test_output = out.stdout
                    if type(test_output) is bytes:
                        test_output = test_output.decode()

                    fail = False
                    if not sanitizer_ok:
                        print("compute-sanitizer returned an error")
                        fail = True
                    elif out.returncode != 0:
                        print(
                            "The test failed due to some other reason: consider running without compute-sanitizer to verify."
                        )
                        print(f"{out.returncode=}")
                        fail = True

                    if fail:
                        print("*****************************************************")
                        print("******************** TEST OUTPUT ********************")
                        print("*****************************************************")
                        print(test_output)
                        print("*****************************************************")
                        print("****************** TEST OUTPUT END ******************")
                        print("*****************************************************")
                        assert None
            else:
                test_fn(*args, **kwargs)

        return wrapper

    return decorator


def compute_actual_scale(x, dtype, per_batch_scale=False):
    max_finite = {
        torch.float8_e5m2: MAX_FINITE_FLOAT8E5,
        torch.float8_e4m3fn: MAX_FINITE_FLOAT8E4NV,
        torch.float8_e4m3fnuz: MAX_FINITE_FLOAT8E4B8,
    }[dtype]
    maxvals = x.abs().amax(dim=tuple(range(1, x.ndim))) if per_batch_scale else x.abs().max()
    return maxvals / max_finite


# --- create tensor ---


@triton.jit(do_not_specialize=["seed", "offset", "rows", "cols", "row_blocks", "col_blocks", "num_elements"])
def _random_block_signs(output, seed: tl.uint64, offset: tl.uint64, rows: tl.int32, cols: tl.int32,
                        row_blocks: tl.int32, col_blocks: tl.int32, block_size: tl.constexpr, num_elements: tl.int32,
                        TILE_SIZE: tl.constexpr):
    indices = tl.program_id(0) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    block = indices // block_size
    lane = indices % block_size
    row_block = (block // col_blocks) % row_blocks
    col_block = block % col_blocks
    random_count = tl.maximum(tl.minimum(block_size, rows - row_block * block_size),
                              tl.minimum(block_size, cols - col_block * block_size))
    counter = offset // 4 + block.to(tl.uint64)
    value, _, _, _ = philox(seed, counter.to(tl.uint32), (counter >> 32).to(tl.uint32), lane.to(tl.uint32),
                            tl.zeros_like(lane).to(tl.uint32))
    tl.store(output + indices, tl.where(lane < random_count, value & 1, 0).to(tl.int64), mask=indices < num_elements)


def _make_random_block_signs(x, batch_size, rows, cols, row_blocks, col_blocks, block_size):
    signs = torch.empty((batch_size, row_blocks, col_blocks, block_size), device=x.device, dtype=torch.int64)
    warming = is_compile_warmup()
    if warming:
        seed, offset = 0, 0
    else:
        index = x.device.index if x.device.index is not None else torch.cuda.current_device()
        generator = torch.cuda.default_generators[index]
        seed, offset = generator.initial_seed(), generator.get_offset()

    _random_block_signs[(triton.cdiv(signs.numel(), 512), )](signs, seed, offset, rows, cols, row_blocks, col_blocks,
                                                             block_size, signs.numel(), TILE_SIZE=512)
    if not warming:
        generator.set_offset(offset + 4 * batch_size * row_blocks * col_blocks)
    return signs


def normalize_blocks(x, BLOCK_SIZE=None):
    if BLOCK_SIZE is None:
        BLOCK_SIZE = int(MXFP_BLOCK_SIZE)
    if x.numel() == 0:
        return x

    batched = x.unsqueeze(0) if x.ndim == 2 else x
    batch_size, rows, cols = batched.shape
    row_padding = (-rows) % BLOCK_SIZE
    col_padding = (-cols) % BLOCK_SIZE
    padded = torch.nn.functional.pad(batched, (0, col_padding, 0, row_padding))
    row_blocks, col_blocks = padded.shape[-2] // BLOCK_SIZE, padded.shape[-1] // BLOCK_SIZE
    blocks = padded.view(batch_size, row_blocks, BLOCK_SIZE, col_blocks, BLOCK_SIZE).permute(0, 1, 3, 2, 4)

    block_maxima = blocks.abs().amax(dim=(-2, -1))
    if torch.version.hip is None:
        signs = _make_random_block_signs(x, batch_size, rows, cols, row_blocks, col_blocks, BLOCK_SIZE)
    else:
        random_signs = []
        for _, row_start, col_start in itertools.product(range(batch_size), range(0, rows, BLOCK_SIZE),
                                                         range(0, cols, BLOCK_SIZE)):
            count = max(min(BLOCK_SIZE, rows - row_start), min(BLOCK_SIZE, cols - col_start))
            signs = torch.randint(0, 2, (count, ), device=x.device)
            if count < BLOCK_SIZE:
                signs = torch.nn.functional.pad(signs, (0, BLOCK_SIZE - count))
            random_signs.append(signs)
        signs = torch.stack(random_signs).view(batch_size, row_blocks, col_blocks, BLOCK_SIZE)
    signs = signs * 2 - 1

    block_heights = (rows - torch.arange(row_blocks, device=x.device) * BLOCK_SIZE).clamp(max=BLOCK_SIZE)
    block_widths = (cols - torch.arange(col_blocks, device=x.device) * BLOCK_SIZE).clamp(max=BLOCK_SIZE)
    height = block_heights[None, :, None, None, None]
    width = block_widths[None, None, :, None, None]
    row = torch.arange(BLOCK_SIZE, device=x.device)[None, None, None, :, None]
    col = torch.arange(BLOCK_SIZE, device=x.device)[None, None, None, None, :]

    diagonal = (row == col) & (row < height) & (col < width)
    extra_row = (width > height) & (row == height - 1) & (col >= height) & (col < width)
    extra_col = (height > width) & (col == width - 1) & (row >= width) & (row < height)
    sign_values = torch.where(width > height, signs[..., None, :], signs[..., :, None])
    replacements = sign_values * block_maxima[..., None, None]
    blocks.copy_(torch.where(diagonal | extra_row | extra_col, replacements, blocks))
    batched.copy_(padded[:, :rows, :cols])
    return x


def alloc_rand(shape, device, dtype, requires_grad=False):
    if is_compile_warmup():
        result = torch.empty(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        if dtype.itemsize != 1 and result.numel() and torch.version.hip is None:
            batch_size, rows, cols = (1, *result.shape) if result.ndim == 2 else result.shape
            block_size = int(MXFP_BLOCK_SIZE)
            _make_random_block_signs(result, batch_size, rows, cols, triton.cdiv(rows, block_size),
                                     triton.cdiv(cols, block_size), block_size)
        return result
    if dtype.itemsize == 1:
        tmp = 2**-(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype).requires_grad_(requires_grad)
    ret = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    ret = normalize_blocks(ret)
    return ret


def _make_slice_sizes(n_slices, total_size, device, generator=None):
    dtype = torch.int32
    if total_size < 0:
        raise ValueError("total_size must be non-negative")
    if n_slices <= 0:
        return torch.zeros((0, ), dtype=dtype, device=device)
    if total_size == 0:
        return torch.zeros((n_slices, ), dtype=dtype, device=device)
    # always set one slice size to zero
    probs = torch.ones(n_slices, device=device) / n_slices
    if n_slices > 1:
        probs[2] += probs[1]
        probs[1] = 0.
    assignments = torch.multinomial(probs, total_size, replacement=True, generator=generator)
    counts = torch.bincount(assignments, minlength=n_slices).to(dtype)
    assert counts.sum().item() == total_size
    assert len(counts) == n_slices
    return counts


def make_slice_sizes(n_slices, total_size, device="cuda"):
    if is_compile_warmup():
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            generator = torch.Generator(device="cpu").manual_seed(0)
            values = _make_slice_sizes(n_slices, total_size, "cpu", generator).tolist()
        result = torch.empty((max(n_slices, 0), ), dtype=torch.int32, device=device)
        result._triton_warmup_values = values
        return result

    torch.manual_seed(0)
    return _make_slice_sizes(n_slices, total_size, device)


def pad_rows_to_multiples(A, indices, multiple=128, pad_value=float('nan')):
    """
    Insert padding so that each row A[i] (for i in indices)
    appears at an output row index that is a multiple of `multiple`.
    """
    D = A.size(1)
    out = []
    for i_cur, i_next in zip(indices[:-1], indices[1:]):
        size = (i_next - i_cur)
        size_padded = ((size + multiple - 1) // multiple) * multiple
        cur = torch.full((size_padded, D), pad_value, dtype=A.dtype, device=A.device)
        cur[:size, :] = A[i_cur:i_next, :]
        out.append(cur)
    return torch.vstack(out)


def pad_ragged_tensor(x, x_ragged_metadata, hbm_swizzling, transpose):
    multiple = 128 if hbm_swizzling else 64
    if is_compile_warmup():
        dimension = 1 if transpose else 0
        shape = list(x.shape)
        values = x_ragged_metadata.slice_sizes._triton_warmup_values
        shape[dimension] = sum((value + multiple - 1) // multiple * multiple for value in values)
        y = torch.empty(shape, dtype=x.dtype, device=x.device)
    elif transpose:
        y = pad_rows_to_multiples(x.T, x_ragged_metadata.slice_offs, multiple=multiple, pad_value=0).T.contiguous()
    else:
        y = pad_rows_to_multiples(x, x_ragged_metadata.slice_offs, multiple=multiple, pad_value=0).contiguous()

    y_ragged_metadata = replace(x_ragged_metadata, slice_offs=x_ragged_metadata.block_offs(multiple) * multiple,
                                slice_sizes_divisibility=multiple)
    return y, y_ragged_metadata


class _WarmupTensorProxy:

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def convert_layout(tensor, layout, **layout_transformation_kwargs):
    converted = _convert_layout(tensor, layout, **layout_transformation_kwargs)
    if not is_compile_warmup() or converted is tensor:
        return converted

    data = tensor.storage.data
    if (not isinstance(layout, BlackwellMXScaleLayout) or data.device.type in ["cpu", "meta"]
            or data.dtype.itemsize != 1 or not converted.storage.data.numel()):
        return converted

    transformation = layout.make_transformation(tensor.shape, tensor.dtype == FP4, **layout_transformation_kwargs)
    data = torch.empty(tensor.shape, dtype=data.dtype, device=data.device)
    transformation.swizzle_data(_WarmupTensorProxy(data))
    return converted


def make_random_tensor(shape, n_slices, ragged_dim, ragged_padding, device, dtype, mxfp_dim, transpose,
                       squeeze_batch_dim, is_mx_rowmajor=False, value_hbm_swizzling=None, scale_hbm_swizzling=None):
    # allocate buffer
    buffer_shape = ((n_slices, ) if ragged_dim is None else tuple()) + shape
    buffer_dtype = torch.bfloat16 if dtype.has_mx_scale else dtype.torch_dtype
    buffer = alloc_rand(buffer_shape, device=device, dtype=buffer_dtype)
    if squeeze_batch_dim:
        buffer = buffer.squeeze(0)
    # handle raggedness
    ragged_metadata = None
    if ragged_dim is not None:
        slice_sizes = make_slice_sizes(n_slices, shape[ragged_dim], device=device)
        ragged_metadata = make_ragged_tensor_metadata(slice_sizes, shape[ragged_dim])
    if ragged_padding:
        buffer, ragged_metadata = pad_ragged_tensor(buffer, ragged_metadata, value_hbm_swizzling is not None
                                                    or scale_hbm_swizzling is not None, ragged_dim == 1)
    # handle transpose
    if transpose:
        buffer = buffer.mT.contiguous().mT
    # handle mxfp
    scales = None
    if mxfp_dim is not None:
        assert dtype.has_mx_scale
        is_nvfp4 = getattr(dtype, "is_nvfp4", False)
        scale_dtype = torch.float8_e4m3fn if is_nvfp4 else torch.uint8
        microblock_size = NVFP_BLOCK_SIZE.value if is_nvfp4 else MXFP_BLOCK_SIZE.value
        if dtype.is_mxfloat4:
            logical_shape = list(buffer.shape)
        if is_mx_rowmajor:
            scales = downcast_to_mxfp(
                buffer,
                dtype.torch_dtype,
                axis=mxfp_dim,
                scale_dtype=scale_dtype,
                microblock_size=microblock_size,
            )[1]
            buffer = downcast_to_mxfp(
                buffer.mT.contiguous(),
                dtype.torch_dtype,
                axis=mxfp_dim,
                scale_dtype=scale_dtype,
                microblock_size=microblock_size,
            )[0].mT
        else:
            buffer, scales = downcast_to_mxfp(
                buffer,
                dtype.torch_dtype,
                axis=mxfp_dim,
                scale_dtype=scale_dtype,
                microblock_size=microblock_size,
            )
        buffer = wrap_torch_tensor(
            buffer,
            FP4 if dtype.is_mxfloat4 else None,
            shape=logical_shape if dtype.is_mxfloat4 else None,
        )
        scales = wrap_torch_tensor(scales)
        if value_hbm_swizzling is not None:
            # convert buffer to swizzled hbm layout
            buffer = convert_layout(buffer, value_hbm_swizzling)
        if scale_hbm_swizzling is not None:
            # hack to avoid circular dependency
            if callable(scale_hbm_swizzling):
                # Segment metadata describes scale rows, never its inner axis.
                scale_hbm_swizzling = scale_hbm_swizzling(ragged_metadata if ragged_dim == 0 else None)
            scales = convert_layout(scales, scale_hbm_swizzling)
    return buffer, scales, ragged_metadata
