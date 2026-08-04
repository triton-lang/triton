import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _strided_offsets(offsets, shape: gl.constexpr, strides: gl.constexpr):
    remaining = offsets
    address = offsets * 0
    for axis in gl.static_range(len(shape) - 1, -1, -1):
        address += (remaining % shape[axis]) * strides[axis]
        remaining = remaining // shape[axis]
    return address


@gluon.jit
def _embed_kernel(x_ptr, out_ptr, n_elements, SHAPE: gl.constexpr, X_STRIDES: gl.constexpr, OUT_STRIDES: gl.constexpr,
                  BLOCK_SIZE: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0])
    offsets = gl.program_id(0).to(gl.int64) * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < n_elements
    x_offsets = _strided_offsets(offsets, SHAPE, X_STRIDES)
    out_offsets = _strided_offsets(offsets, SHAPE, OUT_STRIDES)
    x = gl.load(x_ptr + x_offsets, mask=mask)
    gl.store(out_ptr + out_offsets, gl.experimental_fpsan_embed(x), mask=mask)


@gluon.jit
def _unembed_kernel(x_ptr, out_ptr, n_elements, SHAPE: gl.constexpr, X_STRIDES: gl.constexpr, OUT_STRIDES: gl.constexpr,
                    BLOCK_SIZE: gl.constexpr, THREADS_PER_WARP: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0])
    offsets = gl.program_id(0).to(gl.int64) * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < n_elements
    x_offsets = _strided_offsets(offsets, SHAPE, X_STRIDES)
    out_offsets = _strided_offsets(offsets, SHAPE, OUT_STRIDES)
    x = gl.load(x_ptr + x_offsets, mask=mask)
    gl.store(out_ptr + out_offsets, gl.experimental_fpsan_unembed(x, out_ptr.dtype.element_ty), mask=mask)


def _launch(kernel, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    if n_elements:
        with torch.cuda.device(x.device), triton.knobs.compilation.scope():
            triton.knobs.compilation.instrumentation_mode = "fpsan"
            target = triton.runtime.driver.active.get_current_target()
            kernel[(triton.cdiv(n_elements, 256), )](
                x,
                out,
                n_elements,
                SHAPE=tuple(x.shape),
                X_STRIDES=x.stride(),
                OUT_STRIDES=out.stride(),
                BLOCK_SIZE=256,
                THREADS_PER_WARP=target.warp_size,
                supported_fp8_dtypes=tuple(name for name in gl.dtype.FP_TYPES if name.startswith("fp8")),
            )
    return out


def embed(x: torch.Tensor) -> torch.Tensor:
    """Embed a floating-point GPU tensor in its same-width FPSan integer ring."""
    integer_dtype = getattr(torch, f"int{x.element_size() * 8}")
    return _launch(_embed_kernel, x, torch.empty_like(x, dtype=integer_dtype))


def unembed(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Unembed an FPSan integer payload tensor into the requested float dtype."""
    return _launch(_unembed_kernel, x, torch.empty_like(x, dtype=dtype))
