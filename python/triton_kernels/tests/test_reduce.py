import pytest
import torch
from triton.testing import do_bench
from triton_kernels.reduce import reduce, reduce_torch, PostprocessFn, FnSpecs
from triton_kernels.numerics_details.mxfp import upcast_from_mxfp_torch, downcast_to_mxfp_torch
from triton_kernels.numerics import InFlexData, OutFlexData
import triton


def init_mask(mask_mode, B, M, N, device):
    if mask_mode == "none":
        return None
    if mask_mode == "full":
        mask = (torch.rand((B, M, N), device=device) > 0.3).to(torch.int8)
    if mask_mode == "broadcast_b":
        mask = (torch.rand((1, M, N), device=device) > 0.3).to(torch.int8)
    if mask_mode == "broadcast_m":
        mask = (torch.rand((B, 1, N), device=device) > 0.3).to(torch.int8)
    if mask_mode == "broadcast_n":
        mask = (torch.rand((B, M, 1), device=device) > 0.3).to(torch.int8)
    return mask.expand(B, M, N)


def dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    if dtype_str == "float4_e2m1":
        return torch.uint8
    if dtype_str == "float8":
        return torch.float8_e4m3fn
    return getattr(torch, dtype_str)


@triton.jit
def plus_a(x, a):
    return x + a


@pytest.mark.parametrize("B, M, N, postprocess_fn", [
    (311, 384, 384, None),
    (384, 311, 384, None),
    (384, 384, 311, None),
    (512, 512, 512, None),
    (512, 512, 512, "plus_ten"),
    (4, 4, 4, None),
])
@pytest.mark.parametrize("dtype_str", [
    "float16",
    "float32",
    "mxfloat8",
    "flexfloat8",
])
@pytest.mark.parametrize("mask_mode", [
    "none",  # no mask
    "full",  # full-sized mask [B,M,N]
    "broadcast_b",  # broadcast over B: [1,M,N]
    "broadcast_m",  # broadcast over M: [B,1,N]
    "broadcast_n",  # broadcast over N: [B,M,1]
])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_op(B, M, N, dtype_str, dim, mask_mode, postprocess_fn):
    is_hip = triton.runtime.driver.active.get_current_target().backend == "hip"
    is_pre_h100 = torch.cuda.get_device_capability() < (9, 0)
    if (is_hip or is_pre_h100) and "float8" in dtype_str:
        pytest.skip("float8 not supported on CUDA < 9.0")
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn((B, M, N), device=device, dtype=torch.float32)
    x_mscale, x_flex = None, None
    y_flex_tri, y_flex_ref = None, None
    if is_mx := dtype_str.startswith("mx"):
        dtype = dtype_str_to_torch(dtype_str.removeprefix("mx"))
        x, x_mscale = downcast_to_mxfp_torch(x.to(torch.float16), dtype, axis=-1)
    if is_flex := dtype_str.startswith("flex"):
        dtype = dtype_str_to_torch(dtype_str.removeprefix("flex"))
        expected_scale = torch.tensor([4], device=device, dtype=torch.float32)
        x_flex = InFlexData(scale=torch.tensor([2], device=device, dtype=torch.float32))
        x = x / x_flex.scale
        x = x.to(dtype)
        y_flex_tri = OutFlexData(expected_scale=expected_scale, actual_scale=torch.empty_like(expected_scale))
        y_flex_ref = OutFlexData(expected_scale=expected_scale, actual_scale=torch.empty_like(expected_scale))
    mask = init_mask(mask_mode, B, M, N, device)
    expected_exception = ValueError if dim == 2 and is_mx else None
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            reduce(x, dim=dim, mask=mask, x_mxscale=x_mscale)
        return
    if postprocess_fn == "plus_ten":
        postprocess_fn_tri = PostprocessFn(specs=FnSpecs("plus_a", plus_a, ("a", )), fn_args=(10, ))
        postprocess_fn_ref = lambda x: x + 10
    else:
        postprocess_fn_tri = postprocess_fn_ref = None
    y_tri, y_tri_mxscale = reduce(x, dim=dim, mask=mask, x_mxscale=x_mscale, x_flex=x_flex, y_flex=y_flex_tri,
                                  postprocess_fn=postprocess_fn_tri)
    y_ref, y_ref_mxscale = reduce_torch(x, dim=dim, mask=mask, x_mxscale=x_mscale, x_flex=x_flex, y_flex=y_flex_ref,
                                        postprocess_fn=postprocess_fn_ref)
    if is_mx:
        y_ref = upcast_from_mxfp_torch(y_ref, y_ref_mxscale, torch.float16, axis=-1)
        y_tri = upcast_from_mxfp_torch(y_tri, y_tri_mxscale, torch.float16, axis=-1)
    if is_flex:
        torch.allclose(y_flex_tri.actual_scale, y_flex_ref.actual_scale, atol=1e-3, rtol=1e-3)
    assert torch.allclose(y_tri.float(), y_ref.float(), atol=1e-3, rtol=1e-3)


def bench_reduce(B: int = 4, M: int = 4096, N: int = 4096, *, dim: int = 0, dtype: torch.dtype = torch.float16,
                 iters: int = 200, mask_mode: str = "none"):
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn((B, M, N), device=device, dtype=torch.float32).to(dtype)
    mask = init_mask(mask_mode, B, M, N, device)
    ms = do_bench(lambda: reduce(x, dim=dim, mask=mask), rep=iters)
    nnz = x.numel() if mask is None else (mask.expand(B, M, N) != 0).sum()
    read_bytes = nnz * x.element_size()
    out_elems = (M * N) if dim == 0 else ((B * N) if dim == 1 else (B * M))
    write_bytes = out_elems * x.element_size()
    mask_bytes = 0 if mask is None else (mask.numel() * mask.element_size())
    bytes_total = read_bytes + write_bytes + mask_bytes
    gbps = (bytes_total) / ms / 1e6
    desc = f"reduce: B={B}, M={M}, N={N}, dim={dim}, dtype={str(dtype).split('.')[-1]}, mask={mask_mode}"
    print(f"{desc} -> {gbps:.2f} GB/s")


# bench_reduce(B=4, M=8192, N=8192, dim=0, dtype=torch.float16, mask_mode="none")
# bench_reduce(B=8192, M=4, N=8192, dim=1, dtype=torch.float16, mask_mode="broadcast_n")
# bench_reduce(B=8192, M=4, N=8192, dim=1, dtype=torch.float16, mask_mode="broadcast_m")
# bench_reduce(B=8192, M=4, N=8192, dim=1, dtype=torch.float16, mask_mode="broadcast_b")
