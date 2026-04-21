import sys
import importlib.util
import torch
import triton
import triton.language as tl
import pytest
from triton.tools.tensor_descriptor import TensorDescriptor

from triton.tools.triton_to_gluon_translator.translator import convert_triton_to_gluon
from triton.tools.triton_to_gluon_translator.target import TranslatorTarget
from triton._internal_testing import is_blackwell, is_hopper_or_newer, is_cuda, is_hip_cdna4, is_hip_gfx1250, is_hip_cdna3_or_newer
from triton.language.target_info import current_target


def _convert_host_descriptor(desc):
    """Import and call the target-appropriate convert_host_descriptor."""
    target = current_target()
    if target is not None and target.backend == "hip":
        from triton.tools.triton_to_gluon_translator.amd_helpers import convert_host_descriptor
    else:
        from triton.tools.triton_to_gluon_translator.nvidia_helpers import convert_host_descriptor
    return convert_host_descriptor(desc)


def convert_kernel(kernel, kernel_name, tmp_path, target=None):
    if target is None:
        t = current_target()
        target = TranslatorTarget("nvidia" if t.backend == "cuda" else t.arch)
    converted = convert_triton_to_gluon([kernel], target=target)

    # Write converted kernel to a file so @gluon.jit can retrieve source
    mod_path = tmp_path / "converted_kernel.py"
    mod_path.write_text(converted)

    spec = importlib.util.spec_from_file_location("converted_kernel", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["converted_kernel"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    kernel = getattr(module, kernel_name)
    return kernel


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


def test_simple_kernel(tmp_path):
    kernel = convert_kernel(add_kernel, "add_kernel", tmp_path)

    n = 1024
    BLOCK = 128
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (n // BLOCK, )
    kernel[grid](x, y, out, n, BLOCK)

    ref = torch.empty_like(x)
    add_kernel[grid](x, y, ref, n, BLOCK)

    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@triton.jit
def impl_matmul_tile_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    acc = tl.zeros((M, N), dtype=tl.float32)
    a = tl.load(a_ptr + offs_m * K + (tl.arange(0, K))[None, :])
    b = tl.load(b_ptr + (tl.arange(0, K))[:, None] * N + offs_n)
    acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m * N + offs_n, acc)


@triton.jit
def matmul_tile_kernel(a_ptr, b_ptr, c_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    impl_matmul_tile_kernel(a_ptr, b_ptr, c_ptr, BLOCK_M, BLOCK_N, BLOCK_K)


def test_triton_to_gluon_dot_minimal(tmp_path):
    if not (is_blackwell() or is_hip_cdna3_or_newer() or is_hip_gfx1250()):
        pytest.skip("Requires Blackwell, CDNA3+, or gfx1250")
    kernel = convert_kernel(matmul_tile_kernel, "matmul_tile_kernel", tmp_path)
    M, N, K = 128, 128, 128
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    grid = (1, )

    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    kernel[grid](a, b, c, M, N, K, num_warps=8)

    ref = torch.empty_like(c)
    matmul_tile_kernel[grid](a, b, ref, M, N, K, num_warps=8)
    torch.testing.assert_close(c, ref, atol=0, rtol=0)


@triton.jit
def matmul_kernel(  #
    a_ptr,
    b_ptr,
    output_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), step=1, num_stages=4):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator, out_dtype=output_ptr.dtype.element_ty)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator)


@pytest.mark.parametrize("dtype_src_str", ["float16"])
@pytest.mark.parametrize("dtype_dst_str", ["float32"])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES", [(128, 128, 64, 1)])
@pytest.mark.parametrize("NUM_WARPS", [4])
def test_simple_matmul(dtype_src_str, dtype_dst_str, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_WARPS, tmp_path):
    if not (is_blackwell() or is_hip_cdna4() or is_hip_gfx1250()):
        pytest.skip("Requires Blackwell, CDNA4, or gfx1250")
    device = "cuda"
    M, N, K = 1024, 512, 256
    torch.manual_seed(42)
    dtype_src_str = "float32" if dtype_src_str == "tensorfloat32" else dtype_src_str
    dtype_src = getattr(torch, dtype_src_str)

    kernel = convert_kernel(matmul_kernel, "matmul_kernel", tmp_path)

    a = torch.randn(M, K, dtype=dtype_src, device=device)
    b = torch.randn(K, N, dtype=dtype_src, device=device)
    dtype_dst = getattr(torch, dtype_dst_str)
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    kernel[grid](a, b, output, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                 output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K)

    ref = torch.empty_like(output)
    matmul_kernel[grid](a, b, ref, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                        output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K)
    torch.testing.assert_close(output, ref, atol=0, rtol=0)


@triton.jit
def descriptor_store_kernel(desc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, V: tl.constexpr):
    tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16) + V
    desc.store([0, 0], tile)


def _skip_unless_descriptor_target():
    if is_cuda() and not is_hopper_or_newer():
        pytest.skip("Requires Hopper+")
    elif not is_cuda() and not is_hip_gfx1250():
        pytest.skip("Requires descriptor support")


def test_triton_to_gluon_descriptor_roundtrip(tmp_path):
    _skip_unless_descriptor_target()
    kernel = convert_kernel(descriptor_store_kernel, "descriptor_store_kernel", tmp_path)

    M = N = 64
    y = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    grid = (1, )
    block_shape = [M, N]
    desc = TensorDescriptor(y, y.shape, y.stride(), block_shape)
    gluon_desc = _convert_host_descriptor(desc)
    kernel[grid](gluon_desc, M, N, 1.0)

    y_ref = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    desc_ref = TensorDescriptor(y_ref, y_ref.shape, y_ref.stride(), block_shape)
    descriptor_store_kernel[grid](desc_ref, M, N, 1.0)
    torch.testing.assert_close(y, y_ref, atol=0, rtol=0)


@triton.jit
def descriptor_copy_kernel(in_desc, out_desc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    tile = in_desc.load([0, 0])
    out_desc.store([0, 0], tile)


def test_triton_to_gluon_descriptor_load_roundtrip(tmp_path):
    _skip_unless_descriptor_target()
    kernel = convert_kernel(descriptor_copy_kernel, "descriptor_copy_kernel", tmp_path)

    M = N = 64
    x = torch.ones((M, N), device="cuda", dtype=torch.float16) * 3.0
    y = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    grid = (1, )
    block_shape = [M, N]

    in_desc = TensorDescriptor(x, x.shape, x.stride(), block_shape)
    gluon_desc = _convert_host_descriptor(in_desc)
    out_desc = _convert_host_descriptor(TensorDescriptor(y, y.shape, y.stride(), block_shape))
    kernel[grid](gluon_desc, out_desc, M, N)

    y_ref = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    desc_ref = TensorDescriptor(y_ref, y_ref.shape, y_ref.stride(), block_shape)
    descriptor_copy_kernel[grid](in_desc, desc_ref, M, N)
    torch.testing.assert_close(y, y_ref, atol=0, rtol=0)


@triton.jit
def make_tensor_descriptor_copy_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    out_desc = tl.make_tensor_descriptor(
        y_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    tile = in_desc.load([0, 0])
    out_desc.store([0, 0], tile)


def test_triton_to_gluon_make_tensor_descriptor(tmp_path, with_allocator):
    _skip_unless_descriptor_target()
    kernel = convert_kernel(make_tensor_descriptor_copy_kernel, "make_tensor_descriptor_copy_kernel", tmp_path)

    M = N = 64
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)
    y = torch.zeros_like(x)
    grid = (1, )

    kernel[grid](x, y, M, N, M, N)

    torch.testing.assert_close(y, x, atol=0, rtol=0)


@triton.jit
def reshape_trans_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, TRANS_KIND: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    x = tl.reshape(tl.load(x_ptr + offsets), 16, 16)
    y = tl.load(y_ptr + offsets).reshape(16, 16)
    if TRANS_KIND == "trans_method":
        a = x + y.trans(1, 0)
    elif TRANS_KIND == "tl_trans_separate":
        a = x + tl.trans(y, 1, 0)
    elif TRANS_KIND == "tl_trans_tuple":
        a = x + tl.trans(y, (1, 0))
    elif TRANS_KIND == "tl_trans":
        a = x + tl.trans(y)
    a = a.reshape(256)
    tl.store(out_ptr + offsets, a)


@pytest.mark.parametrize("TRANS_KIND", ["trans_method", "tl_trans_separate", "tl_trans_tuple", "tl_trans"])
def test_triton_reshape_trans(tmp_path, TRANS_KIND):
    kernel = convert_kernel(reshape_trans_kernel, "reshape_trans_kernel", tmp_path)

    n = 1024
    BLOCK = 256
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (n // BLOCK, )
    kernel[grid](x, y, out, n, BLOCK, TRANS_KIND)
    ref = torch.empty_like(x)
    reshape_trans_kernel[grid](x, y, ref, n, BLOCK, TRANS_KIND)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


BLOCK_SPLIT = tl.constexpr(256)


@triton.jit
def split_kernel(x_ptr, out_ptr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SPLIT + tl.arange(0, BLOCK_SPLIT)
    offsets2 = pid * BLOCK_SPLIT + tl.arange(0, 2 * BLOCK_SPLIT)

    s0, s1 = tl.reshape(tl.load(x_ptr + offsets2), BLOCK_SPLIT, 2).split()
    a = s0 + s1
    p = out_ptr + offsets
    tl.store(p, a)


def test_split(tmp_path):
    kernel = convert_kernel(split_kernel, "split_kernel", tmp_path)

    n = 1024
    x = torch.randn(2 * n, device="cuda", dtype=torch.float32)
    grid = (n // BLOCK_SPLIT, )

    out = torch.empty_like(x[:n])
    kernel[grid](x, out)
    ref = torch.empty_like(x[:n])
    split_kernel[grid](x, ref)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@triton.jit
def cat_translation_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, CAN_REORDER: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    z = tl.cat(x, y, can_reorder=CAN_REORDER)
    tl.store(out_ptr + tl.arange(0, 2 * BLOCK), z)


@pytest.mark.parametrize("can_reorder", [False, True])
@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_cat_translation(tmp_path, can_reorder):
    kernel = convert_kernel(cat_translation_kernel, "cat_translation_kernel", tmp_path)

    block = 128
    x = torch.arange(0, block, device="cuda", dtype=torch.int32)
    y = torch.arange(-block, 0, device="cuda", dtype=torch.int32)
    out = torch.empty((2 * block, ), device="cuda", dtype=torch.int32)

    kernel[(1, )](x, y, out, BLOCK=block, CAN_REORDER=can_reorder, num_warps=4)

    ref = torch.cat([x, y], dim=0)
    if can_reorder:
        torch.testing.assert_close(torch.sort(out).values, torch.sort(ref).values, atol=0, rtol=0)
    else:
        torch.testing.assert_close(out, ref, atol=0, rtol=0)


@triton.jit
def reduce_to_scalar_kernel(out_ptr):
    x = tl.arange(0, 16)
    x = tl.sum(x)
    tl.store(out_ptr, x)


def test_reduce_to_scalar(tmp_path):
    kernel = convert_kernel(reduce_to_scalar_kernel, "reduce_to_scalar_kernel", tmp_path)
    grid = (1, )

    out = torch.empty((1, ), device="cuda", dtype=torch.int32)
    kernel[grid](out)
    ref = torch.empty_like(out)
    reduce_to_scalar_kernel[grid](ref)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@triton.jit
def extrema_reduce_kernel(x_ptr, max_ptr, min_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    x = tl.reshape(x, BLOCK // 2, 2)
    tl.store(max_ptr + tl.arange(0, BLOCK // 2), tl.max(x, axis=1))
    tl.store(min_ptr + tl.arange(0, BLOCK // 2), tl.min(x, axis=1))


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_extrema_reduction(tmp_path):
    kernel = convert_kernel(extrema_reduce_kernel, "extrema_reduce_kernel", tmp_path)

    block = 256
    x = torch.randn(block, device="cuda", dtype=torch.float32)
    out_max = torch.empty((block // 2, ), device="cuda", dtype=torch.float32)
    out_min = torch.empty((block // 2, ), device="cuda", dtype=torch.float32)
    kernel[(1, )](x, out_max, out_min, BLOCK=block)

    ref_max = torch.empty_like(out_max)
    ref_min = torch.empty_like(out_min)
    extrema_reduce_kernel[(1, )](x, ref_max, ref_min, BLOCK=block)
    torch.testing.assert_close(out_max, ref_max, atol=0, rtol=0)
    torch.testing.assert_close(out_min, ref_min, atol=0, rtol=0)


@triton.jit
def num_threads_kernel(out_ptr):
    num_threads: tl.constexpr = tl.extra.cuda.num_threads()
    offs = tl.arange(0, num_threads)
    tl.store(out_ptr + offs, 1)


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_num_threads(tmp_path):
    kernel = convert_kernel(num_threads_kernel, "num_threads_kernel", tmp_path)

    num_threads = 256
    out = torch.empty(num_threads, dtype=torch.int32, device="cuda")
    kernel[(1, )](out, num_warps=num_threads // 32)
    ref = torch.empty_like(out)
    num_threads_kernel[(1, )](ref, num_warps=num_threads // 32)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@triton.jit
def atomic_add_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    scalar_mask = True
    tl.atomic_add(out_ptr + idx, idx, mask=scalar_mask, sem="release", scope="cta")


def test_atomic_add(tmp_path):
    kernel = convert_kernel(atomic_add_kernel, "atomic_add_kernel", tmp_path)

    block = 32 * 4
    ref = torch.zeros((block, ), device="cuda")
    atomic_add_kernel[(1, )](ref, BLOCK=block)

    out = torch.zeros((block, ), device="cuda")
    kernel[(1, )](out, BLOCK=block)
    torch.testing.assert_close(out, ref)


# ---- additional op coverage ----


@triton.jit
def cat_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    z = tl.cat(x, y, can_reorder=True)
    tl.store(out_ptr + tl.arange(0, 2 * BLOCK), z)


def test_cat(tmp_path):
    kernel = convert_kernel(cat_kernel, "cat_kernel", tmp_path)

    BLOCK = 256
    x = torch.randn(BLOCK, device="cuda", dtype=torch.float32)
    y = torch.randn(BLOCK, device="cuda", dtype=torch.float32)
    out = torch.empty(2 * BLOCK, device="cuda", dtype=torch.float32)
    kernel[(1, )](x, y, out, BLOCK)

    ref = torch.empty_like(out)
    cat_kernel[(1, )](x, y, ref, BLOCK)
    torch.testing.assert_close(sorted(out.cpu()), sorted(ref.cpu()), atol=0, rtol=0)


@triton.jit
def gather_scatter_roundtrip_kernel(out_ptr, in_ptr, idx_ptr, X: tl.constexpr, Y: tl.constexpr, BLOCK_X: tl.constexpr,
                                    BLOCK_Y: tl.constexpr):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
    in_desc = tl.make_tensor_descriptor(in_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
    out_desc = tl.make_tensor_descriptor(out_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
    data = in_desc.gather(idx, 0)
    out_desc.scatter(data, idx, 0)


# TODO: parametrize over _descriptor_targets once NVIDIA gather/scatter translation is supported.
# The translator currently routes gather/scatter to AMD-specific helpers for AMD targets only.
@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires gfx1250")
def test_gather_scatter_roundtrip(tmp_path):
    kernel = convert_kernel(gather_scatter_roundtrip_kernel, "gather_scatter_roundtrip_kernel", tmp_path,
                            target=TranslatorTarget.GFX1250)

    X, Y, BLOCK_X, BLOCK_Y = 64, 64, 8, 64
    inp = torch.arange(X * Y, device="cuda", dtype=torch.float16).reshape(X, Y)
    idx = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device="cuda", dtype=torch.int32)
    out = torch.zeros((X, Y), device="cuda", dtype=torch.float16)
    kernel[(1, )](out, inp, idx, X, Y, BLOCK_X, BLOCK_Y)

    expected = torch.zeros_like(out)
    for i, row in enumerate(idx.tolist()):
        expected[row] = inp[row]
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
