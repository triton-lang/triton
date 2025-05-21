import torch

import triton
import triton.language as tl
import pytest

import pathlib
import uuid
from triton._internal_testing import is_cuda


def do_bench(kernel_call, quantiles, use_cuda_graph=False):
    if use_cuda_graph:
        return triton.testing.do_bench_cudagraph(kernel_call, quantiles=quantiles)
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)


@pytest.mark.parametrize('use_cuda_graph', [False, True])
def test_kwargs(use_cuda_graph: bool, device: str):
    if use_cuda_graph and not torch.cuda.is_available():
        pytest.xfail("CUDA is not available")

    M, N = 1024, 16
    src = torch.randn(M * N, device=device)
    dst = torch.empty(M * N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

    @triton.autotune(configs=configs, key=["M"],
                     do_bench=lambda kernel, quantiles: do_bench(kernel, quantiles, use_cuda_graph))
    @triton.jit
    def _kernel(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
        offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :])
        tl.store(dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']), )
    _kernel[grid](dst, src, N, M, N)
    # the key word args could be in arbitrary order.
    _kernel[grid](dst=dst, src=src, M=M // 2, stride_m=N, BLOCK_SIZE_N=N)
    assert len(_kernel.cache) == 2


def test_no_do_bench(device: str):
    M, N = 1024, 16
    src = torch.randn(M * N, device=device)
    dst = torch.empty(M * N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

    @triton.autotune(configs=configs, key=["M"])
    @triton.jit
    def _kernel(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
        offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :])
        tl.store(dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']), )
    _kernel[grid](dst, src, N, M, N)
    assert len(_kernel.cache) == 1


@pytest.mark.parametrize('pass_kwargs_to_kernel', [False, True])
def test_restore(pass_kwargs_to_kernel, device):
    N = 1024
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    if pass_kwargs_to_kernel:
        _kernel[grid](src=src, N=N)
    else:
        _kernel[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))


def test_hooks(device):
    # Autotuner's pre- and post- hooks should be called the same number of times
    N = 4096
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 4096}), triton.Config(kwargs={'BLOCK_SIZE': 32})]

    values = {"counter": 0, "has_exception": False}

    def _pre_hook(*args, **kwargs):
        values["counter"] += 1

    def _post_hook(*args, exception):
        values["counter"] -= 1
        if exception is not None:
            values["has_exception"] = True
        assert values["counter"] == 0

    @triton.autotune(configs=configs, key=['N'], do_bench=do_bench, pre_hook=_pre_hook, post_hook=_post_hook)
    @triton.heuristics({"N_STAGES": lambda nargs: 100 if nargs['N'] == 4096 else 4})
    @triton.jit
    def _kernel(src, N, N_STAGES: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        max_iters = tl.cdiv(N, BLOCK_SIZE)
        for _ in tl.range(max_iters, num_stages=N_STAGES):
            x = tl.load(src + offsets, mask=offsets < N)
            tl.store(src + offsets, x, mask=offsets < N)
            offsets += BLOCK_SIZE

    _kernel[(1, )](src, N)

    # On NVIDIA GPUs:
    # The tuning knob `num_stages` can be set by users.
    # This will cause out of resources when N_STAGES = 100
    # shared memory bytes = N_STAGES * BLOCK_SIZE * sizeof(float)
    # On AMD GPUs:
    # `num_stages` is a fixed value of 2, so it won't cause out of resources
    if triton.runtime.driver.active.get_current_target().backend == "cuda":
        assert values["has_exception"] is True
    else:
        assert values["has_exception"] is False


@pytest.mark.parametrize('with_perf_model', [False, True])
def test_prune_configs(with_perf_model: bool, device: str):
    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)
    records = {}

    def early_config_prune(configs, named_args, **kwargs):
        records['run_early_config_prune'] = True
        if "N" in kwargs and kwargs["N"] == 1024:
            records['capture_kwargs'] = True
        if "dst" in named_args and "src" in named_args and len(named_args) == 2:
            records['capture_named_args'] = True
        return [configs[0]]

    def perf_model(*args, **kwargs):
        records['run_perf_model'] = True
        return kwargs['BLOCK_SIZE']

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    if with_perf_model:
        prune_configs_by = {'perf_model': perf_model, 'top_k': 1}
    else:
        prune_configs_by = {'early_config_prune': early_config_prune}

    @triton.autotune(configs=configs, key=['N'], prune_configs_by=prune_configs_by, do_bench=do_bench)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](dst, src, N=N)
    torch.testing.assert_close(src, dst)
    if with_perf_model:
        assert len(records) == 1
        assert records['run_perf_model']
    else:
        assert len(records) == 3
        assert records['run_early_config_prune']
        assert records['capture_kwargs']
        assert records['capture_named_args']


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
                    reason="Requires compute capability >= 9 for NV")
def test_override_ttir(device):
    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)

    ir_src = r"""
module {
  tt.func public @_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+01> : tensor<32xf32>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = arith.mulf %9, %cst : tensor<32xf32>
    %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10, %6 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
    """
    temp_file = pathlib.Path(f"/tmp/test_override_{str(uuid.uuid4())}.ttir")
    temp_file.write_text(ir_src)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32, 'ir_override': str(temp_file)})]

    @triton.autotune(configs=configs, key=['N'], do_bench=do_bench)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](dst, src, N=N)

    # Change the behavior of kernel by overriding PTX
    torch.testing.assert_close(src * 10, dst)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9,
                    reason="Requires compute capability >= 9 for NV")
def test_override_ttgir(device):
    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)

    ir_src = r"""
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+01> : tensor<32xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<32xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<32xi32, #blocked>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32, #blocked>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>, #blocked>
    %10 = arith.mulf %9, %cst : tensor<32xf32, #blocked>
    %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %12, %10, %6 : tensor<32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
    """
    temp_file = pathlib.Path(f"/tmp/test_override_{str(uuid.uuid4())}.ttgir")
    temp_file.write_text(ir_src)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32, 'ir_override': str(temp_file)})]

    @triton.autotune(configs=configs, key=['N'], do_bench=do_bench)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](dst, src, N=N)

    # Change the behavior of kernel by overriding PTX
    torch.testing.assert_close(src * 10, dst)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 9,
                    reason="PTX file in this unit test is only for SM90")
def test_override_ptx(device):
    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)

    ir_src = r"""
//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_90a
.address_size 64

	// .globl	_kernel                 // -- Begin function _kernel
                                        // @_kernel
.visible .entry _kernel(
	.param .u64 .ptr .global .align 1 _kernel_param_0,
	.param .u64 .ptr .global .align 1 _kernel_param_1,
	.param .u32 _kernel_param_2,
	.param .u64 .ptr .global .align 1 _kernel_param_3
)
.reqntid 128
{
	.reg .pred 	%p<4>;
	.reg .b32 	%r<10>;
	.reg .b32 	%f<3>;
	.reg .b64 	%rd<6>;
	.loc	1 180 0
$L__func_begin0:
	.loc	1 180 0

// %bb.0:
	ld.param.u64 	%rd3, [_kernel_param_0];
	ld.param.u64 	%rd4, [_kernel_param_1];
$L__tmp0:
	.loc	1 181 28
	mov.u32 	%r3, %ctaid.x;
	.loc	1 181 33
	shl.b32 	%r4, %r3, 5;
	ld.param.u32 	%r5, [_kernel_param_2];
	.loc	1 181 59
	mov.u32 	%r6, %tid.x;
	and.b32  	%r7, %r6, 31;
	.loc	1 181 46
	or.b32  	%r8, %r4, %r7;
	.loc	1 182 46
	setp.lt.s32 	%p1, %r8, %r5;
	.loc	1 182 22
	mul.wide.s32 	%rd5, %r8, 4;
	add.s64 	%rd1, %rd4, %rd5;
	.loc	1 182 16
	// begin inline asm
	mov.u32 %r1, 0x0;
	@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
	// end inline asm
	mov.b32 	%f1, %r1;
	.loc	1 183 12
	mul.f32 	%f2, %f1, 0f41200000;
	.loc	1 184 19
	add.s64 	%rd2, %rd3, %rd5;
	.loc	1 184 28
	and.b32  	%r9, %r6, 96;
	setp.eq.s32 	%p3, %r9, 0;
	mov.b32 	%r2, %f2;
	and.pred  	%p2, %p3, %p1;
	// begin inline asm
	@%p2 st.global.b32 [ %rd2 + 0 ], { %r2 };
	// end inline asm
	.loc	1 184 4
	ret;
$L__tmp1:
$L__func_end0:
                                        // -- End function
}
    """
    temp_file = pathlib.Path(f"/tmp/test_override_{str(uuid.uuid4())}.ptx")
    temp_file.write_text(ir_src)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32, 'ir_override': str(temp_file)})]

    @triton.autotune(configs=configs, key=['N'], do_bench=do_bench)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        x = x * 10
        tl.store(dst + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](dst, src, N=N)

    # Change the behavior of kernel by overriding PTX
    torch.testing.assert_close(src * 10, dst)


def test_exceed_tmem(device):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 10:
        pytest.skip("Test requires tensor memory.")
    N = 512
    dst = torch.empty((N, ), device=device, dtype=torch.float32)
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 128}), triton.Config(kwargs={'BLOCK_SIZE': 32})]
    exception_out_of_resource = None

    def _post_hook(*args, exception):
        nonlocal exception_out_of_resource
        if exception is not None:
            exception_out_of_resource = exception

    @triton.autotune(configs=configs, key=['N'], do_bench=do_bench, pre_hook=None, post_hook=_post_hook)
    @triton.jit
    def dot_kernel(dst, BLOCK_SIZE: tl.constexpr):
        a = tl.full((BLOCK_SIZE, BLOCK_SIZE), 0.0, tl.float16)
        b = tl.full((BLOCK_SIZE, BLOCK_SIZE), 0.0, tl.float16)
        c0 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        c1 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        c2 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        c3 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        c4 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        for i in range(0, 100):
            c0 = tl.dot(a, b, c0)
            c1 = tl.dot(a, b, c1)
            c2 = tl.dot(a, b, c2)
            c3 = tl.dot(a, b, c3)
            c4 = tl.dot(a, b, c4)
        c = c4 + c3 + c2 + c1 + c0
        c = c.reshape([BLOCK_SIZE * BLOCK_SIZE])
        tl.store(dst + tl.arange(0, BLOCK_SIZE * BLOCK_SIZE), c)

    dot_kernel[(1, )](dst)
    assert exception_out_of_resource is not None and str(
        exception_out_of_resource
    ) == "out of resource: tensor memory, Required: 640, Hardware limit: 512. Reducing block sizes or `num_stages` may help."


def test_exceed_threads(device):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    x = torch.empty(1024, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    output = torch.empty_like(x)

    configs = [
        triton.Config({}, num_warps=128),
        triton.Config({}, num_warps=4),
    ]

    exception_out_of_resource = None

    def _post_hook(*args, exception):
        nonlocal exception_out_of_resource
        if exception is not None:
            exception_out_of_resource = exception

    @triton.autotune(configs=configs, key=['BLOCK_SIZE'], do_bench=do_bench, post_hook=_post_hook)
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def grid(meta):
        return (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )

    add_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=128)

    warp_size = triton.runtime.driver.active.get_current_target().warp_size
    assert exception_out_of_resource is not None and f"out of resource: threads, Required: {128 * warp_size}" in str(
        exception_out_of_resource)
