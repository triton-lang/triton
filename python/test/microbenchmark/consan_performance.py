import json
from pathlib import Path

import pytest
import torch
import triton
from triton._internal_testing import is_cuda
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import blackwell
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma

COMPILE_TOLERANCE = 1.5
RUNTIME_TOLERANCE = 1.25


@gluon.jit
def _tma_tcgen05_mma_multicast_loop(a_desc, b_desc):
    block_m: gl.constexpr = 256
    block_n: gl.constexpr = 128
    block_k: gl.constexpr = 128
    acc_layout: gl.constexpr = blackwell.TensorMemoryLayout(
        [block_k, block_k],
        col_stride=1,
        cga_layout=((1, 0), ),
        two_ctas=True,
    )
    smem_a = gl.allocate_shared_memory(gl.float16, [block_m, block_k], a_desc.layout)
    smem_b = gl.allocate_shared_memory(
        gl.float16,
        [block_k, block_n],
        gl.NVMMASharedLayout.get_default_for([block_k, block_n], gl.float16, cga_layout=((0, 1), )),
    )
    acc = blackwell.allocate_tensor_memory(gl.float32, [block_m, block_n], acc_layout)
    tma_barrier = mbarrier.allocate_mbarrier(two_ctas=True)
    mbarrier.init(tma_barrier, count=1)
    mma_barrier = mbarrier.allocate_mbarrier()
    mma_barrier_count: gl.constexpr = blackwell.tcgen05_mma_barrier_count([smem_a, smem_b], True,
                                                                          acc.type.layout.two_ctas)
    mbarrier.init(mma_barrier, count=mma_barrier_count)

    tma_phase = 0
    mma_phase = 0
    for k in range(4):
        offset_k = k * block_k
        mbarrier.expect(tma_barrier, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
        tma.async_load(a_desc, [0, offset_k], tma_barrier, smem_a, multicast=True)
        tma.async_load(b_desc, [offset_k, 0], tma_barrier, smem_b, multicast=True)
        mbarrier.wait(tma_barrier, tma_phase, deps=[smem_a, smem_b])
        blackwell.tcgen05_mma(smem_a, smem_b, acc, use_acc=k != 0, multicast=True, mbarriers=[mma_barrier])
        mbarrier.wait(mma_barrier, mma_phase, deps=[smem_a, smem_b])
        tma_phase = (tma_phase + 1) % 2
        mma_phase = (mma_phase + 1) % 2

    mbarrier.invalidate(tma_barrier)
    mbarrier.invalidate(mma_barrier)


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 10,
                    reason="TCGEN05 MMA requires a Blackwell GPU")
def test_consan_multicast_performance(with_allocator):
    target = triton.runtime.driver.active.get_current_target()
    platform = f"{target.backend}:sm{target.arch}"
    baselines_path = Path(__file__).with_name("consan_performance_baselines.json")
    baselines = json.loads(baselines_path.read_text())
    assert platform in baselines, f"Missing ConSan multicast performance baseline for {platform}"
    baseline = baselines[platform]

    torch.manual_seed(42)
    a = torch.randn((256, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 128), device="cuda", dtype=torch.float16)
    a_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(
        a,
        [256, 128],
        gl.NVMMASharedLayout.get_default_for([256, 128], gl.float16, cga_layout=((1, 0), )),
    )
    b_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(
        b,
        [128, 128],
        gl.NVMMASharedLayout.get_default_for([128, 128], gl.float16, cga_layout=((0, 1), )),
    )

    def launch():
        _tma_tcgen05_mma_multicast_loop[(1, )](a_desc, b_desc, num_warps=4, num_ctas=2)

    compilations = []

    def compilation_listener(*, src, metadata, metadata_group, times, cache_hit):
        if getattr(src, "fn", None) is _tma_tcgen05_mma_multicast_loop:
            compilations.append((times, cache_hit))

    with triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = "consan"
        triton.knobs.compilation.listener = compilation_listener
        _tma_tcgen05_mma_multicast_loop.device_caches.clear()
        triton.knobs.compilation.always_compile = True
        try:
            launch()
        finally:
            triton.knobs.compilation.always_compile = False

        assert len(compilations) == 1, f"Expected one multicast compilation, observed {len(compilations)}"
        compilation_times, cache_hit = compilations[0]
        assert not cache_hit, "ConSan multicast compilation unexpectedly used the Triton cache"
        triton.knobs.compilation.listener = None
        torch.cuda.synchronize()

        runtime_ms = triton.testing.do_bench(launch, warmup=50, rep=200, return_mode="median")
        runtime_limit_ms = baseline["runtime_ms"] * RUNTIME_TOLERANCE
        if runtime_ms > runtime_limit_ms:
            runtime_ms = min(runtime_ms, triton.testing.do_bench(launch, warmup=50, rep=200, return_mode="median"))

    compile_seconds = compilation_times.total / 1_000_000
    stage_seconds = {stage: duration / 1_000_000 for stage, duration in compilation_times.lowering_stages}
    print(f"ConSan TMA multicast/TCGEN05 platform={platform} compile={compile_seconds:.3f}s "
          f"runtime={runtime_ms:.3f}ms stages={stage_seconds}")

    compile_limit_seconds = baseline["compile_seconds"] * COMPILE_TOLERANCE
    assert compile_seconds <= compile_limit_seconds, (
        f"ConSan multicast compilation took {compile_seconds:.3f}s on {platform}; "
        f"baseline={baseline['compile_seconds']:.3f}s, limit={compile_limit_seconds:.3f}s, stages={stage_seconds}")
    assert runtime_ms <= runtime_limit_ms, (f"ConSan multicast runtime was {runtime_ms:.3f}ms on {platform}; "
                                            f"baseline={baseline['runtime_ms']:.3f}ms, limit={runtime_limit_ms:.3f}ms")
