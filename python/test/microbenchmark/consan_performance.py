import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import triton
from triton._internal_testing import is_cuda

COMPILE_TOLERANCE = 1.5
RUNTIME_TOLERANCE = 1.25


def _load_attention_example():
    python_root = Path(__file__).resolve().parents[2]
    module_path = python_root / "examples" / "gluon" / "01-attention-forward.py"
    # Match pytest's module name so later attention tests reuse the compiled kernel.
    module_name = ".".join(module_path.relative_to(python_root.parent).with_suffix("").parts)
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Gluon attention from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] != 10,
                    reason="Gluon attention requires a Blackwell GPU")
def test_consan_attention_performance(with_allocator):
    target = triton.runtime.driver.active.get_current_target()
    platform = f"{target.backend}:sm{target.arch}"
    baselines_path = Path(__file__).with_name("consan_performance_baselines.json")
    baselines = json.loads(baselines_path.read_text())
    assert platform in baselines, f"Missing ConSan attention performance baseline for {platform}"
    baseline = baselines[platform]

    attention = _load_attention_example()
    torch.manual_seed(42)
    shape = (1, 32, 1024, 64)
    q, k, v = [torch.empty(shape, device="cuda", dtype=torch.float16).normal_(mean=0.0, std=0.5) for _ in range(3)]
    output = torch.empty_like(q)
    max_scores = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    cga_layout = ((1, 0), )
    config = attention.KernelConfig(
        GROUP_SIZE_N=1,
        SPLIT_EXP_FACTOR=4,
        USE_TMEM_RED=False,
        NUM_KV_BUFFERS=4,
        USE_EXP2_TURNSTILE=True,
        CGA_LAYOUT=cga_layout,
    )

    def launch():
        attention.attention_forward(q, k, v, False, 0.5, output, max_scores, p=config, cga_layout=cga_layout)

    compilations = []

    def compilation_listener(*, src, metadata, metadata_group, times, cache_hit):
        if getattr(src, "fn", None) is attention.attention_kernel:
            compilations.append((times, cache_hit))

    with triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = "consan"
        triton.knobs.compilation.listener = compilation_listener
        attention.attention_kernel.device_caches.clear()
        triton.knobs.compilation.always_compile = True
        try:
            launch()
        finally:
            triton.knobs.compilation.always_compile = False

        assert len(compilations) == 1, f"Expected one attention compilation, observed {len(compilations)}"
        compilation_times, cache_hit = compilations[0]
        assert not cache_hit, "ConSan attention compilation unexpectedly used the Triton cache"
        triton.knobs.compilation.listener = None
        torch.cuda.synchronize()

        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=0.5, is_causal=False)
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=0)

        runtime_ms = triton.testing.do_bench(launch, warmup=100, rep=3000, return_mode="median")
        runtime_limit_ms = baseline["runtime_ms"] * RUNTIME_TOLERANCE
        if runtime_ms > runtime_limit_ms:
            runtime_ms = min(runtime_ms, triton.testing.do_bench(launch, warmup=100, rep=3000, return_mode="median"))

    compile_seconds = compilation_times.total / 1_000_000
    stage_seconds = {stage: duration / 1_000_000 for stage, duration in compilation_times.lowering_stages}
    print(f"ConSan attention platform={platform} compile={compile_seconds:.3f}s "
          f"runtime={runtime_ms:.3f}ms stages={stage_seconds}")

    compile_limit_seconds = baseline["compile_seconds"] * COMPILE_TOLERANCE
    assert compile_seconds <= compile_limit_seconds, (
        f"ConSan attention compilation took {compile_seconds:.3f}s on {platform}; "
        f"baseline={baseline['compile_seconds']:.3f}s, limit={compile_limit_seconds:.3f}s, stages={stage_seconds}")
    assert runtime_ms <= runtime_limit_ms, (f"ConSan attention runtime was {runtime_ms:.3f}ms on {platform}; "
                                            f"baseline={baseline['runtime_ms']:.3f}ms, limit={runtime_limit_ms:.3f}ms")
