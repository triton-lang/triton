"""Compare one Triton program with its late reduction optimization off and on.

The 32 basis values are computed entirely in registers from a block-uniform
runtime scalar; no basis tensor is read from global memory. The same source,
inputs, compiler checkout, GPU, and timing method are used for both variants.

Run from a CUDA-capable Triton development checkout:

    PYTHONPATH=python python python/test/microbenchmark/one_hot_xor_reduction.py \
        --output benchmarks/one_hot_xor_reduction/results.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import pathlib
import re
import statistics
import sys
import tempfile
from typing import Any

import torch

import triton
import triton.backends.nvidia.compiler as nvidia_compiler
import triton.language as tl
from triton import knobs
from triton._C import libtriton
from triton._C.libtriton import passes

PASS_WRAPPER = "add_optimize_one_hot_xor_reduction"
SCHEMA_VERSION = 4
TIMING_SAMPLES = 10
DEFAULT_SEED = 0x31415926
DEFAULT_SIZES = ",".join(str(1 << exponent) for exponent in range(8, 26))
MAX_ELEMENTS = (1 << 31) - 1
BOUNDARY_SIZES = (1, 31, 32, 33, 127, 128, 129, 255, 257, 1000, 4097)
BOUNDARY_GUARD = -0x13579BDF


@triton.jit(do_not_specialize=["N", "seed"])
def _classic_one_hot_xor_kernel(
    indices_ptr,
    output_ptr,
    N,
    seed,
    BLOCK: tl.constexpr,
    BENCHMARK_VARIANT: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    index = tl.load(indices_ptr + offsets, mask=mask, other=0).to(tl.uint32)
    bits = tl.arange(0, 32)
    bases = (bits.to(tl.uint32) * 0x9E3779B9) ^ seed.to(tl.uint32)

    result = tl.full(index.shape, 0, tl.uint32)
    for bit in tl.static_range(32):
        basis = tl.xor_sum(tl.where(bits == bit, bases, 0), axis=0)
        result ^= tl.where((index >> bit) & 1, basis, 0)

    tl.store(output_ptr + offsets, result, mask=mask)


@dataclasses.dataclass(frozen=True)
class InstructionCounts:
    ptx_total: int
    ptx_shuffle: int
    ptx_shuffle_indexed: int
    ptx_lop3: int
    ptx_redux: int
    ptx_barrier: int
    ptx_global_load: int
    ptx_global_store: int
    ptx_shared_load: int
    ptx_shared_store: int
    ttgir_reduce: int
    ttgir_global_load: int
    llir_shuffle: int
    registers: int | None
    shared_bytes: int | None


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkout_provenance(program_source: str) -> dict[str, Any]:
    benchmark_path = pathlib.Path(__file__).resolve()
    checkout = benchmark_path.parents[3]
    module_paths = {
        "triton_module": pathlib.Path(triton.__file__).resolve(),
        "nvidia_backend_module": pathlib.Path(nvidia_compiler.__file__).resolve(),
        "native_extension": pathlib.Path(libtriton.__file__).resolve(),
        "kernel_source": pathlib.Path(inspect.getfile(_classic_one_hot_xor_kernel.fn)).resolve(),
    }
    for label, path in module_paths.items():
        if not path.is_relative_to(checkout):
            raise RuntimeError(f"{label} was loaded from {path}, outside benchmark checkout {checkout}; "
                               f"run with PYTHONPATH={checkout / 'python'} after building this checkout")
    if module_paths["kernel_source"] != benchmark_path:
        raise RuntimeError(f"benchmarked JIT source is not this benchmark: {module_paths['kernel_source']}")

    backend_source = checkout / "third_party/nvidia/backend/compiler.py"
    if module_paths["nvidia_backend_module"].read_bytes() != backend_source.read_bytes():
        raise RuntimeError("the imported NVIDIA backend does not match this checkout's "
                           f"{backend_source}; rebuild/reinstall this checkout before benchmarking")
    if f"passes.ttgpuir.{PASS_WRAPPER}(pm)" not in inspect.getsource(nvidia_compiler.CUDABackend.make_llir):
        raise RuntimeError(f"the imported NVIDIA backend does not schedule {PASS_WRAPPER}")
    if not hasattr(passes.ttgpuir, PASS_WRAPPER):
        raise RuntimeError(f"the native extension {module_paths['native_extension']} does not expose "
                           f"passes.ttgpuir.{PASS_WRAPPER}; run make in {checkout}")

    native_inputs = (
        checkout / "lib/Dialect/TritonGPU/Transforms/OptimizeOneHotXorReduction.cpp",
        checkout / "lib/Dialect/TritonGPU/Transforms/CMakeLists.txt",
        checkout / "include/triton/Dialect/TritonGPU/Transforms/Passes.td",
        checkout / "python/src/passes.cc",
    )
    native_mtime = module_paths["native_extension"].stat().st_mtime_ns
    stale_inputs = [path for path in native_inputs if path.stat().st_mtime_ns > native_mtime]
    if stale_inputs:
        raise RuntimeError(f"native extension {module_paths['native_extension']} is older than "
                           f"{', '.join(str(path) for path in stale_inputs)}; run make in {checkout}")

    return {
        "checkout_root": str(checkout),
        **{label: str(path)
           for label, path in module_paths.items()},
        "kernel_source_sha256": hashlib.sha256(program_source.encode()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "nvidia_backend_sha256": hashlib.sha256(backend_source.read_bytes()).hexdigest(),
        "native_extension_sha256": _file_sha256(module_paths["native_extension"]),
        "native_source_sha256":
        {str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
         for path in native_inputs},
    }


@contextlib.contextmanager
def _isolated_compiler_cache():
    if knobs.compilation.override:
        raise RuntimeError("Disable TRITON_KERNEL_OVERRIDE before benchmarking the source compiler pipeline")
    if knobs.runtime.add_stages_inspection_hook is not None:
        raise RuntimeError("Remove custom compiler pipeline hooks before benchmarking the source compiler pipeline")
    previous_cache = knobs.cache.dir
    _classic_one_hot_xor_kernel.device_caches.clear()
    with tempfile.TemporaryDirectory(prefix="triton-one-hot-xor-benchmark-") as cache_dir:
        knobs.cache.dir = cache_dir
        try:
            yield cache_dir
        finally:
            _classic_one_hot_xor_kernel.device_caches.clear()
            knobs.cache.dir = previous_cache


def _instruction_counts(compiled: Any) -> InstructionCounts:
    ptx = compiled.asm["ptx"]
    ttgir = compiled.asm.get("ttgir", "")
    lines = [line.split("//", 1)[0].strip() for line in ptx.splitlines()]
    instructions = [line for line in lines if line.endswith(";") and not line.startswith((".", "//", "{"))]
    metadata = compiled.metadata
    return InstructionCounts(
        ptx_total=len(instructions),
        ptx_shuffle=len(re.findall(r"\bshfl\.sync\.", ptx)),
        ptx_shuffle_indexed=len(re.findall(r"\bshfl\.sync\.idx\.", ptx)),
        ptx_lop3=len(re.findall(r"\blop3\.b32\b", ptx)),
        ptx_redux=len(re.findall(r"\bred(?:ux)?\.sync\.", ptx)),
        ptx_barrier=len(re.findall(r"\b(?:bar\.sync|barrier\.sync)\b", ptx)),
        ptx_global_load=sum(bool(re.search(r"\bld\.global\.", line)) for line in instructions),
        ptx_global_store=sum(bool(re.search(r"\bst\.global\.", line)) for line in instructions),
        ptx_shared_load=len(re.findall(r"\bld\.shared\.", ptx)),
        ptx_shared_store=len(re.findall(r"\bst\.shared\.", ptx)),
        ttgir_reduce=ttgir.count('"tt.reduce"'),
        ttgir_global_load=len(re.findall(r"\btt\.load\b", ttgir)),
        llir_shuffle=compiled.asm.get("llir", "").count("llvm.nvvm.shfl.sync"),
        registers=getattr(compiled, "n_regs", None),
        shared_bytes=getattr(metadata, "shared", None),
    )


@contextlib.contextmanager
def _pass_enabled(enabled: bool):
    if not hasattr(passes.ttgpuir, PASS_WRAPPER):
        raise RuntimeError(f"This Triton build does not expose passes.ttgpuir.{PASS_WRAPPER}; "
                           "rebuild the native extension before benchmarking.")
    original = getattr(passes.ttgpuir, PASS_WRAPPER)
    if not enabled:
        setattr(passes.ttgpuir, PASS_WRAPPER, lambda _pm: None)
    try:
        yield
    finally:
        setattr(passes.ttgpuir, PASS_WRAPPER, original)


def _reference(indices: torch.Tensor, seed: int) -> torch.Tensor:
    indices64 = indices.to(torch.int64) & 0xFFFFFFFF
    bits = torch.arange(32, device=indices.device, dtype=torch.int64)
    bases = ((bits * 0x9E3779B9) & 0xFFFFFFFF) ^ (seed & 0xFFFFFFFF)
    output = torch.zeros_like(indices64)
    for bit in range(32):
        contribution = torch.where(((indices64 >> bit) & 1).bool(), bases[bit], 0)
        output = output ^ contribution
    return output.to(torch.int32)


def _compile_variant(
    indices: torch.Tensor,
    output: torch.Tensor,
    size: int,
    seed: int,
    block: int,
    enabled: bool,
) -> tuple[Any, Any]:
    grid = (triton.cdiv(size, block), )

    def launch():
        return _classic_one_hot_xor_kernel[grid](
            indices,
            output,
            size,
            seed,
            BLOCK=block,
            BENCHMARK_VARIANT=int(enabled),
            num_warps=4,
        )

    with _pass_enabled(enabled):
        compiled = launch()
        torch.cuda.synchronize()
    return compiled, launch


def _measure(launch: Any, repetitions_ms: int) -> dict[str, Any]:
    samples_ms = triton.testing.do_bench_cudagraph(launch, rep=repetitions_ms, return_mode="all")
    samples_us = [sample * 1000 for sample in samples_ms]
    if len(samples_us) != TIMING_SAMPLES or any(not math.isfinite(sample) or sample <= 0 for sample in samples_us):
        raise RuntimeError(f"expected {TIMING_SAMPLES} positive finite CUDA Graph timing samples")
    ordered = sorted(samples_us)
    return {
        "samples_us": samples_us,
        "median_us": statistics.median(samples_us),
        "mean_us": statistics.fmean(samples_us),
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "p20_us": ordered[int(0.20 * (len(ordered) - 1))],
        "p80_us": ordered[int(0.80 * (len(ordered) - 1))],
        "relative_stddev": statistics.pstdev(samples_us) / statistics.fmean(samples_us),
    }


def _sizes(value: str) -> list[int]:
    try:
        sizes = [int(part, 0) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from error
    if not sizes or any(size <= 0 or size > MAX_ELEMENTS for size in sizes):
        raise argparse.ArgumentTypeError(
            f"sizes must be between 1 and {MAX_ELEMENTS}; kernel offsets use signed 32-bit integers")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must not contain duplicates")
    return sizes


def _inputs(size: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260826 + size)
    indices = torch.randint(-(1 << 31), (1 << 31) - 1, (size, ), device="cuda", dtype=torch.int32, generator=generator)
    edge_values = torch.tensor([0, 1, 5, -(1 << 31), -1], device="cuda", dtype=torch.int32)
    count = min(size, edge_values.numel())
    indices[:count] = edge_values[:count]
    return indices


def _assert_compilation_pair(compiled: dict[str, Any], counts: dict[str, InstructionCounts]) -> None:
    baseline = compiled["baseline"]
    optimized = compiled["optimized"]
    if baseline.metadata.hash == optimized.metadata.hash:
        raise RuntimeError("optimization OFF and ON unexpectedly share a compiler-cache key")
    for stage in ("ttir", "ttgir"):
        if baseline.asm.get(stage) != optimized.asm.get(stage):
            raise RuntimeError(f"optimization OFF and ON do not compile the same {stage.upper()}; "
                               "only the late LLVM-stage pass may differ")

    off = counts["baseline"]
    on = counts["optimized"]
    if off.ttgir_reduce != 32 or on.ttgir_reduce != 32:
        raise RuntimeError("the displayed source must contribute exactly 32 reductions to both pre-pass TTGIR stages; "
                           f"got OFF={off.ttgir_reduce}, ON={on.ttgir_reduce}")
    if off.ptx_shuffle_indexed != 0 or on.ptx_shuffle_indexed != 32:
        raise RuntimeError(
            "the native pass must replace the 32 one-hot reductions with exactly 32 indexed warp shuffles; "
            f"got OFF={off.ptx_shuffle_indexed}, ON={on.ptx_shuffle_indexed}")
    if on.ptx_redux != 0 or (off.ptx_redux == 0 and off.ptx_shuffle == 0):
        raise RuntimeError("optimization OFF must contain warp reductions and optimization ON must contain none; "
                           f"got OFF redux={off.ptx_redux}, OFF shuffles={off.ptx_shuffle}, ON redux={on.ptx_redux}")
    if on.ptx_lop3 <= off.ptx_lop3:
        raise RuntimeError("the optimized source must contain the conditional-XOR LOP3 fusion; "
                           f"got OFF={off.ptx_lop3}, ON={on.ptx_lop3}")


def _run_boundary_case(size: int, seed: int, block: int) -> dict[str, Any]:
    indices = _inputs(size)
    expected = _reference(indices, seed)
    variants: dict[str, Any] = {}
    counts: dict[str, InstructionCounts] = {}
    for name, enabled in (("baseline", False), ("optimized", True)):
        output = torch.full((size + block, ), BOUNDARY_GUARD, device="cuda", dtype=torch.int32)
        kernel, _ = _compile_variant(indices, output, size, seed, block, enabled)
        torch.testing.assert_close(output[:size], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            output[size:],
            torch.full((block, ), BOUNDARY_GUARD, device="cuda", dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        variants[name] = kernel
        counts[name] = _instruction_counts(kernel)
        if counts[name].ptx_global_load != 1 or counts[name].ptx_global_store != 1:
            raise RuntimeError(f"boundary N={size} {name} must have exactly one index load and one output store; "
                               f"got {counts[name].ptx_global_load} loads and {counts[name].ptx_global_store} stores")
    _assert_compilation_pair(variants, counts)
    print(f"boundary N={size:4d} OFF=correct ON=correct masked-tail=preserved", flush=True)
    return {
        "frontend": "classic",
        "basis_mode": "computed",
        "basis_storage": "registers",
        "basis_global_loads": 0,
        "elements": size,
        "block_size": block,
        "seed": seed,
        "baseline_correct": True,
        "optimized_correct": True,
        "masked_store_guard_preserved": True,
        "optimization_observed": True,
        "variants": {
            name: {"correct": True, "instructions": dataclasses.asdict(counts[name])}
            for name in ("baseline", "optimized")
        },
    }


def _run_case(
    size: int,
    seed: int,
    block: int,
    repetitions_ms: int,
    output_dir: pathlib.Path,
    case_index: int = 0,
) -> dict[str, Any]:
    indices = _inputs(size)
    expected = _reference(indices, seed)

    variants: dict[str, dict[str, Any]] = {}
    compiled_variants: dict[str, Any] = {}
    instruction_counts: dict[str, InstructionCounts] = {}
    launchers: dict[str, Any] = {}
    for name, enabled in (("baseline", False), ("optimized", True)):
        output = torch.empty_like(indices)
        compiled, launch = _compile_variant(indices, output, size, seed, block, enabled)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        counts = _instruction_counts(compiled)
        if counts.ptx_global_load != 1 or counts.ptx_global_store != 1:
            raise RuntimeError(f"expected exactly one index load and one output store, got "
                               f"{counts.ptx_global_load} global loads and {counts.ptx_global_store} global stores")
        label = f"classic-computed-{size}-{name}"
        (output_dir / f"{label}.ptx").write_text(compiled.asm["ptx"])
        (output_dir / f"{label}.ttgir").write_text(compiled.asm.get("ttgir", ""))
        variants[name] = {
            "correct": True,
            "instructions": dataclasses.asdict(counts),
        }
        compiled_variants[name] = compiled
        instruction_counts[name] = counts
        launchers[name] = launch
    _assert_compilation_pair(compiled_variants, instruction_counts)

    # Alternate measurement order between adjacent sizes to reduce clock drift.
    order = ("baseline", "optimized") if case_index % 2 == 0 else ("optimized", "baseline")
    for name in order:
        timing = _measure(launchers[name], repetitions_ms)
        timing["throughput_gigaelements_per_second"] = size / (timing["median_us"] * 1000)
        timing["effective_bandwidth_gb_per_second"] = size * 8 / (timing["median_us"] * 1000)
        variants[name]["timing"] = timing

    baseline = variants["baseline"]
    optimized = variants["optimized"]
    speedup = baseline["timing"]["median_us"] / optimized["timing"]["median_us"]
    result = {
        "frontend": "classic",
        "basis_mode": "computed",
        "basis_storage": "registers",
        "basis_global_loads": 0,
        "elements": size,
        "block_size": block,
        "seed": seed,
        "measurement_order": list(order),
        "variants": variants,
        "speedup": speedup,
        "optimization_observed": True,
    }
    print(
        f"N={size:8d} "
        f"baseline={baseline['timing']['median_us']:8.3f} us "
        f"optimized={optimized['timing']['median_us']:8.3f} us "
        f"speedup={speedup:6.3f}x "
        f"shfl.idx={baseline['instructions']['ptx_shuffle_indexed']}->"
        f"{optimized['instructions']['ptx_shuffle_indexed']} "
        f"global-loads={baseline['instructions']['ptx_global_load']}->"
        f"{optimized['instructions']['ptx_global_load']}",
        flush=True,
    )
    return result


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    large = [case for case in cases if case["elements"] >= 1_048_576]
    peak = max(cases, key=lambda case: case["speedup"])
    return {
        "performance_cases":
        len(cases),
        "bit_exact_performance_cases":
        sum(case["variants"]["baseline"]["correct"] and case["variants"]["optimized"]["correct"] for case in cases),
        "optimized_cases":
        sum(case["optimization_observed"] for case in cases),
        "geomean_speedup":
        statistics.geometric_mean(case["speedup"] for case in cases),
        "large_problem_geomean_speedup":
        (statistics.geometric_mean(case["speedup"] for case in large) if large else None),
        "large_problem_minimum_elements":
        1_048_576,
        "peak_speedup":
        peak["speedup"],
        "peak_frontend":
        "classic",
        "peak_basis_mode":
        "computed",
        "peak_elements":
        peak["elements"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output JSON file.")
    parser.add_argument("--sizes", type=_sizes, default=_sizes(DEFAULT_SIZES))
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=DEFAULT_SEED)
    parser.add_argument("--repetitions-ms", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("an NVIDIA CUDA GPU is required")
    if args.block < 32 or args.block & (args.block - 1):
        parser.error("--block must be a power of two and at least 32")
    if not -(1 << 31) <= args.seed < 1 << 31:
        parser.error("--seed must fit in a signed 32-bit integer")
    if args.repetitions_ms <= 0:
        parser.error("--repetitions-ms must be positive")

    torch.cuda.set_device(args.device)
    output_dir = args.output.parent / f"{args.output.stem}-artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260826)

    properties = torch.cuda.get_device_properties(args.device)
    program_source = inspect.getsource(_classic_one_hot_xor_kernel.fn)
    provenance = _checkout_provenance(program_source)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run": {"requested_sizes": args.sizes, "complete": False},
        "method": {
            "description":
            ("The same standard Triton program with the one-hot reduction optimization disabled or enabled"),
            "program_source":
            program_source,
            "bases_source":
            "computed-in-registers-no-basis-buffer",
            "basis_expression":
            "(bits.to(tl.uint32) * 0x9E3779B9) ^ seed.to(tl.uint32)",
            "seed":
            args.seed,
            "block_size":
            args.block,
            "pass_wrapper":
            PASS_WRAPPER,
            "timing":
            "triton.testing.do_bench_cudagraph(..., return_mode='all')",
            "timing_description": ("Median steady-state GPU execution time from CUDA Graph replay; "
                                   "identical input/output buffers are reused without flushing L2."),
            "timing_repetitions_ms":
            args.repetitions_ms,
            "timing_samples":
            TIMING_SAMPLES,
            "timing_cache_policy": {"reuse_input_output_buffers": True, "flush_l2": False},
            "effective_bandwidth_bytes_per_element":
            8,
            "global_memory_traffic":
            "one 32-bit input index load and one 32-bit output store per element",
            "basis_global_memory_loads":
            0,
            "cache_policy":
            ("reused input/output buffers, no L2 cache flush; bases are recomputed in registers for every block"),
            "source_compilation":
            "same checkout @triton.jit source compiled twice through the real NVIDIA pipeline",
            "jit_cache_isolation":
            "fresh temporary compiler cache plus unused BENCHMARK_VARIANT constexpr",
            "pre_pass_ir_identity":
            "exact byte equality of optimization-OFF and optimization-ON TTIR and TTGIR",
            "optimization_verification":
            "32 source reductions; indexed shuffles 0 to 32; fused LOP3 increases",
            "correctness":
            "bit-exact independent 32-bit PyTorch CUDA reference",
        },
        "environment": {
            "gpu": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_bytes": properties.total_memory,
            "multiprocessors": properties.multi_processor_count,
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "triton_module": triton.__file__,
            "python_version": sys.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "provenance": provenance,
        },
        "cases": [],
        "boundary_correctness": {
            "sizes": list(BOUNDARY_SIZES),
            "cases": [],
            "masked_store_guard_int32": BOUNDARY_GUARD,
            "guard_elements_per_case": args.block,
            "passed": 0,
            "total": len(BOUNDARY_SIZES),
        },
    }

    def save_report() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    # Save an explicitly incomplete record first, so an interrupted run cannot
    # leave a previous successful result at the requested output path.
    save_report()
    with _isolated_compiler_cache():
        for size in BOUNDARY_SIZES:
            case = _run_boundary_case(size, args.seed, args.block)
            report["boundary_correctness"]["cases"].append(case)
            report["boundary_correctness"]["passed"] += 1
            save_report()

        for case_index, size in enumerate(args.sizes):
            case = _run_case(size, args.seed, args.block, args.repetitions_ms, output_dir, case_index)
            report["cases"].append(case)
            report["summary"] = _summary(report["cases"])
            save_report()

    report["run"]["complete"] = True
    save_report()
    summary = report["summary"]
    print(
        f"Saved {summary['performance_cases']} bit-exact performance cases and "
        f"{report['boundary_correctness']['passed']} masked-tail cases to {args.output}; "
        f"geomean={summary['geomean_speedup']:.3f}x "
        f"peak={summary['peak_speedup']:.3f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
