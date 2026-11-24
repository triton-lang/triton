"""Compare tree and sequential mixed-precision reductions on Rubin.

Run with this checkout's Triton installation, for example::

    PYTHONPATH=./python python python/examples/gluon/07-reduction-schedules.py \
        --output-dir reduction-schedule-results

Each thread owns K narrow values for each of O independent outputs. Loads are
coalesced across rows; the reduction axis stays within registers. Each iteration
widens to FP32 and sums, as in the attention denominator. A recurrence forms the
next iteration's narrow inputs from the original inputs and the previous sum,
preventing loop-invariant reduction hoisting. This is a synthetic recurrent sum,
not an attention kernel. One iteration also measures a simple load/reduce/store.

The reduction layouts pack along the reduction axis. This isolates in-thread
scheduling without shuffles/shared memory. Timing uses alternating CUDA-graph
measurements. Uncapped register allocation is the default; --maxnreg can sweep
budgets. The compiler backend's register-allocation flags are recorded in the
metadata (the current baseline forces --regAllocOptLevel=2). Local-memory bytes
are not a count of spilled registers or instructions.
"""

import argparse
import csv
import itertools
import inspect
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import torch
import triton
import triton.testing
from triton.backends.nvidia.compiler import CUDABackend, get_ptxas
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def recurrent_sum(X, Y, K: gl.constexpr, O: gl.constexpr, ITERS: gl.constexpr, MODE: gl.constexpr, WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([O, K], [32, 1], [WARPS, 1], [1, 0])
    rows: gl.constexpr = WARPS * 32 * O
    r = gl.arange(0, rows, layout=gl.SliceLayout(1, layout))
    k = gl.arange(0, K, layout=gl.SliceLayout(0, layout))
    base = gl.load(X + gl.program_id(0) * rows * K + k[None, :] * rows + r[:, None])
    carry = gl.full((rows, ), 0, gl.float32, gl.SliceLayout(1, layout))
    for _ in range(ITERS):
        narrow = (base.to(gl.float32) + carry[:, None]).to(base.dtype)
        total = gl.sum(narrow.to(gl.float32), 1, in_thread_reduction=MODE)
        carry = total * (0.5 / K)
    gl.store(Y + gl.program_id(0) * rows + r, carry)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", nargs="+", choices=["float16", "bfloat16"], default=["float16", "bfloat16"])
    parser.add_argument("--values-per-output", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--outputs-per-thread", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--iterations", nargs="+", type=int, default=[1, 128])
    parser.add_argument("--maxnreg", nargs="+", type=int, default=[0], help="0 means uncapped")
    parser.add_argument("--num-warps", type=int, choices=[4, 8], default=4)
    parser.add_argument("--ctas-per-sm", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--rep-ms", type=float, default=20)
    parser.add_argument("--warmup-seconds", type=float, default=5)
    args = parser.parse_args()
    for v in args.values_per_output + args.outputs_per_thread:
        if v < 1 or v & (v - 1):
            parser.error("values/output and outputs/thread must be positive powers of two")
    if min(args.iterations) < 1 or min(args.maxnreg) < 0 or min(args.ctas_per_sm, args.pairs) < 1:
        parser.error("iterations, CTAs/SM, and pairs must be positive; register caps must be nonnegative")
    if args.rep_ms <= 0 or args.warmup_seconds < 0:
        parser.error("rep-ms must be positive and warmup-seconds nonnegative")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (10, 7):
        parser.error("this experiment requires a Rubin (10.7) GPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = args.output_dir / "kernels"
    artifacts.mkdir()
    ptxas = get_ptxas(107)
    metadata = dict(ptxas_path=ptxas.path, ptxas_version=subprocess.check_output([ptxas.path, "--version"], text=True),
                    make_cubin_source=inspect.getsource(CUDABackend.make_cubin), enable_fp_fusion=False,
                    arguments={k: str(v) if isinstance(v, Path) else v
                               for k, v in vars(args).items()
                               }, triton_file=triton.__file__, torch_version=torch.__version__, gpu=str(props),
                    revision=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                     text=True).strip(), script_source=Path(__file__).read_text(),
                    complete=False, timing="CUDA graphs, alternating pairs; speedup = tree time / sequential time - 1",
                    local_memory_metric="CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, not spill instruction count")
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    torch.manual_seed(0)
    grid = props.multi_processor_count * args.ctas_per_sm
    modes = ["tree", "sequential"]
    fields = [
        "dtype", "k", "outputs_per_thread", "iterations", "maxnreg", "grid", "mode", "pair", "us", "registers",
        "local_memory_bytes", "mixed_adds", "fp32_packed_adds", "max_abs_error", "kernel_hash"
    ]
    summaries = []
    warmed = False
    with (args.output_dir / "measurements.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        cases = itertools.product(args.dtype, args.values_per_output, args.outputs_per_thread, args.iterations,
                                  args.maxnreg)
        for dtype, k, outputs, iters, cap in cases:
            rows = args.num_warps * 32 * outputs
            x = torch.randn((grid, k, rows), device="cuda", dtype=getattr(torch, dtype)) * 0.1
            y = {mode: torch.empty((grid, rows), device="cuda", dtype=torch.float32) for mode in modes}
            options = dict(num_warps=args.num_warps, enable_fp_fusion=False, ptx_options="")
            if cap:
                options["maxnreg"] = cap
            kernels = {}
            errors = {}
            for mode in modes:
                kernels[mode] = recurrent_sum[(grid, )](x, y[mode], k, outputs, iters, mode, args.num_warps, **options)
                kernel = kernels[mode]
                (artifacts / f"{kernel.hash}.ptx").write_text(kernel.asm["ptx"])
                (artifacts / f"{kernel.hash}.cubin").write_bytes(kernel.asm["cubin"])
            # FP64 accumulation is exact for these short sums of narrow values;
            # iterate the same narrowing recurrence to check both schedules.
            reference = torch.zeros((grid, rows), device="cuda", dtype=torch.float32)
            for _ in range(iters):
                values = (x.float() + reference[:, None, :]).to(x.dtype)
                reference = (values.double().sum(1) * (0.5 / k)).float()
            for mode in modes:
                errors[mode] = (y[mode] - reference).abs().max().item()
                torch.testing.assert_close(y[mode], reference, rtol=1e-4, atol=1e-6)

            def launch(mode):
                recurrent_sum[(grid, )](x, y[mode], k, outputs, iters, mode, args.num_warps, **options)

            if not warmed:
                end = time.monotonic() + args.warmup_seconds
                while time.monotonic() < end:
                    launch("tree")
                    launch("sequential")
                    torch.cuda.synchronize()
                warmed = True
            timings = {mode: [] for mode in modes}
            for pair in range(args.pairs):
                for mode in modes if pair % 2 == 0 else modes[::-1]:
                    us = 1000 * triton.testing.do_bench_cudagraph(lambda: launch(mode), rep=args.rep_ms)
                    timings[mode].append(us)
                    kernel = kernels[mode]
                    writer.writerow(
                        dict(dtype=dtype, k=k, outputs_per_thread=outputs, iterations=iters, maxnreg=cap, grid=grid,
                             mode=mode, pair=pair, us=us, registers=kernel.n_regs,
                             local_memory_bytes=kernel.n_spills * 4,
                             mixed_adds=len(re.findall(r"add\.rn\.f32x2\.(?:f16|bf16)x2\.f32x2", kernel.asm["ptx"])),
                             fp32_packed_adds=len(re.findall(r"add\.rn\.f32x2\s", kernel.asm["ptx"])),
                             max_abs_error=errors[mode], kernel_hash=kernel.hash))
                    file.flush()
            tree, seq = [statistics.median(timings[mode]) for mode in modes]
            summary = dict(
                dtype=dtype, k=k, outputs_per_thread=outputs, iterations=iters, maxnreg=cap, tree_us=tree,
                sequential_us=seq, speedup_percent=100 * (tree / seq - 1),
                tree_cv_percent=100 * statistics.pstdev(timings["tree"]) / statistics.mean(timings["tree"]),
                sequential_cv_percent=100 * statistics.pstdev(timings["sequential"]) /
                statistics.mean(timings["sequential"]), tree_registers=kernels["tree"].n_regs,
                sequential_registers=kernels["sequential"].n_regs, tree_local_bytes=kernels["tree"].n_spills * 4,
                sequential_local_bytes=kernels["sequential"].n_spills * 4)
            summaries.append(summary)
            print(
                f"{dtype} K={k:3} O={outputs} iters={iters:3} cap={cap:3}: "
                f"{tree:.3f} -> {seq:.3f} us ({summary['speedup_percent']:+.2f}%), "
                f"regs {summary['tree_registers']} -> {summary['sequential_registers']}", flush=True)
            with (args.output_dir / "summary.csv").open("w") as summary_file:
                w = csv.DictWriter(summary_file, fieldnames=list(summary))
                w.writeheader()
                w.writerows(summaries)
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
