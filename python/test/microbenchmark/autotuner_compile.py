r"""Compare compile-only wall time for serial, PR #11847, and async warmup.

Use an existing Triton build with this checkout's Python sources (no rebuild):

    git fetch origin pull/11847/head:refs/remotes/origin/pr11847
    PYTHONPATH=python python python/test/microbenchmark/autotuner_compile.py

The default comparison is pinned to the reviewed #11847 revision. Override
--previous-ref to compare another revision. Its autotuner.py is loaded directly
from Git; all modes use the same JIT, compiler, and native extension. Only run
this with a trusted revision, since it executes that revision's Python code.

Each sample uses a fresh subprocess and an empty Triton cache. Imports, tensor
allocation, device initialization, JIT binder initialization, and sysconfig
initialization are outside the timer for all modes. The timer includes warmup,
executor creation/shutdown, and waiting for every compile. No kernels are
launched or benchmarked. This measures a hook-free FP16 matmul configuration
sweep, not end-to-end autotuning or kernel execution. Select the GPU with
CUDA_VISIBLE_DEVICES; all modes use logical device 0 because the old autotuner
workers do not inherit the caller's current device.

Example:
    PYTHONPATH=python python python/test/microbenchmark/autotuner_compile.py \
        --workers 4 --repeats 3 --output /tmp/autotuner-compile.json
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

PREVIOUS_REVISION = "e34ed2d5196d60bed9767a5e346fb3d85824864d"
ROOT = Path(__file__).resolve().parents[3]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def measure(args):
    from concurrent.futures import ThreadPoolExecutor
    import sysconfig
    import time
    import types

    import torch
    import triton
    import triton.language as tl

    if Path(triton.__file__).resolve() != ROOT / "python/triton/__init__.py":
        raise RuntimeError("Use this checkout's Python sources: PYTHONPATH=python")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA or HIP GPU")

    torch.cuda.set_device(0)
    autotune = triton.autotune
    if args.mode == "previous":
        source = git("show", f"{args.previous_ref}:python/triton/runtime/autotuner.py")
        previous = types.ModuleType("triton.runtime._previous_autotuner")
        previous.__package__ = "triton.runtime"
        exec(compile(source, f"{args.previous_ref}:autotuner.py", "exec"), previous.__dict__)
        if not hasattr(previous.Autotuner, "_parallel_bench"):
            raise RuntimeError("--previous-ref must contain the worker autotuner from PR #11847")
        # The comparison adds only these settings, not the previous knobs module.
        triton.knobs.autotuning.compile_workers = args.workers
        triton.knobs.autotuning.overlap_bench = False
        autotune = previous.autotune

    configs = [
        triton.Config({"BM": bm, "BN": bn, "BK": 32}, num_warps=4, num_stages=stages)
        for bm in (32, 64, 128)
        for bn in (32, 64)
        for stages in (2, 3, 4)
    ][:args.configs]

    @triton.jit
    def matmul(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for k in range(triton.cdiv(K, BK)):
            a = tl.load(A + rows[:, None] * K + (ks[None, :] + k * BK),
                        (rows[:, None] < M) & (ks[None, :] + k * BK < K), other=0)
            b = tl.load(B + (ks[:, None] + k * BK) * N + cols[None, :],
                        (ks[:, None] + k * BK < K) & (cols[None, :] < N), other=0)
            acc = tl.dot(a, b, acc)
        tl.store(C + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))

    kernel = autotune(configs=configs, key=[])(matmul)
    a = torch.empty((512, 512), device="cuda:0", dtype=torch.float16)
    b = torch.empty_like(a)
    c = torch.empty_like(a)
    driver = triton.runtime.driver.active
    device = driver.get_current_device()
    target = driver.get_current_target()
    driver.get_current_stream(device)
    # Build the binder on the caller in every mode, excluding unrelated lazy
    # initialization and avoiding concurrent defaultdict initialization in #11847.
    cache = matmul.device_caches[device][0]
    sysconfig.get_config_vars()
    grid = lambda meta: (triton.cdiv(512, meta["BM"]), triton.cdiv(512, meta["BN"]))
    torch.cuda.synchronize()
    start = time.perf_counter()
    if args.mode == "async":
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            with triton.AsyncCompileMode(executor):
                kernels = kernel.warmup(a, b, c, 512, 512, 512, grid=grid)
    else:
        kernels = kernel.warmup(a, b, c, 512, 512, 512, grid=grid)
    elapsed = time.perf_counter() - start
    if len(kernels) != len(configs) or any(k is None for k in kernels) or len(cache) != len(configs):
        raise RuntimeError("Not all configurations compiled into the expected device cache")
    print(
        json.dumps({
            "mode": args.mode,
            "seconds": elapsed,
            "configs": len(configs),
            "workers": args.workers,
            "target": str(target),
            "gpu": torch.cuda.get_device_name(0),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "triton_python": triton.__file__,
        }))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--previous-ref", default=PREVIOUS_REVISION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--configs", type=int, default=18, help="Number of matmul configs (2-18)")
    parser.add_argument("--output", type=Path, help="Write samples and provenance as JSON")
    parser.add_argument("--mode", choices=("serial", "previous", "async"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or args.repeats < 1 or not 2 <= args.configs <= 18:
        parser.error("workers and repeats must be positive; configs must be between 2 and 18")
    if args.mode:
        measure(args)
        return

    previous_sha = git("rev-parse", "--verify", f"{args.previous_ref}^{{commit}}")
    provenance = {
        "head": git("rev-parse", "HEAD"),
        "previous": previous_sha,
        "working_tree_changes": git("status", "--porcelain"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "workers": args.workers,
        "configs": args.configs,
        "repeats": args.repeats,
    }
    samples = []
    modes = ["serial", "previous", "async"]
    for repeat in range(args.repeats):
        # Rotate the order to reduce systematic thermal/order bias.
        for mode in modes[repeat % 3:] + modes[:repeat % 3]:
            with tempfile.TemporaryDirectory(prefix="triton-autotuner-compile-") as cache_dir:
                env = dict(os.environ, TRITON_CACHE_DIR=cache_dir,
                           PYTHONPATH=str(ROOT / "python") + os.pathsep + os.environ.get("PYTHONPATH", ""))
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--mode",
                    mode,
                    "--previous-ref",
                    previous_sha,
                    "--workers",
                    str(args.workers),
                    "--configs",
                    str(args.configs),
                ]
                result = subprocess.run(command, env=env, text=True, capture_output=True)
                if result.returncode:
                    raise RuntimeError(f"{mode} failed:\n{result.stdout}\n{result.stderr}")
                sample = json.loads(result.stdout.strip().splitlines()[-1])
                sample["repeat"] = repeat
                samples.append(sample)
                print(f"{mode:8s} sample {repeat + 1}: {sample['seconds']:.3f} s", flush=True)
    medians = {mode: statistics.median(s["seconds"] for s in samples if s["mode"] == mode) for mode in modes}
    report = {"provenance": provenance, "samples": samples, "median_seconds": medians}
    print("Median compile-only wall time:")
    for mode, seconds in medians.items():
        print(f"  {mode:8s} {seconds:.3f} s")
    print(f"Previous / async: {medians['previous'] / medians['async']:.2f}x")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
