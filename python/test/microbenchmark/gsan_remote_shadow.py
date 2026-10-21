"""Compare local and peer-GPU GSan shadow placement with identical kernels/data.

Run with two idle, peer-atomic-capable CUDA GPUs visible. The default runs each
placement in a fresh process, alternating order across rounds. CUDA event
samples are queued behind a GPU sleep so host-side GSan launch overhead does
not leave gaps inside the measured kernel intervals. No CUDA graph is used.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl
from triton.experimental import gsan


@triton.jit
def vector_add(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    y = tl.load(Y + i, i < N, other=0)
    tl.store(Z + i, x + y, i < N)


@triton.jit
def row_softmax(X, Y, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * COLS + col, col < COLS, other=-float("inf"))
    numerator = tl.exp(x - tl.max(x, 0))
    y = numerator / tl.sum(numerator, 0)
    tl.store(Y + row * COLS + col, y, col < COLS)


@triton.jit
def matmul(A, B, C, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(tl.cdiv(N, BK)):
        ks = step * BK + k
        a = tl.load(A + m[:, None] * N + ks[None, :])
        b = tl.load(B + ks[:, None] * N + n[None, :])
        acc += tl.dot(a, b)
    tl.store(C + m[:, None] * N + n[None, :], acc)


def samples_us(fn, samples):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(samples)]
    # At the GB300's clock rate this is much longer than queuing these launches.
    # Reject a sample batch if the host takes so long that this guard expires.
    guard_start = torch.cuda.Event(enable_timing=True)
    guard_end = torch.cuda.Event(enable_timing=True)
    guard_start.record()
    torch.cuda._sleep(200_000_000)
    guard_end.record()
    begin = time.perf_counter()
    for start, end in events:
        start.record()
        fn()
        end.record()
    queued_ms = (time.perf_counter() - begin) * 1e3
    torch.cuda.synchronize()
    guard_ms = guard_start.elapsed_time(guard_end)
    if queued_ms > guard_ms * 0.8:
        raise RuntimeError(f"Host queue time {queued_ms:.1f}ms exceeded timing guard {guard_ms:.1f}ms")
    return [start.elapsed_time(end) * 1e3 for start, end in events]


def worker(args):
    torch.cuda.set_device(0)
    torch.manual_seed(123)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    instrumented = args.worker != "off"
    pool = None
    if instrumented:
        gsan.configure(device_ranks={0: 0}, num_devices=1, rng_seed=123,
                       shadow_device=1 if args.worker == "remote" else 0)
        pool = gsan.create_mem_pool()
    triton.knobs.compilation.instrumentation_mode = "gsan" if instrumented else ""

    cases = [("vector", 2**16), ("vector", 2**20), ("vector", 2**24), ("softmax", 1024), ("softmax", 4096),
             ("matmul", 512), ("matmul", 1024), ("linear", 1024)]
    results = []
    for kind, size in cases:
        with torch.cuda.use_mem_pool(pool) if instrumented else nullcontext():
            shape = (size, ) if kind == "vector" else ((1024, size) if kind == "softmax" else (size, size))
            dtype = torch.bfloat16 if kind in ("matmul", "linear") else torch.float32
            x = torch.randn(shape, device="cuda", dtype=dtype)
            y = torch.randn(shape, device="cuda", dtype=dtype) if kind not in ("softmax", "linear") else None
            z = torch.empty(shape, device="cuda", dtype=torch.float32)
        if kind == "linear":
            # Allocate the immutable weight matrix outside the GSan pool.
            y = torch.randn(shape, device="cuda", dtype=dtype)
        if kind == "vector":
            fn = partial(vector_add[(triton.cdiv(size, 256), )], x, y, z, size, 256)
            expected = x + y
        elif kind == "softmax":
            fn = partial(row_softmax[(1024, )], x, z, size, triton.next_power_of_2(size), num_warps=4)
            expected = x.softmax(dim=-1)
        else:
            fn = partial(matmul[(size // 32, size // 64)], x, y, z, size, 32, 64, 32, num_warps=4)
            expected = x.float() @ y.float()
        kernel = fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(z, expected, rtol=2e-3, atol=2e-3)
        fingerprint = hashlib.sha256(x.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        timings = samples_us(fn, args.samples)
        torch.testing.assert_close(z, expected, rtol=2e-3, atol=2e-3)
        result = {
            "case": f"{kind}_{size}", "input_sha256": fingerprint, "cubin_sha256":
            hashlib.sha256(kernel.asm["cubin"]).hexdigest(), "registers": kernel.n_regs, "spills": kernel.n_spills,
            "samples_us": timings, "median_us": statistics.median(timings), "data_bytes":
            sum(t.numel() * t.element_size() for t in (x, y, z) if t is not None)
        }
        results.append(result)
        print(json.dumps({"placement": args.worker, **result}), flush=True)
        del fn, x, y, z, expected, kernel
        torch.cuda.synchronize()
    metadata = {
        "placement": args.worker, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton_file":
        triton.__file__, "samples_per_case": args.samples, "results": results
    }
    if instrumented:
        from triton.experimental.gsan._allocator import get_runtime_state_layout
        metadata["runtime_layout"] = get_runtime_state_layout(0)
    Path(args.output).write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("off", "local", "remote"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for round_index in range(args.rounds):
        order = ("off", "local", "remote") if round_index % 2 == 0 else ("remote", "local", "off")
        for placement in order:
            result_file = output / f"{round_index}-{placement}.json"
            with (output / f"{round_index}-{placement}.log").open("w") as log:
                subprocess.run([
                    sys.executable, __file__, "--worker", placement, "--output",
                    str(result_file), "--samples",
                    str(args.samples)
                ], check=True, stdout=log, stderr=subprocess.STDOUT)
            run = json.loads(result_file.read_text())
            runs.append(run)
            print(f"Completed round {round_index + 1}/{args.rounds}: {placement}", flush=True)
    summary = []
    for case_index, case in enumerate(runs[0]["results"]):
        row = {"case": case["case"]}
        for placement in ("off", "local", "remote"):
            matches = [r["results"][case_index] for r in runs if r["placement"] == placement]
            assert all(r["input_sha256"] == case["input_sha256"] for r in matches)
            row[placement + "_us"] = statistics.median(r["median_us"] for r in matches)
            row[placement + "_round_medians_us"] = [r["median_us"] for r in matches]
        instrumented = [r["results"][case_index] for r in runs if r["placement"] != "off"]
        assert len({r["cubin_sha256"] for r in instrumented}) == 1, "instrumented binaries differ"
        row["remote_over_local"] = row["remote_us"] / row["local_us"]
        summary.append(row)
        print(json.dumps(row), flush=True)
    source_files = [
        Path(__file__),
        Path(gsan.__file__).with_name("_allocator.py"),
        Path(gsan.__file__).parent / "src" / "GSanAllocator.cc"
    ]
    report = {
        "summary": summary, "runs": runs, "source_sha256":
        {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in source_files}
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
