"""Native Hopper WGMMA widths in weight-reusing FP8 projections.

Requires an SM90 GPU and Gluon support for non-power-of-two WGMMA widths.
Run this file to check correctness and compare native N with next_power_of_2(N).
Both arms use identical inputs, shared-memory layouts, precision and stores.
Every timed invocation is preceded by a 256 MiB write AND complete read, outside
its Proton/CUPTI timing scope. Compilation and input preparation are untimed.

    python python/examples/gluon/07-native-wgmma-widths.py --case recurrent136

The projection reuses weights across eight row tiles. The recurrent cell keeps
both weights and state on chip across 32 steps. These amortize global-memory
traffic so eliminating padded output-column arithmetic can matter. The baseline
is the same kernel with a power-of-two instruction width, not a separately
tuned implementation using multiple smaller instructions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import triton
import triton.profiler as proton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import async_copy, fence_async_shared, warpgroup_mma


@gluon.jit
def weight_reuse_projection(X, W, Y, N: gl.constexpr, K: gl.constexpr, INSTR_N: gl.constexpr, ROW_TILES: gl.constexpr):
    """Apply the same narrow projection to ROW_TILES groups of 64 rows."""
    BN: gl.constexpr = triton.next_power_of_2(N)
    copy_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    out_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout([3, 0], [4, 1], [16, BN, 32])
    # Tensor shapes stay power-of-two; only the hardware instruction N changes.
    wn = gl.arange(0, BN, layout=gl.SliceLayout(1, copy_layout))
    wk = gl.arange(0, K, layout=gl.SliceLayout(0, copy_layout))
    weights = gl.allocate_shared_memory(gl.float8e4nv, [BN, K], gl.NVMMASharedLayout(128, 8))
    async_copy.async_load(weights, W + wn[:, None] * K + wk[None, :], wn[:, None] < N)
    async_copy.commit_group()
    async_copy.wait_group(0)
    fence_async_shared()
    inputs = gl.allocate_shared_memory(gl.float8e4nv, [64, K], gl.NVMMASharedLayout(128, 8))
    xm = gl.arange(0, 64, layout=gl.SliceLayout(1, copy_layout))
    om = gl.arange(0, 64, layout=gl.SliceLayout(1, out_layout))
    on = gl.arange(0, BN, layout=gl.SliceLayout(0, out_layout))
    for tile in range(ROW_TILES):
        row = (gl.program_id(0) * ROW_TILES + tile) * 64
        async_copy.async_load(inputs, X + (row + xm[:, None]) * K + wk[None, :])
        async_copy.commit_group()
        async_copy.wait_group(0)
        fence_async_shared()
        acc = gl.zeros([64, BN], gl.float32, mma_layout)
        acc = warpgroup_mma(inputs, weights.permute((1, 0)), acc, max_num_imprecise_acc=32, instruction_n=INSTR_N)
        result = gl.convert_layout(acc, out_layout)
        gl.store(Y + (row + om[:, None]) * N + on[None, :], result, on[None, :] < N)


@gluon.jit
def recurrent_projection(X, W, Bias, Y, N: gl.constexpr, INSTR_N: gl.constexpr, STEPS: gl.constexpr):
    """Fuse a tied-weight FP8 recurrent cell, keeping state on chip."""
    BN: gl.constexpr = triton.next_power_of_2(N)
    copy_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout([3, 0], [4, 1], [16, BN, 32])
    wn = gl.arange(0, BN, layout=gl.SliceLayout(1, copy_layout))
    wk = gl.arange(0, BN, layout=gl.SliceLayout(0, copy_layout))
    weights = gl.allocate_shared_memory(gl.float8e4nv, [BN, BN], gl.NVMMASharedLayout(128, 8))
    async_copy.async_load(weights, W + wn[:, None] * BN + wk[None, :], wn[:, None] < N)
    async_copy.commit_group()
    async_copy.wait_group(0)
    m = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    n = gl.arange(0, BN, layout=gl.SliceLayout(0, layout))
    row = gl.program_id(0) * 64
    state = gl.load(X + (row + m[:, None]) * N + n[None, :], n[None, :] < N, 0.0)
    bias = gl.load(Bias + n, n < N, 0.0)
    inputs = gl.allocate_shared_memory(gl.float8e4nv, [64, BN], gl.NVMMASharedLayout(128, 8))
    for step in range(STEPS):
        inputs.store(state)
        fence_async_shared()
        acc = gl.zeros([64, BN], gl.float32, mma_layout)
        acc = warpgroup_mma(inputs, weights.permute((1, 0)), acc, max_num_imprecise_acc=32, instruction_n=INSTR_N)
        result = gl.convert_layout(acc, layout)
        result = gl.minimum(gl.maximum(result * 0.5 + bias[None, :], 0.0), 1.0)
        state = gl.where(n[None, :] < N, result, 0.0).to(gl.float8e4nv)
    gl.store(Y + (row + m[:, None]) * N + n[None, :], state, n[None, :] < N)


def cold_samples(run, samples, path):
    """Return device kernel time in microseconds, excluding cache preparation."""
    for _ in range(5):
        run()
    cache = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    checksum = torch.empty((), dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    session = proton.start(str(path), context="shadow", data="tree")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(samples):
            with proton.scope(f"prepare.{i:05d}"):
                cache.zero_()
                torch.sum(cache.view(torch.int64), dim=0, out=checksum)
            with proton.scope(f"sample.{i:05d}"):
                run()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert checksum.item() == 0
    proton.deactivate(session, flushing=True)
    data = proton.data.get(session)
    proton.finalize(session)
    timings = triton.testing._collect_proton_scope_times(data, "sample.")
    assert len(timings) == samples

    # Verify that each measured scope contains only the requested kernel.
    prepared = []

    def check(node):
        name = node.get("frame", {}).get("name", "")
        children = node.get("children", [])
        # Graph capture also records empty scopes, without executed GPU metrics.
        executed = any("time (ns)" in child.get("metrics", {}) for child in children)
        if name.startswith("sample."):
            if not executed:
                return
            assert len(children) == 1
            assert children[0]["frame"]["name"] in ("weight_reuse_projection", "recurrent_projection")
            assert children[0]["metrics"]["count"] == 1
        elif name.startswith("prepare."):
            if not executed:
                return
            assert len(children) >= 2
            assert all(child["metrics"]["count"] == 1 for child in children)
            prepared.append(name)
        else:
            for child in children:
                check(child)

    for node in data:
        check(node)
    assert len(prepared) == samples
    return np.asarray(timings) * 1000


def projection_case(n, k, row_tiles, blocks):
    m = 64 * row_tiles * blocks
    torch.manual_seed(0)
    x = (torch.randn((m, k), device="cuda") * 0.25).to(torch.float8_e4m3fn)
    w = (torch.randn((n, k), device="cuda") * 0.25).to(torch.float8_e4m3fn)
    y = torch.empty((m, n), device="cuda", dtype=torch.float16)
    expected = x.float().cpu() @ w.float().cpu().T

    def run(width):
        return weight_reuse_projection[(blocks, )](x, w, y, n, k, width, row_tiles, num_warps=4)

    return run, y, expected, {"kind": "projection", "m": m, "n": n, "k": k, "row_tiles": row_tiles}


def recurrent_case(n, steps, blocks):
    m = 64 * blocks
    bn = triton.next_power_of_2(n)
    torch.manual_seed(0)
    x = torch.rand((m, n), device="cuda").to(torch.float8_e4m3fn)
    w = torch.zeros((bn, bn), device="cuda")
    # Dyadic weights make the FP32 reference sums exact before FP8 rounding.
    w[:n, :n] = (torch.randint(0, 2, (n, n), device="cuda") * 2 - 1) * 0.125
    w = w.to(torch.float8_e4m3fn)
    bias = torch.full((n, ), 0.0625, device="cuda")
    y = torch.empty((m, n), device="cuda", dtype=torch.float8_e4m3fn)
    expected = x.float().cpu()
    ref_w = w[:n, :n].float().cpu()
    for _ in range(steps):
        expected = (expected @ ref_w.T * 0.5 + 0.0625).clamp(0, 1).to(torch.float8_e4m3fn).float()

    def run(width):
        return recurrent_projection[(blocks, )](x, w, bias, y, n, width, steps, num_warps=4)

    return run, y, expected, {"kind": "recurrent", "m": m, "n": n, "steps": steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("native-width-results.json"))
    parser.add_argument("--case", choices=["all", "projection136", "projection72", "recurrent80", "recurrent136"],
                        default="all")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    assert torch.cuda.get_device_capability()[0] == 9, "Requires Hopper"
    factories = {
        "projection136": lambda: projection_case(136, 256, 8, 256),
        "projection72": lambda: projection_case(72, 512, 8, 256),
        "recurrent80": lambda: recurrent_case(80, 32, 512),
        "recurrent136": lambda: recurrent_case(136, 32, 512),
    }
    names = list(factories) if args.case == "all" else [args.case]
    results = []
    for name in names:
        run, output, expected, case = factories[name]()
        n = case["n"]
        padded_n = triton.next_power_of_2(n)
        compiled = run(padded_n)
        torch.testing.assert_close(output.float().cpu(), expected, atol=2e-3, rtol=2e-3)
        baseline = output.clone()
        compiled_native = run(n)
        torch.testing.assert_close(output.float(), baseline.float(), atol=0, rtol=0)
        assert f"m64n{n}k32.f32.e4m3.e4m3" in compiled_native.asm["ptx"]
        case["max_abs_error"] = (output.float().cpu() - expected).abs().max().item()
        for width, kernel in [(padded_n, compiled), (n, compiled_native)]:
            samples = cold_samples(lambda: run(width), args.samples,
                                   args.output.with_suffix(f".{len(results)}.profile"))
            result = dict(case, instruction_n=width, registers=kernel.n_regs, local_bytes=kernel.n_spills * 4,
                          median_us=float(np.median(samples)), p10_us=float(np.quantile(samples, .1)),
                          p90_us=float(np.quantile(samples, .9)), samples_us=samples.tolist())
            results.append(result)
            print({key: value for key, value in result.items() if key != "samples_us"}, flush=True)
        print(f"{name}: {results[-2]['median_us'] / results[-1]['median_us']:.3f}x speedup", flush=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
