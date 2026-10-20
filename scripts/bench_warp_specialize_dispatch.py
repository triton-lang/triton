"""Paired benchmark of jump-table and conditional worker dispatch.

Run on the GPU host:
  python scripts/bench_warp_specialize_dispatch.py --output /tmp/dispatch-bench

Both variants use the same optimized LLVM IR, differing only in the kernel's
no-jump-tables attribute. Each loop iteration launches one warp-specialized
region whose partitions write separate shared-memory rows. Global stores occur
only after the loop. Timings include dispatch, barriers, and shared-memory work;
they are not isolated instruction latencies.

The one-worker case is a control: LLVM already uses conditional branches for
that small switch. A fixed maxnreg keeps the register allocation budget equal.
CUDA graphs exclude Python launch overhead. Alternating paired measurements
use warmed kernels and a warm constant cache.
"""

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def write_partition(buffer, iteration, slot: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    offsets = gl.arange(0, 32, layout=layout)
    buffer.index(slot).store(iteration + 100 * slot + offsets)


@gluon.jit(do_not_specialize=["iterations"])
def dispatch_kernel(output, iterations, workers: gl.constexpr):
    rows: gl.constexpr = workers + 1
    buffer = gl.allocate_shared_memory(gl.int32, [rows, 32], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    for iteration in range(iterations):
        if workers == 1:
            gl.warp_specialize([
                (write_partition, (buffer, iteration, 0)),
                (write_partition, (buffer, iteration, 1)),
            ], [1])
        elif workers == 3:
            gl.warp_specialize([
                (write_partition, (buffer, iteration, 0)),
                (write_partition, (buffer, iteration, 1)),
                (write_partition, (buffer, iteration, 2)),
                (write_partition, (buffer, iteration, 3)),
            ], [1, 1, 1])
        else:
            gl.warp_specialize([
                (write_partition, (buffer, iteration, 0)),
                (write_partition, (buffer, iteration, 1)),
                (write_partition, (buffer, iteration, 2)),
                (write_partition, (buffer, iteration, 3)),
                (write_partition, (buffer, iteration, 4)),
                (write_partition, (buffer, iteration, 5)),
                (write_partition, (buffer, iteration, 6)),
                (write_partition, (buffer, iteration, 7)),
            ], [1, 1, 1, 1, 1, 1, 1])
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    col = gl.arange(0, 32, layout=layout)
    for row in gl.static_range(rows):
        values = buffer.index(row).load(layout)
        offsets = gl.program_id(0) * rows * 32 + row * 32 + col
        gl.store(output + offsets, values)


def set_jump_tables(llir, disabled):
    # Keep both variants identical except for this one kernel attribute.
    llir = re.sub(r'\s*"no-jump-tables"="(?:true|false)"', "", llir)
    if disabled:
        match = re.search(r"^define ptx_kernel .* #(\d+)", llir, re.MULTILINE)
        assert match, "missing kernel attribute group"
        prefix = f"attributes #{match[1]} = {{"
        assert prefix in llir
        llir = llir.replace(prefix, prefix + ' "no-jump-tables"="true"', 1)
    return llir


def capture(fn, launches):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            fn()
    return graph


def measure(graph, launches):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / launches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=11)
    parser.add_argument("--launches", type=int, default=16)
    parser.add_argument("--workers", type=int, nargs="+", choices=[1, 3, 7], default=[1, 3, 7])
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 64, 1024])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = triton.runtime.driver.active.get_current_target()
    from triton.backends.nvidia.compiler import get_ptxas
    assembler = get_ptxas(target.arch).path
    report = {
        "gpu": torch.cuda.get_device_name(),
        "target": str(target),
        "triton": triton.__version__,
        "triton_path": triton.__file__,
        "ptxas": subprocess.check_output([assembler, "--version"], text=True),
        "pairs": args.pairs,
        "launches_per_graph": args.launches,
        "kernels": [],
        "results": [],
    }
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    for workers in args.workers:
        output = torch.empty((num_sms, workers + 1, 32), device="cuda", dtype=torch.int32)
        options = {"num_warps": 4, "maxnreg": 128}
        original = dispatch_kernel.warmup(output, 17, workers, grid=(1, ), **options)
        variants = {}
        for name, disabled in [("baseline", False), ("conditional", True)]:
            prefix = args.output / f"workers{workers}_{name}"
            ir_path = prefix.with_suffix(".llir")
            ir_path.write_text(set_jump_tables(original.asm["llir"], disabled))
            variant_options = dict(options, ir_override=str(ir_path))
            compiled = dispatch_kernel[(1, )](output, 17, workers, **variant_options)
            prefix.with_suffix(".ptx").write_text(compiled.asm["ptx"])
            prefix.with_suffix(".sass").write_text(compiled.asm["sass"])
            prefix.with_suffix(".cubin").write_bytes(compiled.asm["cubin"])
            report["kernels"].append({
                "workers": workers,
                "variant": name,
                "registers": compiled.n_regs,
                "spills": compiled.n_spills,
                "shared": compiled.metadata.shared,
                "ptx_brx": compiled.asm["ptx"].count("brx.idx"),
                "sass_brx": len(re.findall(r"\bBRXU?\b", compiled.asm["sass"])),
            })
            if disabled:
                assert "brx.idx" not in compiled.asm["ptx"]
                assert not re.search(r"\bBRXU?\b", compiled.asm["sass"])
            variants[name] = variant_options
        for blocks in [1, num_sms]:
            for iterations in args.iterations:
                expected = (iterations - 1 + 100 * torch.arange(workers + 1, device="cuda")[:, None] +
                            torch.arange(32, device="cuda")[None, :])
                graphs = {}
                for name, variant_options in variants.items():

                    def launch():
                        dispatch_kernel[(blocks, )](output, iterations, workers, **variant_options)

                    output.fill_(-1)
                    launch()
                    torch.testing.assert_close(output[:blocks], expected.expand(blocks, -1, -1), check_dtype=False,
                                               atol=0, rtol=0)
                    graphs[name] = capture(launch, args.launches)
                for _ in range(5):
                    for graph in graphs.values():
                        graph.replay()
                torch.cuda.synchronize()
                samples = {name: [] for name in variants}
                paired_ratios = []
                for pair in range(args.pairs):
                    order = list(variants) if pair % 2 == 0 else list(reversed(variants))
                    values = {name: measure(graphs[name], args.launches) for name in order}
                    for name, value in values.items():
                        samples[name].append(value)
                    paired_ratios.append(values["baseline"] / values["conditional"])
                result = {
                    "workers": workers,
                    "blocks": blocks,
                    "iterations": iterations,
                    "baseline_us": statistics.median(samples["baseline"]),
                    "conditional_us": statistics.median(samples["conditional"]),
                    "median_paired_speedup": statistics.median(paired_ratios),
                    "samples_us": samples,
                    "paired_speedups": paired_ratios,
                }
                report["results"].append(result)
                print(json.dumps({k: v
                                  for k, v in result.items()
                                  if k not in ("samples_us", "paired_speedups")}), flush=True)
                (args.output / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
