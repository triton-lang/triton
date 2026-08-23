"""Same-input, interleaved K=64 versus packed 96+96+64 dense MMA experiment.

Run on GB300 after rebuilding this branch. Operand values remain packed FP4;
only the MMA decomposition changes. The scale range is deliberately bounded
so the FP32 reference is reliable even for large reduction dimensions.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import triton
from triton.tools.mxfp import MXFP4Tensor


def load_example():
    path = Path(__file__).with_name("04-2cta-block-scale-matmul.py")
    spec = importlib.util.spec_from_file_location("block_scaled_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare(size, fmt):
    vec = 16 if fmt == "nvfp4" else 32
    operands, scales, references = [], [], []
    for _ in range(2):
        values = MXFP4Tensor(size=(size, size), device="cuda").random()
        exponent = torch.randint(-3, 1, (size, size // vec), device="cuda")
        scale = torch.exp2(exponent.float())
        references.append(values.to(torch.float32) * scale.repeat_interleave(vec, dim=1))
        operands.append(values.to_packed_tensor(dim=1))
        scales.append(scale.to(torch.float8_e4m3fn) if fmt == "nvfp4" else (exponent + 127).to(torch.uint8))
    return operands, scales, references, vec


def gpu_state():
    return subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,uuid,clocks.sm,temperature.gpu,power.draw,utilization.gpu",
        "--format=csv,noheader",
    ], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--formats", nargs="+", choices=["mxfp4", "nvfp4"], default=["mxfp4"])
    parser.add_argument("--buffers", type=int, nargs="+", default=[5])
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--scheduler", choices=["auto", "sps", "clc"], default="auto")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    example = load_example()
    selected = {}
    original_run = example.mma_scaled_warp_specialized_kernel.run

    def capture(*a, **kw):
        kernel = original_run(*a, **kw)
        selected["kernel"] = kernel
        return kernel

    example.mma_scaled_warp_specialized_kernel.run = capture
    root = Path(__file__).resolve().parents[3]
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)).hexdigest(),
        "torch": torch.__version__,
        "triton": triton.__file__,
        "device": torch.cuda.get_device_name(),
        "method": "same inputs; CUDA graph steady state; alternating AB/BA; no L2 flush",
        "scale_values": [0.125, 0.25, 0.5, 1.0],
        "args": vars(args) | {"output": str(args.output)},
        "cases": [],
    }
    for fmt in args.formats:
        for size in args.sizes:
            torch.manual_seed(123)
            operands, scales, references, vec = prepare(size, fmt)
            expected = references[0] @ references[1].T
            scales = [example.swizzle_scales_packed_block(s) for s in scales]
            for buffers in args.buffers:
                scheduler = args.scheduler if args.scheduler != "auto" else ("sps" if size <= 8192 else "clc")
                config = dict(example.BEST_2CTA_CONFIG, BLOCK_K=args.block_k, num_buffers=buffers,
                              scheduler=example.SCHEDULER_SPS if scheduler == "sps" else example.SCHEDULER_CLC)
                case = dict(format=fmt, size=size, block_k=args.block_k, buffers=buffers, scheduler=scheduler,
                            gpu_before=gpu_state(), variants=[])
                functions = []
                baseline = None
                for enable in [False, True]:
                    fn = lambda enable=enable: example.mma_scaled_warp_specialized(*operands, *scales, vec,
                                                                                   enable_fp4_k96=enable, **config)
                    output = fn()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(output.float(), expected, atol=1e-3, rtol=1e-3)
                    compiled = selected["kernel"]
                    name = f"{fmt}-{size}-bk{args.block_k}-buf{buffers}-{scheduler}-{'k96' if enable else 'k64'}"
                    for ext in ["ptx", "cubin", "ttgir"]:
                        data = compiled.asm[ext]
                        (args.output / f"{name}.{ext}").write_bytes(data if isinstance(data, bytes) else data.encode())
                    result = dict(mode="k96" if enable else "k64", correct=True,
                                  cubin_sha256=hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
                                  shared_bytes=compiled.metadata.shared, tmem_columns=compiled.metadata.tmem_size,
                                  ptx_mmas=compiled.asm["ptx"].count("tcgen05.mma."), ms=[])
                    if enable:
                        result["baseline_max_abs"] = (output.float() - baseline.float()).abs().max().item()
                    else:
                        baseline = output
                    functions.append(fn)
                    case["variants"].append(result)
                base, cand = case["variants"]
                assert cand["ptx_mmas"] * 4 == base["ptx_mmas"] * 3
                assert base["shared_bytes"] == cand["shared_bytes"]
                for fn in functions:
                    for _ in range(3):
                        fn()
                torch.cuda.synchronize()
                for repeat in range(args.repeats):
                    for variant in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                        ms = triton.testing.do_bench_cudagraph(functions[variant], rep=args.rep_ms)
                        case["variants"][variant]["ms"].append(ms)
                for result in case["variants"]:
                    result["median_ms"] = statistics.median(result["ms"])
                    result["pflops"] = 2 * size**3 / (result["median_ms"] * 1e12)
                case["speedup"] = base["median_ms"] / cand["median_ms"]
                case["gpu_after"] = gpu_state()
                report["cases"].append(case)
                (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(case), flush=True)
    return report


if __name__ == "__main__":
    main()
