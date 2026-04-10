# Simple dense matmul benchmark calling in matmul.py
# This is convenient for simple performance measurements and bring up of new dtypes or targets.
# This is not meant to be a comprehensive benchmark of triton_kernels.

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.profiler as proton
from triton_kernels.matmul import PrecisionConfig, matmul
from triton_kernels.tensor import make_ragged_tensor_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dense triton_kernels.matmul cases with and without transposed W layout.")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--semantic-mode",
        choices=("dense", "openai_ragged"),
        default="dense",
        help="dense uses plain 2D inputs; openai_ragged matches the single-expert ragged OpenAI harness.",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=("cuda_graph", "eager"),
        default="cuda_graph",
        help="Timing method. cuda_graph uses triton.testing.do_bench_cudagraph.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup time in ms for eager timing only.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Rep count for cuda_graph timing or rep time in ms for eager timing.",
    )
    parser.add_argument(
        "--transpose-w",
        choices=("both", "false", "true"),
        default="both",
        help="Which W layout(s) to benchmark. 'true' matches transpose-contiguous-transpose layout.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="If set, use Proton as the measurement path and emit .hatchet profiles into this directory.",
    )
    parser.add_argument(
        "--profile-reps",
        type=int,
        default=100,
        help="Number of kernel invocations to record when --profile-dir is set.",
    )
    return parser.parse_args()


def make_weight(k: int, n: int, dtype: torch.dtype, device: torch.device, transpose_w: bool) -> torch.Tensor:
    weight = torch.randn((k, n), device=device, dtype=dtype)
    if not transpose_w:
        return weight.contiguous()
    return weight.transpose(-1, -2).contiguous().transpose(-1, -2)


def make_batched_weight(k: int, n: int, dtype: torch.dtype, device: torch.device, transpose_w: bool) -> torch.Tensor:
    weight = torch.randn((1, k, n), device=device, dtype=dtype)
    if not transpose_w:
        return weight.contiguous()
    return weight.transpose(-1, -2).contiguous().transpose(-1, -2)


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2.0 * m * n * k / (ms * 1e9)


def read_proton_matmul_avg_ms(profile_path: Path) -> float:
    profile = json.loads(profile_path.read_text())
    candidates = []

    def collect(node: dict) -> None:
        frame = node.get("frame", {})
        metrics = node.get("metrics", {})
        if "_p_matmul" in frame.get("name", ""):
            candidates.append(metrics)
        for child in node.get("children", []):
            collect(child)

    for node in profile:
        collect(node)

    if not candidates:
        raise RuntimeError(f"No matmul kernel found in Proton profile {profile_path}")

    matmul_metrics = max(candidates, key=lambda metrics: metrics.get("flops16", 0))
    return matmul_metrics["time (ns)"] / matmul_metrics["count"] / 1e6


def make_inputs(
    *,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    transpose_w: bool,
    semantic_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict, tuple[int, ...]]:
    if semantic_mode == "dense":
        a = torch.randn((m, k), device=device, dtype=dtype)
        b = make_weight(k, n, dtype, device, transpose_w)
        return a, b, None, {}, tuple(b.stride())

    slice_sizes = torch.tensor([m], dtype=torch.int32, device=device)
    a_ragged_metadata = make_ragged_tensor_metadata(slice_sizes, m)
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = make_batched_weight(k, n, dtype, device, transpose_w)
    bias = torch.randn((1, n), device=device, dtype=torch.float32)
    return a, b, bias, {"a_ragged_metadata": a_ragged_metadata}, tuple(b.stride())


def make_precision_config(dtype: torch.dtype, semantic_mode: str) -> PrecisionConfig:
    # KI's matmul wrapper forwards flexpoint_saturate_inf=True into triton_kernels
    # even for the fp16 benchmark path. Match that when reproducing the OpenAI harness.
    if semantic_mode == "openai_ragged":
        return PrecisionConfig(flexpoint_saturate_inf=True)
    return PrecisionConfig(
        out_dtype=dtype,
        flexpoint_saturate_inf=False,
    )


def benchmark_case(
    *,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    transpose_w: bool,
    semantic_mode: str,
    benchmark_mode: str,
    warmup: int,
    rep: int,
    profile_dir: Path | None,
    profile_reps: int,
) -> tuple[float, float, tuple[int, ...]]:
    a, b, bias, matmul_kwargs, b_stride = make_inputs(
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        device=device,
        transpose_w=transpose_w,
        semantic_mode=semantic_mode,
    )
    precision_config = make_precision_config(dtype, semantic_mode)
    dtype_name = str(dtype).removeprefix("torch.")

    def run() -> torch.Tensor:
        return matmul(a, b, bias, precision_config=precision_config, **matmul_kwargs)

    # Compile outside the measured region so the reported numbers reflect steady-state execution.
    run()
    torch.cuda.synchronize(device)

    if profile_dir is not None:
        if profile_reps <= 0:
            raise ValueError("--profile-reps must be positive")
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_name = profile_dir / f"{semantic_mode}_{dtype_name}_wt_{str(transpose_w).lower()}"
        proton.start(str(profile_name), hook="triton")
        for _ in range(profile_reps):
            run()
        torch.cuda.synchronize(device)
        proton.finalize()
        ms = read_proton_matmul_avg_ms(profile_name.with_suffix(".hatchet"))
        return ms, tflops(m, n, k, ms), b_stride

    if benchmark_mode == "cuda_graph":
        ms = triton.testing.do_bench_cudagraph(run, rep=rep)
    else:
        ms = triton.testing.do_bench(run, warmup=warmup, rep=rep)
    return ms, tflops(m, n, k, ms), b_stride


def iter_transpose_values(choice: str) -> list[bool]:
    if choice == "both":
        return [False, True]
    return [choice == "true"]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    dtype = getattr(torch, args.dtype)
    measurement_mode = "proton" if args.profile_dir is not None else args.benchmark_mode

    print(f"Benchmarking dense triton_kernels.matmul with M={args.m}, N={args.n}, K={args.k}, "
          f"dtype={args.dtype}, device={device}, semantic_mode={args.semantic_mode}, "
          f"measurement_mode={measurement_mode}")
    print("transpose_w  b.stride()      time_ms   tflops")
    print("-----------  -------------  --------  -------")

    for transpose_w in iter_transpose_values(args.transpose_w):
        ms, perf_tflops, b_stride = benchmark_case(
            m=args.m,
            n=args.n,
            k=args.k,
            dtype=dtype,
            device=device,
            transpose_w=transpose_w,
            semantic_mode=args.semantic_mode,
            benchmark_mode=args.benchmark_mode,
            warmup=args.warmup,
            rep=args.rep,
            profile_dir=args.profile_dir,
            profile_reps=args.profile_reps,
        )
        print(f"{str(transpose_w):>11}  {str(b_stride):>13}  {ms:8.3f}  {perf_tflops:7.3f}")

    if args.profile_dir is not None:
        print(f"Proton profiles written to {args.profile_dir}")


if __name__ == "__main__":
    main()
