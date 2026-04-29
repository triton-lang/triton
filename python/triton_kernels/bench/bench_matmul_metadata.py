import argparse
import dataclasses
import math
import time

import torch

from triton_kernels.matmul_details._common import _matmul_flops_and_bytes_from_slices


@dataclasses.dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype

    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    def numel(self) -> int:
        return math.prod(self.shape)


def _old_flops_and_bytes_from_slices(args, M, N, K, X, Y, W, slice_sizes, nbits, batch_size):
    n_tokens = slice_sizes.sum()
    z = 1 if args["RAGGED_DIMENSION"] == "K" else batch_size

    if args["RAGGED_DIMENSION"] == "K":
        flops = n_tokens.to(torch.float64) * (2.0 * M * N * z)
        n_x_bytes = n_tokens * X.shape[-2] * X.element_size()
        n_y_bytes = Y.numel() * Y.element_size() * (2 if args["OutAcc"] is not None else 1)
        n_w_bytes = n_tokens * W.shape[-1] * W.element_size()
    else:
        if M is None:
            flops = n_tokens.to(torch.float64) * (2.0 * N * K * z)
        elif K is None:
            flops = n_tokens.to(torch.float64) * (2.0 * M * N * z)
        else:
            flops = torch.empty((), dtype=torch.float64, device=slice_sizes.device)
            flops.fill_(2.0 * M * N * K * z)
        n_x_bytes = n_tokens * X.shape[-1] * X.element_size()
        n_y_bytes = n_tokens * Y.shape[-1] * Y.element_size()
        n_w_bytes = (W.numel() * W.element_size() // slice_sizes.numel()) * (slice_sizes > 0).sum()

    return {f"flops{nbits}": flops, "bytes": n_x_bytes + n_y_bytes + n_w_bytes}


def _make_slice_sizes(n_slices, max_slice_size, active_fraction, device):
    n_active = max(1, min(n_slices, round(n_slices * active_fraction)))
    slice_sizes = torch.zeros((n_slices, ), dtype=torch.int32, device=device)
    active = torch.randperm(n_slices, device=device)[:n_active]
    slice_sizes[active] = torch.randint(1, max_slice_size + 1, (n_active, ), dtype=torch.int32, device=device)
    return slice_sizes


def _make_case(args, mode, device):
    dtype = getattr(torch, args.dtype)
    slice_sizes = _make_slice_sizes(args.n_slices, args.max_slice_size, args.active_fraction, device)
    total_capacity = args.n_slices * args.max_slice_size

    if mode == "ragged_k":
        metadata_args = {"RAGGED_DIMENSION": "K", "OutAcc": object() if args.out_acc else None}
        X = FakeTensor((args.m, total_capacity), dtype)
        Y = FakeTensor((args.m, args.n), dtype)
        W = FakeTensor((total_capacity, args.n), dtype)
        return metadata_args, args.m, args.n, None, X, Y, W, slice_sizes, dtype.itemsize * 8, 1

    metadata_args = {"RAGGED_DIMENSION": "M", "OutAcc": None}
    X = FakeTensor((total_capacity, args.k), dtype)
    Y = FakeTensor((total_capacity, args.n), dtype)
    W = FakeTensor((args.n_slices, args.k, args.n), dtype)
    return metadata_args, None, args.n, args.k, X, Y, W, slice_sizes, dtype.itemsize * 8, args.batch_size


def _bench(fn, iters, warmup, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    last = None
    for _ in range(iters):
        last = fn()
    end_event.record()
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - wall_start
    # Touch the result once after timing to catch bad kernels without putting .item()
    # synchronization in the measured loop.
    for value in last.values():
        assert value.numel() == 1
    return start_event.elapsed_time(end_event) * 1000.0 / iters, wall_s * 1e6 / iters


def _run_case(args, mode, device):
    case = _make_case(args, mode, device)
    metadata_args, M, N, K, X, Y, W, slice_sizes, nbits, batch_size = case

    def old_fn():
        return _old_flops_and_bytes_from_slices(metadata_args, M, N, K, X, Y, W, slice_sizes, nbits, batch_size)

    def new_fn():
        return _matmul_flops_and_bytes_from_slices(metadata_args, M, N, K, X, Y, W, slice_sizes, nbits, batch_size)

    old = old_fn()
    new = new_fn()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(new[f"flops{nbits}"].cpu(), old[f"flops{nbits}"].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(new["bytes"].cpu(), old["bytes"].to(torch.int64).cpu(), rtol=0, atol=0)

    old_gpu_us, old_wall_us = _bench(old_fn, args.iters, args.warmup, device)
    new_gpu_us, new_wall_us = _bench(new_fn, args.iters, args.warmup, device)
    print(
        f"{mode},{args.n_slices},{args.max_slice_size},{args.active_fraction},"
        f"{old_gpu_us:.3f},{new_gpu_us:.3f},{old_gpu_us / new_gpu_us:.2f},"
        f"{old_wall_us:.3f},{new_wall_us:.3f},{old_wall_us / new_wall_us:.2f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark ragged matmul launch-metadata flops/bytes counters.")
    parser.add_argument("--mode", choices=("ragged_m", "ragged_k", "both"), default="both")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-slices", type=int, default=256)
    parser.add_argument("--max-slice-size", type=int, default=256)
    parser.add_argument("--active-fraction", type=float, default=0.8)
    parser.add_argument("--out-acc", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(0)

    modes = ("ragged_m", "ragged_k") if args.mode == "both" else (args.mode, )
    print(
        "mode,n_slices,max_slice_size,active_fraction,old_gpu_us,new_gpu_us,gpu_speedup,old_wall_us,new_wall_us,wall_speedup"
    )
    for mode in modes:
        _run_case(args, mode, device)


if __name__ == "__main__":
    main()
