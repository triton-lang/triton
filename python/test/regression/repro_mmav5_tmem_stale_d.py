#!/usr/bin/env python3
"""Minimal MMAv5/TMEM repro.

Run on gb200:
  TRITON_ALWAYS_COMPILE=1 DISABLE_PTXAS_OPT=1 \
    python python/test/regression/repro_mmav5_tmem_stale_d.py \
      --iters 200 --progress 25 --noise-size 4096

Optimized ptxas path with inline-asm sleep:
  TRITON_ALWAYS_COMPILE=1 DISABLE_PTXAS_OPT=0 \
    python python/test/regression/repro_mmav5_tmem_stale_d.py \
      --iters 200 --progress 25 --noise-size 4096 --sleep-cycles 1
"""

import argparse
import threading

import torch
import triton
import triton.language as tl


BLOCK_N = 64
BLOCK_HEADS = 16
DREL = 16
ATOL = 1e-5
SEED = 0


@triton.jit
def tmem_mmav5_leak_repro_kernel(
    X,
    DY,
    DWT_OUT,
    stride_x_pid,
    stride_x_head,
    stride_dy_pid,
    stride_dy_head,
    stride_out_pid,
    stride_out_row,
    SLEEP_CYCLES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    DREL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_HEADS)
    offs_d = tl.arange(0, DREL)

    x_ptr = X + pid * stride_x_pid + offs_h[:, None] * stride_x_head + offs_d[None, :]
    x = tl.load(x_ptr)
    dy_ptr = DY + pid * stride_dy_pid + offs_h[:, None] * stride_dy_head + offs_n[None, :]
    dy = tl.load(dy_ptr)
    dw_t = tl.dot(tl.trans(dy), x)
    if SLEEP_CYCLES > 0:
        tl.inline_asm_elementwise(
            f"nanosleep.u32 {SLEEP_CYCLES}; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    tl.store(
        DWT_OUT + pid * stride_out_pid + offs_n[:, None] * stride_out_row + offs_d[None, :],
        dw_t,
    )


def make_inputs(num_pids: int, device: torch.device):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED)
    x = torch.randn((num_pids, BLOCK_HEADS, DREL), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    dy = torch.randn((num_pids, BLOCK_HEADS, BLOCK_N), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    return x.to(device), dy.to(device)


class NoiseThread(threading.Thread):
    def __init__(self, stop_event: threading.Event, size: int):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.size = size

    def run(self):
        stream = torch.cuda.Stream()
        x = torch.randn((self.size, self.size), dtype=torch.bfloat16, device="cuda")
        while not self.stop_event.is_set():
            with torch.cuda.stream(stream):
                x = (x @ x.mT) @ x


def run_once(x: torch.Tensor, dy: torch.Tensor, num_pids: int, sleep_cycles: int):
    out = torch.zeros((num_pids, BLOCK_N, DREL), dtype=torch.float32, device=x.device)
    tmem_mmav5_leak_repro_kernel[(num_pids,)](
        x,
        dy,
        out,
        *x.stride()[:2],
        *dy.stride()[:2],
        *out.stride()[:-1],
        SLEEP_CYCLES=sleep_cycles,
        BLOCK_N=BLOCK_N,
        BLOCK_HEADS=BLOCK_HEADS,
        DREL=DREL,
        num_warps=4,
        maxnreg=255,
        num_stages=1,
    )
    return out


def first_diff(lhs: torch.Tensor, rhs: torch.Tensor):
    lhs_nan = torch.isnan(lhs)
    rhs_nan = torch.isnan(rhs)
    bad_mask = (lhs_nan != rhs_nan) | ((~lhs_nan & ~rhs_nan) & ((lhs - rhs).abs() > ATOL))
    idx = torch.nonzero(bad_mask, as_tuple=False)
    if idx.numel() == 0:
        return None
    first = idx[0].tolist()
    lhs_val = lhs[tuple(first)].item()
    rhs_val = rhs[tuple(first)].item()
    diff_val = float("nan") if (lhs_nan[tuple(first)] or rhs_nan[tuple(first)]) else abs(lhs_val - rhs_val)
    return first, lhs_val, rhs_val, diff_val


def stress(iters: int, progress: int, noise_size: int, sleep_cycles: int):
    num_pids = 2 * torch.cuda.get_device_properties(0).multi_processor_count
    x, dy = make_inputs(num_pids, torch.device("cuda"))
    reference = run_once(x, dy, num_pids, sleep_cycles).cpu()

    stop_event = threading.Event()
    noise_thread = None
    if noise_size > 0:
        noise_thread = NoiseThread(stop_event, noise_size)
        noise_thread.start()

    try:
        for outer in range(1, iters + 1):
            out = run_once(x, dy, num_pids, sleep_cycles)
            bad = first_diff(out.cpu(), reference)
            if bad is not None:
                idx, cur, ref, diff = bad
                print(
                    f"failure outer={outer} first_diff={idx} "
                    f"cur={cur} ref={ref} abs_err={diff}"
                )
                return 1
            if outer % progress == 0:
                print(f"outer={outer} clean")
        print(f"clean for {iters} iterations")
        return 0
    finally:
        if noise_thread is not None:
            stop_event.set()
            noise_thread.join(timeout=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--progress", type=int, default=25)
    parser.add_argument("--noise-size", type=int, default=4096)
    parser.add_argument("--sleep-cycles", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(stress(args.iters, args.progress, args.noise_size, args.sleep_cycles))


if __name__ == "__main__":
    main()
