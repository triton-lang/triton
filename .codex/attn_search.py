import argparse
import csv
import importlib.util
import itertools
import math
import os
import pathlib
import time
import traceback
from dataclasses import replace

import torch
import triton


ROOT = pathlib.Path(__file__).resolve().parents[1]
ATTN_PATH = ROOT / "python/examples/gluon/01-attention-forward.py"


def load_attention_module():
    spec = importlib.util.spec_from_file_location("attn_fwd", ATTN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def dtype_from_name(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp8":
        return torch.float8_e5m2
    raise ValueError(name)


def candidate_configs(mod, case, stage):
    _, _, n_ctx, head_dim, dtype_name, causal = case
    dtype = dtype_from_name(dtype_name)
    selected = mod.select_kernel_config(head_dim, n_ctx, dtype, causal, use_tmem_red=False)
    if stage == "schedule":
        block_n_options = [64, 128]
        group_size_options = [1, 2, 4, 8] if causal else [1, 2, 4]
        for block_n, group_size_n in itertools.product(block_n_options, group_size_options):
            yield replace(selected, BLOCK_N=block_n, GROUP_SIZE_N=group_size_n, MAXNREG=128)
        return

    if stage == "buffer":
        kv_options = [2, 3] if head_dim == 128 else [2, 3, 4, 5, 6, 8]
        turnstile_options = [False, True] if head_dim == 64 else [False]
        for kv_buffers, use_turnstile in itertools.product(kv_options, turnstile_options):
            yield replace(selected, NUM_KV_BUFFERS=kv_buffers, USE_EXP2_TURNSTILE=use_turnstile)
        return

    split_options = [1, 2, 4, 8]
    maxnreg_options = [96, 112, 128]

    if stage == "pilot":
        split_options = sorted(set([selected.SPLIT_EXP_FACTOR, 4]))
        maxnreg_options = [128]

    use_tmem_options = [False]
    if not causal and dtype_name in {"fp16", "bf16", "fp8"} and torch.cuda.get_device_capability()[0:2] == (10, 3):
        use_tmem_options.append(True)

    for split, maxnreg, use_tmem in itertools.product(split_options, maxnreg_options, use_tmem_options):
        if use_tmem and causal:
            continue
        yield replace(selected, SPLIT_EXP_FACTOR=split, MAXNREG=maxnreg, USE_TMEM_RED=use_tmem)


def all_cases(stage):
    z = 4
    h = 32
    n_ctxs = [1024, 2048, 4096, 8192]
    head_dims = [64, 128]
    dtypes = [dtype for dtype in os.environ.get("ATTN_SEARCH_DTYPES", "fp16,bf16").split(",") if dtype]
    causals = [False, True]
    cases = list(itertools.product([z], [h], n_ctxs, head_dims, dtypes, causals))
    if stage == "pilot":
        return [
            (z, h, 1024, 64, "fp16", True),
            (z, h, 2048, 128, "fp16", False),
            (z, h, 2048, 128, "bf16", True),
            (z, h, 8192, 128, "fp16", False),
        ]
    return cases


def make_jobs(mod, stage):
    jobs = []
    for case in all_cases(stage):
        for p in candidate_configs(mod, case, stage):
            jobs.append((case, p))
    return jobs


def tflops(z, h, n_ctx, head_dim, causal, ms):
    flops = 4.0 * z * h * n_ctx * n_ctx * head_dim
    if causal:
        flops *= 0.5
    return flops * 1e-12 / (ms * 1e-3)


def run_job(mod, case, p, rep, validate, compile_only):
    z, h, n_ctx, head_dim, dtype_name, causal = case
    dtype = dtype_from_name(dtype_name)
    torch.manual_seed(1000 + n_ctx + head_dim + 17 * int(causal) + (1 if dtype_name == "bf16" else 0))
    q = torch.empty((z, h, n_ctx, head_dim), device="cuda").normal_(0.0, 0.5).to(dtype)
    k = torch.empty((z, h, n_ctx, head_dim), device="cuda").normal_(0.0, 0.5).to(dtype)
    v = torch.empty((z, h, n_ctx, head_dim), device="cuda").normal_(0.0, 0.5).to(dtype)
    o = torch.empty_like(q)
    m = torch.empty((z, h, n_ctx), device="cuda", dtype=torch.float32)

    start = time.perf_counter()
    mod.attention_forward(q, k, v, causal, 1.3, o, m, p.USE_TMEM_RED, p=p)
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - start

    if compile_only:
        return {
            "status": "ok",
            "Z": z,
            "H": h,
            "N_CTX": n_ctx,
            "HEAD_DIM": head_dim,
            "dtype": dtype_name,
            "causal": causal,
            "BLOCK_M": p.BLOCK_M,
            "BLOCK_N": p.BLOCK_N,
            "GROUP_SIZE_N": p.GROUP_SIZE_N,
            "SPLIT_EXP_FACTOR": p.SPLIT_EXP_FACTOR,
            "NUM_WARPS": p.NUM_WARPS,
            "MAXNREG": p.MAXNREG,
            "OCCUPANCY": p.OCCUPANCY,
            "USE_TMEM_RED": p.USE_TMEM_RED,
            "NUM_KV_BUFFERS": p.NUM_KV_BUFFERS,
            "USE_EXP2_TURNSTILE": p.USE_EXP2_TURNSTILE,
            "ms": math.nan,
            "tflops": math.nan,
            "compile_s": compile_s,
            "bench_s": 0.0,
            "max_abs": "",
            "error": "",
        }

    max_abs = ""
    if validate:
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.3, is_causal=causal)
        torch.cuda.synchronize()
        max_abs = (ref - o).abs().max().item()

    bench_start = time.perf_counter()
    ms = triton.testing.do_bench_cudagraph(
        lambda: mod.attention_forward(q, k, v, causal, 1.3, o, m, p.USE_TMEM_RED, p=p),
        rep=rep,
    )
    bench_s = time.perf_counter() - bench_start

    return {
        "status": "ok",
        "Z": z,
        "H": h,
        "N_CTX": n_ctx,
        "HEAD_DIM": head_dim,
        "dtype": dtype_name,
        "causal": causal,
        "BLOCK_M": p.BLOCK_M,
        "BLOCK_N": p.BLOCK_N,
        "GROUP_SIZE_N": p.GROUP_SIZE_N,
        "SPLIT_EXP_FACTOR": p.SPLIT_EXP_FACTOR,
        "NUM_WARPS": p.NUM_WARPS,
        "MAXNREG": p.MAXNREG,
        "OCCUPANCY": p.OCCUPANCY,
        "USE_TMEM_RED": p.USE_TMEM_RED,
        "NUM_KV_BUFFERS": p.NUM_KV_BUFFERS,
        "USE_EXP2_TURNSTILE": p.USE_EXP2_TURNSTILE,
        "ms": ms,
        "tflops": tflops(z, h, n_ctx, head_dim, causal, ms),
        "compile_s": compile_s,
        "bench_s": bench_s,
        "max_abs": max_abs,
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "broad", "schedule", "buffer"], default="pilot")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / ".codex/attn_search_results"))
    args = parser.parse_args()

    torch.cuda.set_device(0)
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    mod = load_attention_module()
    jobs = make_jobs(mod, args.stage)
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard]
    if args.limit is not None:
        jobs = jobs[:args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = pathlib.Path(args.out_dir) / f"{args.stage}_shard{args.shard}_of_{args.num_shards}.csv"
    fields = [
        "status",
        "Z",
        "H",
        "N_CTX",
        "HEAD_DIM",
        "dtype",
        "causal",
        "BLOCK_M",
        "BLOCK_N",
        "GROUP_SIZE_N",
        "SPLIT_EXP_FACTOR",
        "NUM_WARPS",
        "MAXNREG",
        "OCCUPANCY",
        "USE_TMEM_RED",
        "NUM_KV_BUFFERS",
        "USE_EXP2_TURNSTILE",
        "ms",
        "tflops",
        "compile_s",
        "bench_s",
        "max_abs",
        "error",
    ]

    start = time.perf_counter()
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, (case, p) in enumerate(jobs, start=1):
            try:
                row = run_job(mod, case, p, args.rep, args.validate, args.compile_only)
            except Exception:
                z, h, n_ctx, head_dim, dtype_name, causal = case
                row = {
                    "status": "error",
                    "Z": z,
                    "H": h,
                    "N_CTX": n_ctx,
                    "HEAD_DIM": head_dim,
                    "dtype": dtype_name,
                    "causal": causal,
                    "BLOCK_M": p.BLOCK_M,
                    "BLOCK_N": p.BLOCK_N,
                    "GROUP_SIZE_N": p.GROUP_SIZE_N,
                    "SPLIT_EXP_FACTOR": p.SPLIT_EXP_FACTOR,
                    "NUM_WARPS": p.NUM_WARPS,
                    "MAXNREG": p.MAXNREG,
                    "OCCUPANCY": p.OCCUPANCY,
                    "USE_TMEM_RED": p.USE_TMEM_RED,
                    "NUM_KV_BUFFERS": p.NUM_KV_BUFFERS,
                    "USE_EXP2_TURNSTILE": p.USE_EXP2_TURNSTILE,
                    "ms": math.nan,
                    "tflops": math.nan,
                    "compile_s": math.nan,
                    "bench_s": math.nan,
                    "max_abs": "",
                    "error": traceback.format_exc(limit=2).replace("\n", " | "),
                }
            writer.writerow(row)
            f.flush()
            elapsed = time.perf_counter() - start
            print(
                f"shard {args.shard}/{args.num_shards} {idx}/{len(jobs)} "
                f"{row['status']} N={row['N_CTX']} D={row['HEAD_DIM']} "
                f"dtype={row['dtype']} causal={row['causal']} split={row['SPLIT_EXP_FACTOR']} "
                f"maxnreg={row['MAXNREG']} tmem={row['USE_TMEM_RED']} "
                f"kv={row['NUM_KV_BUFFERS']} turnstile={row['USE_EXP2_TURNSTILE']} elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
