import argparse
import csv
import glob
import importlib.util
import math
import os
import pathlib
import time
from dataclasses import replace

import torch
import triton


ROOT = pathlib.Path(__file__).resolve().parents[1]
SEARCH_PATH = ROOT / ".codex/attn_search.py"


def load_search_module():
    spec = importlib.util.spec_from_file_location("attn_search", SEARCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


search = load_search_module()


FIELDS = [
    "label",
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


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def case_key(row):
    return (
        int(row["Z"]),
        int(row["H"]),
        int(row["N_CTX"]),
        int(row["HEAD_DIM"]),
        row["dtype"],
        parse_bool(row["causal"]),
    )


def config_key(p):
    return (
        p.BLOCK_M,
        p.BLOCK_N,
        p.GROUP_SIZE_N,
        p.SPLIT_EXP_FACTOR,
        p.NUM_WARPS,
        p.MAXNREG,
        p.OCCUPANCY,
        p.USE_TMEM_RED,
        p.NUM_KV_BUFFERS,
        p.USE_EXP2_TURNSTILE,
    )


def fill_config(mod, p, case):
    _, _, n_ctx, head_dim, dtype_name, causal = case
    dtype = search.dtype_from_name(dtype_name)
    if p.GROUP_SIZE_N is None:
        p = replace(p, GROUP_SIZE_N=4 if causal else 1)
    if p.SPLIT_EXP_FACTOR is None:
        p = replace(p, SPLIT_EXP_FACTOR=max(1, 256 // head_dim))
    if p.NUM_KV_BUFFERS is None:
        p = replace(p, NUM_KV_BUFFERS=mod._default_num_kv_buffers(head_dim, dtype))
    if p.USE_EXP2_TURNSTILE is None:
        p = replace(p, USE_EXP2_TURNSTILE=head_dim == 64)
    return p


def legacy_config(mod, case):
    _, _, _, head_dim, dtype_name, causal = case
    dtype = search.dtype_from_name(dtype_name)
    return mod.KernelConfig(
        BLOCK_M=256,
        BLOCK_N=128,
        GROUP_SIZE_N=4 if causal else 1,
        SPLIT_EXP_FACTOR=max(1, 256 // head_dim),
        NUM_WARPS=4,
        MAXNREG=128,
        OCCUPANCY=1,
        USE_TMEM_RED=False,
        NUM_KV_BUFFERS=mod._default_num_kv_buffers(head_dim, dtype),
        USE_EXP2_TURNSTILE=head_dim == 64,
    )


def selected_config(mod, case):
    _, _, n_ctx, head_dim, dtype_name, causal = case
    dtype = search.dtype_from_name(dtype_name)
    return fill_config(mod, mod.select_kernel_config(head_dim, n_ctx, dtype, causal, False), case)


def row_config(mod, row):
    case = case_key(row)
    p = selected_config(mod, case)
    int_fields = [
        "BLOCK_M",
        "BLOCK_N",
        "GROUP_SIZE_N",
        "SPLIT_EXP_FACTOR",
        "NUM_WARPS",
        "MAXNREG",
        "OCCUPANCY",
        "NUM_KV_BUFFERS",
    ]
    bool_fields = ["USE_TMEM_RED", "USE_EXP2_TURNSTILE"]
    for field in int_fields:
        if field in row and row[field] not in ("", "None"):
            p = replace(p, **{field: int(row[field])})
    for field in bool_fields:
        if field in row and row[field] not in ("", "None"):
            p = replace(p, **{field: parse_bool(row[field])})
    return fill_config(mod, p, case)


def add_candidate(dst, case, label, p):
    seen = {config_key(existing_p) for _, existing_p in dst[case]}
    if config_key(p) not in seen:
        dst[case].append((label, p))


def source_candidates(mod, top_k_source, search_root):
    cases = search.all_cases("broad")
    by_case = {case: [] for case in cases}
    for case in cases:
        add_candidate(by_case, case, "current", selected_config(mod, case))
        add_candidate(by_case, case, "legacy", legacy_config(mod, case))

    for dirname in ["broad", "schedule", "buffer"]:
        rows = []
        for path in glob.glob(str(pathlib.Path(search_root) / dirname / "*.csv")):
            rows.extend(row for row in csv.DictReader(open(path)) if row.get("status") == "ok")
        grouped = {}
        for row in rows:
            grouped.setdefault(case_key(row), []).append(row)
        for case, case_rows in grouped.items():
            for rank, row in enumerate(sorted(case_rows, key=lambda r: float(r["ms"]))[:top_k_source], start=1):
                add_candidate(by_case, case, f"{dirname}_top{rank}", row_config(mod, row))
    return by_case


def narrow_candidates(mod, narrow_dir, top_k):
    rows = []
    for path in glob.glob(str(pathlib.Path(narrow_dir) / "*.csv")):
        rows.extend(row for row in csv.DictReader(open(path)) if row.get("status") == "ok")
    grouped = {}
    for row in rows:
        grouped.setdefault(case_key(row), []).append(row)

    by_case = {}
    for case, case_rows in grouped.items():
        by_case[case] = []
        add_candidate(by_case, case, "current", selected_config(mod, case))
        add_candidate(by_case, case, "legacy", legacy_config(mod, case))
        for rank, row in enumerate(sorted(case_rows, key=lambda r: float(r["ms"]))[:top_k], start=1):
            add_candidate(by_case, case, f"narrow_top{rank}_{row.get('label', '')}", row_config(mod, row))
    return by_case


def make_jobs(by_case):
    jobs = []
    for case in sorted(by_case):
        for label, p in by_case[case]:
            jobs.append((case, label, p))
    return jobs


def run_labeled_job(mod, case, label, p, rep):
    row = search.run_job(mod, case, p, rep=rep, validate=False, compile_only=False)
    row["label"] = label
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", choices=["source", "narrow"], default="source")
    parser.add_argument("--narrow-dir", default=str(ROOT / ".codex/attn_finalist_results/narrow"))
    parser.add_argument("--search-root", default=str(ROOT / ".codex/attn_search_results"))
    parser.add_argument("--top-k-source", type=int, default=4)
    parser.add_argument("--top-k-narrow", type=int, default=4)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out-dir", default=str(ROOT / ".codex/attn_finalist_results"))
    args = parser.parse_args()

    torch.cuda.set_device(0)
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    mod = search.load_attention_module()
    by_case = source_candidates(mod, args.top_k_source, args.search_root) if args.input == "source" else narrow_candidates(
        mod, args.narrow_dir, args.top_k_narrow)
    jobs = make_jobs(by_case)
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = pathlib.Path(args.out_dir) / f"{args.input}_shard{args.shard}_of_{args.num_shards}.csv"
    start = time.perf_counter()
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for idx, (case, label, p) in enumerate(jobs, start=1):
            try:
                row = run_labeled_job(mod, case, label, p, args.rep)
            except Exception as exc:
                z, h, n_ctx, head_dim, dtype_name, causal = case
                row = {
                    "label": label,
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
                    "error": repr(exc),
                }
            writer.writerow(row)
            f.flush()
            elapsed = time.perf_counter() - start
            print(
                f"{idx}/{len(jobs)} {row['status']} {label} N={row['N_CTX']} D={row['HEAD_DIM']} "
                f"dtype={row['dtype']} causal={row['causal']} ms={row['ms']} elapsed={elapsed:.1f}s",
                flush=True,
            )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
