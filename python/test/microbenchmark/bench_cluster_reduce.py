import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl


@gluon.constexpr_function
def _make_cga_bases_2d(cluster_size):
    if cluster_size <= 1:
        return []
    n = cluster_size.bit_length() - 1
    return [[1 << i, 0] for i in range(n)]


@gluon.jit
def cluster_reduce_vector_kernel(x_ptr, y_ptr, tile_size: ttgl.constexpr, cluster_size: ttgl.constexpr):
    num_warps: ttgl.constexpr = ttgl.num_warps()
    cga_bases: ttgl.constexpr = _make_cga_bases_2d(cluster_size)
    layout_cga: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0], cga_bases)
    rows = ttgl.arange(0, cluster_size, layout=ttgl.SliceLayout(dim=1, parent=layout_cga))[:, None]
    cols = ttgl.arange(0, tile_size, layout=ttgl.SliceLayout(dim=0, parent=layout_cga))[None, :]
    offs = rows * tile_size + cols
    x = ttgl.load(x_ptr + offs).to(ttgl.float32)
    y = ttgl.sum(x, axis=0)
    ttgl.store(y_ptr + offs, y[None, :])


def _parse_csv_ints(s):
    return [int(x) for x in s.split(",") if x]


def _run_case(cluster_size, tile_size, iters, warmup):
    torch.manual_seed(0)
    x = torch.randn(cluster_size * tile_size, device="cuda", dtype=torch.float16)
    y = torch.zeros(cluster_size * tile_size, device="cuda", dtype=torch.float32)
    compiled = cluster_reduce_vector_kernel[(1, )](
        x,
        y,
        tile_size=tile_size,
        cluster_size=cluster_size,
        num_warps=4,
        num_ctas=cluster_size,
    )
    torch.cuda.synchronize()
    ref = x.view(cluster_size, tile_size).float().sum(dim=0)
    got = y.view(cluster_size, tile_size)[0]
    ok = torch.allclose(got, ref, atol=1e-3, rtol=1e-3)
    ptx = compiled.asm["ptx"]
    ms = triton.testing.do_bench(
        lambda: cluster_reduce_vector_kernel[(1, )](
            x,
            y,
            tile_size=tile_size,
            cluster_size=cluster_size,
            num_warps=4,
            num_ctas=cluster_size,
        ),
        warmup=warmup,
        rep=iters,
    )
    return {
        "cluster_size": cluster_size,
        "tile_size": tile_size,
        "ms": ms,
        "ok": int(ok),
        "bulk": int("cp.async.bulk.shared::cluster.shared::cta" in ptx),
        "cluster_ld": int("ld.shared::cluster" in ptx),
    }


def _run_child(args):
    res = _run_case(args.cluster_size, args.tile_size, args.iters, args.warmup)
    print(
        f"{args.mode},{res['cluster_size']},{res['tile_size']},{res['ms']:.6f},"
        f"{res['ok']},{res['bulk']},{res['cluster_ld']}",
        flush=True,
    )


def _write_svg(rows, out_svg):
    import math

    width, height = 1200, 900
    margin = 70
    plot_w = (width - 3 * margin) / 2
    plot_h = (height - 3 * margin) / 2
    colors = {"old": "#4c78a8", "new": "#f58518"}
    clusters = sorted({r["cluster_size"] for r in rows})
    max_y = max(r["ms"] for r in rows) * 1.15
    min_x = min(r["tile_size"] for r in rows)
    max_x = max(r["tile_size"] for r in rows)
    log_min, log_max = math.log2(min_x), math.log2(max_x)
    if log_min == log_max:
        log_max = log_min + 1.0

    def sx(x, left):
        return left + (math.log2(x) - log_min) / (log_max - log_min) * plot_w

    def sy(y, top):
        return top + plot_h - y / max_y * plot_h

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append('<text x="40" y="35" font-size="24" font-family="sans-serif">Cluster Reduce old vs new</text>')
    for idx, cs in enumerate(clusters):
        col = idx % 2
        row = idx // 2
        left = margin + col * (plot_w + margin)
        top = margin + row * (plot_h + margin)
        svg.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#ddd"/>')
        svg.append(f'<text x="{left+5}" y="{top-10}" font-size="18" font-family="sans-serif">cluster_size={cs}</text>')
        svg.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#999"/>')
        svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#999"/>')
        for mode in ["old", "new"]:
            sub = sorted((r["tile_size"], r["ms"]) for r in rows if r["cluster_size"] == cs and r["mode"] == mode)
            path = " ".join(
                ("M" if i == 0 else "L") + f"{sx(x, left):.1f},{sy(y, top):.1f}" for i, (x, y) in enumerate(sub))
            svg.append(f'<path d="{path}" fill="none" stroke="{colors[mode]}" stroke-width="3"/>')
            for x, y in sub:
                svg.append(f'<circle cx="{sx(x, left):.1f}" cy="{sy(y, top):.1f}" r="3.5" fill="{colors[mode]}"/>')
        if idx == 0:
            svg.append(f'<rect x="{left+plot_w-150}" y="{top+10}" width="140" height="50" fill="white" stroke="#ddd"/>')
            svg.append(
                f'<line x1="{left+160}" y1="{top+28}" x2="{left+190}" y2="{top+28}" stroke="{colors["old"]}" stroke-width="3"/>'
            )
            svg.append(f'<text x="{left+198}" y="{top+32}" font-size="12" font-family="sans-serif">old</text>')
            svg.append(
                f'<line x1="{left+160}" y1="{top+45}" x2="{left+190}" y2="{top+45}" stroke="{colors["new"]}" stroke-width="3"/>'
            )
            svg.append(f'<text x="{left+198}" y="{top+49}" font-size="12" font-family="sans-serif">new</text>')
    svg.append("</svg>")
    out_svg.write_text("\n".join(svg))


def _run_parent(args):
    rows = []
    for cluster_size in _parse_csv_ints(args.cluster_sizes):
        for tile_size in _parse_csv_ints(args.tile_sizes):
            for mode in ["old", "new"]:
                env = os.environ.copy()
                if mode == "old":
                    env["TRITON_DISABLE_CLUSTER_REDUCE_BULK"] = "1"
                else:
                    env.pop("TRITON_DISABLE_CLUSTER_REDUCE_BULK", None)
                env["TRITON_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), f"triton_cluster_reduce_vector_{mode}")
                proc = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--mode",
                        mode,
                        "--cluster-size",
                        str(cluster_size),
                        "--tile-size",
                        str(tile_size),
                        "--iters",
                        str(args.iters),
                        "--warmup",
                        str(args.warmup),
                    ],
                    cwd=os.getcwd(),
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                line = proc.stdout.strip()
                print(line)
                mode_s, cs, ts, ms, ok, bulk, cluster_ld = line.split(",")
                rows.append({
                    "mode": mode_s,
                    "cluster_size": int(cs),
                    "tile_size": int(ts),
                    "ms": float(ms),
                    "ok": int(ok),
                    "bulk": int(bulk),
                    "cluster_ld": int(cluster_ld),
                })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "cluster_size", "tile_size", "ms", "ok", "bulk", "cluster_ld"])
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for r in rows:
        key = (r["cluster_size"], r["tile_size"])
        grouped.setdefault(key, {})[r["mode"]] = r
    summary = []
    for (cs, ts), modes in sorted(grouped.items()):
        old = modes["old"]
        new = modes["new"]
        delta_ms = new["ms"] - old["ms"]
        delta_pct = delta_ms / old["ms"] * 100.0
        summary.append({
            "cluster_size": cs,
            "tile_size": ts,
            "old_ms": old["ms"],
            "new_ms": new["ms"],
            "delta_ms": delta_ms,
            "delta_pct": delta_pct,
        })

    out_summary = Path(args.out_summary)
    with out_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f,
                                fieldnames=["cluster_size", "tile_size", "old_ms", "new_ms", "delta_ms", "delta_pct"])
        writer.writeheader()
        writer.writerows(summary)

    out_svg = Path(args.out_svg)
    _write_svg(rows, out_svg)
    print(f"summary_csv={out_summary}")
    print(f"plot_svg={out_svg}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vector multi-CTA cluster reduce lowering.")
    parser.add_argument("--mode", choices=["old", "new", "both"], default="both")
    parser.add_argument("--cluster-sizes", default="2,4,8,16")
    parser.add_argument("--tile-sizes", default="64,256,1024,4096,8192,16384")
    parser.add_argument("--cluster-size", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--out-csv", default="/tmp/cluster_reduce_vector_results.csv")
    parser.add_argument("--out-summary", default="/tmp/cluster_reduce_vector_summary.csv")
    parser.add_argument("--out-svg", default="/tmp/cluster_reduce_vector_summary.svg")
    args = parser.parse_args()

    if args.mode == "both":
        _run_parent(args)
    else:
        _run_child(args)


if __name__ == "__main__":
    main()
