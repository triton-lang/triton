#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--engine-tpm-csv", required=True)
p.add_argument("--zen-tpm-csv", required=True)
p.add_argument("--target-tpm-csv", required=True)
p.add_argument("--prod-case1-tbt", type=float, default=12.0)
p.add_argument("--prod-case1-tpm-low", type=float, default=600_000.0)
p.add_argument("--prod-case1-tpm-high", type=float, default=1_000_000.0)
p.add_argument("--prod-case2-tbt", type=float, default=9.0)
p.add_argument("--prod-case2-tpm-low", type=float, default=200_000.0)
p.add_argument("--prod-case2-tpm-high", type=float, default=600_000.0)
p.add_argument("--output", required=True)
p.add_argument("--csv-output", required=True)
a = p.parse_args()


def fit_curve(df, x, y, bins=36, points=160):
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(x)
    if len(d) < 3 or d[x].nunique() < 3:
        return d
    d = d.copy()
    if len(d) > bins:
        d["bin"] = pd.cut(
            d[x],
            np.linspace(d[x].min(), d[x].max(), bins + 1),
            include_lowest=True,
            duplicates="drop",
        )
        d = (
            d.groupby("bin", observed=True)
            .agg(**{x: (x, "median"), y: (y, "median")})
            .dropna()
            .reset_index(drop=True)
        )
    degree = min(3, d[x].nunique() - 1)
    if degree < 1:
        return d[[x, y]]
    coeff = np.polyfit(d[x], d[y], degree)
    xs = np.linspace(d[x].min(), d[x].max(), points)
    ys = np.maximum(np.polyval(coeff, xs), 0)
    return pd.DataFrame({x: xs, y: ys})


frames = []
if Path(a.engine_tpm_csv).exists():
    df = pd.read_csv(a.engine_tpm_csv)
    if {"tbt_ms", "tpm_per_a100e"}.issubset(df.columns):
        frames.append(
            pd.DataFrame(
                {
                    "source": "engine",
                    "tbt_ms": pd.to_numeric(df["tbt_ms"], errors="coerce"),
                    "tpm_per_a100e": pd.to_numeric(df["tpm_per_a100e"], errors="coerce"),
                }
            )
        )

if Path(a.zen_tpm_csv).exists():
    df = pd.read_csv(a.zen_tpm_csv)
    if {"source", "tbt_ms", "tpm_per_a100e"}.issubset(df.columns):
        z = df[df["source"].isin(["zen replay"])].copy()
        frames.append(
            pd.DataFrame(
                {
                    "source": z["source"],
                    "tbt_ms": pd.to_numeric(z["tbt_ms"], errors="coerce"),
                    "tpm_per_a100e": pd.to_numeric(z["tpm_per_a100e"], errors="coerce"),
                }
            )
        )

out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["source", "tbt_ms", "tpm_per_a100e"])
out = out.replace([np.inf, -np.inf], np.nan).dropna()
Path(a.csv_output).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(a.csv_output, index=False)

plt.figure(figsize=(10, 6.4))
if out.empty:
    plt.text(0.5, 0.5, "No TPM/A100e curves", ha="center", va="center", transform=plt.gca().transAxes)
else:
    colors = {
        "engine": "#155eef",
        "zen replay": "#111827",
    }
    labels = {
        "engine": "engine with optimal ratio",
        "zen replay": "zen bench replay with optimal ratio",
    }
    for source, g in out.groupby("source", sort=False):
        g = g.sort_values("tbt_ms")
        color = colors.get(source)
        curve = fit_curve(g, "tbt_ms", "tpm_per_a100e")
        plt.scatter(g.tbt_ms, g.tpm_per_a100e, s=22, alpha=0.22, color=color, label="_nolegend_")
        plt.plot(curve.tbt_ms, curve.tpm_per_a100e, lw=2.5, color=color, label=labels.get(source, source))

prod_cases = [
    ("prod serving case 1", a.prod_case1_tbt, a.prod_case1_tpm_low, a.prod_case1_tpm_high, "#059669"),
    ("prod serving case 2", a.prod_case2_tbt, a.prod_case2_tpm_low, a.prod_case2_tpm_high, "#dc2626"),
]
for label, tbt, low, high, color in prod_cases:
    if not all(np.isfinite(v) for v in [tbt, low, high]) or high <= 0 or low < 0:
        continue
    if low > high:
        low, high = high, low
    mid = (low + high) / 2
    plt.vlines(tbt, low, high, color=color, lw=10, alpha=0.8, label=f"{label}: {tbt:g} ms, {low/1e6:.1f}-{high/1e6:.1f}M")
    plt.scatter([tbt], [mid], color=color, s=180, edgecolor="white", linewidth=2.0, zorder=5)

plt.xlabel("TBT (ms)")
plt.ylabel("TPM per A100e")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
plt.grid(alpha=0.3)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig(a.output, dpi=170)
print(f"[done] all tpm rows={len(out)}")
