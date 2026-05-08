#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--sampling-csv", required=True)
p.add_argument("--prefill-csv", required=True)
p.add_argument("--input-sampling-ratio", type=float, default=256)
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


sdf = pd.read_csv(a.sampling_csv)
pdf = pd.read_csv(a.prefill_csv)
ratio = a.input_sampling_ratio

rows = []
zen_uniform_prefill = pdf[pdf["source"] == "zen uniform"].copy()
zen_uniform_pmax = pd.to_numeric(zen_uniform_prefill.get("input_tpgs"), errors="coerce").dropna().max() if not zen_uniform_prefill.empty else np.nan
for source in ["engine", "zen replay", "zen uniform"]:
    s = sdf[sdf["source"] == source].copy()
    psrc = pdf[pdf["source"] == source].copy()
    if s.empty or psrc.empty:
        continue
    if source in {"zen replay", "zen uniform"} and np.isfinite(zen_uniform_pmax) and zen_uniform_pmax > 0:
        pmax = zen_uniform_pmax
    else:
        pmax = pd.to_numeric(psrc["input_tpgs"], errors="coerce").dropna().max()
    if not np.isfinite(pmax) or pmax <= 0:
        continue
    sampling_tpgs = pd.to_numeric(s["tpgs"], errors="coerce")
    tbt = pd.to_numeric(s["tbt_ms"], errors="coerce")
    prefill_gpus_per_sampling_gpu = ratio * sampling_tpgs / pmax
    tpm = 60 * (ratio + 1) * sampling_tpgs / (1 + prefill_gpus_per_sampling_gpu)
    out = pd.DataFrame(
        {
            "source": source,
            "tbt_ms": tbt,
            "sampling_tpgs": sampling_tpgs,
            "prefill_input_tpgs_max": pmax,
            "prefill_gpus_per_sampling_gpu": prefill_gpus_per_sampling_gpu,
            "tpm_per_a100e": tpm,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    rows.extend(out.to_dict("records"))

out = pd.DataFrame(rows)
Path(a.csv_output).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(a.csv_output, index=False)

plt.figure(figsize=(9, 6))
if out.empty:
    plt.text(
        0.5,
        0.5,
        "No Zen TPM/A100e points",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
else:
    colors = {"engine": "#155eef", "zen replay": "#111827", "zen uniform": "#d55e00"}
    for source, g in out.groupby("source", sort=False):
        g = g.sort_values("tbt_ms")
        color = colors.get(source)
        curve_label = "connected" if source == "zen uniform" else "fit"
        curve = g[["tbt_ms", "tpm_per_a100e"]] if source == "zen uniform" else fit_curve(g, "tbt_ms", "tpm_per_a100e")
        pmax = g["prefill_input_tpgs_max"].iloc[0]
        plt.scatter(g.tbt_ms, g.tpm_per_a100e, s=24, alpha=0.28, color=color, label="_nolegend_")
        plt.plot(
            curve.tbt_ms,
            curve.tpm_per_a100e,
            lw=2.5,
            color=color,
            label=f"{source} {curve_label} (n={len(g)}, max prefill={pmax:.1f})",
        )
plt.xlabel("TBT (ms)")
plt.ylabel("TPM per A100e")
plt.title(f"Zen TPM/A100e vs TBT\ninput:sampling={ratio:g}:1")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(a.output, dpi=170)
print(f"[done] zen tpm rows={len(out)}")
