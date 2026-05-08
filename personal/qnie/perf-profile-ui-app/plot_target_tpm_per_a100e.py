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


def fit_curve(df, x, y, points=160):
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(x)
    if len(d) < 3 or d[x].nunique() < 3:
        return d
    degree = min(3, d[x].nunique() - 1)
    coeff = np.polyfit(d[x], d[y], degree)
    xs = np.linspace(d[x].min(), d[x].max(), points)
    ys = np.maximum(np.polyval(coeff, xs), 0)
    return pd.DataFrame({x: xs, y: ys})


sdf = pd.read_csv(a.sampling_csv)
pdf = pd.read_csv(a.prefill_csv)

s = sdf[sdf["curve"] == "perf target"].copy()
p = pdf[pdf["curve"] == "perf target"].copy()
if p.empty:
    p = pdf.copy()

prefill_col = "input_tpgs" if "input_tpgs" in p.columns else "prefill_uncached_tpgs"
pmax = pd.to_numeric(p[prefill_col], errors="coerce").dropna().max()

if s.empty or not np.isfinite(pmax) or pmax <= 0:
    out = pd.DataFrame(columns=["tbt_ms", "sampling_tpgs", "prefill_input_tpgs_max", "prefill_gpus_per_sampling_gpu", "tpm_per_a100e"])
else:
    ratio = a.input_sampling_ratio
    sampling_tpgs = pd.to_numeric(s["tpgs"], errors="coerce")
    tbt = pd.to_numeric(s["tbt_ms"], errors="coerce")
    prefill_gpus_per_sampling_gpu = ratio * sampling_tpgs / pmax
    tpm = 60 * (ratio + 1) * sampling_tpgs / (1 + prefill_gpus_per_sampling_gpu)
    out = pd.DataFrame(
        {
            "tbt_ms": tbt,
            "sampling_tpgs": sampling_tpgs,
            "prefill_input_tpgs_max": pmax,
            "prefill_gpus_per_sampling_gpu": prefill_gpus_per_sampling_gpu,
            "tpm_per_a100e": tpm,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()

Path(a.csv_output).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(a.csv_output, index=False)

plt.figure(figsize=(9, 6))
if out.empty:
    plt.text(0.5, 0.5, "No target TPM/A100e points", ha="center", va="center", transform=plt.gca().transAxes)
else:
    o = out.sort_values("tbt_ms")
    plt.scatter(o.tbt_ms, o.tpm_per_a100e, s=28, alpha=0.35, color="#b794f4", label=f"target points (n={len(o)})")
    plt.plot(o.tbt_ms, o.tpm_per_a100e, lw=2.6, color="#7c3aed", label=f"target connected, max prefill={pmax:.1f}")
plt.xlabel("Target TBT (ms)")
plt.ylabel("TPM per A100e")
plt.title(f"Perf Target TPM/A100e vs TBT\ninput:sampling={a.input_sampling_ratio:g}:1")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(a.output, dpi=170)
print(f"[done] target tpm rows={len(out)}")
