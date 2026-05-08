#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


p = argparse.ArgumentParser()
p.add_argument("--sampling-csv", required=True)
p.add_argument("--sampling-gpus", type=float, required=True)
p.add_argument("--prefill-csv", required=True)
p.add_argument("--prefill-link", required=True)
p.add_argument("--sampling-output", required=True)
p.add_argument("--prefill-output", required=True)
p.add_argument("--sampling-csv-output", required=True)
p.add_argument("--prefill-csv-output", required=True)
p.add_argument("--step-seconds", type=int, default=60)
a = p.parse_args()


def link_params(raw: str):
    q = parse_qs(urlparse(raw).query)

    def first(*ks: str, default: str = ""):
        for k in ks:
            if q.get(k) and q[k][0] and q[k][0] != "$__all":
                return q[k][0]
        return default

    def ts(v: str) -> int:
        if v.isdigit():
            n = int(v)
            return n // 1000 if n > 10_000_000_000 else n
        return int(dt.datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp())

    return {
        "env": first("var-env", default="prod"),
        "vm": first("var-virtual_model", "var-model", default=".*"),
        "start": ts(first("from", default="now-1h")),
        "end": ts(first("to", default="now")),
    }


def esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"')


def query_prefill_tbt(raw_link: str) -> pd.DataFrame:
    info = link_params(raw_link)
    promql = (
        'histogram_avg(sum by (virtual_model, shadow) ('
        'sum_over_time(api_grpc_inference_stream_mean_time_between_tokens{'
        f'env=~"{esc(info["env"])}",grpc="true",shadow=~".*",success="true",'
        f'virtual_model=~"{esc(info["vm"])}"'
        "}[1m])))"
    )
    api_key = os.getenv("CHRONOSPHERE_API_KEY", "")
    base = os.getenv("CHRONO_PROM_BASE_URL", "https://openai.chronosphere.io/data/metrics").rstrip("/")
    if not api_key:
        print("[warn] prefill TBT query skipped: CHRONOSPHERE_API_KEY is not set")
        return pd.DataFrame(columns=["timestamp", "tbt_ms"])
    url = (
        f"{base}/api/v1/query_range?query={quote(promql, safe='')}"
        f"&start={info['start']}&end={info['end']}&step={a.step_seconds}"
    )
    req = Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        print(f"[warn] prefill TBT query failed: Chronosphere HTTP {exc.code}")
        return pd.DataFrame(columns=["timestamp", "tbt_ms"])
    if payload.get("status") != "success":
        print(f"[warn] prefill TBT query failed: {payload}")
        return pd.DataFrame(columns=["timestamp", "tbt_ms"])
    result = payload.get("data", {}).get("result", payload.get("result", []))
    vals: dict[int, list[float]] = defaultdict(list)
    for series in result:
        for raw_t, raw_v in series.get("values", []):
            try:
                vals[int(float(raw_t))].append(float(raw_v))
            except Exception:
                pass
    rows = [{"timestamp": t, "tbt_ms": sum(v) / len(v)} for t, v in vals.items() if v]
    return pd.DataFrame(rows).sort_values("timestamp")


def fit_curve(df: pd.DataFrame, x: str, y: str, bins: int = 36, points: int = 160) -> pd.DataFrame:
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
    ys = np.polyval(coeff, xs)
    ys = np.maximum(ys, 0)
    return pd.DataFrame({x: xs, y: ys})


def plot_xy(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: str):
    plt.figure(figsize=(9, 5.5))
    clean = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        plt.text(0.5, 0.5, "No data points", ha="center", va="center", transform=plt.gca().transAxes)
    else:
        plt.scatter(clean[x], clean[y], s=24, alpha=0.28, color="#155eef")
        line = fit_curve(clean, x, y)
        plt.plot(line[x], line[y], lw=2.5, color="#155eef", label=f"engine (n={len(clean)})")
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=170)


def short_replica_name(value: str) -> str:
    value = str(value)
    return value.rsplit("-", 1)[-1] if "-" in value else value


def plot_prefill_replicas(df: pd.DataFrame, path: str):
    plt.figure(figsize=(9, 5.5))
    clean = (
        df[["replica", "batch_size", "prefill_input_tpgs"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["batch_size", "prefill_input_tpgs"])
        .sort_values("batch_size")
    )
    if clean.empty:
        plt.text(0.5, 0.5, "No data points", ha="center", va="center", transform=plt.gca().transAxes)
    else:
        colors = plt.get_cmap("tab10").colors
        for idx, (replica, group) in enumerate(clean.groupby("replica", sort=True)):
            color = colors[idx % len(colors)]
            label = f"replica {short_replica_name(replica)}"
            plt.scatter(group.batch_size, group.prefill_input_tpgs, s=16, alpha=0.18, color=color, label="_nolegend_")
            line = fit_curve(group, "batch_size", "prefill_input_tpgs", bins=24)
            plt.plot(line.batch_size, line.prefill_input_tpgs, lw=1.6, alpha=0.8, color=color, label=label)

        all_line = fit_curve(clean, "batch_size", "prefill_input_tpgs", bins=36)
        plt.plot(
            all_line.batch_size,
            all_line.prefill_input_tpgs,
            lw=3.2,
            color="#111827",
            label=f"all replicas (n={len(clean)})",
            zorder=10,
        )
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend(fontsize=8, ncols=2)
    plt.xlabel("Batch size")
    plt.ylabel("Prefill input TPGS")
    plt.title("Engine Prefill: Input TPGS vs Batch Size")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=170)


sampling = pd.read_csv(a.sampling_csv)
if "tbt" not in sampling.columns and "tbt_ms" in sampling.columns:
    sampling["tbt"] = sampling["tbt_ms"]
sampling_out = pd.DataFrame(
    {
        "timestamp": pd.to_numeric(sampling.get("timestamp"), errors="coerce"),
        "tbt_ms": pd.to_numeric(sampling["tbt"], errors="coerce"),
        "sampling_tpgs": pd.to_numeric(sampling["sampled_tokens"], errors="coerce") / a.sampling_gpus,
    }
).dropna(subset=["tbt_ms", "sampling_tpgs"])

prefill = pd.read_csv(a.prefill_csv)
y_col = "prefill_input_tpgs" if "prefill_input_tpgs" in prefill.columns else "prefill_input_tokens_per_second"
prefill_out = pd.DataFrame(
    {
        "replica": prefill.get("api_pipereplica_id", "all"),
        "timestamp": pd.to_numeric(prefill["timestamp"], errors="coerce"),
        "batch_size": pd.to_numeric(prefill["avg_batch_size"], errors="coerce"),
        "prefill_input_tpgs": pd.to_numeric(prefill[y_col], errors="coerce"),
    }
).dropna(subset=["batch_size", "prefill_input_tpgs"]).sort_values("batch_size")

Path(a.sampling_csv_output).parent.mkdir(parents=True, exist_ok=True)
sampling_out.to_csv(a.sampling_csv_output, index=False)
prefill_out.to_csv(a.prefill_csv_output, index=False)

plot_xy(
    sampling_out,
    "tbt_ms",
    "sampling_tpgs",
    "Engine Sampling: TPGS vs TBT",
    "TBT (ms)",
    "Sampling TPGS",
    a.sampling_output,
)
plot_prefill_replicas(prefill_out, a.prefill_output)
print(f"[done] engine sampling rows={len(sampling_out)}")
print(f"[done] engine prefill batch rows={len(prefill_out)} replicas={prefill_out['replica'].nunique()}")
