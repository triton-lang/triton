#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument("--sampling-csv",required=True); p.add_argument("--prefill-csv",required=True); p.add_argument("--sampling-gpus",type=float,required=True); p.add_argument("--input-sampling-ratio",type=float,default=256); p.add_argument("--blackwell-multiplier",type=float,default=3); p.add_argument("--output",required=True); p.add_argument("--csv-output",required=True); a=p.parse_args()
def fit_curve(df,x,y,bins=36,points=160):
    d=df[[x,y]].replace([np.inf,-np.inf],np.nan).dropna().sort_values(x)
    if len(d)<3 or d[x].nunique()<3: return d
    d=d.copy()
    if len(d)>bins:
        d["bin"]=pd.cut(d[x],np.linspace(d[x].min(),d[x].max(),bins+1),include_lowest=True,duplicates="drop")
        d=d.groupby("bin",observed=True).agg(**{x:(x,"median"),y:(y,"median")}).dropna().reset_index(drop=True)
    degree=min(3,d[x].nunique()-1)
    if degree<1: return d[[x,y]]
    coeff=np.polyfit(d[x],d[y],degree); xs=np.linspace(d[x].min(),d[x].max(),points); ys=np.maximum(np.polyval(coeff,xs),0)
    return pd.DataFrame({x:xs,y:ys})
sdf=pd.read_csv(a.sampling_csv)
if "tbt" not in sdf.columns and "tbt_ms" in sdf.columns: sdf["tbt"]=sdf["tbt_ms"]
s=pd.to_numeric(sdf["sampled_tokens"],errors="coerce")/a.sampling_gpus; tbt=pd.to_numeric(sdf["tbt"],errors="coerce")
pdf=pd.read_csv(a.prefill_csv); col="prefill_input_tpgs" if "prefill_input_tpgs" in pdf.columns else "prefill_input_tokens_per_second"; pmax=float(pd.to_numeric(pdf[col],errors="coerce").dropna().max())
ratio=a.input_sampling_ratio; pg=ratio*s/pmax; tpm=60*(ratio+1)*s/(1+pg)
out=pd.DataFrame({"tbt_ms":tbt,"sampling_tpgs":s,"prefill_input_tpgs_max":pmax,"prefill_gpus_per_sampling_gpu":pg,"tpm_per_a100e":tpm}).replace([np.inf,-np.inf],np.nan).dropna()
Path(a.csv_output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.csv_output,index=False)
o=out.sort_values("tbt_ms"); fit=fit_curve(o,"tbt_ms","tpm_per_a100e"); plt.figure(figsize=(9,6)); plt.scatter(o.tbt_ms,o.tpm_per_a100e,s=24,alpha=.28,color="#8ab6f9",label=f"A100e points (n={len(o)})"); plt.plot(fit.tbt_ms,fit.tpm_per_a100e,lw=2.8,color="#155eef",label="A100e fitted curve"); plt.xlabel("TBT (ms)"); plt.ylabel("TPM per A100e"); plt.title(f"TPM/A100e vs TBT\ninput:sampling={ratio:g}:1, max prefill={pmax:.1f} input TPGS"); plt.xlim(left=0); plt.ylim(bottom=0); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(a.output,dpi=170); print(f"[done] max prefill_input_tpgs={pmax:.9g}")
