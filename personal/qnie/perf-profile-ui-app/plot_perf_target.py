#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, io, re, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument("--sampling-target",required=True); p.add_argument("--prefill-target",required=True)
p.add_argument("--sampling-output",required=True); p.add_argument("--prefill-output",required=True)
p.add_argument("--sampling-csv-output",required=True); p.add_argument("--prefill-csv-output",required=True)
p.add_argument("--prefill-input-tpgs-multiplier",type=float,default=1.0)
a=p.parse_args()

def txt(path):
    if path.startswith("az://"):
        r=subprocess.run(["bbb","cat",path],capture_output=True,text=True,timeout=240)
        if r.returncode: raise SystemExit(f"bbb cat failed for {path}\n{r.stderr}")
        return r.stdout
    return Path(path).read_text()
def rows(path):
    raw=txt(path); lines=[l.strip() for l in raw.splitlines() if l.strip()]
    if lines and lines[0].startswith("|"):
        h=[c.strip() for c in lines[0].strip("|").split("|")]; out=[]
        for l in lines[1:]:
            c=[x.strip() for x in l.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?",x) for x in c): continue
            if len(c)==len(h): out.append(dict(zip(h,c)))
        return out
    return list(csv.DictReader(io.StringIO(raw)))
def n(r,*ks):
    for k in ks:
        if k in r:
            try: v=float(r[k])
            except (TypeError,ValueError): continue
            if np.isfinite(v): return v
    return None
def fit_curve(df,x,y,points=160):
    d=df[[x,y]].replace([np.inf,-np.inf],np.nan).dropna().sort_values(x)
    if len(d)<3 or d[x].nunique()<3: return d
    degree=min(3,d[x].nunique()-1)
    coeff=np.polyfit(d[x],d[y],degree); xs=np.linspace(d[x].min(),d[x].max(),points); ys=np.maximum(np.polyval(coeff,xs),0)
    return pd.DataFrame({x:xs,y:ys})
def series(df, preferred, fallback=None, default=None):
    if preferred in df.columns:
        return pd.to_numeric(df[preferred],errors="coerce")
    if fallback and fallback in df.columns:
        return pd.to_numeric(df[fallback],errors="coerce")
    if default is not None:
        return pd.Series([default]*len(df),index=df.index,dtype=float)
    return pd.Series([np.nan]*len(df),index=df.index,dtype=float)

spts=[]
sraw=pd.DataFrame(rows(a.sampling_target))
if not sraw.empty:
    base_tpgs=series(sraw,"avg_tpgs","tokens_per_gpu_second")
    sampling_batch=series(sraw,"batch_size","sampling_n_requests")
    if sampling_batch.isna().all():
        sampling_batch=series(sraw,"n_requests")
    sampling_world_size=series(sraw,"world_size",default=1.0).where(lambda x:x>0)
    useful_plus_roofline=series(sraw,"latency_useful_only_plus_roofline_ms_per_token","latency_useful_only_plus_roofline_ms")
    curves=[
        ("tbt",series(sraw,"avg_ms_per_token","latency_ms")),
        ("proton latency",series(sraw,"proton_latency_ms_per_token","proton_latency_ms")),
        ("perf target",useful_plus_roofline),
    ]
    for label,xs in curves:
        tpgs=sampling_batch*1000.0/xs/sampling_world_size
        tpgs=tpgs.fillna(base_tpgs)
        d=pd.DataFrame({"curve":label,"tbt_ms":xs,"tpgs":tpgs}).replace([np.inf,-np.inf],np.nan).dropna()
        spts.extend(d.to_dict("records"))
sdf=pd.DataFrame(spts)

ppts=[]
praw=pd.DataFrame(rows(a.prefill_target))
if not praw.empty:
    batch=series(praw,"batch_size","sampling_n_requests")
    if batch.isna().all():
        batch=series(praw,"n_requests")
    world_size=series(praw,"world_size",default=1.0).where(lambda x:x>0)
    curves=[
        ("tbt",series(praw,"latency_ms")),
        ("proton latency",series(praw,"proton_latency_ms")),
        ("perf target",series(praw,"latency_useful_only_plus_roofline_ms")),
    ]
    for label,latency_ms in curves:
        prefill_uncached_tpgs=batch*1000.0/latency_ms/world_size
        input_tpgs=prefill_uncached_tpgs*a.prefill_input_tpgs_multiplier
        d=pd.DataFrame({"curve":label,"batch_size":batch,"prefill_uncached_tpgs":prefill_uncached_tpgs,"input_tpgs":input_tpgs}).replace([np.inf,-np.inf],np.nan).dropna()
        ppts.extend(d.to_dict("records"))
pdf=pd.DataFrame(ppts)

Path(a.sampling_csv_output).parent.mkdir(parents=True,exist_ok=True)
sdf.to_csv(a.sampling_csv_output,index=False); pdf.to_csv(a.prefill_csv_output,index=False)

plt.figure(figsize=(9,6))
if sdf.empty:
    plt.text(.5,.5,"No sampling target points",ha="center",va="center",transform=plt.gca().transAxes)
else:
    colors={"tbt":"#155eef","proton latency":"#111827","perf target":"#7c3aed"}
    for label,g in sdf.groupby("curve",sort=False):
        g=g.sort_values("tbt_ms"); color=colors.get(label)
        plt.scatter(g.tbt_ms,g.tpgs,s=28,alpha=.45,color=color,label="_nolegend_")
        plt.plot(g.tbt_ms,g.tpgs,lw=2.2,color=color,label=f"{label} (n={len(g)})")
plt.xlabel("Target TBT (ms)"); plt.ylabel("Sampling TPGS"); plt.title("Perf Target: Sampling"); plt.xlim(left=0); plt.ylim(bottom=0); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(a.sampling_output,dpi=170)

plt.figure(figsize=(9,6))
if pdf.empty:
    plt.text(.5,.5,"No prefill target points",ha="center",va="center",transform=plt.gca().transAxes)
else:
    colors={"tbt":"#155eef","proton latency":"#111827","perf target":"#7c3aed"}
    for label,g in pdf.groupby("curve",sort=False):
        g=g.sort_values("batch_size"); color=colors.get(label)
        plt.scatter(g.batch_size,g.prefill_uncached_tpgs,s=28,alpha=.45,color=color,label="_nolegend_")
        plt.plot(g.batch_size,g.prefill_uncached_tpgs,lw=2.2,color=color,label=f"{label} (n={len(g)})")
plt.xlabel("Batch size"); plt.ylabel("Prefill Uncached TPGS"); plt.title("Perf Target: Prefill"); plt.xlim(left=0); plt.ylim(bottom=0); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(a.prefill_output,dpi=170)
print(f"[done] perf target sampling rows={len(sdf)}")
print(f"[done] perf target prefill rows={len(pdf)}")
