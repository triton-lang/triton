#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, io, re, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
p=argparse.ArgumentParser()
for x in ["engine-sampling-csv","engine-prefill-csv","sampling-output","prefill-output","sampling-csv-output","prefill-csv-output"]: p.add_argument("--"+x,required=True)
p.add_argument("--engine-sampling-gpus",type=float,required=True); p.add_argument("--zen-sampling-replay",default=""); p.add_argument("--zen-prefill-replay",default=""); p.add_argument("--zen-sampling-uniform",default=""); p.add_argument("--zen-prefill-uniform",default=""); p.add_argument("--zen-prefill-tpgs-multiplier",type=float,default=20)
a=p.parse_args()
warnings=[]
def txt(path):
    if path.startswith("az://"):
        r=subprocess.run(["bbb","cat",path],capture_output=True,text=True,timeout=240)
        if r.returncode: raise RuntimeError(f"bbb cat failed for {path}\n{r.stderr}")
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
            except ValueError: continue
            if np.isfinite(v): return v
    return None
def zens(path,label):
    pts=[]
    for r in rows(path):
        tpgs=n(r,"tpgs","tokens_per_gpu_second"); tbt=n(r,"avg_ms_per_token","tbt","tbt_ms")
        if tbt is None:
            us=n(r,"elapsed_in_us"); tbt=None if us is None else us/1000
        if tpgs is not None and tbt is not None: pts.append({"source":label,"tbt_ms":tbt,"tpgs":tpgs})
    if not pts: raise SystemExit(f"No sampling points from {path}")
    return pd.DataFrame(pts)
def zenp(path,label,mult):
    pts=[]
    for r in rows(path):
        b=n(r,"batch_size","sampling_n_requests","n_requests"); t=n(r,"tpgs","tokens_per_gpu_second")
        if b is not None and t is not None: pts.append({"source":label,"batch_size":b,"input_tpgs":t*mult})
    if not pts: raise SystemExit(f"No prefill points from {path}")
    return pd.DataFrame(pts)
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
def overlay_curve(df,x,y,label):
    d=df[[x,y]].replace([np.inf,-np.inf],np.nan).dropna().sort_values(x)
    if label=="zen uniform":
        return d
    return fit_curve(d,x,y)
s=pd.read_csv(a.engine_sampling_csv)
if "tbt" not in s.columns and "tbt_ms" in s.columns: s["tbt"]=s["tbt_ms"]
sframes=[pd.DataFrame({"source":"engine","tbt_ms":pd.to_numeric(s["tbt"],errors="coerce"),"tpgs":pd.to_numeric(s["sampled_tokens"],errors="coerce")/a.engine_sampling_gpus}).dropna()]
if a.zen_sampling_replay:
    try: sframes.append(zens(a.zen_sampling_replay,"zen replay"))
    except Exception as e:
        warnings.append(f"skipped zen sampling replay: {e}")
if a.zen_sampling_uniform:
    try: sframes.append(zens(a.zen_sampling_uniform,"zen uniform"))
    except Exception as e:
        warnings.append(f"skipped zen sampling uniform: {e}")
sdf=pd.concat(sframes,ignore_index=True)
ep=pd.read_csv(a.engine_prefill_csv); y="prefill_input_tpgs" if "prefill_input_tpgs" in ep.columns else "prefill_input_tokens_per_second"
pframes=[pd.DataFrame({"source":"engine","batch_size":pd.to_numeric(ep["avg_batch_size"],errors="coerce"),"input_tpgs":pd.to_numeric(ep[y],errors="coerce")}).dropna()]
if a.zen_prefill_replay:
    try: pframes.append(zenp(a.zen_prefill_replay,"zen replay",a.zen_prefill_tpgs_multiplier))
    except Exception as e:
        warnings.append(f"skipped zen prefill replay: {e}")
if a.zen_prefill_uniform:
    try: pframes.append(zenp(a.zen_prefill_uniform,"zen uniform",a.zen_prefill_tpgs_multiplier))
    except Exception as e:
        warnings.append(f"skipped zen prefill uniform: {e}")
pdf=pd.concat(pframes,ignore_index=True)
Path(a.sampling_csv_output).parent.mkdir(parents=True,exist_ok=True); sdf.to_csv(a.sampling_csv_output,index=False); pdf.to_csv(a.prefill_csv_output,index=False)
c={"engine":"#155eef","zen replay":"#111827","zen uniform":"#d55e00"}
plt.figure(figsize=(9,6))
for lab,g in sdf.groupby("source",sort=False):
    curve_label="connected" if lab=="zen uniform" else "fit"
    plt.scatter(g.tbt_ms,g.tpgs,s=24,alpha=.25,color=c.get(lab),label="_nolegend_"); r=overlay_curve(g,"tbt_ms","tpgs",lab); plt.plot(r.tbt_ms,r.tpgs,lw=2.5,color=c.get(lab),label=f"{lab} {curve_label} (n={len(g)})")
plt.xlabel("TBT (ms)"); plt.ylabel("TPGS"); plt.title("Sampling: engine + Zen Bench"); plt.xlim(left=0); plt.ylim(bottom=0); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(a.sampling_output,dpi=170)
plt.figure(figsize=(9,6))
for lab,g in pdf.groupby("source",sort=False):
    curve_label="connected" if lab in {"zen replay","zen uniform"} else "fit"
    plt.scatter(g.batch_size,g.input_tpgs,s=24,alpha=.25,color=c.get(lab),label="_nolegend_"); r=g[["batch_size","input_tpgs"]].replace([np.inf,-np.inf],np.nan).dropna().sort_values("batch_size") if lab in {"zen replay","zen uniform"} else overlay_curve(g,"batch_size","input_tpgs",lab); plt.plot(r.batch_size,r.input_tpgs,lw=2.5,color=c.get(lab),label=f"{lab} {curve_label} (n={len(g)})")
plt.xlabel("Batch size"); plt.ylabel("Input TPGS"); plt.title("Prefill: engine + Zen Bench"); plt.xlim(left=0); plt.ylim(bottom=0); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(a.prefill_output,dpi=170)
for w in warnings:
    print(f"[warn] {w}")
print(f"[done] sampling rows={len(sdf)}"); print(f"[done] prefill rows={len(pdf)}")
