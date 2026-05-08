#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, datetime as dt, json, os, re
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

p=argparse.ArgumentParser(); p.add_argument("--agentwolf-link",required=True); p.add_argument("--output",required=True); p.add_argument("--step-seconds",type=int,default=60); a=p.parse_args()
q=parse_qs(urlparse(a.agentwolf_link).query)
def first(*ks, default=""):
    for k in ks:
        if q.get(k) and q[k][0] and q[k][0] != "$__all": return q[k][0]
    return default
def ts(v):
    if v.isdigit():
        n=int(v); return n//1000 if n>10_000_000_000 else n
    return int(dt.datetime.fromisoformat(v.replace("Z","+00:00")).timestamp())
def esc(v): return v.replace("\\","\\\\").replace('"','\\"')
def matcher(label, value, regex=False):
    op = "=~" if regex or value == ".*" else "="
    return f'{label}{op}"{esc(value)}"'
engine=first("var-engine","var-api_engine")
group=first("var-api_pipereplica_group_id", default="default")
cluster=first("var-engine_cluster","var-unified_cluster", default=".*")
env=first("var-env", default="prod")
vm=first("var-virtual_model","var-model", default=".*")
start=ts(first("from", default="now-1h")); end=ts(first("to", default="now"))
queries={
 "stream_tbt": 'histogram_avg(sum by (virtual_model, shadow) (sum_over_time(api_grpc_inference_stream_mean_time_between_tokens{env=~"'+esc(env)+'",grpc="true",shadow=~".*",success="true",virtual_model=~"'+esc(vm)+'"}[1m])))',
 "engine_decoder_tbt": 'histogram_avg(sum by (api_engine_id, api_pipereplica_group_id) (sum_over_time(enginev3_inference_pipereplica_gpt_gpu_sub_step_time_ms{'+",".join([matcher("api_engine_id",engine),matcher("api_pipereplica_group_id",group),matcher("cluster",cluster),matcher("env",env),'sub_step="decoder"'])+'}[1m])))',
 "sampled_tokens": 'avg by (api_engine_id, api_pipereplica_group_id) (sum_per_second(enginev3_inference_pipereplica_gpt_accepted_tokens{'+",".join([matcher("api_engine_id",engine),matcher("api_pipereplica_group_id",group),matcher("cluster",cluster),matcher("env",env)])+'}[4m0s]))',
}
def fmt(t): return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).isoformat().replace("+00:00","Z")
def query(promql):
    api_key=os.getenv("CHRONOSPHERE_API_KEY","")
    base=os.getenv("CHRONO_PROM_BASE_URL","https://openai.chronosphere.io/data/metrics").rstrip("/")
    if not api_key:
        raise SystemExit("CHRONOSPHERE_API_KEY is required for direct Chronosphere queries")
    url=f"{base}/api/v1/query_range?query={quote(promql,safe='')}&start={start}&end={end}&step={a.step_seconds}"
    req=Request(url,method="GET")
    req.add_header("Authorization",f"Bearer {api_key}")
    try:
        with urlopen(req,timeout=90) as resp:
            payload=json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        raise SystemExit(f"Chronosphere HTTP {e.code}: {e.read().decode('utf-8',errors='replace')}") from e
    if payload.get("status")!="success":
        raise SystemExit(f"Chronosphere query failed: {payload}")
    result=payload.get("data",{}).get("result",payload.get("result",[]))
    vals=defaultdict(list)
    for s in result:
        for raw_t,raw_v in s.get("values",[]):
            try: vals[int(float(raw_t))].append(float(raw_v))
            except Exception: pass
    return {t:sum(v)/len(v) for t,v in vals.items() if v}
series={k:query(v) for k,v in queries.items()}
tbt_source="stream_tbt" if series["stream_tbt"] else "engine_decoder_tbt"
series["tbt"]=series[tbt_source]
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
all_ts=sorted(set(series["tbt"]) | set(series["sampled_tokens"]))
with open(a.output,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["timestamp","tbt","sampled_tokens"])
    for t in all_ts: w.writerow([t,series["tbt"].get(t,""),series["sampled_tokens"].get(t,"")])
for k in ["stream_tbt","engine_decoder_tbt","sampled_tokens"]:
    print(f"[ok] {k}: {len(series[k])} points ({start} -> {end})")
print(f"[ok] tbt_source: {tbt_source}")
print(f"[done] wrote: {a.output}")
