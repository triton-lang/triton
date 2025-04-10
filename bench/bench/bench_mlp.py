from pathlib import Path
import json
import triton.profiler as proton
import triton.profiler.viewer as viewer
import torch
import triton_bench.swiglu
from triton_bench.mxfp import downcast_to_mxfp
from triton_bench.matmul_ogs import MicroscalingCtx, matmul_ogs, PrecisionConfig, FlexCtx
from triton_bench.numerics import InFlexData
from triton_bench.routing import routing
from triton_bench.meta import cuda_capability_geq, is_hip, get_cdna_version

if torch.cuda.is_available() and not is_hip():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def quantize(w, dtype, dev, **opt):
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), MicroscalingCtx()
    elif dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 \
            else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), \
                   MicroscalingCtx()
    else:
        assert dtype == "mx4", f"{dtype=}"
        swizzle_mx_scale = opt["swizzle_mx_scale"]
        swizzle_axis = 2 if swizzle_mx_scale else None
        w = w.to(torch.bfloat16)
        w, mx_scales, weight_scale_shape = downcast_to_mxfp(w, torch.uint8, axis=1, swizzle_axis=swizzle_axis)
        return w, InFlexData(), MicroscalingCtx(weight_scale=mx_scales, swizzle_mx=swizzle_mx_scale,
                                                actual_weight_scale_shape=weight_scale_shape)


def bench_mlp(batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype,
              # tensor / expert parallelism
              TP=1, EP=1, name=""):
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    dev = "cuda"

    # input
    # weights
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = torch.randn((n_expts_tot, ), device=dev)
    b1 = torch.randn((dim2 // TP, ), device=dev)
    b2 = torch.randn((dim1, ), device=dev)

    # -- numerics --
    optg = dict()
    opt1 = {"swizzle_mx_scale": True} if w_dtype == "mx4" else dict()
    opt2 = {"swizzle_mx_scale": True} if w_dtype == "mx4" else dict()
    wg, wg_flex, wg_mx = quantize(wg, "bf16", dev, **optg)
    w1, w1_flex, w1_mx = quantize(w1, w_dtype, dev, **opt1)
    w2, w2_flex, w2_mx = quantize(w2, w_dtype, dev, **opt2)
    pcg = PrecisionConfig(mx_ctx=wg_mx, flex_ctx=FlexCtx(rhs_data=wg_flex))
    pcs = triton_bench.swiglu.PrecisionConfig(limit=1.0)
    pc1 = PrecisionConfig(mx_ctx=w1_mx, flex_ctx=FlexCtx(rhs_data=w1_flex))
    pc2 = PrecisionConfig(mx_ctx=w2_mx, flex_ctx=FlexCtx(rhs_data=w2_flex))

    # -- benchmark --
    fpath = Path(f"logs/{name}/{batch}-{dim1}-{dim2}-{n_expts_tot}-{n_expts_act}-{x_dtype}-{w_dtype}.hatchet")
    fpath.parent.mkdir(parents=True, exist_ok=True)
    x_dtype = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
    if x_dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        x_dtype = torch.float8_e4m3fnuz

    x = torch.randn((batch, dim1), device=dev)
    xg = x.to(wg.dtype if n_expts_tot > 1 else x_dtype)
    x = x.to(x_dtype)
    # run layer
    proton.start(str(fpath.with_suffix('')), hook="triton")
    for i in range(100):
        if n_expts_tot > 1:
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            rdata, gather_indx, scatter_indx = routing(logits, n_expts_act, simulated_ep=EP)
        else:
            rdata, gather_indx, scatter_indx = None, None, None
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        x = triton_bench.swiglu.swiglu(x, 1.0, pcs)
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2)
    proton.finalize()

    # -- analyze --
    gf, inclusive_metrics, exclusive_metrics, device_info = viewer.read(fpath)
    # Now the dataframe only contains leave nodes (i.e., kernels) that perform matmuls
    matmuls = gf.filter("MATCH ('*', c) WHERE c.'name' =~ '.*matmul.*' AND c IS LEAF").dataframe
    tot_bytes = matmuls["bytes"].sum()
    tot_flops = sum(matmuls[[c for c in ['flops8', 'flops16'] if c in matmuls.columns]].sum())
    tot_time = matmuls["time (ns)"].sum()

    # Calculate theoretical min time based on hardware limits.
    device_type = matmuls["device_type"].iloc[0]
    device_id = matmuls["device_id"].iloc[0]
    info = device_info[device_type][device_id]

    min_time_flops_sec = sum(
        (matmuls[f"flops{width}"].sum() if f"flops{width}" in matmuls.columns else 0) /
        proton.specs.max_flops(device_type, info["arch"], width, info["num_sms"], info["clock_rate"])
        for width in [8, 16])
    min_time_bytes_sec = tot_bytes / proton.specs.max_bytes(info["bus_width"], info["memory_clock_rate"])

    util = max(min_time_flops_sec, min_time_bytes_sec) / (tot_time / 1e9)
    tflops = tot_flops / tot_time * 1e-3
    tbps = tot_bytes / tot_time * 1e-3

    return {"util": float(util), "tflops": float(tflops), "tbps": float(tbps)}


if __name__ == "__main__":
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    qxdtype = "fp8" if has_native_mx4 else "bf16"
    print(bench_mlp(8192, 8192, 8192, 1, 1, "fp8", "fp8", TP=1, EP=1, name="dense"))
    print(bench_mlp(8192, 8192, 8192, 1, 1, qxdtype, "mx4", TP=1, EP=1, name="dense"))
    print(bench_mlp(2048, 5120, 8192, 128, 4, "fp8", "fp8", TP=4, EP=2, name="llama4"))
    print(bench_mlp(2048, 5120, 8192, 128, 4, qxdtype, "mx4", TP=4, EP=2, name="llama4"))
