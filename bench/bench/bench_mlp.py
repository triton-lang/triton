from pathlib import Path
import json
import triton
import triton.profiler as proton
import torch
import triton_bench.swiglu
from triton_bench.mxfp import downcast_to_mxfp
from triton_bench.matmul_ogs import MicroscalingCtx, matmul_ogs, PrecisionConfig, FlexCtx
from triton_bench.numerics import InFlexData
from triton_bench.routing import routing, simulate_expert_sharded_routing
from triton_bench.meta import cuda_capability_geq


def is_hip_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx950'


if torch.cuda.is_available() and not is_hip_cdna4():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def _query_gpu_specs():
    if is_hip_cdna4():
        # no spec data yet.
        return None
    import subprocess
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i=0"]
    output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    name = output.splitlines()[0]
    return {
        "NVIDIA H100 80GB HBM3": {"MAX_TFLOPS8": 1979, "MAX_TFLOPS16": 989, "MAX_TBPS": 3.35}, "HGX GB200":
        {"MAX_TFLOPS8": 4500, "MAX_TFLOPS16": 2250, "MAX_TBPS": 8.0}
    }[name]


SPECS = _query_gpu_specs()


def quantize(w, dtype, dev, **opt):
    if dtype == "bf16":
        return w.to(torch.bfloat16), InFlexData(), MicroscalingCtx()
    elif dtype == "fp8":
        wq = w.to(torch.float8_e4m3fn).transpose(-1, -2).contiguous().transpose(-1, -2)
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
    proton.start(str(fpath.with_suffix('')), hook="triton")
    proton.deactivate()
    # run layer
    x_dtype = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    for i in range(100):
        x = torch.randn((batch, dim1), device=dev)
        x = x.to(wg.dtype if n_expts_tot > 1 else x_dtype)
        proton.activate()
        if n_expts_tot > 1:
            logits = matmul_ogs(x, wg, bg, precision_config=pcg)
            rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)
            if EP > 1:
                proton.deactivate()
                # TODO: activate proton here when fast expert parallelism simulation is done
                m = logits.shape[0] * EP
                _, rdata, gather_indx, scatter_indx = simulate_expert_sharded_routing(m, rdata, EP, device=dev)
                proton.activate()
            x = x.to(x_dtype)
        else:
            rdata, gather_indx, scatter_indx = None, None, None
        # c0 = torch.empty((x.shape[0], w1.shape[-1]), device=dev, dtype=x.dtype)
        # c1 = torch.empty((x.shape[0], w2.shape[-1]), device=dev, dtype=x.dtype)
        # cublas.matmul(x, w1.squeeze(0), c0)
        # cublas.matmul(c0, w2.squeeze(0), c1)
        x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1)
        x = triton_bench.swiglu.swiglu(x, 1.0, pcs)
        x = matmul_ogs(x, w2, b2, rdata, scatter_indx=scatter_indx, precision_config=pc2)
        proton.deactivate()
    proton.finalize()

    # -- analyze --
    with open(f"{fpath}") as fd:
        data = json.load(fd)
        # TODO: this will be broken if kernels use scopes themselves
        # compute useful (a.k.a. matmul) bytes and flops
        matmuls = [x for x in data[0]["children"] if "matmul" in x["frame"]["name"]]
        tot_bytes = sum([x["metrics"]["bytes"] for x in matmuls])
        tot_flops = {w: sum([x["metrics"].get(f"flops{w}", 0) for x in matmuls]) for w in [8, 16]}
        # compute total time (incl. "not useful" work)
        # TODO: proton should really be recording that in the json instead of
        # relying on the user to aggregate
        tot_time = sum(x["metrics"].get("time (ns)", 0) for x in data[0]["children"])
        min_time_flops = min_time_bytes = 0
        if SPECS is not None:
            min_time_flops = sum([tot_flops[w] / SPECS[f"MAX_TFLOPS{w}"] for w in [8, 16]]) * 1e-3
            min_time_bytes = tot_bytes / SPECS["MAX_TBPS"] * 1e-3
        min_time = max(min_time_flops, min_time_bytes)
        util = min_time / tot_time
        tflops = sum([tot_flops[w] for w in [8, 16]]) / tot_time * 1e-3
        tbps = tot_bytes / tot_time * 1e-3

    return util, tflops, tbps


if __name__ == "__main__":
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or is_hip_cdna4()
    qxdtype = "fp8" if has_native_mx4 else "bf16"
    print(bench_mlp(8192, 8192, 8192, 1, 1, "fp8", "fp8", TP=1, EP=1, name="dense"))
    print(bench_mlp(8192, 8192, 8192, 1, 1, qxdtype, "mx4", TP=1, EP=1, name="dense"))
    print(bench_mlp(2048, 5120, 8192, 128, 4, "fp8", "fp8", TP=4, EP=1, name="llama4"))
    print(bench_mlp(2048, 5120, 8192, 128, 4, qxdtype, "mx4", TP=4, EP=1, name="llama4"))
