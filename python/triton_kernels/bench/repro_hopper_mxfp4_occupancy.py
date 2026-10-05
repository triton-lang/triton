"""Random-data Hopper MXFP4 reproducer: torch + triton + public triton_kernels only."""

import argparse
import json
import torch
import triton
from triton.backends.nvidia.compiler import get_ptxas
from triton_kernels.matmul import matmul, matmul_torch, PrecisionConfig, FusedActivation, FnSpecs
from triton_kernels.matmul_details import opt_flags
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.swiglu import swiglu, swiglu_fn, PrecisionConfig as SwiGLUConfig
from triton_kernels.tensor import FP4, wrap_torch_tensor, convert_layout, make_ragged_tensor_metadata
from triton_kernels.tensor_details import layout
from triton_kernels.testing import assert_close

p = argparse.ArgumentParser()
p.add_argument("--rows", type=int, default=64)
p.add_argument("--experts", type=int, default=8)
p.add_argument("--k", type=int, default=2048)
p.add_argument("--n", type=int, default=4096)
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--rep-ms", type=int, default=200)
p.add_argument("--check-only", action="store_true")
a = p.parse_args()
assert torch.cuda.get_device_capability()[0] == 9
print(
    json.dumps(
        dict(
            torch=torch.__version__,
            triton=triton.__version__,
            ptxas=get_ptxas(90).path,
            gpu=torch.cuda.get_device_name(),
        )
    ),
    flush=True,
)
torch.manual_seed(a.seed)
x = torch.randn(a.rows, a.k, device="cuda", dtype=torch.bfloat16)
w = torch.randn(a.experts, a.n, a.k, device="cuda", dtype=torch.bfloat16).transpose(-1, -2)
q, s = downcast_to_mxfp(w, torch.uint8, axis=-2)
w = convert_layout(wrap_torch_tensor(q, dtype=FP4), layout.make_default_matmul_mxfp4_w_layout(-2))
s = convert_layout(wrap_torch_tensor(s), layout.make_default_matmul_mxfp4_w_scale_layout(-2, num_warps=8))
ids = torch.randint(a.experts, (a.rows,), device="cuda")
sizes = torch.bincount(ids, minlength=a.experts).to(torch.int32)
ragged = make_ragged_tensor_metadata(sizes, a.rows)
gather = torch.randint(a.rows, (a.rows,), device="cuda", dtype=torch.int32)
scatter = torch.randperm(a.rows, device="cuda", dtype=torch.int32)
precision = PrecisionConfig(out_dtype=torch.bfloat16, b_mx_scale=s, b_microblock_size=32)
original = opt_flags.make_default_opt_flags_nvidia
mode = "baseline"
record = {}


def flags(*args, **kwargs):
    f = original(*args, **kwargs)
    if mode == "baseline":
        f.target_kernel_kwargs["maxnreg"] = None
    else:
        assert f.target_kernel_kwargs["maxnreg"] == 128, "Apply hopper-register-cap.patch first"
    record.update(vars(f))
    return f


opt_flags.make_default_opt_flags_nvidia = flags
constraints = dict(block_m=16, block_n=256, block_k=128, num_warps=8, num_stages=5, is_persistent=False, split_k=1)
for activation in (False, True):
    fused = (
        FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2), (1.702, 7.0))
        if activation
        else None
    )
    results = {}
    out = torch.empty((a.rows, a.n // (2 if activation else 1)), device="cuda", dtype=torch.bfloat16)
    for mode in ("baseline", "patched"):

        def run():
            return matmul(
                x,
                w,
                None,
                a_ragged_metadata=ragged,
                gather_indx=gather,
                scatter_indx=scatter,
                precision_config=precision,
                fused_activation=fused,
                c=out,
            )

        with opt_flags.scoped_opt_flags_constraints(constraints):
            y = run()
            torch.cuda.synchronize()
            results[mode] = y.clone()
            times = [] if a.check_only else [triton.testing.do_bench_cudagraph(run, rep=a.rep_ms) for _ in range(3)]
        print(
            json.dumps(
                dict(mode=mode, swiglu=activation, shape=[a.rows, a.n, a.k, a.experts], ms=times, flags=record),
                default=str,
            ),
            flush=True,
        )
    assert torch.equal(results["baseline"], results["patched"]), "baseline/patch bitwise mismatch"
    ref = matmul_torch(
        x, w, None, a_ragged_metadata=ragged, gather_indx=gather, scatter_indx=scatter, precision_config=precision
    )
    if activation:
        ref = swiglu(ref, alpha=1.702, precision_config=SwiGLUConfig(7.0))
    assert_close(results["patched"], ref, maxtol=3e-2)
    print("PASS bitwise baseline parity and independent torch reference", activation, flush=True)
