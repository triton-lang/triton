#!/usr/bin/env python3
"""H100 BF16 x MXFP4 grouped-matmul benchmark for ptxas 13.2 versus 13.3.

Requirements: H100 80GB, PyTorch, Triton compiler at
 a34c2ab7c9700e33085aedc732352e7ce4c7ddfc, and this branch's triton_kernels.
Keep the same compiler, kernel source, GPU, and Python environment for both runs.
Only change ptxas. Changing CUDA_HOME alone does not select Triton's assembler.
From this checkout's root, run each command in a fresh shell:

  export PYTHONPATH="$PWD/python/triton_kernels:$PYTHONPATH"
  export TRITON_PTXAS_PATH=/path/to/cuda-13.2/bin/ptxas
  export TRITON_PTXAS_BLACKWELL_PATH="$TRITON_PTXAS_PATH"
  export TRITON_CACHE_DIR=$(mktemp -d)
  python python/triton_kernels/bench/repro_hopper_mxfp4_ptxas.py

Repeat with /path/to/cuda-13.3/bin/ptxas and another fresh cache directory.
Both overrides are set because this compiler can select either assembler slot.
The measured versions were 13.2.51 and 13.3.33; 13.4.59 also regressed.
The original environment used PyTorch 2.11, CUDA runtime 13.0, driver 580.126.09.
Do not replace the fixed compiler when making triton_kernels importable.

This calls the public triton_kernels matmul with random weights and routing.
The explicit configuration reproduces the affected 16x256x128 tile without the
PR's register cap; PTX 8.7 is supported by both assemblers. --maxnreg 128 tests
its occupancy-preserving cap. There are no CUDA-driver calls or worker processes.

Expected uncapped timings from the earlier controlled H100 experiment:
  rows     ptxas 13.2     ptxas 13.3
    64       121.5 us       142.3 us
   512       431.5 us       500.5 us
Absolute timings vary. Run on an idle GPU and repeat both versions before
comparing. This is a synthetic kernel benchmark, not an end-to-end model test.
"""

import argparse
import statistics
import subprocess


def bench(rows, maxnreg):
    import torch
    from triton.testing import do_bench_cudagraph
    from triton_kernels.matmul import PrecisionConfig, matmul, matmul_torch
    from triton_kernels.matmul_details.opt_flags import OptFlags, scoped_opt_flags
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
    from triton_kernels.tensor import FP4, convert_layout, make_ragged_tensor_metadata, wrap_torch_tensor
    from triton_kernels.tensor_details import layout
    from triton_kernels.testing import assert_close

    torch.manual_seed(1234)
    experts, k, n = 256, 2048, 4096
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    dense = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16).transpose(-1, -2)
    q, scales = downcast_to_mxfp(dense, torch.uint8, axis=-2)
    del dense
    w = convert_layout(wrap_torch_tensor(q, dtype=FP4), layout.make_default_matmul_mxfp4_w_layout(-2))
    s = convert_layout(wrap_torch_tensor(scales), layout.make_default_matmul_mxfp4_w_scale_layout(-2, num_warps=8))
    del q, scales
    ids = torch.randint(experts, (rows, ), device="cuda")
    sizes = torch.bincount(ids, minlength=experts).to(torch.int32)
    ragged = make_ragged_tensor_metadata(sizes, rows)
    gather = torch.randint(rows, (rows, ), device="cuda", dtype=torch.int32)
    scatter = torch.randperm(rows, device="cuda", dtype=torch.int32)
    precision = PrecisionConfig(out_dtype=torch.bfloat16, b_mx_scale=s, b_microblock_size=32)
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(a_ragged_metadata=ragged, gather_indx=gather, scatter_indx=scatter, precision_config=precision)
    flags = OptFlags(
        block_m=16,
        block_n=256,
        block_k=128,
        num_warps=8,
        num_stages=5,
        group_m=8,
        xcd_swizzle=1,
        w_cache_modifier=None,
        split_k=1,
        is_persistent=False,
        idle_sms=0,
        epilogue_subtile=1,
        arch=None,
        occupancy_target=2,
        target_kernel_kwargs={"maxnreg": maxnreg, "FLATTEN_LOOPS": False, "ptx_version": 87},
    )
    with scoped_opt_flags(flags):

        def run():
            return matmul(x, w, None, c=out, **kwargs)

        ref = matmul_torch(x, w, None, **kwargs)
        assert_close(run(), ref, maxtol=3e-2)
        del ref
        torch.cuda.empty_cache()
        for _ in range(20):
            run()
        samples = [1000 * do_bench_cudagraph(run, rep=150) for _ in range(9)]
    print(f"rows={rows}, maxnreg={maxnreg}, median={statistics.median(samples):.3f} us")
    print("  samples (us):", ", ".join(f"{x:.3f}" for x in samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maxnreg", type=int, default=None, help="optional register cap; try 128")
    args = parser.parse_args()
    import torch
    import triton
    from triton.backends.nvidia.compiler import get_ptxas

    print("GPU:", torch.cuda.get_device_name())
    print("Triton:", triton.__version__, triton.__file__)
    assembler = get_ptxas(90).path
    print("ptxas:", assembler)
    print(subprocess.check_output([assembler, "--version"], text=True).strip())
    for rows in (64, 512):
        bench(rows, args.maxnreg)


if __name__ == "__main__":
    main()
