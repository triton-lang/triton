#!/usr/bin/env python3
"""Reproduce Hopper MXFP4 ptxas register-allocation regression using random data.

This file is on a reproducer-only branch, not part of PR12132's proposed merge.
Requirements: H100 80GB, PyTorch, and ONE fixed Triton compiler installation at
 a34c2ab7c9700e33085aedc732352e7ce4c7ddfc. Kernel sources must be from
 c7f0bf56bf32c0cca58959b70b944fa41f2038f2 (the base of this branch).
Do not install this kernel checkout's compiler over that compiler environment.
The original run used PyTorch 2.11, CUDA runtime 13.0, driver 580.126.09.

From any directory, using that environment's Python:
  python repro_hopper_mxfp4_ptxas.py \
    --old /absolute/path/to/ptxas-13.2.51 \
    --new /absolute/path/to/ptxas-13.3.33 \
    --kernels /absolute/path/to/this-checkout/python/triton_kernels \
    --output /absolute/path/to/new-results

Only ptxas changes between arms: both assembler environment overrides are set,
PTX is fixed at 8.7, and each worker gets its own empty cache. Changing CUDA_HOME
alone is insufficient. No toolkit/runtime/driver or Triton source change between
arms is needed. The adjacent Triton commit a34c2ab7 vs parent 83a381bb only
explains when Hopper started selecting the newer assembler by default.

Default: uncapped plain BF16 x MXFP4 grouped matmul, 256 experts, K=2048,
N=4096, 64/512 rows; reference checks, output/PTX hash equality, warmup and nine
shuffled interleaved CUDA-graph timing rounds. Expected ptxas13.2 ->13.3 result:
121 ->153 registers/thread, two ->one resident blocks/SM, roughly 17%/16% slower.
13.4 retains the regression. Resource attributes are saved with the PTX/cubin.

Add --with-fix to compare the PR's resource-based register policy as well.
Add --rows 64 32768 --swiglu to investigate the separate large-tile spill case.
That additional case is not an established target-encoder bottleneck.
The policy remains unchanged for that tile; newer assemblers add spills with a
roughly 2-3% deficit in a separate repeat. Dynamic clocks limit timing precision.

Results, selected assembler versions, compiler path, telemetry and compiled
artifacts are written beneath --output. All rounds are retained; medians are
reported, not a production performance claim. No model weights or private data.
This single file includes its worker; no companion Python scripts are needed.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path


def resources(path, metadata):
    """Read compiled resource usage with the CUDA driver, without launching."""
    import ctypes as c

    cu = c.CDLL("libcuda.so.1")

    def call(name, *args):
        status = getattr(cu, name)(*args)
        if status:
            raise RuntimeError((name, status))

    module, function = c.c_void_p(), c.c_void_p()
    call("cuModuleLoad", c.byref(module), str(path).encode())
    try:
        call("cuModuleGetFunction", c.byref(function), module, metadata["name"].encode())
        call("cuFuncSetAttribute", function, 8, metadata["shared"])
        registers, local, blocks = c.c_int(), c.c_int(), c.c_int()
        call("cuFuncGetAttribute", c.byref(registers), 4, function)
        call("cuFuncGetAttribute", c.byref(local), 3, function)
        call(
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            c.byref(blocks),
            function,
            32 * metadata["num_warps"],
            c.c_size_t(metadata["shared"]),
        )
        return dict(
            registers=registers.value,
            local_bytes=local.value,
            blocks_per_sm=blocks.value,
            shared_bytes=metadata["shared"],
        )
    finally:
        call("cuModuleUnload", module)


def worker():
    import json
    import os
    import pathlib
    import subprocess
    import sys

    import torch
    import triton
    from triton.backends.nvidia.compiler import get_ptxas
    from triton_kernels.matmul import FnSpecs, FusedActivation, PrecisionConfig, matmul, matmul_torch
    from triton_kernels.matmul_details import opt_flags
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
    from triton_kernels.swiglu import PrecisionConfig as SwiGLUConfig
    from triton_kernels.swiglu import swiglu, swiglu_fn
    from triton_kernels.tensor import FP4, convert_layout, make_ragged_tensor_metadata, wrap_torch_tensor
    from triton_kernels.tensor_details import layout
    from triton_kernels.testing import assert_close

    mode = os.environ["AUDIT_MODE"]
    ptx = int(os.environ["AUDIT_PTX"])
    record = {}
    original = opt_flags.make_default_opt_flags_nvidia

    def flags(*args, **kw):
        f = original(*args, **kw)
        if mode == "baseline" and not f.is_persistent:
            f.target_kernel_kwargs["maxnreg"] = None
        f.target_kernel_kwargs["ptx_version"] = ptx
        record.update(vars(f))
        return f

    opt_flags.make_default_opt_flags_nvidia = flags

    def emit(x):
        print("AUDIT " + json.dumps(x, default=str), flush=True)

    emit({
        "ready": True,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "triton_path": triton.__file__,
        "ptxas": get_ptxas(90).path,
        "assembler_version": subprocess.check_output([get_ptxas(90).path, "--version"], text=True),
        "device": torch.cuda.get_device_name(),
        "mode": mode,
        "ptx": ptx,
    })
    for line in sys.stdin:
        cmd = json.loads(line)
        if cmd["op"] == "init":
            # Release the previous case before allocating this case.
            x = w = s = out = ragged = gather = scatter = precision = fused = y = None
            torch.cuda.empty_cache()
            torch.manual_seed(1234)
            a = cmd["shape"]
            rows, n, k, e = a["rows"], a["n"], a["k"], a["experts"]
            dtype = getattr(torch, a.get("dtype", "bfloat16"))
            x = torch.randn(rows, k, device="cuda", dtype=dtype)
            dense = torch.randn(e, n, k, device="cuda", dtype=dtype).transpose(-1, -2)
            q, scales = downcast_to_mxfp(dense, torch.uint8, axis=-2)
            del dense
            w = convert_layout(
                wrap_torch_tensor(q, dtype=FP4),
                layout.StridedLayout(-2) if dtype == torch.float16 else layout.make_default_matmul_mxfp4_w_layout(-2),
            )
            del q
            s = convert_layout(wrap_torch_tensor(scales),
                               layout.make_default_matmul_mxfp4_w_scale_layout(-2, num_warps=8))
            del scales
            ids = torch.randint(e, (rows, ), device="cuda")
            if a.get("routing") == "skew":
                ids[:rows * 3 // 4] = 0
            sizes = torch.bincount(ids, minlength=e).to(torch.int32)
            ragged = make_ragged_tensor_metadata(sizes, rows)
            gather = torch.randint(rows, (rows, ), device="cuda", dtype=torch.int32)
            scatter = torch.randperm(rows, device="cuda", dtype=torch.int32)
            precision = PrecisionConfig(out_dtype=dtype, b_mx_scale=s, b_microblock_size=32)
            fused = (FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
                                     (1.702, 7.0)) if a["activation"] else None)
            out = torch.empty(rows, n // (2 if a["activation"] else 1), device="cuda", dtype=dtype)

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

            y = run()
            torch.cuda.synchronize()
            ref = matmul_torch(
                x,
                w,
                None,
                a_ragged_metadata=ragged,
                gather_indx=gather,
                scatter_indx=scatter,
                precision_config=precision,
            )
            if a["activation"]:
                ref = swiglu(ref, alpha=1.702, precision_config=SwiGLUConfig(7.0))
            assert_close(y, ref, maxtol=3e-2)
            del ref
            torch.cuda.empty_cache()
            for _ in range(20):
                run()
            torch.cuda.synchronize()
            digest = hashlib.sha256(y.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            pts = []
            for p in pathlib.Path(os.environ["TRITON_CACHE_DIR"]).glob("*/*.ptx"):
                if "_matmul" not in p.name:
                    continue
                m = json.loads(p.with_suffix(".json").read_text())
                pts.append({
                    "path": str(p),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "metadata": m,
                    "resources": resources(p.with_suffix(".cubin"), m),
                })
            emit({"initialized": a, "flags": record, "output_sha256": digest, "reference_passed": True, "ptx": pts})
        elif cmd["op"] == "bench":
            for _ in range(10):
                run()
            ms = triton.testing.do_bench_cudagraph(run, rep=150)
            emit({"ms": ms})
        elif cmd["op"] == "quit":
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, help="absolute path to older ptxas executable")
    parser.add_argument("--new", required=True, help="absolute path to newer ptxas executable")
    parser.add_argument("--kernels", required=True, help="checkout's python/triton_kernels directory")
    parser.add_argument("--output", required=True, help="new results directory (must not exist)")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--with-fix", action="store_true")
    parser.add_argument("--swiglu", action="store_true", help="include fused SwiGLU in addition to plain")
    args = parser.parse_args()
    if args.rounds < 1 or any(n < 1 for n in args.rows):
        parser.error("rounds and row counts must be positive")
    for name in ("old", "new"):
        p = Path(getattr(args, name))
        if not p.is_absolute() or not p.is_file() or not os.access(p, os.X_OK):
            parser.error(f"--{name} must be an absolute executable path")
    kernels = Path(args.kernels).resolve()
    if not (kernels / "triton_kernels" / "matmul.py").exists():
        parser.error("--kernels must contain triton_kernels/matmul.py")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    configs = [("old-base", args.old, "baseline"), ("new-base", args.new, "baseline")]
    if args.with_fix:
        configs += [("old-fix", args.old, "fixed"), ("new-fix", args.new, "fixed")]
    workers, logs = {}, {}
    measurements = []

    def receive(process):
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"Worker exited ({process.poll()}); inspect {root}/*.stderr")
            if line.startswith("AUDIT "):
                return json.loads(line[6:])

    def call(name, command):
        p = workers[name]
        p.stdin.write(json.dumps(command) + "\n")
        p.stdin.flush()
        return receive(p)

    with (root / "results.jsonl").open("w", buffering=1) as output:

        def record(data):
            output.write(json.dumps(data, default=str) + "\n")

        try:
            for name, binary, mode in configs:
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=args.gpu,
                    PYTHONPATH=str(kernels) + os.pathsep + env.get("PYTHONPATH", ""),
                    AUDIT_MODE=mode,
                    AUDIT_PTX="87",
                    TRITON_PTXAS_PATH=binary,
                    TRITON_PTXAS_BLACKWELL_PATH=binary,
                    TRITON_CACHE_DIR=str(root / ("cache-" + name)),
                )
                logs[name] = (root / (name + ".stderr")).open("w")
                p = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker"],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=logs[name],
                    text=True,
                    bufsize=1,
                )
                workers[name] = p
                info = receive(p)
                record(dict(event="environment", config=name, result=info))
                if Path(info["ptxas"]).resolve() != Path(binary).resolve():
                    raise RuntimeError(f"Unexpected assembler: {info}")
                print(name, info["assembler_version"].strip(), flush=True)
            for rows in args.rows:
                for activation in [False, True] if args.swiglu else [False]:
                    shape = dict(
                        rows=rows,
                        experts=256,
                        k=2048,
                        n=4096,
                        dtype="bfloat16",
                        routing="uniform",
                        activation=activation,
                    )
                    case = json.dumps(shape, sort_keys=True)
                    initialized = {}
                    for name, _, _ in configs:
                        info = call(name, dict(op="init", shape=shape))
                        initialized[name] = info
                        record(dict(event="init", case=case, config=name, result=info))
                    if len({v["output_sha256"] for v in initialized.values()}) != 1:
                        raise RuntimeError("Assembler/policy variants produced different outputs")
                    for suffix in ["base", "fix"] if args.with_fix else ["base"]:
                        old = {x["sha256"] for x in initialized["old-" + suffix]["ptx"]}
                        new = {x["sha256"] for x in initialized["new-" + suffix]["ptx"]}
                        if not old or old != new:
                            raise RuntimeError("PTX differs across assembler arms; comparison is not isolated")
                    times = {name: [] for name in workers}
                    for round_index in range(args.rounds):
                        order = list(workers)
                        random.Random(947 + round_index).shuffle(order)
                        for name in order:
                            ms = call(name, dict(op="bench"))["ms"]
                            times[name].append(ms)
                            record(dict(event="timing", case=case, config=name, round=round_index, ms=ms))
                        telemetry = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "-i",
                                args.gpu,
                                "--query-gpu=uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate",
                                "--format=csv,noheader",
                            ],
                            text=True,
                        ).strip()
                        record(dict(event="telemetry", case=case, round=round_index, data=telemetry))
                    summary = dict(
                        shape=shape,
                        median_ms={n: statistics.median(v)
                                   for n, v in times.items()},
                        paired_slowdown_percent=100 *
                        (statistics.median([b / a for a, b in zip(times["old-base"], times["new-base"])]) - 1),
                    )
                    measurements.append(summary)
                    print(json.dumps(summary), flush=True)
            record(dict(event="complete"))
            (root / "summary.json").write_text(json.dumps(measurements, indent=2) + "\n")
        finally:
            for p in workers.values():
                if p.poll() is None:
                    p.terminate()
            for p in workers.values():
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for log in logs.values():
                log.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
    else:
        main()
