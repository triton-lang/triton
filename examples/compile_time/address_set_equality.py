"""Synthetic cold-compilation benchmark for equal shared-memory footprints.

Run on a compatible NVIDIA GPU; defaults were measured on GB300.
See address_set_equality.md for the input contract and comparison procedure.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
import triton as tr
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


# With STAGES=1, both runtime descriptor indices must be zero.
@g.jit
def descriptor_swap(flags, out, WIDTH: gl.constexpr, STEPS: gl.constexpr, CTAS: gl.constexpr, STAGES: gl.constexpr):
    cga: gl.constexpr = [[0]] * (CTAS.bit_length() - 1)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=cga)
    parent = gl.allocate_shared_memory(gl.int8, [STAGES, WIDTH], gl.SwizzledSharedLayout(1, 1, 1, [0], cga_layout=cga))
    for stage in gl.static_range(STAGES):
        parent.index(stage).slice(0, 128).store(gl.full([128], 3 + 4 * stage, gl.int8, layout))
    ia = gl.load(flags)
    ib = gl.load(flags + 1)
    if STAGES != 1:
        ia = ia % STAGES
        ib = ib % STAGES
    a = parent.index(ia)
    b = parent.index(ib)
    for i in gl.static_range(STEPS):
        cond = gl.load(flags + 2 + i) != 0
        if cond:
            a, b = b, a
    x = a.slice(0, 128).load(layout)
    y = b.slice(0, 128).load(layout)
    gl.store(out + gl.arange(0, 128, layout=layout), x.to(gl.int32) + 2 * y.to(gl.int32))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=131072)
    p.add_argument("--ctas", type=int, default=8)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--output", type=Path, required=True, help="Directory for timings and compiled artifacts")
    args = p.parse_args()
    if args.width < 128 or args.width & (args.width - 1):
        p.error("--width must be a power of two, at least 128")
    if args.ctas not in (1, 2, 4, 8) or args.stages < 1 or args.steps < 1:
        p.error("use 1, 2, 4, or 8 CTAs and positive stages/steps")
    dest = args.output
    dest.mkdir(parents=True, exist_ok=True)
    # A new cache and process keep each invocation cold.
    cache_dir = tempfile.TemporaryDirectory(prefix="address-set-compile-")
    tr.knobs.cache.dir = cache_dir.name
    torch.manual_seed(1)
    flags = torch.randint(0, 2, (args.steps + 2,), device="cuda", dtype=torch.int32)
    flags[:2] = torch.tensor([0, 1 % args.stages], device="cuda", dtype=torch.int32)
    out = torch.empty(128, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    tr.runtime.driver.active.get_current_target()
    start = time.perf_counter()
    compiled = descriptor_swap.warmup(
        flags, out, args.width, args.steps, args.ctas, args.stages, num_warps=4, num_ctas=args.ctas, grid=(1,)
    )
    elapsed = time.perf_counter() - start
    descriptor_swap[(1,)](flags, out, args.width, args.steps, args.ctas, args.stages, num_warps=4, num_ctas=args.ctas)
    torch.cuda.synchronize()
    values = flags.cpu().tolist()
    a, b = values[:2]
    for flag in values[2:]:
        if flag:
            a, b = b, a
    expected = (3 + 4 * a) + 2 * (3 + 4 * b)
    assert torch.equal(out, torch.full_like(out, expected))
    for ext, data in compiled.asm.items():
        path = dest / ("kernel." + ext)
        path.write_bytes(data) if isinstance(data, bytes) else path.write_text(str(data))
    result = dict(
        seconds=elapsed,
        width=args.width,
        steps=args.steps,
        ctas=args.ctas,
        stages=args.stages,
        expected=expected,
        shared=compiled.metadata.shared,
        gpu=torch.cuda.get_device_name(),
        triton_version=tr.__version__,
    )
    (dest / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
