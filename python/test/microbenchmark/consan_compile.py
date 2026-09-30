"""Measure ConSan compilation of a synthetic affine map and reduction.

Use a fresh TRITON_CACHE_DIR for each baseline or candidate run, for example:
  TRITON_CACHE_DIR=$(mktemp -d) python consan_compile.py \
      --width 32768 --depth 4 --output /tmp/consan-compile-run --run
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['TRITON_INSTRUMENTATION_MODE'] = 'consan'
import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.backends.nvidia.compiler import CUDABackend


@g.jit
def add_pair(left, right):
    return left + right


@g.jit
def wide_integer_map(source, destination, WIDTH: gl.constexpr, DEPTH: gl.constexpr, WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [WARPS], [0])
    offsets = gl.arange(0, WIDTH, layout=layout)
    values = offsets + gl.load(source)
    for _ in gl.static_range(DEPTH):
        values = values * 1664525 + 1013904223
    selected = gl.reduce(values, 0, add_pair)
    gl.store(destination, selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--depth', type=int, default=0)
    parser.add_argument('--warps', type=int, default=4)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    stages = {}
    for name in ('gluon_to_ttgir', 'make_llir', 'make_ptx', 'make_cubin'):
        original = getattr(CUDABackend, name, None)
        if original is None:
            continue

        def measured(self, *a, _name=name, _original=original, **kw):
            start = time.monotonic()
            result = _original(self, *a, **kw)
            stages[_name] = time.monotonic() - start
            print('STAGE', _name, stages[_name], flush=True)
            return result

        setattr(CUDABackend, name, measured)
    source = torch.arange(args.width, dtype=torch.int32, device='cuda') + 17
    destination = torch.empty(1, dtype=source.dtype, device='cuda')
    torch.cuda.synchronize()
    start = time.monotonic()
    kernel = wide_integer_map.warmup(source, destination, args.width, args.depth, args.warps, grid=(1, ),
                                     num_warps=args.warps)
    elapsed = time.monotonic() - start
    hashes = {}
    sizes = {}
    for name in ('ttgir', 'llir', 'ptx', 'cubin'):
        value = kernel.asm[name]
        value = value.encode() if isinstance(value, str) else value
        (out / ('kernel.' + name)).write_bytes(value)
        hashes[name] = hashlib.sha256(value).hexdigest()
        sizes[name] = len(value)
    correct = None
    if args.run:
        wide_integer_map[(1, )](source, destination, args.width, args.depth, args.warps, num_warps=args.warps)
        reference = torch.arange(args.width, dtype=torch.int32, device="cuda") + source[0]
        for _ in range(args.depth):
            reference = reference * 1664525 + 1013904223
        reference = reference.sum(dtype=torch.int32).reshape(1)
        torch.cuda.synchronize()
        torch.testing.assert_close(destination, reference, rtol=0, atol=0)
        correct = True
    result = dict(width=args.width, depth=args.depth, warps=args.warps, compiler=triton.__file__, seconds=elapsed,
                  stages=stages, hashes=hashes, sizes=sizes, correct=correct)
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
