"""Interleaved exact-input FP4 controls and all-K96 Gluon experiments on sm103.

Use reductions divisible by 768 for the packed stream. Example:
  python bench-tcgen05-pure-k96.py --size 16384 --k 16128 --buffers 6 --epilogue 32 --output RESULTS
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import triton
from triton.tools.mxfp import MXFP4Tensor


def load_experiment():
    path = Path(__file__).with_name('experimental-tcgen05-k96.py')
    spec = importlib.util.spec_from_file_location('k96_experiment', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare(m, n, k, fmt):
    vec = 16 if fmt == 'nvfp4' else 32
    operands, scales, refs = [], [], []
    for rows in (m, n):
        values = MXFP4Tensor(size=(rows, k), device='cuda').random()
        exponent = torch.randint(-3, 1, (rows, k // vec), device='cuda')
        scale = torch.exp2(exponent.float())
        refs.append(values.to(torch.float32) * scale.repeat_interleave(vec, dim=1))
        operands.append(values.to_packed_tensor(dim=1))
        scales.append(scale.to(torch.float8_e4m3fn) if vec == 16 else (exponent + 127).to(torch.uint8))
    return operands, scales, refs, vec


def gpu_state():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,clocks.sm,temperature.gpu,power.draw,utilization.gpu',
        '--format=csv,noheader'
    ], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=8192, help='M=N; K can be set separately without padding')
    parser.add_argument('--k', type=int)
    parser.add_argument('--format', choices=['mxfp4', 'nvfp4'], default='mxfp4')
    parser.add_argument('--modes', nargs='+', choices=['k64', 'mixed', 'raw', 'pure', 'exact192', 'exact384'],
                        default=['k64', 'mixed', 'raw', 'pure'])
    parser.add_argument('--buffers', type=int, default=6,
                        help='Experimental modes only; native controls retain five buffers')
    parser.add_argument('--epilogue', type=int, default=32, help='Experimental modes only; native controls retain N64')
    parser.add_argument('--scheduler', choices=['auto', 'sps', 'clc'], default='auto')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--rep-ms', type=int, default=500)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    k = args.k if args.k is not None else args.size // 768 * 768
    experiment = load_experiment()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    operands, scales, refs, vec = prepare(args.size, args.size, k, args.format)
    expected = refs[0] @ refs[1].T
    scales = [experiment.base.swizzle_scales_packed_block(s) for s in scales]
    scheduler = args.scheduler if args.scheduler != 'auto' else ('sps' if k <= 8192 else 'clc')
    scheduler_id = experiment.SCHEDULER_SPS if scheduler == 'sps' else experiment.SCHEDULER_CLC
    config = dict(experiment.base.BEST_2CTA_CONFIG, scheduler=scheduler_id, num_buffers=5)
    selected = {}
    original_run = experiment.base.mma_scaled_warp_specialized_kernel.run

    def capture(*a, **kw):
        compiled = original_run(*a, **kw)
        selected['kernel'] = compiled
        return compiled

    experiment.base.mma_scaled_warp_specialized_kernel.run = capture
    root = Path(__file__).resolve().parents[3]
    report = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)).hexdigest(),
                  source_sha256=hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest(),
                  torch=torch.__version__, triton=triton.__file__, gpu_before=gpu_state(),
                  method='same inputs; alternating order; CUDA graphs; no L2 flush', seed=123,
                  scale_values=[0.125, 0.25, 0.5, 1.0],
                  args=vars(args) | dict(output=str(args.output), k=k, scheduler=scheduler), results=[])
    (args.output / 'source.py').write_bytes(Path(experiment.__file__).read_bytes())
    functions = []
    baseline = None
    for mode in args.modes:
        if mode in ('k64', 'mixed'):

            def fn(mode=mode):
                return experiment.base.mma_scaled_warp_specialized(*operands, *scales, vec,
                                                                   enable_fp4_k96=mode == 'mixed', **config)
        else:

            def fn(mode=mode):
                out, compiled = experiment.matmul(*operands, *scales, vec, mode=mode, buffers=args.buffers,
                                                  epilogue=args.epilogue, scheduler=scheduler_id)
                selected['kernel'] = compiled
                return out

        out = fn()
        compiled = selected['kernel']
        for ext in ('ptx', 'ttgir', 'cubin'):
            data = compiled.asm[ext]
            (args.output / f'{mode}.{ext}').write_bytes(data if isinstance(data, bytes) else data.encode())
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=1e-3)
        if baseline is None:
            baseline = out
        row = dict(mode=mode, correct=True, shared=compiled.metadata.shared, tmem=compiled.metadata.tmem_size,
                   cubin_sha256=hashlib.sha256(compiled.asm['cubin']).hexdigest(),
                   ptx_mmas=compiled.asm['ptx'].count('tcgen05.mma.'),
                   baseline_max_abs=(out.float() - baseline.float()).abs().max().item(), ms=[])
        report['results'].append(row)
        functions.append(fn)
        print(json.dumps(row), flush=True)
    for fn in functions:
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    for repeat in range(args.repeats):
        order = range(len(functions)) if repeat % 2 == 0 else reversed(range(len(functions)))
        for i in order:
            report['results'][i]['ms'].append(triton.testing.do_bench_cudagraph(functions[i], rep=args.rep_ms))
    for row in report['results']:
        row['median_ms'] = statistics.median(row['ms'])
        row['pflops'] = 2 * args.size**2 * k / (row['median_ms'] * 1e12)
        print(json.dumps(row), flush=True)
    report['gpu_after'] = gpu_state()
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
