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
from types import SimpleNamespace

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


def _json_constant(value):
    if isinstance(value, triton.language.constexpr):
        return _json_constant(value.value)
    if isinstance(value, (tuple, list)):
        return [_json_constant(x) for x in value]
    return value


class FrozenKernel:
    """Replay an archived cubin through the normal launcher, without compilation."""

    def __init__(self, directory):
        from triton.compiler.compiler import CompiledKernel
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'launch.json').read_text())
        self.constants = {tuple(k): v for k, v in self.manifest['constants']}
        source = SimpleNamespace(signature=dict(self.manifest['signature']), constants=self.constants,
                                 fn=SimpleNamespace(arg_names=self.manifest['arg_names']))
        group = {name: str(self.directory / name) for name in self.manifest['files']}
        self.kernel = CompiledKernel(source, group, self.manifest['cache_hash'])
        assert hashlib.sha256(self.kernel.asm['cubin']).hexdigest() == self.manifest['cubin_sha256']

    @staticmethod
    def archive(compiled, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, source in compiled.metadata_group.items():
            data = Path(source).read_bytes()
            (directory / name).write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
        manifest = dict(cache_hash=compiled.hash, cubin_sha256=hashlib.sha256(compiled.asm['cubin']).hexdigest(),
                        arg_names=compiled.src.fn.arg_names, signature=list(compiled.src.signature.items()),
                        constants=[[list(k), _json_constant(v)]
                                   for k, v in compiled.src.constants.items()], files=files)
        (directory / 'launch.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest

    def invoke(self, example, operands, scales, vec, config):
        m, n, k = operands[0].shape[0], operands[1].shape[0], operands[0].shape[1] * 2
        ad, bd, cd, asd, bsd = example.make_dummy_descriptors(*operands, *scales, torch.float16, m, n)
        values = dict(a_desc=ad, b_desc=bd, c_desc=cd, a_scale_desc=asd, b_scale_desc=bsd, M=m, N=n, K=k,
                      A_ELEM_PER_BYTE=2, B_ELEM_PER_BYTE=2, BLOCK_M=256, BLOCK_N=256, BLOCK_K=config['block_k'],
                      EPILOGUE_BLOCK_N=config['epilogue'], CGA_LAYOUT=((1, 0), ))
        example.mma_scaled_tma_set_block_size_hook(values)
        args = [
            values[name] if name in values else self.constants[(i, )]
            for i, name in enumerate(self.manifest['arg_names'])
        ]
        scheduler = example.SCHEDULER_SPS if config['scheduler'] == 'sps' else example.SCHEDULER_CLC
        grid = example.mma_scaled_warp_specialized_grid(m, n, 256, 256, 2, scheduler, operands[0].device)
        grid = tuple(grid) + (1, ) * (3 - len(grid))
        self.kernel[grid](*args)
        return cd.base


def control_key(config):
    return '-'.join(str(config[x]) for x in ('format', 'mode', 'block_k', 'buffers', 'epilogue', 'scheduler'))


def recorded_controls():
    root = Path(__file__).parent
    mixed = json.loads((root / 'k96-measurements.json').read_text())['main']
    pure = json.loads((root / 'pure-k96-measurements.json').read_text())['final_sweep']
    controls = []
    for case in mixed['cases']:
        for row in case['variants']:
            controls.append(
                dict(format=case['format'], size=case['size'], k=case['size'],
                     mode='mixed' if row['mode'] == 'k96' else 'k64', block_k=case['block_k'], buffers=case['buffers'],
                     epilogue=64, scheduler=case['scheduler'], cubin_sha256=row['cubin_sha256'],
                     historical_ms=row['median_ms']))
    for report in pure:
        args = report['args']
        for row in report['results']:
            native = row['mode'] in ('k64', 'mixed')
            controls.append(
                dict(format=args['format'], size=args['size'], k=args['k'], mode=row['mode'], block_k=256,
                     buffers=5 if native else args['buffers'], epilogue=64 if native else args['epilogue'],
                     scheduler=args['scheduler'], cubin_sha256=row['cubin_sha256'], historical_ms=row['median_ms']))
    return controls


def freeze_controls(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    experiment = load_experiment()
    selected = {}
    original = experiment.base.mma_scaled_warp_specialized_kernel.run

    def capture(*args, **kwargs):
        compiled = original(*args, **kwargs)
        selected['kernel'] = compiled
        return compiled

    experiment.base.mma_scaled_warp_specialized_kernel.run = capture
    controls = recorded_controls()
    configs = {}
    for config in controls:
        key = control_key(config)
        if key in configs:
            assert configs[key]['cubin_sha256'] == config['cubin_sha256']
        configs[key] = config
    verified = []
    for key, config in configs.items():
        torch.manual_seed(123)
        operands, scales, refs, vec = prepare(256, 256, 768, config['format'])
        scales = [experiment.base.swizzle_scales_packed_block(s) for s in scales]
        scheduler = experiment.SCHEDULER_SPS if config['scheduler'] == 'sps' else experiment.SCHEDULER_CLC
        if config['mode'] in ('k64', 'mixed'):
            options = dict(experiment.base.BEST_2CTA_CONFIG, scheduler=scheduler, num_buffers=config['buffers'])
            output = experiment.base.mma_scaled_warp_specialized(*operands, *scales, vec,
                                                                 enable_fp4_k96=config['mode'] == 'mixed', **options)
            compiled = selected['kernel']
        else:
            output, compiled = experiment.matmul(*operands, *scales, vec, mode=config['mode'],
                                                 buffers=config['buffers'], epilogue=config['epilogue'],
                                                 scheduler=scheduler)
        digest = hashlib.sha256(compiled.asm['cubin']).hexdigest()
        assert digest == config['cubin_sha256'], (key, digest, config['cubin_sha256'])
        FrozenKernel.archive(compiled, directory / digest)
        frozen = FrozenKernel(directory / digest)
        replayed = frozen.invoke(experiment.base, operands, scales, vec, config)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = frozen.invoke(experiment.base, operands, scales, vec, config)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(replayed.float(), refs[0] @ refs[1].T, atol=1e-3, rtol=1e-3)
        assert torch.equal(output.view(torch.int16), replayed.view(torch.int16))
        assert torch.equal(output.view(torch.int16), graph_output.view(torch.int16))
        verified.append(dict(key=key, cubin_sha256=digest, graph_replays=10, bit_identical=True))
        print(json.dumps(verified[-1]), flush=True)
    for name in ('experimental-tcgen05-k96.py', '04-2cta-block-scale-matmul.py', 'k96-measurements.json',
                 'pure-k96-measurements.json'):
        (directory / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    manifest = dict(checkpoint='d93e07b76de305b651df579ce50b64ab7d237618', controls=controls, verified=verified,
                    gpu=gpu_state())
    (directory / 'index.json').write_text(json.dumps(manifest, indent=2) + '\n')


def verify_frozen_cases(directory):
    directory = Path(directory)
    index = json.loads((directory / 'index.json').read_text())
    experiment = load_experiment()
    groups = {}
    for config in index['controls']:
        key = (config['format'], config['size'], config['k'])
        groups.setdefault(key, []).append(config)
    checked = []
    for (fmt, size, k), configs in groups.items():
        torch.manual_seed(123)
        operands, scales, refs, vec = prepare(size, size, k, fmt)
        expected = refs[0] @ refs[1].T
        scales = [experiment.base.swizzle_scales_packed_block(s) for s in scales]
        baseline = None
        for config in configs:
            frozen = FrozenKernel(directory / config['cubin_sha256'])
            output = frozen.invoke(experiment.base, operands, scales, vec, config)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = frozen.invoke(experiment.base, operands, scales, vec, config)
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output.float(), expected, atol=1e-3, rtol=1e-3)
            assert torch.equal(output.view(torch.int16), replayed.view(torch.int16))
            if baseline is None:
                baseline = output
            assert torch.equal(output.view(torch.int16), baseline.view(torch.int16))
            checked.append(
                dict(format=fmt, size=size, k=k, mode=config['mode'], cubin_sha256=config['cubin_sha256'], correct=True,
                     bit_identical=True, graph_replays=10))
            print(json.dumps(checked[-1]), flush=True)
        (directory / 'validation.json').write_text(json.dumps(checked, indent=2) + '\n')


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
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze-controls', type=Path, help='Archive and verify the recorded control binaries')
    parser.add_argument('--frozen', type=Path, help='Add paired archived controls to this measurement')
    parser.add_argument('--verify-frozen', type=Path, help='Validate archived controls at every recorded shape')
    args = parser.parse_args()
    if args.freeze_controls is not None:
        freeze_controls(args.freeze_controls)
        return
    if args.verify_frozen is not None:
        verify_frozen_cases(args.verify_frozen)
        return
    if args.output is None:
        parser.error('--output is required for benchmarking')
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
    if args.frozen is not None:
        index = json.loads((args.frozen / 'index.json').read_text())
        for mode in args.modes:
            native = mode in ('k64', 'mixed')
            config = dict(format=args.format, mode=mode, block_k=256, buffers=5 if native else args.buffers,
                          epilogue=64 if native else args.epilogue, scheduler=scheduler)
            matched = next(c for c in index['controls'] if control_key(c) == control_key(config))
            frozen = FrozenKernel(args.frozen / matched['cubin_sha256'])

            def fn(frozen=frozen, config=config):
                return frozen.invoke(experiment.base, operands, scales, vec, config)

            out = fn()
            torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=1e-3)
            report['results'].append(
                dict(mode='frozen-' + mode, correct=True, cubin_sha256=matched['cubin_sha256'], ms=[]))
            functions.append(fn)
        count = len(args.modes)
        paired = [i for pair in zip(range(count), range(count, 2 * count)) for i in pair]
        functions = [functions[i] for i in paired]
        report['results'] = [report['results'][i] for i in paired]
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
    if args.frozen is not None:
        rows = {row['mode']: row for row in report['results']}
        for mode in args.modes:
            row = rows[mode]
            row['latency_ratio_vs_frozen'] = row['median_ms'] / rows['frozen-' + mode]['median_ms']
            row['within_two_percent'] = row['latency_ratio_vs_frozen'] <= 1.02
    report['gpu_after'] = gpu_state()
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
