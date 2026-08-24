"""Interleaved native FP4 kernels and immutable binary controls on sm103.

Use reductions divisible by 768 for the packed stream. Example:
  python bench-tcgen05-pure-k96.py --size 16384 --k 16128 --buffers 6 --epilogue 32 --output RESULTS
"""
import argparse
from dataclasses import asdict
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
from triton.experimental.gluon.language import NVMMASharedLayout
from triton.tools.mxfp import MXFP4Tensor


def load_experiment(filename="experimental-tcgen05-k96.py"):
    path = Path(__file__).with_name(filename)
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
        for name, digest in self.manifest['files'].items():
            assert hashlib.sha256((self.directory / name).read_bytes()).hexdigest() == digest, name
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

    def launch_config(self):

        def saved(name):
            return self.constants[(self.manifest['arg_names'].index(name), )]

        return dict(block_k=saved('BLOCK_K'), buffers=saved('num_buffers'), epilogue=saved('EPILOGUE_BLOCK_N'),
                    scheduler='clc' if saved('scheduler') == 'cluster_launch_control' else 'sps')

    def invoke(self, example, operands, scales, vec, config=None):
        if config is None:
            config = self.launch_config()
        m, n, k = operands[0].shape[0], operands[1].shape[0], operands[0].shape[1] * 2
        ad, bd, cd, asd, bsd = example.make_dummy_descriptors(*operands, *scales, torch.float16, m, n)
        values = dict(a_desc=ad, b_desc=bd, c_desc=cd, a_scale_desc=asd, b_scale_desc=bsd, M=m, N=n, K=k,
                      A_ELEM_PER_BYTE=2, B_ELEM_PER_BYTE=2, BLOCK_M=256, BLOCK_N=256, BLOCK_K=config['block_k'],
                      EPILOGUE_BLOCK_N=config['epilogue'], CGA_LAYOUT=((1, 0), ))
        example.mma_scaled_tma_set_block_size_hook(values)
        if 'c_final' in self.manifest['arg_names']:
            layout = NVMMASharedLayout(128, 16, cga_layout=((1, 0), ))
            values['c_final'] = example.TensorDescriptor.from_tensor(cd.base, [256, 256], layout)
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
    parser.add_argument('--example', choices=['experimental-tcgen05-k96.py', '07-pure-k96-matmul.py'],
                        default="experimental-tcgen05-k96.py")
    parser.add_argument('--size', type=int, default=8192, help='M=N; K can be set separately without padding')
    parser.add_argument('--m', type=int, help='Override the row dimension for a rectangular problem')
    parser.add_argument('--n', type=int, help='Override the column dimension for a rectangular problem')
    parser.add_argument('--k', type=int)
    parser.add_argument('--out-dtype', choices=['float16', 'bfloat16', 'float32'], default='float16',
                        help='Example 07 output type; input operands remain packed FP4')
    parser.add_argument('--format', choices=['mxfp4', 'nvfp4'], default='mxfp4')
    parser.add_argument('--modes', nargs='+', choices=['k64', 'mixed', 'native'], default=['k64', 'mixed', 'native'])
    parser.add_argument('--buffers', type=int, help='Pure K96 only; mixed/K64 controls retain five buffers')
    parser.add_argument('--epilogue', type=int, help='Pure K96 only; mixed/K64 controls retain N64')
    parser.add_argument('--scheduler', choices=['auto', 'sps', 'clc'], default='auto')
    parser.add_argument('--tile-width', type=int, help='Example 07 traversal width')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--rep-ms', type=int, default=500)
    parser.add_argument('--output', type=Path)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument('--compare-native', action='store_true',
                            help='Compare with the original native K96 kernel and its original scheduling')
    comparison.add_argument('--frozen', type=Path, help='Add paired archived controls to this measurement')
    comparison.add_argument('--frozen-native', type=Path,
                            help='Compare with one archived native binary using its saved launch configuration')
    parser.add_argument('--verify-frozen', type=Path, help='Validate archived controls at every recorded shape')
    args = parser.parse_args()
    if args.tile_width is not None and args.example != '07-pure-k96-matmul.py':
        parser.error('--tile-width requires example 07')
    if args.out_dtype != 'float16' and (args.example != '07-pure-k96-matmul.py' or args.modes != ['native']
                                        or args.compare_native or args.frozen or args.frozen_native):
        parser.error('non-FP16 output requires example 07 native mode without historical controls')
    out_dtype = getattr(torch, args.out_dtype)
    tuned_dense = args.example == '07-pure-k96-matmul.py' and args.frozen is None
    if args.buffers is None:
        args.buffers = 6 if args.format == 'mxfp4' or (tuned_dense and out_dtype.itemsize == 2) else 5
    if args.epilogue is None:
        split_scales = tuned_dense and args.format == 'nvfp4' and args.buffers > 5
        args.epilogue = 16 if split_scales else (64 if args.format == 'mxfp4' else 128) // out_dtype.itemsize
    if args.verify_frozen is not None:
        verify_frozen_cases(args.verify_frozen)
        return
    if args.output is None:
        parser.error('--output is required for benchmarking')
    m = args.m if args.m is not None else args.size
    n = args.n if args.n is not None else args.size
    k = args.k if args.k is not None else args.size // 768 * 768
    experiment = load_experiment(args.example)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    operands, scales, refs, vec = prepare(m, n, k, args.format)
    expected = refs[0] @ refs[1].T
    if out_dtype == torch.bfloat16:
        expected = expected.to(out_dtype).float()
    scales = [experiment.base.swizzle_scales_packed_block(s) for s in scales]
    scheduler = args.scheduler
    if scheduler == 'auto':
        if tuned_dense:
            scheduler = 'sps'
        elif args.example == '07-pure-k96-matmul.py':
            scheduler = 'clc' if vec == 16 and k > 16384 else 'sps'
        else:
            scheduler = 'sps' if k <= 8192 else 'clc'
    scheduler_id = experiment.SCHEDULER_SPS if scheduler == 'sps' else experiment.SCHEDULER_CLC
    config = dict(experiment.base.BEST_2CTA_CONFIG, scheduler=scheduler_id, num_buffers=5)
    selected = {}

    def capture(run):

        def wrapped(*a, **kw):
            compiled = run(*a, **kw)
            selected['kernel'] = compiled
            return compiled

        return wrapped

    base_kernel = experiment.base.mma_scaled_warp_specialized_kernel
    base_kernel.run = capture(base_kernel.run)
    if args.example == '07-pure-k96-matmul.py':
        experiment.dense_k96_kernel.run = capture(experiment.dense_k96_kernel.run)
    root = Path(__file__).resolve().parents[3]
    from triton.backends.nvidia.compiler import get_ptxas, get_ptxas_version
    assembler = get_ptxas(torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1])
    report = dict(
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)).hexdigest(),
        source_sha256=hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest(),
        compiler_library_sha256=hashlib.sha256(Path(triton._C.libtriton.__file__).read_bytes()).hexdigest(),
        torch=torch.__version__, triton=triton.__file__, gpu_before=gpu_state(), ptxas_path=assembler.path,
        ptxas_sha256=hashlib.sha256(Path(assembler.path).read_bytes()).hexdigest(),
        ptxas_version=get_ptxas_version(103), method='same inputs; alternating order; CUDA graphs; no L2 flush',
        seed=123, scale_values=[0.125, 0.25, 0.5, 1.0],
        args={key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()} | dict(m=m, n=n, k=k, scheduler=scheduler), inputs=[
                  dict(shape=list(t.shape), dtype=str(t.dtype),
                       sha256=hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                  for t in operands + scales
              ], results=[])
    (args.output / 'source.py').write_bytes(Path(experiment.__file__).read_bytes())
    if args.compare_native:
        reference_path = Path(__file__).with_name('experimental-tcgen05-k96.py')
        (args.output / 'reference-source.py').write_bytes(reference_path.read_bytes())
        report['reference_source_sha256'] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    functions = []
    baseline = None
    for mode in args.modes + (["reference"] if args.compare_native else []):
        if mode in ('k64', 'mixed'):

            def fn(mode=mode, config=config):
                return experiment.base.mma_scaled_warp_specialized(*operands, *scales, vec,
                                                                   enable_fp4_k96=mode == 'mixed', **config)
        elif mode == "reference":
            reference = load_experiment()

            def fn():
                out, compiled = reference.matmul(*operands, *scales, vec)
                selected['kernel'] = compiled
                return out
        else:

            def fn(mode=mode):
                tuning = {} if args.tile_width is None else dict(tile_width=args.tile_width)
                if args.example == '07-pure-k96-matmul.py':
                    return experiment.matmul(*operands, *scales, buffers=args.buffers, epilogue=args.epilogue,
                                             scheduler=scheduler_id, out_dtype=out_dtype, **tuning)
                out, compiled = experiment.matmul(*operands, *scales, vec, buffers=args.buffers, epilogue=args.epilogue,
                                                  scheduler=scheduler_id)
                selected['kernel'] = compiled
                return out

        out = fn()
        compiled = selected['kernel']
        for ext in ('ptx', 'ttgir', 'cubin'):
            data = compiled.asm[ext]
            (args.output / f'{mode}.{ext}').write_bytes(data if isinstance(data, bytes) else data.encode())
        FrozenKernel.archive(compiled, args.output / mode)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=1e-3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = fn()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(replayed, out, atol=0, rtol=0)
        if baseline is None:
            baseline = out
        row = dict(mode=mode, correct=True, graph_replays=10, kernel_hash=compiled.hash,
                   compiler_metadata=compiled.metadata._asdict() | dict(target=asdict(compiled.metadata.target)),
                   shared=compiled.metadata.shared, tmem=compiled.metadata.tmem_size, registers=compiled.n_regs,
                   spills=compiled.n_spills, cubin_sha256=hashlib.sha256(compiled.asm['cubin']).hexdigest(),
                   ptx_mmas=compiled.asm['ptx'].count('tcgen05.mma.'),
                   baseline_max_abs=(out.float() - baseline.float()).abs().max().item(), ms=[])
        report['results'].append(row)
        functions.append(fn)
        print(json.dumps(row), flush=True)
    if args.frozen is not None:
        index = json.loads((args.frozen / 'index.json').read_text())
        for mode in args.modes:
            native = mode in ('k64', 'mixed')
            config = dict(format=args.format, mode='pure' if mode == 'native' else mode, block_k=256,
                          buffers=5 if native else args.buffers, epilogue=64 if native else args.epilogue,
                          scheduler=scheduler)
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
        # Keep the historical K64 and same-producer inline-mixed controls as
        # replay-only diagnostics; the supported kernel has no inline MMA path.
        for diagnostic in ('k64', 'raw'):
            label = 'frozen-' + diagnostic
            if any(row['mode'] == label for row in report['results']):
                continue
            config = dict(format=args.format, mode=diagnostic, block_k=256,
                          buffers=5 if diagnostic == 'k64' else args.buffers,
                          epilogue=64 if diagnostic == 'k64' else args.epilogue, scheduler=scheduler)
            matched = next(c for c in index['controls'] if control_key(c) == control_key(config))
            frozen = FrozenKernel(args.frozen / matched['cubin_sha256'])

            def fn(frozen=frozen, config=config):
                return frozen.invoke(experiment.base, operands, scales, vec, config)

            out = fn()
            torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=1e-3)
            report['results'].append(
                dict(mode=label, diagnostic=True, correct=True, cubin_sha256=matched['cubin_sha256'], ms=[]))
            functions.append(fn)
    if args.frozen_native is not None:
        frozen = FrozenKernel(args.frozen_native)

        def fn(frozen=frozen):
            return frozen.invoke(experiment.base, operands, scales, vec)

        out = fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, atol=1e-3, rtol=1e-3)
        report['results'].append(
            dict(mode='frozen-native', correct=True, launch_config=frozen.launch_config(),
                 shared=frozen.kernel.metadata.shared, tmem=frozen.kernel.metadata.tmem_size,
                 registers=frozen.kernel.n_regs, spills=frozen.kernel.n_spills,
                 cubin_sha256=frozen.manifest['cubin_sha256'], ms=[]))
        functions.append(fn)
    for fn in functions:
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    for repeat in range(args.repeats):
        order = range(len(functions)) if repeat % 2 == 0 else reversed(range(len(functions)))
        for i in order:
            calls = 0

            def timed():
                nonlocal calls
                calls += 1
                return functions[i]()

            report['results'][i]['ms'].append(triton.testing.do_bench_cudagraph(timed, rep=args.rep_ms))
            # do_bench_cudagraph makes six warmup/estimation calls, captures the
            # remaining calls, and replays that graph ten times.
            executions = 10 * (calls - 6)
            report['results'][i].setdefault('device_executions', []).append(executions)
            if args.repeats >= 7:
                assert executions >= 300, "increase --rep-ms to capture at least 300 device executions"
    for row in report['results']:
        row['median_ms'] = statistics.median(row['ms'])
        row['pflops'] = 2 * m * n * k / (row['median_ms'] * 1e12)
        row['cv_percent'] = 100 * statistics.pstdev(row['ms']) / statistics.mean(row['ms'])
        print(json.dumps(row), flush=True)
    if args.compare_native:
        reference_ms = next(row['median_ms'] for row in report['results'] if row['mode'] == 'reference')
        for row in report['results']:
            row['latency_ratio_vs_reference'] = row['median_ms'] / reference_ms
    if args.frozen is not None:
        rows = {row['mode']: row for row in report['results']}
        for mode in args.modes:
            row = rows[mode]
            row['latency_ratio_vs_frozen'] = row['median_ms'] / rows['frozen-' + mode]['median_ms']
            row['within_two_percent'] = row['latency_ratio_vs_frozen'] <= 1.02
    if args.frozen_native is not None:
        frozen_ms = next(row['median_ms'] for row in report['results'] if row['mode'] == 'frozen-native')
        for row in report['results']:
            if row['mode'] != 'frozen-native':
                row['latency_ratio_vs_frozen'] = row['median_ms'] / frozen_ms
                row['within_two_percent'] = row['latency_ratio_vs_frozen'] <= 1.02
    report['gpu_after'] = gpu_state()
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
