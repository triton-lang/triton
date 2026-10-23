"""Validate and benchmark experimental Hopper MXFP4-to-FP8 matmul layouts."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import statistics
from pathlib import Path

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
import hopper_mxfp4_gluon as hg


def reference_weight(codes, scales):
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=codes.device, dtype=torch.float64)
    values = table[(codes & 7).long()]
    values = torch.where(codes & 8 != 0, -values, values)
    factors = torch.exp2(scales.to(torch.float64) - 127)
    factors = torch.where(scales == 255, float('nan'), factors)
    values = values * factors.repeat_interleave(32, dim=1)
    # Match saturating FP8 conversion, preserving NaNs and signed zeros.
    return values.clamp(-448, 448).float().to(torch.float8_e4m3fn)


def check_output(output, reference):
    error = (output.float() - reference).abs()
    relative_l2 = (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)).item()
    normalized_max = (error.max() / reference.abs().max()).item()
    assert relative_l2 < .005 and normalized_max < .005, (relative_l2, normalized_max)
    return {
        'max_abs_error': error.max().item(), 'relative_l2_error': relative_l2, 'normalized_max_error': normalized_max
    }


def timed_samples(run, cache, rounds=5):
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    samples = []
    if cache == 'warm':
        for _ in range(rounds):
            samples.append(triton.testing.do_bench_cudagraph(run, rep=30) * 1000)
    else:
        flush = torch.empty(64 * 1024 * 1024, device='cuda', dtype=torch.int32)
        graph = torch.cuda.CUDAGraph()
        events = [(torch.cuda.Event(enable_timing=True,
                                    external=True), torch.cuda.Event(enable_timing=True, external=True))
                  for _ in range(10)]
        with torch.cuda.graph(graph):
            for start, end in events:
                flush.zero_()
                start.record()
                run()
                end.record()
        for _ in range(rounds):
            graph.replay()
            torch.cuda.synchronize()
            samples.extend(start.elapsed_time(end) * 1000 for start, end in events)
    return samples


def measure(run, rounds):
    return {
        cache: {'median_us': statistics.median(samples), 'samples_us': samples}
        for cache in ('warm', 'cold')
        for samples in [timed_samples(run, cache, rounds)]
    }


def inputs(shape, seed):
    m, n, k = shape
    torch.manual_seed(seed)
    ca = torch.randint(0, 16, (m, k), device='cuda', dtype=torch.uint8)
    sa = torch.randint(122, 126, (m, k // 32), device='cuda', dtype=torch.uint8)
    cw = torch.randint(0, 16, (n, k), device='cuda', dtype=torch.uint8)
    sw = torch.randint(123, 132, (n, k // 32), device='cuda', dtype=torch.uint8)
    return ca, sa, cw, sw


FORMATS = list(itertools.product((8, 16, 32, 64), (32, 64, 128, 256)))


def _runner(a, sa, w, sw, shape, bm, split, lhs_format, *, rhs_format=hg.RhsFormat.SWIZZLED, lhs_tile=(16, 64),
            mapping=(16, 1)):
    config = hg.KernelConfig(bm, split, *mapping)
    return hg.make_runner(a, sa, w, sw, shape, config, lhs_format=lhs_format, rhs_format=rhs_format, lhs_tile=lhs_tile)


@g.jit
def unpack(Q, S, O, M: gl.constexpr, K: gl.constexpr, TM: gl.constexpr, TK: gl.constexpr, LR: gl.constexpr,
           WR: gl.constexpr):
    x = hg.load_lhs_words(Q, S, gl.program_id(0), gl.program_id(1), M, K, 32, TM, TK, LR, WR)
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    x = gl.convert_layout(x, L)
    m = gl.program_id(0) * 32 + gl.arange(0, 32, layout=gl.SliceLayout(1, L))
    k = gl.program_id(1) * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, L))
    gl.store(O + m[:, None] * K + k[None, :], x, (m[:, None] < M) & (k[None, :] < K))


@g.jit
def unpack_kernel(W, S, O, N: gl.constexpr, K: gl.constexpr, WN: gl.constexpr, WK: gl.constexpr, SN: gl.constexpr,
                  SK: gl.constexpr, SWIZZLED: gl.constexpr):
    value = hg.load_weight(W, S, gl.program_id(0), gl.program_id(1), N, K, WN, WK, SN, SK, SWIZZLED)
    OL: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    value = gl.convert_layout(value, OL)
    n = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, OL))
    k = gl.program_id(1) * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, OL))
    gl.store(O + n[:, None] * K + k[None, :], value, (n[:, None] < N) & (k[None, :] < K))


def _strided(tensor):
    storage = torch.empty((tensor.shape[0], tensor.shape[1] * 2), device=tensor.device, dtype=tensor.dtype)
    storage[:, ::2] = tensor
    return storage[:, ::2]


def validate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This experiment requires a Hopper GPU")
    torch.manual_seed(7)
    for m, k in ((19, 384), (65, 160)):
        codes = torch.arange(16, device='cuda', dtype=torch.uint8).repeat(m, k // 16)
        scales = (torch.arange(m * k // 32, device='cuda').reshape(m, k // 32) % 12 + 122).to(torch.uint8)
        expected = reference_weight(codes, scales)
        canonical, _ = hg.pack_rhs(codes, scales, False)
        for tile in FORMATS:
            q, s = hg.pack_lhs_words(codes, scales, tile)
            pq, ps, _ = hg.make_lhs_packer(canonical, scales, tile)
            torch.testing.assert_close(pq.view(torch.int32), q.view(torch.int32), rtol=0, atol=0)
            torch.testing.assert_close(ps, s, rtol=0, atol=0)
            for lr, wr in itertools.product((8, 16, 32), (1, 4)):
                out = torch.empty_like(expected)
                unpack[(triton.cdiv(m, 32), triton.cdiv(k, 128))](q, s, out, m, k, *tile, lr, wr)
                torch.testing.assert_close(out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
        n = 65
        cw = torch.arange(16, device='cuda', dtype=torch.uint8).repeat(n, k // 16)
        sw = (torch.arange(n * k // 32, device='cuda').reshape(n, k // 32) % 12 + 122).to(torch.uint8)
        w = reference_weight(cw, sw)
        for swizzled in (False, True):
            qw, qs = hg.pack_rhs(cw, sw, swizzled)
            if swizzled:
                raw, _ = hg.pack_rhs(cw, sw, False)
                packed, packed_s, _ = hg.make_rhs_packer(raw, sw, k)
                torch.testing.assert_close(packed.flatten().view(torch.int16),
                                           qw.flatten().view(torch.int16), rtol=0, atol=0)
                torch.testing.assert_close(packed_s.flatten(), qs.flatten(), rtol=0, atol=0)
            else:
                qw, qs = _strided(qw), _strided(qs)
            decoded = torch.empty_like(w)
            unpack_kernel[(triton.cdiv(n, 64), triton.cdiv(k, 128))](qw, qs, decoded, n, k,
                                                                     *((0, 0) if swizzled else qw.stride()),
                                                                     *((0, 0) if swizzled else qs.stride()), swizzled)
            torch.testing.assert_close(decoded.view(torch.uint8), w.view(torch.uint8), rtol=0, atol=0)
            for bm, split in itertools.product((16, 32, 64), (1, 4)):
                base, ref = _runner(expected, scales, w, sw, (m, n, k), bm, split, hg.LhsFormat.FP8,
                                    rhs_format=hg.RhsFormat.FP8)
                base()
                for mode in hg.LhsFormat:
                    if mode == hg.LhsFormat.FP8:
                        qa, qsa = expected, scales
                    elif mode in (hg.LhsFormat.WORD, hg.LhsFormat.WORD_CONVERT):
                        qa, qsa = hg.pack_lhs_words(codes, scales)
                    else:
                        qa, qsa = hg.pack_lhs_rows(codes, scales, (16, 64) if mode == hg.LhsFormat.BLOCKED else None)
                    if mode in (hg.LhsFormat.FP8, hg.LhsFormat.CANONICAL):
                        qa, qsa = _strided(qa), _strided(qsa)
                    run, got = _runner(qa, qsa, qw, qs, (m, n, k), bm, split, mode,
                                       rhs_format=hg.RhsFormat.SWIZZLED if swizzled else hg.RhsFormat.CANONICAL)
                    run()
                    torch.testing.assert_close(got, ref, rtol=0, atol=0)
        for invalid in (121, 134):
            try:
                hg.pack_rhs(codes, torch.full_like(scales, invalid), False)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsupported scale was accepted")
    print('PASS: exact decode, packing, tails, signed zero, scale contract, and matched FP8 matmul', flush=True)


SHAPES = [(m, n, k)
          for m in (1, 16, 32, 64, 256, 1024)
          for n, k in ((512, 512), (2048, 4096), (8192, 8192), (8192, 2048))]
SHAPES += [(7, 1024, 768), (17, 4096, 2048), (65, 2048, 4096), (257, 512, 512), (511, 4096, 1024), (2048, 512, 512),
           (4, 4096, 16384), (128, 16384, 1024)]
MAPS = list(itertools.product((8, 16, 32), (1, 4)))
ARMS = [('canonical', None, None)] + [(f'blocked_{m}x{k}', (m, k), None) for m, k in ((8, 64), (8, 128), (16, 128))]
ARMS += [(f'word_{m}x{k}', None, (m, k)) for m, k in FORMATS]


def summarize(root):
    cases = [json.loads(p.read_text()) for p in sorted(root.glob('case_*.json'))]
    if not cases:
        return
    metrics = ('warm', 'cold')
    rows = []
    ranks = {name: {cache: [] for cache in metrics} for name, _, _ in ARMS}
    cases.sort(key=lambda c: tuple(c['shape']))
    for c in cases:
        for cache in metrics:
            winner = min(c['arms'], key=lambda n: c['arms'][n][cache]['median_us'])
            floor = c['arms'][winner][cache]['median_us']
            for name, arm in c['arms'].items():
                us = arm[cache]['median_us']
                regret = us / floor - 1
                ranks[name][cache].append(regret)
                rows.append({
                    'm': c['shape'][0], 'n': c['shape'][1], 'k': c['shape'][2], 'format': name, 'cache': cache,
                    'latency_us': us, 'winner': winner, 'best_us': floor, 'regret_pct': regret * 100, 'compute':
                    arm['selected']['compute'], 'mapping': arm['selected']['mapping'], 'storage_bytes':
                    arm['storage_bytes']
                })
    with (root / 'regret_per_size.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for name, by_cache in ranks.items():
        values = [v for vs in by_cache.values() for v in vs]
        summary.append({
            'format': name, 'cases': len(cases), 'worst_regret_pct': max(values) * 100, 'geomean_slowdown_pct':
            (statistics.geometric_mean([1 + v for v in values]) - 1) * 100,
            **{f'{cache}_worst_regret_pct': max(vs) * 100
               for cache, vs in by_cache.items()}, **{
                f'{cache}_geomean_slowdown_pct': (statistics.geometric_mean([1 + v for v in vs]) - 1) * 100
                for cache, vs in by_cache.items()
            }
        })
    summary.sort(key=lambda r: (r['cold_worst_regret_pct'], r['cold_geomean_slowdown_pct']))
    (root / 'ranking.json').write_text(json.dumps(summary, indent=2))
    (root / 'results.json').write_text(json.dumps(cases, indent=2))


def case(shape, root):
    m, n, k = shape
    ca, sa, cw, sw = inputs(shape, 123)
    a, w = reference_weight(ca, sa), reference_weight(cw, sw)
    qw, qs = hg.pack_rhs(cw, sw, True)
    reference = a.float() @ w.float().T
    compute = [(bm, sp) for bm in (16, 32, 64) if bm <= max(16, triton.next_power_of_2(m)) for sp in (1, 4, 16, 64)
               if sp * 128 <= k]
    cached = {}
    packed = {
        name: hg.pack_lhs_words(ca, sa, word) if word else hg.pack_lhs_rows(ca, sa, tile)
        for name, tile, word in ARMS
    }
    arm_info = {name: (tile, word) for name, tile, word in ARMS}

    def make(name, cfg, mapping):
        tile, word = arm_info[name]
        qa, qsa = packed[name]
        return _runner(qa, qsa, qw, qs, shape, *cfg,
                       hg.LhsFormat.WORD if word else hg.LhsFormat.BLOCKED if tile else hg.LhsFormat.CANONICAL,
                       lhs_tile=word or tile or (16, 64), mapping=mapping)

    def trial(name, cfg, mapping):
        run, out = make(name, cfg, mapping)
        compiled = run()
        if cfg not in cached:
            base, expected = _runner(a, sa, w, sw, shape, *cfg, hg.LhsFormat.FP8, rhs_format=hg.RhsFormat.FP8)
            base()
            cached[cfg] = expected
        torch.testing.assert_close(out, cached[cfg], rtol=0, atol=0)
        error = check_output(out, reference)
        return {
            'compute': cfg, 'mapping': mapping, 'us': triton.testing.do_bench_cudagraph(run, rep=12) * 1000,
            'registers': compiled.n_regs, 'shared': compiled.metadata.shared, 'spills': compiled.n_spills, **error
        }

    anchor_trials = [trial('word_32x32', cfg, (32, 1)) for cfg in compute]
    anchor = tuple(min(anchor_trials, key=lambda r: r['us'])['compute'])
    result = {
        'shape': shape, 'seed': 123, 'anchor_trials': anchor_trials, 'arms': {}, 'input_sha256': {
            name: hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()
            for name, t in [('ca', ca), ('sa', sa), ('cw', cw), ('sw', sw)]
        }
    }
    for name, tile, word in ARMS:
        if word:
            trials = [trial(name, anchor, mp) for mp in MAPS]
            mapping = tuple(min(trials, key=lambda r: r['us'])['mapping'])
            trials += [trial(name, cfg, mapping) for cfg in compute if cfg != anchor]
            best = min(trials, key=lambda r: r['us'])
            cfg = tuple(best['compute'])
            if cfg != anchor:
                trials += [trial(name, cfg, mp) for mp in MAPS if mp != mapping]
        else:
            trials = [trial(name, cfg, (32, 1)) for cfg in compute]
        best = min(trials, key=lambda r: r['us'])
        run, out = make(name, tuple(best['compute']), tuple(best['mapping']))
        run()
        qa, qsa = packed[name]
        result['arms'][name] = {
            'selected': best, 'trials': trials, 'storage_bytes':
            qa.numel() * qa.element_size() + qsa.numel() * qsa.element_size(), **measure(run, 5)
        }
    result['baselines'] = {}
    for name, dense_rhs in (('fp8_lhs', False), ('fp8_both', True)):
        weight, scale = (w, sw) if dense_rhs else (qw, qs)
        trials = []
        for cfg in compute:
            run, out = _runner(a, sa, weight, scale, shape, *cfg, hg.LhsFormat.FP8,
                               rhs_format=hg.RhsFormat.FP8 if dense_rhs else hg.RhsFormat.SWIZZLED)
            run()
            torch.testing.assert_close(out, cached[cfg], rtol=0, atol=0)
            trials.append({'compute': cfg, 'us': triton.testing.do_bench_cudagraph(run, rep=12) * 1000})
        selected = min(trials, key=lambda trial: trial['us'])
        run, out = _runner(a, sa, weight, scale, shape, *selected['compute'], hg.LhsFormat.FP8,
                           rhs_format=hg.RhsFormat.FP8 if dense_rhs else hg.RhsFormat.SWIZZLED)
        result['baselines'][name] = {'selected': selected, 'trials': trials, **measure(run, 5)}
    path = root / ('case_' + '_'.join(map(str, shape)) + '.json')
    path.write_text(json.dumps(result, indent=2))
    summarize(root)
    rank = json.loads((root / 'ranking.json').read_text())
    print(json.dumps({'complete': shape, 'cases': rank[0]['cases'], 'leading_fixed_formats': rank[:3]}), flush=True)
    return result


def confirm(root, names):
    cases = json.loads((root / 'results.json').read_text())
    records = []
    for c in cases:
        shape = tuple(c['shape'])
        m, n, k = shape
        ca, sa, cw, sw = inputs(shape, 391)
        qw, qs = hg.pack_rhs(cw, sw, True)
        a, w = reference_weight(ca, sa), reference_weight(cw, sw)
        reference = a.float() @ w.float().T
        cached = {}
        # Always retain the previously selected per-size winner as a fixed comparator.
        winners = {min(c['arms'], key=lambda name: c['arms'][name][cache]['median_us']) for cache in ('warm', 'cold')}
        chosen = list(dict.fromkeys(['canonical', *names, *sorted(winners)]))
        record = {'shape': shape, 'seed': 391, 'arms': {}}
        for name in chosen:
            _, tile, word = next(t for t in ARMS if t[0] == name)
            best = c['arms'][name]['selected']
            cfg = tuple(best['compute'])
            mp = tuple(best['mapping'])
            qa, qsa = hg.pack_lhs_words(ca, sa, word) if word else hg.pack_lhs_rows(ca, sa, tile)
            run, out = _runner(qa, qsa, qw, qs, shape, *cfg,
                               hg.LhsFormat.WORD if word else hg.LhsFormat.BLOCKED if tile else hg.LhsFormat.CANONICAL,
                               lhs_tile=word or tile or (16, 64), mapping=mp)
            compiled = run()
            if cfg not in cached:
                base, expected = _runner(a, sa, w, sw, shape, *cfg, hg.LhsFormat.FP8, rhs_format=hg.RhsFormat.FP8)
                base()
                cached[cfg] = expected
            torch.testing.assert_close(out, cached[cfg], rtol=0, atol=0)
            row = {'selected': best, **measure(run, 5), **check_output(out, reference)}
            if word:
                canonical, _ = hg.pack_rhs(ca, sa, False)
                pq, ps, prepare = hg.make_lhs_packer(canonical, sa, word)
                torch.testing.assert_close(pq.view(torch.int32), qa.view(torch.int32), rtol=0, atol=0)
                torch.testing.assert_close(ps, qsa, rtol=0, atol=0)
                matmul, got = _runner(pq, ps, qw, qs, shape, *cfg, hg.LhsFormat.WORD, lhs_tile=word, mapping=mp)

                def pipeline():
                    prepare()
                    return matmul()

                pipeline()
                torch.testing.assert_close(got, cached[cfg], rtol=0, atol=0)
                row['prepack_warm_samples_us'] = timed_samples(prepare, 'warm', 5)
                row['pipeline'] = measure(pipeline, 5)
            record['arms'][name] = row
            if name in names:
                stem = '_'.join(map(str, shape)) + '_' + name
                for ext in ('ptx', 'ttgir'):
                    (root / f'{stem}.{ext}').write_text(compiled.asm[ext])
        records.append(record)
        (root / 'confirmation.json').write_text(json.dumps(records, indent=2))
        print(json.dumps({'confirmed': shape, 'count': len(records)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--confirm', nargs='*')
    p.add_argument('--shapes', help='Semicolon-separated M,N,K triples; defaults to the synthetic sweep')
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    validate()
    if args.validate_only:
        return
    shapes = [tuple(map(int, shape.split(','))) for shape in args.shapes.split(';')] if args.shapes else SHAPES
    hashes = {}
    for source in (Path(__file__), Path(hg.__file__)):
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        'shapes': shapes, 'torch': torch.__version__, 'triton': triton.__version__, 'cuda': torch.version.cuda, 'gpu':
        torch.cuda.get_device_name(), 'source_sha256': hashes, 'visible_devices':
        os.environ.get('CUDA_VISIBLE_DEVICES'), 'selection': 'worst cold-cache regret; warm-selected configurations'
    }
    metadata_path = root / 'metadata.json'
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous['source_sha256'] != hashes:
            raise ValueError('Output directory was produced by different sources; choose a new directory')
        if args.confirm is None and previous['shapes'] != [list(shape) for shape in shapes]:
            raise ValueError('Output directory contains a different shape set')
    if args.confirm is not None:
        if not metadata_path.exists():
            raise ValueError('Confirmation requires an existing sweep')
        unknown = set(args.confirm) - {name for name, _, _ in ARMS}
        if unknown:
            raise ValueError(f'Unknown formats: {sorted(unknown)}')
        confirm(root, args.confirm)
        return
    metadata_path.write_text(json.dumps(metadata, indent=2))
    for source in (Path(__file__), Path(hg.__file__)):
        (root / source.name).write_bytes(source.read_bytes())
    for shape in shapes:
        path = root / ('case_' + '_'.join(map(str, shape)) + '.json')
        if not path.exists():
            case(shape, root)
    summarize(root)


if __name__ == '__main__':
    main()
