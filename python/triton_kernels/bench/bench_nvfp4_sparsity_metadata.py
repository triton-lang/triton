"""Time paired NVFP4 selector loads and sparse matmul on SM100/SM103."""

import argparse
import statistics

import torch
import triton
import triton.language as tl

from triton_kernels.matmul import PrecisionConfig, matmul


@triton.jit
def _read_selectors(E, Out, N: tl.constexpr, K: tl.constexpr, STRIDES: tl.constexpr, TILED: tl.constexpr,
                    TILES_PER_CTA: tl.constexpr):
    # One CTA reads consecutive 128x2 selector tiles and writes a checksum per row and word.
    n_tile, k_chunk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = tl.arange(0, 128)[:, None]
    word = tl.arange(0, 2)[None, :]
    ptr = E + batch.to(tl.int64) * STRIDES[0]
    total = tl.full((128, 2), 0, tl.uint32)
    for step in range(TILES_PER_CTA):
        tile = k_chunk * TILES_PER_CTA + step
        if TILED:
            offset = n_tile * STRIDES[1] + tile * STRIDES[2] + row * STRIDES[3] + word * STRIDES[4]
        else:
            offset = (128 * n_tile + row) * STRIDES[1] + (2 * tile + word) * STRIDES[2]
        total += tl.load(ptr + offset).to(tl.uint32)
    chunk = (batch * (N // 128) + n_tile) * (K // (128 * TILES_PER_CTA)) + k_chunk
    tl.store(Out + chunk * 256 + row * 2 + word, total)


def _measure(calls):
    # Capture separately, then rotate measurement order on each round.
    graphs = []
    for run in calls:
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(8):
                run()
        graphs.append(graph)
    times = [[] for _ in graphs]
    for repeat in range(24):
        for i in range(len(graphs)):
            idx = (i + repeat) % len(graphs)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[idx].replay()
            end.record()
            end.synchronize()
            if repeat >= 3:
                times[idx].append(start.elapsed_time(end) * 1000 / 8)
    return [round(statistics.median(t), 2) for t in times]


def _selectors(batch, n, k):
    # Eight valid selectors per word. Vary the word so address errors affect the checksum.
    codes = torch.tensor([4, 8, 9, 12, 13, 14], device="cuda") * 0x11111111
    return codes[torch.randint(6, (batch, n, k // 64), device="cuda")].int()


def bench_loads(batch, n, k, steps=16):
    row = _selectors(batch, n, k)
    tiled = row.view(batch, n // 128, 128, k // 128, 2).transpose(2, 3).contiguous()
    layouts = (row, row.mT.contiguous().mT, tiled)
    grid = (n // 128, k // (128 * steps), batch)
    ref = layouts[0].view(batch, n // 128, 128, grid[1], steps, 2).sum(4).transpose(2, 3).int()
    out = torch.empty_like(ref, memory_format=torch.contiguous_format)
    calls = []
    for i, data in enumerate(layouts):

        def run(data=data, tiled=i == 2):
            _read_selectors[grid](data, out, n, k, data.stride(), tiled, steps, num_warps=4)

        run()
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        calls.append(run)
    print("selector loads", batch, n, k, steps, _measure(calls), flush=True)


def bench_matmul(batch, m, n, k, scale_k):
    sparsity_metadata = _selectors(batch, n, k).mT.contiguous().mT

    def packed(shape):
        return torch.randint(256, shape, dtype=torch.uint8, device="cuda")

    def scales(shape):
        return torch.randint(1, 4, shape, device="cuda").to(torch.float8_e4m3fn)

    a, b = packed((batch, m, k // 2)), packed((batch, n, k // 4)).mT
    config = PrecisionConfig(a_mx_scale=scales((batch, m, k // scale_k)),
                             b_mx_scale=scales((batch, n, k // scale_k)),
                             a_microblock_size=scale_k, b_microblock_size=scale_k, out_dtype=torch.float32)
    out = torch.empty((batch, m, n), device="cuda")

    def run():
        matmul(a, b, None, c=out, precision_config=config, b_sparsity_metadata=sparsity_metadata)

    print("matmul (column)", batch, m, n, k, scale_k, _measure([run])[0], flush=True)



if __name__ == "__main__":
    torch.manual_seed(17)
    print(torch.cuda.get_device_name(), "median microseconds; selector loads [row, column, tile]", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="sweep CTA access depth and independent weight count")
    parser.add_argument("--aspect-sweep", action="store_true", help="vary N and K at fixed weight bytes")
    args = parser.parse_args()
    if args.aspect_sweep:
        for n, k in ((512, 65536), (2048, 16384), (8192, 4096), (32768, 1024)):
            for scale_k in (16, 32):
                bench_matmul(16, 16, n, k, scale_k)
    elif args.sweep:
        for batch in (1, 64):
            for steps in (1, 4, 16, 64):
                bench_loads(batch, 8192, 8192, steps)
        for batch in (1, 2, 4, 8, 16, 32, 64):
            for scale_k in (16, 32):
                bench_matmul(batch, 16, 4096, 8192, scale_k)
    else:
        for batch in (1, 64):
            bench_loads(batch, 8192, 8192)
        for batch, m in ((1, 16), (1, 512), (16, 16)):
            for scale_k in (16, 32):
                bench_matmul(batch, m, 4096, 8192, scale_k)
