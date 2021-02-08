import itertools
import torch
import triton as tt
import pytest

def sparsify_tensor(x, mask, block):
    ret = torch.empty((x.size(0), mask.sum(), block, block), dtype=x.dtype, device=x.device)
    for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
        ret[:, idx, :, :] = x[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block]
    return ret

def mask_tensor(x, mask, block, value=0):
    ret = x.clone()
    for h, i, j in zip(*(mask == 0).nonzero(as_tuple=True)):
        ret[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block] = value
    return ret

## -----------------------------------------------------------------------------
## Unit Tests
## -----------------------------------------------------------------------------

@pytest.mark.parametrize("MODE, TRANS_A, TRANS_B, BLOCK",
    [
    (mode, at, bt, block) for mode in ['sdd', 'dsd', 'dds']\
                          for at   in [False, True]\
                          for bt   in [False, True]\
                          for block in [16, 32, 64]
    ]
                         )
def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE=torch.float16, Z=3, H=2, M=128, N=256, K=384):
    # set seed
    torch.random.manual_seed(0)
    # create inputs
    a = torch.randn((Z, H, K, M) if TRANS_A else (Z, H, M, K), dtype=DTYPE, device='cuda')
    b = torch.randn((Z, H, N, K) if TRANS_B else (Z, H, K, N), dtype=DTYPE, device='cuda')
    shape = {'sdd': (M, N), 'dsd': (a.shape[2], a.shape[3]), 'dds': (b.shape[2], b.shape[3])}[MODE]
    layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
    # triton result
    op = tt.ops.blocksparse.matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B)
    ra = sparsify_tensor(a, layout, BLOCK) if MODE == 'dsd' else a
    rb = sparsify_tensor(b, layout, BLOCK) if MODE == 'dds' else b
    rc = op(ra, rb)
    # torch result
    ta = mask_tensor(a, layout, BLOCK) if MODE == 'dsd' else a
    tb = mask_tensor(b, layout, BLOCK) if MODE == 'dds' else b
    ta = ta.transpose(2, 3) if TRANS_A else ta
    tb = tb.transpose(2, 3) if TRANS_B else tb
    tc = torch.matmul(ta, tb)
    tc = mask_tensor(tc, layout, BLOCK) if MODE == 'sdd' else tc
    tc = sparsify_tensor(tc, layout, BLOCK) if MODE == 'sdd' else tc
    # compare
    rtol, atol = {torch.float32: (1e-4, 1e-5), torch.float16: (1e-2, 1e-3)}[DTYPE]
    assert torch.allclose(rc, tc, rtol=rtol, atol=atol)


@pytest.mark.parametrize("BLOCK, WIDTH",
                         [
    (block, width) for block in [32]\
                   for width in [256, 576, 1024, 2048, 4096]
                         ]
                         )
def test_softmax(BLOCK, WIDTH, DTYPE=torch.float16):
    # set seed
    torch.random.manual_seed(0)
    Z, H, M, N = 2, 4, WIDTH, WIDTH
    scale = 0.4
    # create inputs
    layout = torch.randint(2, (H, M // BLOCK, N // BLOCK))
    x = torch.randn((Z, H, M, N), dtype=DTYPE, requires_grad=True, device='cuda')
    at_mask = torch.randint(low=0, high=2, size=(N, N), \
                            dtype=torch.bool, requires_grad=False, device='cuda')
    kp_mask = torch.randint(low=0, high=2, size=(Z, N), \
                            dtype=DTYPE, requires_grad=False, device='cuda')
    kp_mask[kp_mask == 1.] = float('-inf')
    # triton result
    op = tt.ops.blocksparse.softmax(layout, BLOCK)
    tx = sparsify_tensor(x, layout, BLOCK)
    ty = op(tx, scale=scale)
    # torch result
    rx = mask_tensor(x, layout, BLOCK, value=float('-inf'))
    # if at_mask is not None:
    #   # broadcast at_mask to the same shape as rx
    #   M = at_mask[None, None, :, :] + torch.zeros_like(rx)
    #   rx[M == 0] = float('-inf')
    # if kp_mask is not None:
    #   rx += kp_mask[:, None, None, :]
    ry = torch.softmax(rx * scale, -1)
    ry = sparsify_tensor(ry, layout, BLOCK)
    # compare
    rtol, atol = {torch.float32: (1e-4, 1e-5), torch.float16: (1e-2, 1e-3)}[DTYPE]
    assert torch.allclose(ry, ty, rtol=rtol, atol=atol)

## -----------------------------------------------------------------------------
## Performance Tests
## -----------------------------------------------------------------------------

def do_bench(fn, warmup=10, rep=50):
    import torch as th
    start_event = th.cuda.Event(enable_timing=True)
    end_event = th.cuda.Event(enable_timing=True)
    ret = fn()
    for i in range(warmup):
        fn()
    th.cuda.synchronize()
    start_event.record()
    for i in range(rep):
        fn()
    end_event.record()
    th.cuda.synchronize()
    time_ms = start_event.elapsed_time(end_event) / rep
    return time_ms

def perf_matmul(BLOCK=64,
                LAYOUT_MODE='tril',
                OP_MODE='sdd',
                TRANS_A=False,
                TRANS_B=False,
                DTYPE=torch.float16,
                warmup=10,
                rep=50):
    Z, H = 1, 1
    K = 512
    make_layout = {
        'tril': lambda H, M, N: torch.tril(torch.ones((H, M, N), dtype=torch.int64)),
        'dense': lambda H, M, N: torch.ones(H, M, N, dtype=torch.int64),
    }[LAYOUT_MODE]
    for N in [128, 256, 512, 1024, 2048, 4096]:
        # create layout
        M, N, K = N, N, N
        shape = {'sdd': (M, N), 'dsd': (K, M) if TRANS_A else (M, K), 'dds': (N, K) if TRANS_B else (K, N)}[OP_MODE]
        layout = make_layout(H, shape[0] // BLOCK, shape[1] // BLOCK)
        # create op
        op = tt.ops.blocksparse.matmul(layout, BLOCK, OP_MODE, trans_a=TRANS_A, trans_b=TRANS_B)
        # inputs
        a = torch.randn((Z, H, K, M) if TRANS_A else (Z, H, M, K), dtype=DTYPE, device='cuda')
        b = torch.randn((Z, H, N, K) if TRANS_B else (Z, H, K, N), dtype=DTYPE, device='cuda')
        a = sparsify_tensor(a, layout, BLOCK) if OP_MODE == 'dsd' else a
        b = sparsify_tensor(b, layout, BLOCK) if OP_MODE == 'dds' else b
        ms = do_bench(lambda: op(a, b), warmup=warmup, rep=rep)
        num_flops = {
            'sdd': 2 * Z * K * float(layout.sum()) * BLOCK * BLOCK * 1e-12, 'dsd':
            2 * Z * N * float(layout.sum()) * BLOCK * BLOCK * 1e-12, 'dds':
            2 * Z * M * float(layout.sum()) * BLOCK * BLOCK * 1e-12
        }[OP_MODE]
        triton_tflops = num_flops / ms * 1e3

def perf_softmax(BLOCK=64, LAYOUT_MODE='tril', DTYPE=torch.float16, warmup=10, rep=50):
    Z, H = 1, 1
    K = 512
    make_layout = {
        'tril': lambda H, M, N: torch.tril(torch.ones((H, M, N), dtype=torch.int64)),
        'dense': lambda H, M, N: torch.ones(H, M, N, dtype=torch.int64),
    }[LAYOUT_MODE]
    for N in [128, 256, 512, 1024, 2048, 4096]:
        layout = make_layout(H, N // BLOCK, N // BLOCK)
        a = torch.randn((Z, H, N, N), dtype=DTYPE, device='cuda')
        a = sparsify_tensor(a, layout, BLOCK)
        op = tt.ops.blocksparse.softmax(layout, BLOCK)
        ms = do_bench(lambda: op(a), warmup=warmup, rep=rep)
        nbytes = 2 * a.numel() * a.element_size()
        triton_gbyps = (nbytes * 1e-9) / (ms * 1e-3)
        print(triton_gbyps)
