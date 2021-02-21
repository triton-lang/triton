import torch
import triton
import pytest

@pytest.mark.parametrize(
    "MODE, TRANS_A, TRANS_B, BLOCK",
    [(mode, at, bt, block) for mode in ["sdd", "dsd", "dds"] for at in [False, True] for bt in [False, True]
     for block in [16, 32, 64]],
)
def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE=torch.float16, Z=3, H=2, M=128, N=256, K=384):
    # set seed
    torch.random.manual_seed(0)
    # create inputs
    a = torch.randn((Z, H, K, M) if TRANS_A else (Z, H, M, K), dtype=DTYPE, device="cuda")
    b = torch.randn((Z, H, N, K) if TRANS_B else (Z, H, K, N), dtype=DTYPE, device="cuda")
    shape = {
        "sdd": (M, N),
        "dsd": (a.shape[2], a.shape[3]),
        "dds": (b.shape[2], b.shape[3]),
    }[MODE]
    layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
    # triton result
    op = triton.ops.blocksparse.matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B)
    ra = triton.testing.sparsify_tensor(a, layout, BLOCK) if MODE == "dsd" else a
    rb = triton.testing.sparsify_tensor(b, layout, BLOCK) if MODE == "dds" else b
    rc = op(ra, rb)
    # torch result
    ta = triton.testing.mask_tensor(a, layout, BLOCK) if MODE == "dsd" else a
    tb = triton.testing.mask_tensor(b, layout, BLOCK) if MODE == "dds" else b
    ta = ta.transpose(2, 3) if TRANS_A else ta
    tb = tb.transpose(2, 3) if TRANS_B else tb
    tc = torch.matmul(ta, tb)
    tc = triton.testing.mask_tensor(tc, layout, BLOCK) if MODE == "sdd" else tc
    tc = triton.testing.sparsify_tensor(tc, layout, BLOCK) if MODE == "sdd" else tc
    # compare
    assert triton.testing.allclose(rc, tc)

@pytest.mark.parametrize(
    "BLOCK, WIDTH",
    [(block, width) for block in [32] for width in [256, 576, 1024, 1792]],
)
def test_softmax(BLOCK, WIDTH, DTYPE=torch.float16):
    # set seed
    torch.random.manual_seed(0)
    Z, H, M, N = 2, 4, WIDTH, WIDTH
    scale = 0.4
    # create inputs
    layout = torch.randint(2, (H, M // BLOCK, N // BLOCK))
    x = torch.randn((Z, H, M, N), dtype=DTYPE, requires_grad=True, device="cuda")
    at_mask = torch.randint(low=0, high=2, size=(N, N), dtype=torch.bool, requires_grad=False, device="cuda")
    kp_mask = torch.randint(low=0, high=2, size=(Z, N), dtype=DTYPE, requires_grad=False, device="cuda")
    kp_mask[kp_mask == 1.0] = float("-inf")
    # triton result
    op = triton.ops.blocksparse.softmax(layout, BLOCK)
    tx = triton.testing.sparsify_tensor(x, layout, BLOCK)
    ty = op(
        tx,
        scale=scale,
        key_padding_mask=kp_mask,
        key_padding_mask_mode="add",
        attn_mask=at_mask.to(DTYPE),
        attn_mask_mode="mul",
    )
    # torch result
    rx = triton.testing.mask_tensor(x, layout, BLOCK, value=float("-inf"))
    if at_mask is not None:
        # broadcast at_mask to the same shape as rx
        M = at_mask[None, None, :, :] + torch.zeros_like(rx)
        rx[M == 0] = float("-inf")
    if kp_mask is not None:
        rx += kp_mask[:, None, None, :]
    ry = torch.softmax(rx * scale, -1)
    ry = torch.softmax(rx * scale, -1)
    ry = triton.testing.sparsify_tensor(ry, layout, BLOCK)
    # compare
    assert triton.testing.allclose(ry, ty)
