import torch
import triton
import pytest


@pytest.mark.parametrize(
    "MODE, TRANS_A, TRANS_B, BLOCK, DTYPE",
    [
        (mode, at, bt, block, dtype) for dtype in ["float16"] for mode in ["sdd"] for at in [False] for bt in [False]
        for block in [16]
    ],
)
def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, Z=3, H=2, M=128, N=128, K=128):
    DTYPE = {"float16": torch.float16, "float32": torch.float32}[DTYPE]
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


# @pytest.mark.parametrize(
#     "BLOCK, WIDTH",
#     [(block, width) for block in [32] for width in [256]],
# )
# def test_softmax(BLOCK, WIDTH, DTYPE=torch.float16):
#     # set seed
#     torch.random.manual_seed(0)
#     Z, H, M, N = 2, 4, WIDTH, WIDTH
#     scale = 0.4
#     # create inputs
#     layout = torch.randint(2, (H, M // BLOCK, N // BLOCK))
#     x = torch.randn((Z, H, M, N), dtype=DTYPE, requires_grad=True, device="cuda")
#     at_mask = torch.randint(low=0, high=2, size=(N, N), dtype=torch.bool, requires_grad=False, device="cuda")
#     kp_mask = torch.randint(low=0, high=2, size=(Z, N), dtype=DTYPE, requires_grad=False, device="cuda")
#     kp_mask[kp_mask == 1.0] = float("-inf")
#     # triton result
#     op = triton.ops.blocksparse.softmax(layout, BLOCK)
#     tx = triton.testing.sparsify_tensor(x, layout, BLOCK)
#     ty = op(
#         tx,
#         scale=scale,
#         key_padding_mask=kp_mask,
#         key_padding_mask_mode="add",
#         attn_mask=at_mask.to(DTYPE),
#         attn_mask_mode="mul",
#     )
#     # torch result
#     rx = triton.testing.mask_tensor(x, layout, BLOCK, value=float("-inf"))
#     if at_mask is not None:
#         # broadcast at_mask to the same shape as rx
#         M = at_mask[None, None, :, :] + torch.zeros_like(rx)
#         rx[M == 0] = float("-inf")
#     if kp_mask is not None:
#         rx += kp_mask[:, None, None, :]
#     ry = torch.softmax(rx * scale, -1)
#     ry = torch.softmax(rx * scale, -1)
#     ry = triton.testing.sparsify_tensor(ry, layout, BLOCK)
#     # compare
#     assert triton.testing.allclose(ry, ty)

# def test_attention_fwd_bwd(
#     input_scale=1.0,
#     tol=2e-2,
#     scale=1 / 8.0,
#     n_ctx=256,
#     dtype=torch.float16,
#     batch_size=2,
#     n_heads=2,
#     block=64,
# ):
#     # inputs
#     qkv_shape = (batch_size, n_heads, n_ctx, 64)
#     qkvs = [torch.nn.Parameter(input_scale * torch.randn(qkv_shape), requires_grad=True).to(dtype).cuda() for _ in range(3)]
#     attn_mask = torch.tril(
#         torch.ones(
#             [n_ctx, n_ctx],
#             device="cuda",
#             dtype=dtype,
#         ),
#         diagonal=0,
#     )

#     # Triton:
#     n_blocks = n_ctx // block
#     layout = torch.tril(torch.ones([n_heads, n_blocks, n_blocks], dtype=torch.long))
#     query, key, value = [x.clone() for x in qkvs]
#     query.retain_grad()
#     key.retain_grad()
#     value.retain_grad()
#     attn_out = triton_attention(layout, block, attn_mask, query=query, key=key, value=value, scale=scale)
#     # ad hoc loss
#     loss = (attn_out**2).mean()
#     loss.backward()
#     grads = [query.grad, key.grad, value.grad]

#     # Torch version:
#     torch_q, torch_k, torch_v = [x.clone() for x in qkvs]
#     attn_mask = 1e6 * (-1 + (attn_mask.reshape((1, 1, n_ctx, n_ctx)).cuda()))
#     torch_q.retain_grad()
#     torch_k.retain_grad()
#     torch_v.retain_grad()
#     scores = scale * torch.einsum("bhsd,bhtd->bhst", torch_q, torch_k)
#     scores = scores + attn_mask
#     probs = torch.softmax(scores, dim=-1)
#     torch_attn_out = torch.einsum("bhst,bhtd->bhsd", probs, torch_v)
#     # ad hoc loss
#     torch_loss = (torch_attn_out**2).mean()
#     torch_loss.backward()
#     torch_grads = [torch_q.grad, torch_k.grad, torch_v.grad]

#     # comparison
#     # print(f"Triton loss {loss} and torch loss {torch_loss}.  Also checking grads...")
#     torch.testing.assert_allclose(loss, torch_loss, rtol=tol, atol=tol)
#     for g1, g2 in zip(grads, torch_grads):
#         torch.testing.assert_allclose(g1, g2, rtol=tol, atol=tol)

# def triton_attention(
#     layout,
#     block: int,
#     attn_mask: torch.Tensor,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     scale: float,
# ):
#     sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True)
#     sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(layout, block, "dsd", trans_a=False, trans_b=False)
#     sparse_softmax = triton.ops.blocksparse.softmax(
#         layout,
#         block,
#     )

#     w = sparse_dot_sdd_nt(query, key)
#     w = sparse_softmax(w, scale=scale, attn_mask=attn_mask, attn_mask_mode="mul")
#     a = sparse_dot_dsd_nn(w, value)
#     return a
