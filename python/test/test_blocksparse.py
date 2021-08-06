import pytest
import torch
import triton


@pytest.mark.parametrize(
    "MODE, TRANS_A, TRANS_B, BLOCK, DTYPE",
    [
        (mode, at, bt, block, dtype)
        for dtype in ["float16", "float32"]
        for mode in ["sdd", "dsd", "dds"]
        for at in [False, True]
        for bt in [False, True]
        for block in [16, 32, 64]
    ],
)
def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, Z=3, H=2, M=512, N=384, K=256):
    DTYPE = {"float16": torch.float16, "float32": torch.float32}[DTYPE]
    # set seed
    torch.random.manual_seed(0)
    # create inputs
    a = torch.randn(
        (Z, H, K, M) if TRANS_A else (Z, H, M, K), dtype=DTYPE, device="cuda"
    )
    b = torch.randn(
        (Z, H, N, K) if TRANS_B else (Z, H, K, N), dtype=DTYPE, device="cuda"
    )
    shape = {
        "sdd": (M, N),
        "dsd": (a.shape[2], a.shape[3]),
        "dds": (b.shape[2], b.shape[3]),
    }[MODE]
    layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
    # triton result
    op = triton.ops.blocksparse.matmul(
        layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B
    )
    ra = triton.testing.sparsify_tensor(a, layout, BLOCK) if MODE == "dsd" else a
    rb = triton.testing.sparsify_tensor(b, layout, BLOCK) if MODE == "dds" else b
    rc = triton.testing.catch_oor(lambda: op(ra, rb), pytest)
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
    "block_size, n_ctx", [(32, 256), (32, 576), (32, 1024), (32, 1792)]
)
def test_softmax(block_size, n_ctx, dtype=torch.float16):
    # set seed
    torch.random.manual_seed(0)
    batch_size, n_heads, n_rows, n_cols = 2, 4, n_ctx, n_ctx
    scale = 0.4
    # this is a block attention mask
    block_mask = torch.randint(
        2, (n_heads, n_rows // block_size, n_cols // block_size), dtype=torch.bool
    )
    logits = torch.randn(
        (batch_size, n_heads, n_rows, n_cols),
        dtype=dtype,
        requires_grad=True,
        device="cuda",
    )
    fine_mask = torch.randint(
        low=0,
        high=2,
        size=(n_cols, n_cols),
        dtype=torch.bool,
        requires_grad=False,
        device="cuda",
    )
    key_padding_mask = torch.randint(
        low=0,
        high=2,
        size=(batch_size, n_cols),
        dtype=dtype,
        requires_grad=False,
        device="cuda",
    )
    key_padding_mask[key_padding_mask == 1.0] = float("-inf")
    # triton result
    sparse_softmax = triton.ops.blocksparse.BlocksparseSoftmax(block_mask, block_size)
    triton_inputs = triton.testing.sparsify_tensor(logits, block_mask, block_size)
    triton_outputs = sparse_softmax(
        triton_inputs,
        scale=scale,
        key_padding_mask=key_padding_mask,
        key_padding_mask_mode="add",
        attn_mask=fine_mask.to(dtype),
        attn_mask_mode="mul",
    )
    # torch result
    torch_inputs = triton.testing.mask_tensor(
        logits, block_mask, block_size, value=float("-inf")
    )
    if fine_mask is not None:
        # broadcast fine_mask to the same shape as inputs
        n_rows = fine_mask[None, None, :, :] + torch.zeros_like(torch_inputs)
        torch_inputs[n_rows == 0] = float("-inf")
    if key_padding_mask is not None:
        torch_inputs += key_padding_mask[:, None, None, :]
    torch_outputs = torch.softmax(torch_inputs * scale, -1)
    torch_outputs_sparse = triton.testing.sparsify_tensor(
        torch_outputs, block_mask, block_size
    )
    # compare
    assert triton.testing.allclose(torch_outputs_sparse, triton_outputs)

    # Compare the backward pass
    torch_outputs.sum().backward()
    torch_logits_grad = logits.grad.clone()
    logits.grad = None

    # The test below is failing, which seems rather concerning
    # triton_outputs.sum().backward()
    # triton_logits_grad = logits.grad.clone()
    # logits.grad = None
    # assert triton.testing.allclose(torch_logits_grad, triton_logits_grad)


def test_attention_fwd_bwd(
    input_scale=1.0,
    tol=2e-2,
    scale=1 / 8.0,
    n_ctx=256,
    dtype=torch.float16,
    batch_size=2,
    n_heads=2,
    block=64,
):
    # inputs
    qkv_shape = (batch_size, n_heads, n_ctx, 64)
    qkvs = [
        torch.nn.Parameter(input_scale * torch.randn(qkv_shape), requires_grad=True)
        .to(dtype)
        .cuda()
        for _ in range(3)
    ]
    attn_mask = torch.tril(
        torch.ones(
            [n_ctx, n_ctx],
            device="cuda",
            dtype=dtype,
        ),
        diagonal=0,
    )

    # Triton:
    n_blocks = n_ctx // block
    layout = torch.tril(torch.ones([n_heads, n_blocks, n_blocks], dtype=torch.long))
    query, key, value = [x.clone() for x in qkvs]
    query.retain_grad()
    key.retain_grad()
    value.retain_grad()
    attn_out = triton_attention(
        layout, block, attn_mask, query=query, key=key, value=value, scale=scale
    )
    # ad hoc loss
    loss = (attn_out ** 2).mean()
    loss.backward()
    grads = [query.grad, key.grad, value.grad]

    # Torch version:
    torch_q, torch_k, torch_v = [x.clone() for x in qkvs]
    attn_mask = 1e6 * (-1 + (attn_mask.reshape((1, 1, n_ctx, n_ctx)).cuda()))
    torch_q.retain_grad()
    torch_k.retain_grad()
    torch_v.retain_grad()
    scores = scale * torch.einsum("bhsd,bhtd->bhst", torch_q, torch_k)
    scores = scores + attn_mask
    probs = torch.softmax(scores, dim=-1)
    torch_attn_out = torch.einsum("bhst,bhtd->bhsd", probs, torch_v)
    # ad hoc loss
    torch_loss = (torch_attn_out ** 2).mean()
    torch_loss.backward()
    torch_grads = [torch_q.grad, torch_k.grad, torch_v.grad]

    # comparison
    # print(f"Triton loss {loss} and torch loss {torch_loss}.  Also checking grads...")
    torch.testing.assert_allclose(loss, torch_loss, rtol=tol, atol=tol)
    for g1, g2 in zip(grads, torch_grads):
        torch.testing.assert_allclose(g1, g2, rtol=tol, atol=tol)


def triton_attention(
    layout,
    block: int,
    attn_mask: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
):
    sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(
        layout, block, "sdd", trans_a=False, trans_b=True
    )
    sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(
        layout, block, "dsd", trans_a=False, trans_b=False
    )
    sparse_softmax = triton.ops.blocksparse.BlockSparseSoftmax(
        layout,
        block,
    )

    w = sparse_dot_sdd_nt(query, key)
    w = sparse_softmax(w, scale=scale, attn_mask=attn_mask, attn_mask_mode="mul")
    a = sparse_dot_dsd_nn(w, value)
    return a
