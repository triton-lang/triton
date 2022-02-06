from numpy import require
import pytest
import torch
import triton


@pytest.mark.parametrize("MODE", ["sdd", "dds", "dsd"])
@pytest.mark.parametrize("TRANS_A", [False, True])
@pytest.mark.parametrize("TRANS_B", [False, True])
@pytest.mark.parametrize("BLOCK", [16, 32, 64])
@pytest.mark.parametrize("DTYPE", [torch.float16])
def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, Z=3, H=2, M=512, N=384, K=256):
    seed = 0
    torch.manual_seed(seed)
    is_sdd = MODE == "sdd"
    is_dsd = MODE == "dsd"
    is_dds = MODE == "dds"
    do_sparsify = lambda x: triton.testing.sparsify_tensor(x, layout, BLOCK)
    do_mask = lambda x: triton.testing.mask_tensor(x, layout, BLOCK)
    # create inputs
    # create op
    a_shape = (Z, H, K, M) if TRANS_A else (Z, H, M, K)
    b_shape = (Z, H, N, K) if TRANS_B else (Z, H, K, N)
    c_shape = (Z, H, M, N)
    shape = {
        "sdd": (M, N),
        "dsd": (a_shape[2], a_shape[3]),
        "dds": (b_shape[2], b_shape[3]),
    }[MODE]
    layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
    # create data
    a_ref, a_tri  = triton.testing.make_pair(a_shape, alpha=.1)
    b_ref, b_tri  = triton.testing.make_pair(b_shape, alpha=.1)
    dc_ref, dc_tri = triton.testing.make_pair(c_shape)
    # compute [torch]
    dc_ref = do_mask(dc_ref) if is_sdd else dc_ref
    a_ref  = do_mask(a_ref)  if is_dsd else a_ref
    b_ref  = do_mask(b_ref)  if is_dds else b_ref
    a_ref.retain_grad()
    b_ref.retain_grad()
    c_ref  = torch.matmul(a_ref.transpose(2, 3) if TRANS_A else a_ref,
                            b_ref.transpose(2, 3) if TRANS_B else b_ref)
    c_ref.backward(dc_ref)
    c_ref  = do_sparsify(c_ref)  if is_sdd else c_ref
    da_ref = do_sparsify(a_ref.grad) if is_dsd else a_ref.grad
    db_ref = do_sparsify(b_ref.grad) if is_dds else b_ref.grad
    # triton result
    dc_tri = do_sparsify(dc_tri) if is_sdd else dc_tri
    a_tri = do_sparsify(a_tri)   if is_dsd else a_tri
    b_tri = do_sparsify(b_tri)   if is_dds else b_tri
    a_tri.retain_grad()
    b_tri.retain_grad()
    op = triton.ops.blocksparse.matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device="cuda")
    c_tri = op(a_tri, b_tri)       
    c_tri.backward(dc_tri)
    da_tri = a_tri.grad
    db_tri = b_tri.grad
    # compare
    triton.testing.assert_almost_equal(c_ref, c_tri)
    triton.testing.assert_almost_equal(da_ref, da_tri)
    triton.testing.assert_almost_equal(db_ref, db_tri)


@pytest.mark.parametrize("BLOCK", [16, 32, 64])
@pytest.mark.parametrize("WIDTH", [256, 576, 1024, 1792])
@pytest.mark.parametrize("DTYPE", [torch.float16, torch.float32])
def test_softmax(BLOCK, WIDTH, DTYPE):
    is_causal = True
    # set seed
    torch.random.manual_seed(0)
    Z, H, M, N = 1, 1, WIDTH, WIDTH
    scale = 0.4
    # create inputs
    layout = torch.randint(2, (H, M // BLOCK, N // BLOCK))
    x = torch.randn((Z, H, M, N), dtype=DTYPE, requires_grad=True, device="cuda")
    at_mask = torch.randint(low=0, high=2, size=(N, N), dtype=torch.bool, requires_grad=False, device="cuda")
    # make sure each row has at least one non-zero element
    torch.diagonal(layout)[:] = 1
    torch.diagonal(at_mask)[:] = 1
    kp_mask = torch.randint(low=0, high=2, size=(Z, N), dtype=DTYPE, requires_grad=False, device="cuda")
    kp_mask[:] = 0
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
        is_causal=is_causal,
    )
    # torch result
    rx = triton.testing.mask_tensor(x, layout, BLOCK, value=float("-inf"))
    # broadcast at_mask to the same shape as rx
    if is_causal:
        at_mask = torch.tril(at_mask)
    M = at_mask[None, None, :, :] + torch.zeros_like(rx)
    rx[M == 0] = float("-inf")
    # rx += kp_mask[:, None, None, :]
    ry = torch.softmax(rx * scale, -1)
    ry = triton.testing.sparsify_tensor(ry, layout, BLOCK)
    # compare
    triton.testing.assert_almost_equal(ry, ty)


@pytest.mark.parametrize("block", [16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_attention_fwd_bwd(
    block,
    dtype,
    input_scale=1.0,
    scale=1 / 8.0,
    n_ctx=256,
    batch_size=2,
    n_heads=2,
):
    # inputs
    qkv_shape = (batch_size, n_heads, n_ctx, 64)
    qkvs = [
        torch.nn.Parameter(input_scale * torch.randn(qkv_shape), requires_grad=True).to(dtype).cuda() for _ in range(3)
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
    attn_out = triton_attention(layout, block, attn_mask, query=query, key=key, value=value, scale=scale)
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
    triton.testing.assert_almost_equal(loss, torch_loss)
    for g1, g2 in zip(grads, torch_grads):
        triton.testing.assert_almost_equal(g1, g2)


@pytest.mark.parametrize("block", [16, 32, 64])
def triton_attention(
    layout,
    block: int,
    attn_mask: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
):
    sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True, device=value.device)
    sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(layout, block, "dsd", trans_a=False, trans_b=False, device=value.device)
    sparse_softmax = triton.ops.blocksparse.softmax(
        layout,
        block,
    )

    w = sparse_dot_sdd_nt(query, key)
    w = sparse_softmax(w, scale=scale, attn_mask=attn_mask, attn_mask_mode="mul")
    a = sparse_dot_dsd_nn(w, value)
    return a
