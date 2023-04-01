import pytest
import torch

import triton
import triton.ops


def sparsify_tensor(x, mask, block):
    ret = torch.empty((x.size(0), mask.sum(), block, block), dtype=x.dtype, device=x.device)
    for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
        ret[:, idx, :, :] = x[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block]
    return ret


def make_pair(shape, device="cuda", alpha=1e-2, beta=0., trans=False, data=None, dtype=torch.float32):
    if data is None:
        data = torch.randn(shape, dtype=torch.float32, requires_grad=True, device=device)
    ref_ret = data
    ref_ret = ref_ret * alpha + beta
    ref_ret = ref_ret.half().to(dtype)
    if trans:
        ref_ret = ref_ret.t().requires_grad_()
    ref_ret = ref_ret.detach().requires_grad_()
    tri_ret = ref_ret.clone().detach().requires_grad_()
    return ref_ret, tri_ret


def mask_tensor(x, mask, block, value=0):
    ret = x.clone()
    for h, i, j in zip(*(mask == 0).nonzero(as_tuple=True)):
        ret[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block] = value
    return ret


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
    do_sparsify = lambda x: sparsify_tensor(x, layout, BLOCK)
    do_mask = lambda x: mask_tensor(x, layout, BLOCK)
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
    layout[1, 2, :] = 0
    layout[1, :, 1] = 0
    # create data
    a_ref, a_tri = make_pair(a_shape, alpha=.1, dtype=DTYPE)
    b_ref, b_tri = make_pair(b_shape, alpha=.1, dtype=DTYPE)
    dc_ref, dc_tri = make_pair(c_shape, dtype=DTYPE)
    # compute [torch]
    dc_ref = do_mask(dc_ref) if is_sdd else dc_ref
    a_ref = do_mask(a_ref) if is_dsd else a_ref
    b_ref = do_mask(b_ref) if is_dds else b_ref
    a_ref.retain_grad()
    b_ref.retain_grad()
    c_ref = torch.matmul(a_ref.transpose(2, 3) if TRANS_A else a_ref,
                         b_ref.transpose(2, 3) if TRANS_B else b_ref)
    c_ref.backward(dc_ref)
    c_ref = do_sparsify(c_ref) if is_sdd else c_ref
    da_ref = do_sparsify(a_ref.grad) if is_dsd else a_ref.grad
    db_ref = do_sparsify(b_ref.grad) if is_dds else b_ref.grad
    # triton result
    dc_tri = do_sparsify(dc_tri) if is_sdd else dc_tri
    a_tri = do_sparsify(a_tri) if is_dsd else a_tri
    b_tri = do_sparsify(b_tri) if is_dds else b_tri
    a_tri.retain_grad()
    b_tri.retain_grad()
    op = triton.ops.blocksparse.matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device="cuda")
    try:
        c_tri = op(a_tri, b_tri)
        c_tri.backward(dc_tri)
        da_tri = a_tri.grad
        db_tri = b_tri.grad
        # compare
        torch.testing.assert_allclose(c_ref, c_tri)
        torch.testing.assert_allclose(da_ref, da_tri)
        torch.testing.assert_allclose(db_ref, db_tri)
    except triton.OutOfResourcesError as e:
        pytest.skip(str(e))


configs = [
    (16, 256),
    (32, 576),
    (64, 1871),
    (128, 2511),
]


@pytest.mark.parametrize("is_dense", [False, True])
@pytest.mark.parametrize("BLOCK, WIDTH", configs)
def test_softmax(BLOCK, WIDTH, is_dense, Z=2, H=2, is_causal=True, scale=0.4):
    # set seed
    torch.random.manual_seed(0)
    Z, H, M, N = 2, 3, WIDTH, WIDTH
    # initialize layout
    # make sure each row has at least one non-zero element
    layout = torch.randint(2, (H, M // BLOCK, N // BLOCK))
    if is_dense:
        layout[:] = 1
    else:
        layout[1, 2, :] = 0
        layout[1, :, 1] = 0
    # initialize data
    a_shape = (Z, H, M, N)
    a_ref, a_tri = make_pair(a_shape)
    dout_ref, dout_tri = make_pair(a_shape)
    # compute [torch]
    a_ref = mask_tensor(a_ref, layout, BLOCK, value=float("-inf"))
    a_ref.retain_grad()
    at_mask = torch.ones((M, N), device="cuda")
    if is_causal:
        at_mask = torch.tril(at_mask)
    M = at_mask[None, None, :, :] + torch.zeros_like(a_ref)
    a_ref[M == 0] = float("-inf")
    out_ref = torch.softmax(a_ref * scale, -1)
    out_ref.backward(dout_ref)
    out_ref = sparsify_tensor(out_ref, layout, BLOCK)
    da_ref = sparsify_tensor(a_ref.grad, layout, BLOCK)
    # compute [triton]
    a_tri = sparsify_tensor(a_tri, layout, BLOCK)
    a_tri.retain_grad()
    dout_tri = sparsify_tensor(dout_tri, layout, BLOCK)
    op = triton.ops.blocksparse.softmax(layout, BLOCK, device="cuda", is_dense=is_dense)
    out_tri = op(a_tri, scale=scale, is_causal=is_causal)
    out_tri.backward(dout_tri)
    da_tri = a_tri.grad
    # compare
    torch.testing.assert_allclose(out_tri, out_ref)
    torch.testing.assert_allclose(da_tri, da_ref)


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
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:
        pytest.skip("Only test tl.dot() on devices with sm >= 70")

    # inputs
    qkv_shape = (batch_size, n_heads, n_ctx, 64)
    qkvs = [
        torch.nn.Parameter(input_scale * torch.randn(qkv_shape), requires_grad=True).to(dtype).cuda() for _ in range(3)
    ]

    # Triton:
    n_blocks = n_ctx // block
    layout = torch.tril(torch.ones([n_heads, n_blocks, n_blocks], dtype=torch.long))
    query, key, value = [x.clone() for x in qkvs]
    query.retain_grad()
    key.retain_grad()
    value.retain_grad()
    attn_out = triton_attention(layout, block, query=query, key=key, value=value, scale=scale)
    # ad hoc loss
    loss = (attn_out ** 2).mean()
    loss.backward()
    grads = [query.grad, key.grad, value.grad]

    # Torch version:
    torch_q, torch_k, torch_v = [x.clone() for x in qkvs]
    attn_mask = torch.ones([n_ctx, n_ctx], device="cuda", dtype=dtype)
    attn_mask = torch.tril(attn_mask, diagonal=0)
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
    torch.testing.assert_allclose(loss, torch_loss, atol=1e-3, rtol=0)
    for g1, g2 in zip(grads, torch_grads):
        torch.testing.assert_allclose(g1, g2)


@pytest.mark.parametrize("block", [16, 32, 64])
def triton_attention(
    layout,
    block: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
):
    sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True, device=value.device)
    sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(layout, block, "dsd", trans_a=False, trans_b=False, device=value.device)
    sparse_softmax = triton.ops.blocksparse.softmax(layout, block, device=value.device)

    w = sparse_dot_sdd_nt(query, key)
    w = sparse_softmax(w, scale=scale, is_causal=True)
    a = sparse_dot_dsd_nn(w, value)
    return a
