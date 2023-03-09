import pytest
import torch

import triton


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(4, 48, 1024, 64)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_op(Z, H, N_CTX, D_HEAD, dtype):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        pytest.skip("Flash attention only supported for compute capability < 80")
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    sm_scale = 0.2
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = triton.ops.attention(q, k, v, sm_scale)
    # print(ref_out)
    # print(tri_out)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    triton.testing.assert_almost_equal(ref_out, tri_out)
    decimal = 1 if dtype == torch.bfloat16 else 2
    triton.testing.assert_almost_equal(ref_dv, tri_dv, decimal=decimal)
    triton.testing.assert_almost_equal(ref_dk, tri_dk)
    triton.testing.assert_almost_equal(ref_dq, tri_dq)
