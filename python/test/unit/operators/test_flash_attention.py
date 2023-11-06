import pytest
import torch

import triton
import triton.ops


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(2, 4, 512, 16),
                                                 (2, 4, 512, 32),
                                                 (2, 4, 512, 64),
                                                 (2, 4, 512, 128)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('seq_par', [True, False])
def test_op(Z, H, N_CTX, D_HEAD, dtype, causal, seq_par):
    import os
    enable_tma = os.environ.get('ENABLE_TMA', 'not found').lower()
    if enable_tma in ["on", "true", "1"]:
        if dtype == torch.bfloat16:
            pytest.skip('bfloat16 tma not support currently')

    capability = torch.cuda.get_device_capability()
<<<<<<< HEAD
    if torch.version.hip is not None:
        if dtype != torch.float16:
            pytest.skip("Currently flash attention on AMD gpu is only supported in fp16.")
        if D_HEAD < 32:
            pytest.skip("D_HEAD < 32 is not supported. It will be enabled once smaller tile size is supported in MFMA pipeline.")
        if D_HEAD > 64:
            pytest.skip("D_HEAD > 64 is not supported. Currently it causes shared memory out of resource error.")

    if capability[0] < 8:
        pytest.skip("Flash attention only supported for compute capability < 80")
=======
    interpreter = os.environ.get("TRITON_INTERPRET", 'not found') in ["on", "true", "1"]
    if not interpreter and capability[0] < 8:
        pytest.skip("Flash attention only supported for compute capability >= 80")
>>>>>>> ac9fa68d18c777e421bd3f6fb1ddcfd60b6fda33
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)

    if torch.version.hip is None:
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None

    # # triton implementation
    tri_out = triton.ops.attention(q, k, v, causal, sm_scale, seq_par)
    # print(ref_out)
    # print(tri_out)
    if torch.version.hip is None:
        tri_out.backward(dout)
        tri_dv, v.grad = v.grad.clone(), None
        tri_dk, k.grad = k.grad.clone(), None
        tri_dq, q.grad = q.grad.clone(), None
    # compare
    atol = 1e-1 if dtype == torch.bfloat16 else 1e-2
<<<<<<< HEAD
    torch.testing.assert_allclose(ref_out, tri_out, atol=atol, rtol=0)
    if torch.version.hip is None:
        torch.testing.assert_allclose(ref_dv, tri_dv, atol=atol, rtol=0)
        torch.testing.assert_allclose(ref_dk, tri_dk, atol=atol, rtol=0)
        torch.testing.assert_allclose(ref_dq, tri_dq, atol=atol, rtol=0)
=======
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=0)
>>>>>>> ac9fa68d18c777e421bd3f6fb1ddcfd60b6fda33
