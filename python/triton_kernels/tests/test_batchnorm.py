import os
import pytest
import torch


@pytest.mark.parametrize("shape", [
    (8, 16),
    (64, 128),
    (2, 8, 32, 32),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
def test_batchnorm_forward_matches_torch(shape, dtype, training):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    x = torch.randn(*shape, device=device, dtype=dtype)
    if x.ndim == 2:
        N, C = x.shape
        gamma = torch.randn(C, device=device, dtype=torch.float32)
        beta = torch.randn(C, device=device, dtype=torch.float32)
    else:
        N, C, H, W = x.shape
        gamma = torch.randn(C, device=device, dtype=torch.float32)
        beta = torch.randn(C, device=device, dtype=torch.float32)

    eps = 1e-5
    momentum = 0.1
    running_mean = torch.zeros(C, device=device, dtype=torch.float32)
    running_var = torch.ones(C, device=device, dtype=torch.float32)

    # reference
    y_ref = torch.nn.functional.batch_norm(
        x.float(),
        running_mean.clone() if not training else None,
        running_var.clone() if not training else None,
        gamma, beta, training=training, momentum=momentum, eps=eps,
    ).to(dtype)

    # under test
    from triton_kernels.batchnorm import batchnorm_forward
    y_tri, saved_mean, saved_var = batchnorm_forward(
        x, gamma, beta, eps=eps, training=training,
        running_mean=running_mean, running_var=running_var, momentum=momentum, layout="NCHW"
    )

    rtol = 1e-5 if dtype is torch.float32 else 3e-2
    atol = 1e-6 if dtype is torch.float32 else 3e-3
    torch.testing.assert_close(y_ref, y_tri.to(dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_batchnorm_eps_and_identity(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    x = torch.randn(4, 8, device=device, dtype=dtype)
    C = x.shape[1]
    gamma = torch.ones(C, device=device, dtype=torch.float32)
    beta = torch.zeros(C, device=device, dtype=torch.float32)

    for eps in (1e-5, 1e-3):
        y_ref = torch.nn.functional.batch_norm(x.float(), None, None, gamma, beta, training=True, eps=eps).to(dtype)
        from triton_kernels.batchnorm import batchnorm_forward
        y_tri, m, v = batchnorm_forward(x, gamma, beta, eps=eps, training=True, layout="NCHW")
        rtol = 1e-5 if dtype is torch.float32 else 3e-2
        atol = 1e-6 if dtype is torch.float32 else 3e-3
        torch.testing.assert_close(y_ref, y_tri.to(dtype), rtol=rtol, atol=atol)


