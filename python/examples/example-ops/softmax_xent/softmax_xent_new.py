import os
import warnings

import lazy_import
import torch
from sympy.utilities.exceptions import SymPyDeprecationWarning

# Ignore an annoying warning printed when we import triton
warnings.simplefilter("ignore", SymPyDeprecationWarning)
lazy_import.lazy_module("triton")

import triton


class _softmax_xent_loss_new(torch.autograd.Function):
    """This modifies logits in place, turning them into negative logprobs
    on the forward pass.  It should not copy the logits at all.
    """

    fwd_src = triton.read(
        os.path.join(os.path.dirname(__file__), "softmax_xent_kernels_new.c"),
        kernel_names=["softmax_fwd"],
    )
    bwd_src = triton.read(
        os.path.join(os.path.dirname(__file__), "softmax_xent_kernels_new.c"),
        kernel_names=["softmax_bwd"],
    )

    # Need TILE = n_vocab for this approach to work:
    input_config_to_kernel_fwd = {}
    input_config_to_kernel_bwd = {}

    @classmethod
    def forward(cls, ctx, logits, indices):
        n_vocab = logits.shape[-1]
        assert indices.dtype == torch.int64
        assert n_vocab % 16 == 0, "Number of logit options must be divisible by 16."

        if not (logits.dtype, n_vocab) in cls.input_config_to_kernel_fwd:
            cls.input_config_to_kernel_fwd[(logits.dtype, n_vocab)] = triton.kernel(
                cls.fwd_src,
                device=logits.device,
                defines={
                    "TILE": n_vocab,
                    "TYPE": logits.dtype,
                },
                num_warps=[4],
            )
        kernel_fwd = cls.input_config_to_kernel_fwd[(logits.dtype, n_vocab)]

        N = logits.numel()
        result = torch.empty_like(indices, dtype=logits.dtype).cuda()
        neg_logprobs = torch.empty_like(logits, dtype=logits.dtype).cuda()
        grid = lambda opt: (triton.cdiv(N, opt.TILE),)
        kernel_fwd(
            logits.data_ptr(),
            neg_logprobs.data_ptr(),
            # neg_logprobs.data_ptr(),
            indices.data_ptr(),
            result.data_ptr(),
            grid=grid,
        )
        # logits -> neg_logprobs via an in place modification by kernel_fwd
        ctx.save_for_backward(neg_logprobs, indices)

        return result

    @classmethod
    def backward(cls, ctx, dneg_logprobs):
        """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        so we initialize the gradient as neg_logprobs, so we can just exponentiate
        to get p[k], which is most of what we need...  neg_logprobs will be
        modified in place to become the gradient we want
        """
        neg_logprobs, indices = ctx.saved_tensors
        assert indices.dtype == torch.int64
        assert (
            dneg_logprobs.dtype == neg_logprobs.dtype
        ), f"Backward flowing derivatives of type {dneg_logprobs.dtype} != logits type {neg_logprobs.dtype}"
        n_vocab = neg_logprobs.shape[-1]
        N = neg_logprobs.numel()

        if not (neg_logprobs.dtype, n_vocab) in cls.input_config_to_kernel_bwd:
            cls.input_config_to_kernel_bwd[
                (neg_logprobs.dtype, n_vocab)
            ] = triton.kernel(
                cls.bwd_src,
                device=neg_logprobs.device,
                defines={"TILE": n_vocab, "TYPE": neg_logprobs.dtype},
                num_warps=[4],
            )
        kernel_bwd = cls.input_config_to_kernel_bwd[(neg_logprobs.dtype, n_vocab)]
        grid = lambda opt: (triton.cdiv(N, opt.TILE),)

        # neg_logprobs will be modified in place to become our gradient:
        kernel_bwd(
            neg_logprobs.data_ptr(),
            indices.data_ptr(),
            dneg_logprobs.data_ptr(),
            grid=grid,
        )

        return neg_logprobs, torch.zeros_like(indices)


triton_softmax_new = _softmax_xent_loss_new.apply
