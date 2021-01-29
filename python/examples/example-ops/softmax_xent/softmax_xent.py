import os
import warnings

import lazy_import
import torch
from sympy.utilities.exceptions import SymPyDeprecationWarning

# Ignore an annoying warning printed when we import triton
warnings.simplefilter("ignore", SymPyDeprecationWarning)
lazy_import.lazy_module("triton")
import triton


class _softmax_xent_loss(torch.autograd.Function):
    """ This makes one copy of the logits. """

    fwd_src = open(os.path.join(os.path.dirname(__file__), "softmax_xent_fwd.c")).read()
    bwd_src = open(os.path.join(os.path.dirname(__file__), "softmax_xent_bwd.c")).read()

    # Need TILE = n_vocab for this approach to work:
    input_config_to_kernel_fwd = {}
    input_config_to_kernel_bwd = {}

    @classmethod
    def forward(cls, ctx, logits, indices):
        """ expects logits in the shape (..., n_vocab) """
        initial_logit_shape = logits.shape
        n_vocab = logits.shape[-1]
        assert indices.dtype == torch.int64
        assert (n_vocab == 512) or (
            n_vocab % 1024 == 0
        ), "Triton softmax op won't work unless n_vocab is 512 or is divisible by 1024."

        logits = logits.reshape((-1, n_vocab))
        indices = indices.reshape((-1,))

        if not (logits.dtype, n_vocab) in cls.input_config_to_kernel_fwd:
            cls.input_config_to_kernel_fwd[(logits.dtype, n_vocab)] = triton.kernel(
                cls.fwd_src,
                defines={"TILE": n_vocab, "TYPE": logits.dtype},
                num_warps=[8],
            )
        kernel_fwd = cls.input_config_to_kernel_fwd[(logits.dtype, n_vocab)]

        N = logits.numel()
        neg_logprobs = torch.zeros_like(logits, dtype=logits.dtype).cuda()
        result = torch.zeros_like(indices, dtype=logits.dtype).cuda()
        grid = lambda opt: (triton.cdiv(N, opt.d("TILE")),)
        kernel_fwd(logits, indices, result, neg_logprobs, grid=grid)
        # logits should be unmodified, but now neg_logprobs are full
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
                defines={"TILE": n_vocab, "TYPE": neg_logprobs.dtype},
                num_warps=[8],
            )
        kernel_bwd = cls.input_config_to_kernel_bwd[(neg_logprobs.dtype, n_vocab)]
        grid = lambda opt: (triton.cdiv(N, opt.d("TILE")),)

        # neg_logprobs will be modified in place to become our gradient:
        kernel_bwd(
            neg_logprobs,
            indices,
            dneg_logprobs,
            grid=grid,
        )

        return neg_logprobs, torch.zeros_like(indices)


triton_softmax = _softmax_xent_loss.apply
