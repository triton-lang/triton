import os
import warnings

import lazy_import
import torch
from sympy.utilities.exceptions import SymPyDeprecationWarning
import numpy as np
import platform

# Ignore an annoying warning printed when we import triton
warnings.simplefilter("ignore", SymPyDeprecationWarning)


def make_power_of_two(x):
    return int(2 ** (np.ceil(np.log(x) / np.log(2.0))))


triton_softmax = None

# only load triton and the softmax on the cluster
if platform.system() != "Darwin":
    import triton

    class _softmax_xent_loss(torch.autograd.Function):
        """This modifies logits in place, turning them into negative logprobs
        on the forward pass.  It should not copy the logits at all.
        """

        fwd_src = triton.read(
            os.path.join(
                os.path.dirname(__file__),
                "softmax_xent_kernels.c",
            ),
            kernel_names=["softmax_fwd"],
        )
        bwd_src = triton.read(
            os.path.join(
                os.path.dirname(__file__),
                "softmax_xent_kernels.c",
            ),
            kernel_names=["softmax_bwd"],
        )

        # Need TILE = n_vocab for this approach to work:
        input_config_to_kernel_fwd = {}
        input_config_to_kernel_bwd = {}

        @classmethod
        def forward(cls, ctx, logits, indices):
            # make sure we can use triton
            n_vocab = logits.shape[-1]
            assert (
                indices.dtype == torch.int64
            ), "Indices are expected to be of type long."

            # compile a new kernel if needed; otherwise load from a cache
            if not (logits.dtype, n_vocab) in cls.input_config_to_kernel_fwd:
                infinities = {
                    torch.float16: "F16_INFINITY",
                    torch.float32: "F32_INFINITY",
                }
                cls.input_config_to_kernel_fwd[(logits.dtype, n_vocab)] = triton.kernel(
                    cls.fwd_src,
                    device=logits.device,
                    defines={
                        "TILE": make_power_of_two(n_vocab),
                        "TYPE": logits.dtype,
                        "INFINITY": infinities[logits.dtype],
                    },
                )
            kernel_fwd = cls.input_config_to_kernel_fwd[(logits.dtype, n_vocab)]

            # flatten logits and be prepared to restore them to their original shape
            original_logits_shape = logits.shape
            if len(original_logits_shape) > 2:
                logits = logits.reshape((-1, n_vocab))
                indices = indices.reshape((-1,))

            # run the kernel and assign the result in place
            result = torch.empty_like(indices, dtype=logits.dtype).cuda()
            neg_logprobs = torch.empty_like(logits, dtype=logits.dtype).cuda()
            grid = lambda opt: (logits.shape[0],)
            kernel_fwd(
                logits.data_ptr(),
                neg_logprobs.data_ptr(),
                indices.data_ptr(),
                result.data_ptr(),
                n_vocab,
                grid=grid,
            )

            if len(original_logits_shape) > 2:
                logits = logits.reshape(original_logits_shape)
                indices = indices.reshape(*original_logits_shape[:-1])

            ctx.save_for_backward(neg_logprobs, indices)
            ctx.original_logits_shape = original_logits_shape

            return result

        @classmethod
        def backward(cls, ctx, dneg_logprobs):
            """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
            so we initialize the gradient as neg_logprobs, so we can just exponentiate
            to get p[k], which is most of what we need...  neg_logprobs will be
            modified in place to become the gradient we want
            """
            # load saved tensors and ensure correct types
            neg_logprobs, indices = ctx.saved_tensors
            original_logits_shape = ctx.original_logits_shape
            assert (
                dneg_logprobs.dtype == neg_logprobs.dtype
            ), f"Backward flowing derivatives of type {dneg_logprobs.dtype} != logits type {neg_logprobs.dtype}"
            n_vocab = neg_logprobs.shape[-1]

            # generate or load kernel
            if not (neg_logprobs.dtype, n_vocab) in cls.input_config_to_kernel_bwd:
                cls.input_config_to_kernel_bwd[
                    (neg_logprobs.dtype, n_vocab)
                ] = triton.kernel(
                    cls.bwd_src,
                    device=neg_logprobs.device,
                    defines={
                        "TILE": make_power_of_two(n_vocab),
                        "TYPE": neg_logprobs.dtype,
                    },
                )
            kernel_bwd = cls.input_config_to_kernel_bwd[(neg_logprobs.dtype, n_vocab)]
            grid = lambda opt: (neg_logprobs.shape[0],)

            # neg_logprobs will be modified in place to become our gradient:
            kernel_bwd(
                neg_logprobs.data_ptr(),
                indices.data_ptr(),
                dneg_logprobs.data_ptr(),
                n_vocab,
                grid=grid,
            )

            # reshape results based on shape of original logits passed to forward
            if len(original_logits_shape) > 2:
                neg_logprobs = neg_logprobs.reshape(original_logits_shape)

            return neg_logprobs, torch.zeros_like(indices)

    triton_softmax = _softmax_xent_loss.apply
