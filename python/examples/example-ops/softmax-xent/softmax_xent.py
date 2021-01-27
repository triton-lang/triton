import os
import torch
import triton


class _softmax_xent_loss(torch.autograd.Function):
    fwd_src = open(os.path.join(os.path.dirname(__file__), "softmax_xent_fwd.c")).read()
    bwd_src = open(os.path.join(os.path.dirname(__file__), "softmax_xent_bwd.c")).read()

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
                defines={"TILE": n_vocab, "TYPE": logits.dtype},
                num_warps=[2, 4, 8, 16],
            )
        kernel_fwd = cls.input_config_to_kernel_fwd[(logits.dtype, n_vocab)]

        N = logits.numel()
        result = torch.empty_like(indices, dtype=logits.dtype).cuda()
        grid = lambda opt: (triton.cdiv(N, opt.d("TILE")),)
        kernel_fwd(logits, indices, result, grid=grid)
        # logits -> neg_logprobs via an in place modification by kernel_fwd
        ctx.save_for_backward(logits, indices)

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
        n_vocab = neg_logprobs.shape[-1]
        N = neg_logprobs.numel()
        useful_int = torch.zeros_like(indices).int()

        if not (neg_logprobs.dtype, n_vocab) in cls.input_config_to_kernel_bwd:
            cls.input_config_to_kernel_bwd[
                (neg_logprobs.dtype, n_vocab)
            ] = triton.kernel(
                cls.bwd_src,
                defines={"TILE": n_vocab, "TYPE": neg_logprobs.dtype},
                num_warps=[16],
            )
        kernel_bwd = cls.input_config_to_kernel_bwd[(neg_logprobs.dtype, n_vocab)]
        grid = lambda opt: (triton.cdiv(N, opt.d("TILE")),)

        # neg_logprobs will be modified in place to become our gradient:
        kernel_bwd(
            neg_logprobs,
            indices,
            useful_int,
            dneg_logprobs,
            grid=grid,
        )

        return neg_logprobs, torch.zeros_like(indices)


triton_softmax = _softmax_xent_loss.apply


def test_softmax(num_seq=16, n_vocab=32 * 1024):
    for dtype in [torch.float32, torch.float16]:
        x = torch.randn(num_seq, n_vocab).to(dtype)
        indices = 4 + torch.ones(num_seq).long()

        triton_input = x.cuda()
        triton_indices = indices.cuda()
        triton_result = triton_softmax(triton_input, triton_indices)
        print("Triton:", triton_result)

        torch_input = x.cuda()
        torch_indices = indices.cuda()
        torch_xent_fn = torch.nn.CrossEntropyLoss(reduction="none")
        torch_result = torch_xent_fn(torch_input, torch_indices)
        print("Torch:", torch_result)
        torch.testing.assert_allclose(torch_result, triton_result)


def test_grad(num_seq=4, n_vocab=512):
    print(num_seq, n_vocab)
    for dtype in [torch.float32, torch.float16]:
        logit = torch.randn(num_seq, n_vocab, requires_grad=True, device="cuda").to(
            dtype
        )
        indices = torch.arange(num_seq).long()

        triton_logit = torch.nn.Parameter(
            logit.clone().detach().cuda(), requires_grad=True
        )
        triton_indices = indices.clone().detach().cuda()
        triton_result = triton_softmax(triton_logit, triton_indices)
        triton_result.mean().backward()
        print("Triton grad:\n", (triton_logit.grad[:, :6]))

        torch_logit = torch.nn.Parameter(
            logit.clone().detach().cuda(), requires_grad=True
        )
        torch_indices = indices.clone().detach().cuda()
        torch_xent_fn = torch.nn.CrossEntropyLoss(reduction="none")
        torch_result = torch_xent_fn(torch_logit, torch_indices)
        torch_result.mean().backward()
        print("Torch grad:\n", (torch_logit.grad[:, :6]))

        torch.testing.assert_allclose(torch_logit.grad, triton_logit.grad)



if __name__ == "__main__":
    test_grad()
