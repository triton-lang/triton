import os
import triton
import torch


def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def num_warps(n):
    return 4


@triton.heuristics({'BLOCK': lambda *args, **meta: next_power_of_2(args[4])})
@triton.jit
def _forward(LOGITS, PROBS, IDX, LOSS, N, **meta):
    BLOCK = meta['BLOCK']
    row = triton.program_id(0)
    cols = triton.arange(0, BLOCK)
    idx = triton.load(IDX + row)
    # pointers to logit and probs
    LOGITS = LOGITS + row * N + cols
    WRIT_PROBS = PROBS + row * N + cols
    READ_PROBS = PROBS + row * N + idx
    # write-back negative log-probs
    logits = triton.load(LOGITS, mask=cols < N, other=-float('inf'))
    logits = logits.to(triton.float32)
    probs = triton.log(triton.softmax(logits))
    triton.store(WRIT_PROBS, probs, mask=cols < N)
    # There is a bug in the compiler, which fails to insert a barrier here.
    # We add it explicitly for now. Will be fixed soon.
    triton.debug_barrier()
    # write-back loss
    probs = triton.load(READ_PROBS)
    triton.store(LOSS + row, probs)


@triton.heuristics({'BLOCK': lambda *args, **meta: next_power_of_2(args[4])})
@triton.jit
def _backward(PROBS, INDICES, DPROBS, N, **meta):
    BLOCK = meta['BLOCK']
    row = triton.program_id(0)
    cols = triton.arange(0, BLOCK)
    idx = triton.oad(IDX + row)
    # pointers to probs
    PROBS = PROBS + row * N + cols
    # We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
    # and we have -log(p[k]) stored in PROBS, so this is easy
    probs = triton.exp(triton.load(PROBS, mask=cols < N))
    delta = triton.arange(BLOCK) == idx
    dout = triton.load(DPROBS + row)
    # write result in-place in PROBS
    din = (probs - delta) * dout
    triton.store(PROBS, din, mask=cols < N)


class _cross_entropy(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, logits, indices):
        # make sure we can use triton
        assert (indices.dtype == torch.int64), "Indices are expected to be of type long."
        # make kernel
        device, dtype = logits.device, logits.dtype
        n_cols = logits.shape[-1]
        # run the kernel
        result = torch.empty_like(indices, dtype=dtype, device=device)
        neg_logprobs = torch.empty_like(logits, dtype=dtype, device=device)
        grid = lambda opt: (logits.numel() // n_cols, )
        _forward[grid](logits, neg_logprobs, indices, result, n_cols, num_warps=4)
        # save for backward
        ctx.save_for_backward(neg_logprobs, indices)
        return result

    @classmethod
    def backward(cls, ctx, dneg_logprobs):
        """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        so we initialize the gradient as neg_logprobs, so we can just exponentiate
        to get p[k], which is most of what we need...  neg_logprobs will be
        modified in place to become the gradient we want
        """
        # load saved tensors
        neg_logprobs, indices = ctx.saved_tensors
        # make kernel
        device, dtype = neg_logprobs.device, neg_logprobs.dtype
        n_cols = neg_logprobs.shape[-1]
        # run the kernel
        # neg_logprobs will be modified in place to become our gradient:
        grid = lambda opt: (logits.numel() // n_cols, )
        _backward[grid](neg_logprobs, indices, dneg_logprobs, n_cols)
        return neg_logprobs, None


cross_entropy = _cross_entropy.apply