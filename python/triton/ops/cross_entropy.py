import torch

import triton
import triton.language as tl


def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def num_warps(N):
    if N < 2048:
        return 4
    elif N < 8192:
        return 8
    return 16


@triton.heuristics({'num_warps': lambda nargs: num_warps(nargs['N'])})
@triton.heuristics({'BLOCK': lambda nargs: next_power_of_2(nargs['N'])})
@triton.jit
def _forward(LOGITS, PROBS, IDX, LOSS, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    idx = tl.load(IDX + row)
    # pointers to logit and probs
    LOGITS = LOGITS + row * N + cols
    WRIT_PROBS = PROBS + row * N + cols
    READ_PROBS = PROBS + row * N + idx
    # write-back negative log-probs
    logits = tl.load(LOGITS, mask=cols < N, other=-float('inf'))
    logits = logits.to(tl.float32)
    logits = logits - tl.max(logits, 0)
    probs = tl.log(tl.sum(tl.exp(logits), 0)) - logits
    tl.store(WRIT_PROBS, probs, mask=cols < N)
    # There is a bug in the compiler, which fails to insert a barrier here.
    # We add it explicitly for now. Will be fixed soon.
    tl.debug_barrier()
    # write-back loss
    probs = tl.load(READ_PROBS)
    tl.store(LOSS + row, probs)


@triton.heuristics({'num_warps': lambda nargs: num_warps(nargs['N'])})
@triton.heuristics({'BLOCK': lambda nargs: next_power_of_2(nargs['N'])})
@triton.jit
def _backward(PROBS, IDX, DPROBS, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    idx = tl.load(IDX + row)
    # pointers to probs
    PROBS = PROBS + row * N + cols
    # We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
    # and we have -log(p[k]) stored in PROBS, so this is easy
    probs = -tl.load(PROBS, mask=cols < N, other=float('inf'))
    probs = tl.exp(probs.to(tl.float32))
    delta = cols == idx
    # write result in-place in PROBS
    dout = tl.load(DPROBS + row)
    din = (probs - delta) * dout
    tl.store(PROBS, din.to(PROBS.dtype.element_ty), mask=cols < N)


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
        _forward[grid](logits, neg_logprobs, indices, result, n_cols)
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
        # run the kernel
        # neg_logprobs will be modified in place to become our gradient:
        n_cols = neg_logprobs.shape[-1]
        grid = lambda opt: (neg_logprobs.numel() // n_cols, )
        _backward[grid](neg_logprobs, indices, dneg_logprobs, n_cols)
        return neg_logprobs, None


cross_entropy = _cross_entropy.apply
