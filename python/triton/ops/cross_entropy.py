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

def largest_pow2_divisor(N):
    if N % 8 == 0: return 8
    if N % 4 == 0: return 4
    if N % 2 == 0: return 2
    return 1

def make_kernel(device, dtype, n_cols, cache, name):
    rounded = next_power_of_2(n_cols)
    div = largest_pow2_divisor(n_cols)
    key = (dtype, rounded, div)
    if key not in cache:
        fname = os.path.join(os.path.dirname(__file__), "cross_entropy.c")
        src = triton.read(fname, kernel_names=[name])
        infinities = {
            torch.float16: "F16_INFINITY",
            torch.float32: "F32_INFINITY",
        }
        defines = {"TILE": rounded, "TYPE": dtype, "INFINITY": infinities[dtype], "N_COLS_MULT": div}
        cache[key] = triton.kernel(src, device=device, defines=defines, num_warps=4)
    return cache[key]

# forward kernel
fwd_kernels = dict()
make_fwd_kernel = lambda device, dtype, n_cols: make_kernel(device, dtype, n_cols, fwd_kernels, "forward")

# backward kernel
bwd_kernels = dict()
make_bwd_kernel = lambda device, dtype, n_cols: make_kernel(device, dtype, n_cols, bwd_kernels, "backward")

class _cross_entropy(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, logits, indices):
        # make sure we can use triton
        assert (indices.dtype == torch.int64), "Indices are expected to be of type long."
        # make kernel
        device, dtype = logits.device, logits.dtype
        n_cols = logits.shape[-1]
        kernel = make_fwd_kernel(device, dtype, n_cols)
        # run the kernel
        result = torch.empty_like(indices, dtype=dtype, device=device)
        neg_logprobs = torch.empty_like(logits, dtype=dtype, device=device)
        kernel(logits.data_ptr(),
               neg_logprobs.data_ptr(),
               indices.data_ptr(),
               result.data_ptr(),
               n_cols,
               grid=lambda opt: (logits.numel() // n_cols, ))
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
        kernel = make_bwd_kernel(device, dtype, n_cols)
        # run the kernel
        # neg_logprobs will be modified in place to become our gradient:
        kernel(neg_logprobs.data_ptr(),
               indices.data_ptr(),
               dneg_logprobs.data_ptr(),
               n_cols,
               grid=lambda opt: (neg_logprobs.numel() // n_cols, ))
        return neg_logprobs, None

cross_entropy = _cross_entropy.apply