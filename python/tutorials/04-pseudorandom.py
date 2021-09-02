"""
Pseudorandom
=================

.. image:: random_bits.png
  :width: 400

In this tutorial, you will write a dropout kernel using Triton. This kernel will different from the
usual implementations - the state will be represented by a single int32 seed, as opposed to bit mask
of the same shape as a tensor. You will learn about:

- how to generate pseudorandom numbers in parallel fashion on a GPU
- the api for `triton.random.philox` (based on [SALMON2011]_)
"""

# %%
# Baseline
# -------------
# Dropout is an aritchemtic operation, which is often utilized to improve performance of deep
# learning in low-data regime (i.e. regularization). It was first introduced in [SRIVASTAVA2014]_.
#
# It takes a vector as input and produces an vector of the same shape as output. Each scalar in the
# output has a probability :math:`p` of being changed to zero and otherwise it is copied from the input.
# This forces the network to perform well even if only :math:`1 - p` scalars from the input are available.
#
# At evaluation time we want to use the full power of the network so we set :math:`p=0`. Naively this would
# increase the norm of the output (which can be a bad thing, e.g. it can lead to artificial decrease
# in the output softmax temperature). To prevent this we multiply the output by :math:`\frac{1}{1 - p}`, which
# keeps the norm consistent regardless of the dropout probability.
#
# Let's first take a look at the baseline implementation.


import tabulate
import torch
import triton
import triton.language as tl

@triton.jit
def _dropout(
        x_ptr,
        x_keep_ptr,
        output_ptr,
        n_elements,
        p,
        **meta,
):
    BLOCK_SIZE = meta['BLOCK_SIZE']
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output

# Input tensor
x = torch.randn(size=(10,)).cuda()
# Dropout mask
p = 0.5
x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()
#
output = dropout(x, x_keep=x_keep, p=p)
print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist()
]))

# %%
# Seeded dropout
# -------------
# Above implementation of dropout works fine, but it can be a bit awkward to deal with. Firstly
# we need to store the dropout mask for backpropagation. Secondly, dropout state management can get
# very tricky when using recompute/checkpointing (e.g. see all the notes about `preserve_rng_state` in
# https://pytorch.org/docs/1.9.0/checkpoint.html). In this tutorial we'll produce an implementation
# that reduces the memory usage and lowers the implementation barrier to persisting randomness across
# multiple invocations of the kernel.
#
# Pseudorandom number generation in Triton is simple! We'll use the function
# `triton.language.random.philox.random_4x`. It takes offset and seed as input and produces 4 random
# int32's as output. To be maximally efficient one should use all 4 numbers at the same time, but
# we're not going to worry about that for now in the interest of simplicity.
#
# The final ingredient we need for our dropout implementation is
# `triton.language.random.philox.uint32_to_uniform_float`, which is a numerically stable way to
# convert random integer to float sampled from [0, 1). This is originally designed from uint32, but
# it works with int32 too as long as the int32 uniformly covers all the possible values it can take.
# Both random int32 and uint32 is just a source of 32 random bits after all!
#
# Let's put it all together.

from triton.language.random.philox import *

@triton.jit
def _seeded_dropout(
        x_ptr,
        output_ptr,
        n_elements,
        p,
        seed,
        **meta,
):
    BLOCK_SIZE = meta['BLOCK_SIZE']
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # generate random int32 based on seed and offset within the input tensor
    random_int, _, _, _ = random_4x(seed, offsets)
    # convert that int32 to float in range [0, 1)
    random_float = uint32_to_uniform_float(random_int)
    # compute dropout mask based on the value of the random float.
    x_keep = random_float > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10,)).cuda()
# Compare this to the baseline - dropout mask is never instantiated!
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["output (seed = 123)"] + output.tolist(),
    ["output (seed = 123)"] + output2.tolist(),
    ["output (seed = 512)"] + output3.tolist()
]))

# %%
# Et Voilà! We have a triton kernel that applies the same dropout mask provided the seed is the same!
# If you'd like explore further applications of pseudorandomness in GPU programming, we encourage you
# to explore the `triton/language/random` folder!

# %%
# Exercises
# -------------
# 1. Extend the kernel to operate over a matrix and use a vector of seeds - one per row.
# 2. Add support for striding.
# 3. (challenge) Implement a kernel for sparse Johnson-Lindenstrauss transform which generates the projection matrix one the fly each time using a seed.

# %%
# References
# --------------
#
# .. [SALMON2011] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", 2011
# .. [SRIVASTAVA2014] Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014
