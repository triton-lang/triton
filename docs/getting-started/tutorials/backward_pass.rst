.. _backward_pass:

Implementing Backward Propagation with Triton
==============================================

Triton does **not** automatically generate backward functions for your kernels.
You must manually define both forward and backward kernels, and wrap them in a
``torch.autograd.Function`` to enable gradient computation.

This tutorial demonstrates how to implement a simple element-wise multiplication
operation with custom forward and backward Triton kernels, and how to use it in
a PyTorch training loop.

Forward Kernel
--------------

We start by writing a Triton kernel that computes ``c = a * b`` element-wise.

.. code-block:: python

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def mul_kernel(
        a_ptr, b_ptr, c_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        c = a * b
        tl.store(c_ptr + offsets, c, mask=mask)

Backward Kernel
---------------

The backward kernel computes the gradients of the loss with respect to the inputs.
For element-wise multiplication, the gradient of ``c = a * b`` with respect to
``a`` is ``b``, and with respect to ``b`` is ``a``. The backward kernel receives
the gradient of the output (``dc``) and stores the gradients for ``a`` and ``b``.

.. code-block:: python

    @triton.jit
    def mul_backward_kernel(
        a_ptr, b_ptr, dc_ptr,
        da_ptr, db_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        dc = tl.load(dc_ptr + offsets, mask=mask)
        da = dc * b
        db = dc * a
        tl.store(da_ptr + offsets, da, mask=mask)
        tl.store(db_ptr + offsets, db, mask=mask)

Wrapping in ``torch.autograd.Function``
-----------------------------------------

We define a custom ``torch.autograd.Function`` that calls the forward and backward
kernels. The forward method launches the forward kernel, and the backward method
launches the backward kernel using the saved tensors.

.. code-block:: python

    class MulFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            c = torch.empty_like(a)
            n_elements = a.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            mul_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)
            return c

        @staticmethod
        def backward(ctx, dc):
            a, b = ctx.saved_tensors
            da = torch.empty_like(a)
            db = torch.empty_like(b)
            n_elements = a.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            mul_backward_kernel[grid](a, b, dc, da, db, n_elements, BLOCK_SIZE=1024)
            return da, db

Using the Custom Function
-------------------------

Now we can use ``MulFunction`` in a PyTorch training loop. The gradients will be
computed correctly via our custom backward kernel.

.. code-block:: python

    # Example training loop
    a = torch.randn(1024, device='cuda', requires_grad=True)
    b = torch.randn(1024, device='cuda', requires_grad=True)
    target = torch.randn(1024, device='cuda')

    # Forward pass using custom function
    c = MulFunction.apply(a, b)
    loss = (c - target).pow(2).sum()

    # Backward pass
    loss.backward()

    print("Gradient w.r.t a:", a.grad)
    print("Gradient w.r.t b:", b.grad)

Conclusion
----------

This tutorial showed how to manually implement forward and backward kernels in
Triton and wrap them in a ``torch.autograd.Function``. This pattern is essential
for any custom operation that needs gradient support in PyTorch.
