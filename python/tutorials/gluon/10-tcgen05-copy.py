"""
TCGen05 Copy Instruction
========================

This tutorial will cover the `tcgen05_copy` instruction: how to use it and its
applications.

The `tcgen05_copy` instruction is an asynchronous tensorcore operation that
copies data from shared memory to tensor memory. `tcgen05_copy` is implicitly
pipelined with `tcgen05_mma` and `tcgen05_commit`. The completion of
`tcgen05_copy` is tracked with `tcgen05_commit` on an mbarrier just like
`tcgen05_mma`.

`tcgen05_copy` can be used to copy data into tensor memory that is fed into a
`tcgen05_mma` instruction. Because `tcgen05_copy` is implicitly pipelined with
`tcgen05_mma`, even though it is asynchronous, the MMA is guaranteed to start
after the copy is complete:

```python
tcgen05_copy(smem, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
tcgen05_commit(bar)
mbarrier.wait(bar, phase=phase)
```

The completion of a single or multiple `tcgen05_copy` operations can be tracked
with `tcgen05_commit`:

```python
tcgen05_copy(lhs_smem, lhs_tmem)
tcgen05_copy(acc_smem, acc_tmem)
tcgen05_commit(bar)
mbarrier.wait(bar, phase=phase)
acc = acc_tmem.load(acc_reg_layout)
lhs = lhs_tmem.load(lhs_reg_layout)
```

The implicit pipelining is because the PTX-level `tcgen05.copy` and `tcgen05.mma`
instructions are executed by the tensor core pipe on the SM, which you can think
of as a single thread running tensor core specific instructions on the SM,
asynchronously from the rest of the SM.

The following is also valid.

```python
tcgen05_copy(lhs_smem0, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
tcgen05_commit(bar)

tcgen05_copy(lhs_smem1, lhs_tmem)
tcgen05_mma(lhs_tmem, rhs_smem, acc_tmem)
```

Because the second `tcgen05_copy` will only execute after the preceeding
`tcgen05_mma` is complete,

`tcgen05_copy` accesses shared memory via the async proxy, just like `tcgen05_mma`.
Make sure to insert fences as appropriate:

```python
lhs_smem.store(value1)
fence_async_shared()
tcgen05_copy(lhs_smem, lhs_tmem)
tcgen05_commit(bar)

mbarrier.wait(bar, phase=phase)
lhs_smem.store(value0)
```

Note that a fence is not needed between `tcgen05_copy` and the second write to
`lhs_smem` because waiting on the completion of the `tcgen05_copy` operation
via the mbarrier implicitly fences the generic and async proxies.

What makes using `tcgen05_copy` particularly tricky is selecting the right
shared memory and tensor memory layouts, as `tcgen05_copy` only supports a
limited set of instruction shapes for copy data from shared to tensor memory.
"""

import triton
import torch


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")
