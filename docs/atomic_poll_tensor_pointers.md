# Tensor-pointer `atomic_poll`

## Summary

Extend `tl.atomic_poll` to accept a tensor of pointers and a scalar or
same-shaped tensor of expected values.  Each unique logical pointer element
polls independently, and the result has the corresponding `int1` shape.

## Scalar compatibility

The scalar form retains its current behavior: one elected thread polls the
pointer, and the completion result is made available to the participating
threads.  This proposal only adds a path for non-scalar pointer tensors.

## Tensor semantics

For a non-scalar pointer tensor, every unique logical element repeatedly
loads its corresponding pointer until the expected value is observed.  A
successful elementwise poll applies the requested acquire semantics to that
lane.  Replicated physical instances of one logical element execute once and
receive the same result.

The tensor form does not add a CTA-wide rendezvous.  Callers that need a
warp- or CTA-wide rendezvous may use an explicit synchronization operation
after the poll.

## Initial scope

The initial implementation should support `acquire` and `relaxed` semantics,
the existing 16-, 32-, and 64-bit integer element types, and no timeout for
tensor pointers.  Scalar timeout behavior remains unchanged.  Tensor timeout
and masked polling can be added once their result and synchronization
semantics are settled.

## Example

```python
offsets = tl.arange(0, 32)
matched = tl.atomic_poll(flags + offsets, 1, sem="acquire", scope="gpu")
```

Each unique pointer in `flags + offsets` is polled independently.  The
caller may add an explicit synchronization operation separately from the
memory-ordering effect of the acquire poll.
