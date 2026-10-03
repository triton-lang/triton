# Hash-based buffer-view deduplication benchmark

Buffer-region analysis joins sets of possible descriptor views. Ordered-set
insertion compares their physical footprints lexicographically, walking each
set address when two footprints are equal.

`RegionInfo::ViewList` uses `std::unordered_set` with a hash of inexpensive
descriptor metadata. Full `BufferRegionView` equality resolves collisions,
including exact `SparseBitVector` equality for physical addresses. Distinct
geometries may share a hash but remain distinct views. Bitvector equality
compares machine words rather than iterating over every set address.

Diagnostic views and emitted concurrency-sanitizer cases are sorted at their
output boundaries for deterministic output. Address construction and set
operations continue to use `SparseBitVector`.

The benchmark allocates a 128 KiB shared-memory region replicated across eight
CTAs, obtains two views with runtime indices, and conditionally swaps them 128
times using block-uniform conditions. Both views have identical possible
footprints. Only a 128-byte slice is loaded at the end. This deliberately
amplifies compiler analysis work; it is not a useful workload or a prediction
of typical application speedup.

The default input contract requires `flags[0] == flags[1] == 0`. The driver sets
both indices to zero, fills the remaining flags with booleans, and checks every
output against the expected value 9. The compiler cannot infer the runtime
index values. No lane-varying scalar control flow or inline assembly is used.

## Reproduce

Use a source-build environment with PyTorch and a compatible NVIDIA GPU.
The defaults were tested on GB300. Smaller `--width`, `--steps`, or `--ctas`
values can be used for exploration on other supported devices.

From a clean checkout, save the same script at a fixed path for both builds:

```sh
git show lyu-oai/symbolic-address-sets:examples/compile_time/address_set_equality.py > /tmp/triton-address-set-equality.py

git switch --detach dd7167259f38efd489620fd09d35a53420e2f80b
make
TRITON_DISABLE_LINE_INFO=1 PYTHONPATH="$PWD/python" python /tmp/triton-address-set-equality.py --output /tmp/address-set-before-1
TRITON_DISABLE_LINE_INFO=1 PYTHONPATH="$PWD/python" python /tmp/triton-address-set-equality.py --output /tmp/address-set-before-2

git switch lyu-oai/symbolic-address-sets
make
TRITON_DISABLE_LINE_INFO=1 PYTHONPATH="$PWD/python" python /tmp/triton-address-set-equality.py --output /tmp/address-set-after-1
TRITON_DISABLE_LINE_INFO=1 PYTHONPATH="$PWD/python" python /tmp/triton-address-set-equality.py --output /tmp/address-set-after-2

cmp /tmp/address-set-before-1/kernel.cubin /tmp/address-set-after-1/kernel.cubin
cmp /tmp/address-set-before-2/kernel.cubin /tmp/address-set-after-2/kernel.cubin
```

Each invocation creates a fresh temporary cache and measures the complete JIT
`warmup` call, including frontend compilation and assembly. Input allocation,
driver initialization, GPU execution, and artifact export are outside the timer.
The script executes the compiled kernel afterward, checks its outputs, and
writes `result.json` and the compiler-stage artifacts into the requested output
directory. Keep the script at the same absolute path when comparing artifacts
so source-location differences do not obscure the comparison.

## Measurements

Linux AArch64, NVIDIA GB300, LLVM `b010a18d2b648cab83c83967ff26b8fde11acdc6`.
Master below denotes the unpatched upstream `main` revision
`dd7167259f38efd489620fd09d35a53420e2f80b`. Both compilers use the same
Python sources, fresh caches, and disabled line information. Times are seconds
of compilation, not GPU execution.

| Workload | Master | New | Speedup |
| --- | ---: | ---: | ---: |
| Synthetic | 70.308 | 1.453 | 48.40x |
| Overlapping accumulator | 2.325 | 0.663 | 3.51x |
| Multi-CTA tutorial | 1.477 | 0.624 | 2.37x |
| Attention | 6.531 | 3.707 | 1.76x |
| Broad tutorial/example sweep total | 72.545 | 50.545 | 1.44x |

The individual rows average two cold runs. The synthetic runs took
70.141 and 70.475 seconds on master, and 1.450 and 1.455 seconds on the
candidate. The broad sweep is one run per compiler, summing 135 kernel
compilations from 142 sampled tests across the Gluon tutorials and examples
and the standard fused attention tutorial. It is a sampled suite, not the
full tutorial parameter grid.

The focused tutorial/example cases were:

- `python/examples/gluon/06-overlapping-accumulator.py::test_mma_scaled_overlap_accumulator[True-mxfp8-mxfp8]`
- `python/tutorials/gluon/14-multicta.py::test_matmul_multicta`
- `python/examples/gluon/01-attention-forward.py::test_op[4ctas-True-dtype2-True-128-8192-32-4]`

All compared GPU binaries were byte-identical; the synthetic case also produced
identical source IR, Gluon IR, Triton GPU IR, LLVM IR, and PTX. Collision coverage
constructs 32 distinct footprints with the same metadata hash and checks exact
membership, insertion-order independence, and idempotent joins. Existing
barrier, alias, padding, and tensor-memory tests check the analysis consumers.

Metadata-only hashing can put distinct geometries in the same bucket, where
full equality determines membership. It does not guarantee constant-time
lookups, and footprint construction and copying still have a cost. The
synthetic intentionally amplifies repeated equal-footprint comparisons and
should not be treated as a typical application-wide speedup.
