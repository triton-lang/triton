# Equal shared-memory footprint compilation benchmark

`AddressSet::operator<` compares physical addresses lexicographically. When two
large footprints are equal, walking the set-bit iterator visits every address.
Buffer-region dataflow joins can repeat this comparison many times while
inserting views into ordered sets. The equality fast path compares the packed
sparse-bitvector blocks first and avoids those element-by-element walks.

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
git show codex/address-set-equality-fast-path:examples/compile_time/address_set_equality.py > /tmp/triton-address-set-equality.py

git switch --detach dd7167259f38efd489620fd09d35a53420e2f80b
make
PYTHONPATH="$PWD/python" python /tmp/triton-address-set-equality.py --output /tmp/address-set-before-1
PYTHONPATH="$PWD/python" python /tmp/triton-address-set-equality.py --output /tmp/address-set-before-2

git switch codex/address-set-equality-fast-path
make
PYTHONPATH="$PWD/python" python /tmp/triton-address-set-equality.py --output /tmp/address-set-after-1
PYTHONPATH="$PWD/python" python /tmp/triton-address-set-equality.py --output /tmp/address-set-after-2

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
The native compiler baseline matches `dd7167259f38efd489620fd09d35a53420e2f80b`;
the candidate adds only the `AddressSet::operator<` equality fast path.
Both use the same Python sources and the default benchmark parameters.

| Run | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 70.288 s | 2.334 s | 30.11x |
| 2 | 70.463 s | 2.339 s | 30.13x |

All four runs executed the kernel and checked every output. The source IR,
Gluon IR, Triton GPU IR, LLVM IR, PTX, and cubin were byte-identical within
each before/after pair. These are cold compilation measurements of this
synthetic case, not GPU execution speedups.
