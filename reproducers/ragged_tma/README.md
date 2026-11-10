# Blackwell ragged TMA allocation-mode reproducer

Run from the repository root on one NVIDIA Blackwell GPU:

```bash
python3 reproducers/ragged_tma/repro.py
```

Requirements: Linux, Python 3.9+, and a working NVIDIA CUDA driver with its PTX JIT compiler. The script uses only Python's standard library and `libcuda.so.1`. No Torch, Triton installation/build, CUDA toolkit, `ptxas`, checkpoint, or multi-GPU launch is needed. The PTX targets generic `sm_90` and is JIT-compiled by the driver for the current GPU, avoiding a GB200-specific cubin. The descriptor-bit diagnostic is intentionally limited to Blackwell (SM 10.x, 11.x, 12.x).

The default command launches four fresh child processes. This is essential because an illegal address poisons the CUDA context.

| Case | Descriptor allocation mode | Accessible backing from tensor base | Expected on the original affected setup |
| --- | --- | --- | --- |
| `original` | Driver default, reported in output | 64 KiB | LARGE, then illegal address |
| `small` | Force SMALL | 64 KiB | Correct output |
| `large` | Force LARGE | 64 KiB | Illegal address |
| `backed` | Driver default, reported in output | 128 KiB | Correct output |

A confirmation prints `CONFIRMED` and exits 0, even though the two expected failing child processes exit nonzero. `FORCED-ONLY` or `NOT REPRODUCED` exits 2. Setup errors, failed controls, or unexpected outcomes print `INCONCLUSIVE` and exit 1. A forced-LARGE failure by itself does **not** establish a driver default-selection bug.

Select a CUDA-visible GPU with `--device 1`. For one case only:

```bash
python3 reproducers/ragged_tma/repro.py --case original
python3 reproducers/ragged_tma/repro.py --case small
```

The single-case command exits 1 on an illegal address. A CUDA exception during PTX compilation or allocation is reported as a setup error, not counted as reproducing the TMA fault.

## Exact load

- Physical input: uint8 `[1, 32, 2048]`, byte strides `[0, 2048, 1]`, 64 KiB.
- One live row at slice offset 0; output is `[128, 128]` bytes, one valid row followed by 127 zero-filled rows.
- Driver-order TMA dimensions: `[2048, 2**30, 1, 0x7fff0000, 0x7fff0000]`.
- Driver-order byte strides: `[2048, 0, 2048, 2**34 - 2048]`.
- TMA box: `[128, 128, 1, 1, 1]`.
- Load coordinates: `[0, 2**30 - 1, 0, 1, 2**30]`.
- One CTA of 32 threads. One thread issues the 5D TMA load; an mbarrier and CTA barrier precede output reads.
- No matmul or FP4 arithmetic is involved. The input bytes model the packed activation storage from the failing NVFP4 matmul.

The input is positioned immediately before an **unmapped CUDA VMM guard**. It cannot accidentally borrow accessible padding from a caching allocator. Simply allocating a 64 KiB tensor with a normal allocator does not reproduce this condition reliably.

The original matmul used M capacity 32, one live row, N=5120, K=4096, NVFP4 A/B with microblock 16, BF16 output, and a persistent 128x256x256 tile with 8 warps and 2 stages. Those matmul settings are unnecessary for this isolated load.

## Diagnosing a non-reproduction

Share the complete output: it reports GPU, compute capability, CUDA driver API version, actual loaded `libcuda` paths, allocation class, tensor base, unmapped guard, and stage of any error. The mode is read from Blackwell descriptor bit 85, rather than requiring PTX 9.5's new query instruction. The host driver version from `nvidia-smi` can differ from the compatibility `libcuda` loaded in a container.

The current fix PR intentionally requests SMALL mode; running its regression test unchanged should pass. This standalone program neither imports that helper nor depends on which Triton package is installed.

The original mode-isolation experiment was confirmed on GB200 with host driver 580.178.04 and compatibility `libcuda` 610.43.02 (driver API 13050). Other Blackwell models and driver combinations are not assumed to reproduce automatically.

## Validation of this standalone command

Confirmed on a GB200 using host driver 580.178.04 and loaded compatibility library `/usr/local/cuda-13.4/compat/libcuda.so.620.07` (CUDA driver API 13050). The unmodified command printed `CONFIRMED`: original/LARGE and forced-LARGE both raised CUDA error 700, while SMALL and 128 KiB backing both passed the byte-for-byte output check. See [the complete example output](example-output.txt).

## References

- [Fix PR](https://github.com/triton-lang/triton/pull/12700)
- [NVIDIA's documented Blackwell descriptor workaround](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-595-91-07/index.html#known-issues)
- [PTX out-of-boundary access requirements](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode-oob-access)

The diagnostic clears or sets bit 85 only on Blackwell. Its layout is architecture-specific; do not extend the architecture guard to future GPUs without checking the descriptor format.
