# Proton Documentation

This directory contains the long-form Proton documentation. The top-level
[`README.md`](../README.md) stays focused on installation and a minimal quick
start.

## Guides

- [Python profiling API](python-api.md): sessions, decorators, scopes, states,
  CPU timed scopes, and custom metrics.
- [Backends and modes](backends-and-modes.md): backend selection, NVIDIA and
  AMD support, instruction sampling, periodic flushing, and instrumentation
  modes.
- [Periodic profiling](periodic-profiling.md): `periodic_flushing`, phase
  advancement, partial output files, and in-memory phase APIs.
- [Intra-kernel profiling](intra-kernel.md): Proton DSL instrumentation and
  TTGIR override workflows.
- [Command line and viewer](cli-and-viewer.md): `proton`, `proton-viewer`,
  metrics, filters, sorted output, trace visualization, and diff profiles.
- [Advanced features](advanced.md): CUDA graphs, knobs, thread safety, known
  issues, and third-party backend registration.

## Runnable Examples

- [`../tutorials/dynamic-net.py`](../tutorials/dynamic-net.py): PyTorch model
  profiling with scopes.
- [`../tutorials/matmul.py`](../tutorials/matmul.py): Triton matmul profiling,
  launch metadata, and instruction sampling.
- [`../tutorials/intra_kernel/example_dsl.py`](../tutorials/intra_kernel/example_dsl.py):
  DSL-level intra-kernel instrumentation.
- [`../tutorials/intra_kernel/example_override.py`](../tutorials/intra_kernel/example_override.py):
  TTGIR override instrumentation.
