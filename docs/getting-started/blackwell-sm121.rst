
Blackwell (sm_121) Triton Bring-Up Notes
========================================

Problem Description
-------------------
On NVIDIA Blackwell (sm_121) systems (e.g., GB10 / DGX Spark), Triton-lowered kernels
may fail with:

::

    RuntimeError: no kernel image is available for execution on the device

This typically occurs when using Triton with mismatched CUDA toolchain components
or incorrect architecture selection.

Affected Systems
----------------
- NVIDIA Blackwell (sm_121)
- GB10 / DGX Spark / consumer Blackwell hardware
- PyTorch + Triton JIT compilation paths

Observed Behavior
------------------
Failure occurs when:

- Using bundled or outdated PTXAS
- Using incorrect architecture override flags (e.g. sm90 fallback)
- Running Triton kernels compiled for incompatible architecture targets

Environment Matrix
------------------

===============================  ==============================
Configuration                   Result
===============================  ==============================
Bundled PTXAS (pre CUDA 13)     Failure
CUDA 13+ system PTXAS           Works
TRITON_OVERRIDE_ARCH=sm90       Invalid / failure on sm_121
===============================  ==============================

Validated Setup
---------------

The following configuration has been observed to work on sm_121 systems:

::

    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
    export TORCH_CUDA_ARCH_LIST="12.1+PTX"
    unset TRITON_OVERRIDE_ARCH

Notes
-----
- sm_121 requires CUDA 13+ capable PTXAS
- Avoid forcing sm90 architecture overrides
- Behavior may vary depending on PyTorch packaging

Status
------
This document describes observed behavior and validated environment configurations.
It does not modify Triton execution logic.