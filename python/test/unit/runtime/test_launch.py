import gc
import importlib
import os
import sys
import textwrap
import time
import tracemalloc
from typing import Tuple

import torch

import triton
import triton.language as tl


def test_memory_leak() -> None:

    @triton.jit
    def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
        xnumel = 10
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)

    tracemalloc.start()
    try:
        inp = torch.randn(10, device='cuda')
        out = torch.randn(10, device='cuda')
        kernel[(10,)](inp, out, 10, XBLOCK=16)
        gc.collect()
        begin, _ = tracemalloc.get_traced_memory()
        for _ in range(100):
            kernel[(10,)](inp, out, 10, XBLOCK=16)
        gc.collect()
        end, _ = tracemalloc.get_traced_memory()
        assert end - begin < 1000
    finally:
        tracemalloc.stop()


def test_kernel_launch_latency() -> None:
    def define_empty_kernel(file_path, num_tensor_args):
        arg_str = ",".join([f"arg{i}: torch.Tensor" for i in range(num_tensor_args)])
        arg_str += ", n_elements: int, BLOCK_SIZE: tl.constexpr"
        func_str = f"""
        import torch

        import triton
        import triton.language as tl

        @triton.jit
        def empty_kernel({arg_str}):
            pass
        """
        with open(file_path, "w") as f:
            f.write(textwrap.dedent(func_str))

    def import_empty_kernel(file_path):
        directory, filename = os.path.split(file_path)
        module_name, _ = os.path.splitext(filename)
        sys.path.insert(0, directory)

        module = importlib.import_module(module_name)
        empty_kernel = module.empty_kernel
        return empty_kernel

    def empty(*kernel_args: Tuple[torch.Tensor]):
        first_arg = kernel_args[0]
        n_elements = first_arg.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        # Warmup
        empty_kernel[grid](*kernel_args, n_elements, BLOCK_SIZE=1024, device=torch.cuda.current_device())
        torch.cuda.synchronize()
        # Measure launch overhead at steady state
        num_runs = 1000
        latency = []
        for i in range(num_runs):
            single_start_time = time.time()
            empty_kernel[grid](*kernel_args, n_elements, BLOCK_SIZE=1024, device=torch.cuda.current_device())
            single_end_time = time.time()
            latency.append((single_end_time - single_start_time) * 1e6)

        latency.sort()
        mid = len(latency) // 2
        median = (latency[mid] + latency[~mid]) / 2
        # print(f"Median latency (usec) = {median}")
        assert median < 35, "Kernel launch time has increased!"

    # Define a jitted empty_kernel in /tmp and import
    file_path = '/tmp/empty_kernel.py'
    num_tensor_args = 40
    define_empty_kernel(file_path, num_tensor_args)
    empty_kernel = import_empty_kernel(file_path)

    # Initialize random tensors for the empty_kernel
    torch.manual_seed(0)
    size = 1024
    kernel_args = (torch.rand(size, device='cuda') for i in range(num_tensor_args))

    # Run empty, which would run empty_kernel internally
    empty(*kernel_args)
