"""Test the compilation steps but do not require a GPU to run.
"""

import os
from pathlib import Path
import functools
import torch
import triton.language as tl
import triton
import triton.common.backend as tcb
from triton.runtime.driver import DriverBase
import sys
import unittest
from unittest import mock
from triton.compiler.compiler import parse_mlir_module, optimize_ttgir, ttir_to_ttgir, CudaTargetDescriptor, ClusterInfo, ttgir_to_llir, TMAInfos, ir


class MockCudaDesc(CudaTargetDescriptor, dict):
    def __init__(self, *, num_stages, **kw):
        super().__init__(**kw)
        self['num_warps'] = kw['num_warps']
        self['num_stages'] = num_stages

class MockGpuBackend(tcb.BaseBackend):
    def __init__(self, device_type: str) -> None:
        super(MockGpuBackend, self).__init__(device_type)
        self.driver = DriverBase()
        self.version_key = None

    def add_stages(self, arch, extern_libs, stages):
        enable_persistent = False
        enable_warp_specialization = False
        optimize_epilogue = False
        target = self.get_architecture_descriptor()
        num_warps = target.num_warps
        num_stages = 3
        num_ctas = 1
        stages["ttgir"] = (lambda path: parse_mlir_module(path, ir.context()),
                           lambda src: optimize_ttgir(
                               ttir_to_ttgir(src, num_warps, num_ctas, target),
                               num_stages, num_warps, num_ctas, target, ClusterInfo(),
                               enable_warp_specialization, enable_persistent, optimize_epilogue))
        stages["llir"] = (lambda path: Path(path).read_text(),
                          lambda src: ttgir_to_llir(src, extern_libs, target, TMAInfos()))

    def add_meta_info(self, ir, cur_module, next_module, metadata, asm):
        metadata["name"] = "extension_backend_name"

    def get_driver(self):
        return self.driver

    def get_stream(self):
        return ""

    @functools.lru_cache(None)
    def get_device_properties(self, device):
        return self.driver.utils.get_device_properties()

    def get_current_device(self):
        return torch.device("cpu")

    def set_current_device(self, device):
        pass

    def get_load_binary_fn(self):
        return self.driver.utils.load_binary

    def get_kernel_bin(self):
        return "ttgir"

    def get_architecture_descriptor(self, **kwargs):
        return MockCudaDesc(capability=80, num_stages=3, num_warps=4, enable_fp_fusion=False)

    def get_version_key(self):
        return "vesion_key"

    def make_launcher_stub(self, name, signature, constants, ids):
        return '/tmp/test.py'

def get_kernel_asm(kernel, *args, **kw):
    """Compile the kernel but don't run it, return the cached IRs.

    Useful for testing without a local GPU.
    """

    tcb.register_backend('mock_gpu', MockGpuBackend)
    kw['device_type'] = 'mock_gpu'
    kw['num_stages'] = 3
    kw['warmup'] = True
    with mock.patch.dict('sys.modules', {'importlib': mock.MagicMock()}):
        pgm = kernel[(1,)](*args, **kw)

    return pgm.asm


def test_pipelined_load():
    """Test the pipeline argument.

    A lonely load in a for{} will certainly not be matched for
    pipelining, but if we explicitly request pipelining it should be
    pipelined.

    """

    inp = torch.rand((100,), dtype=torch.float32)

    @triton.jit
    def _kernel(in_ptr, pipeline: tl.constexpr):
        off = tl.arange(0, 10)
        for i in range(10):
            piped = tl.load(in_ptr + off + i, pipeline=pipeline)

    # TODO(cperivol): test that a load() that would otherwse be
    # pipelined is actually not if we tell it pipeline=False.

    # Return true if the kernel was pipelined
    check_papielined = lambda kern_asm: "triton_gpu.async_wait" in kern_asm['ttgir']

    asm_true = get_kernel_asm(_kernel, inp, True)
    asm_false = get_kernel_asm(_kernel, inp, False)
    asm_none = get_kernel_asm(_kernel, inp, None)

    assert asm_none['ttir'] == asm_false['ttir']

    # With true there is pipelinin
    assert check_pipelined(asm_true)
    # With false or none there is not pipelining
    assert not check_pipelined(asm_false)
    assert not check_pipelined(asm_none)
