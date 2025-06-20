import os
import torch
import pytest
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma
from triton._internal_testing import is_cuda
import multiprocessing
import tempfile

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass  # start method already set


class ProcessResult:

    def __init__(self, exc, driver_stderr_output):
        self.exc = exc
        self.driver_stderr_output = driver_stderr_output


def target(client_fn, queue, *args):
    # Prepare temp file for capturing low-level stderr
    with tempfile.TemporaryFile(mode="w+") as tmp_stderr:
        saved_stderr_fd = os.dup(2)
        os.dup2(tmp_stderr.fileno(), 2)  # Redirect fd 2 to tmp_stderr
        exc = None

        try:
            client_fn(*args)
        except Exception as e:
            exc = e
        finally:
            # Restore original stderr
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)

            # Read driver stderr
            tmp_stderr.seek(0)
            driver_stderr_output = tmp_stderr.read()
            queue.put(ProcessResult(exc, driver_stderr_output))


def run_in_process(client_fn, args):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(client_fn, queue, *args))
    p.start()
    p.join()
    result = queue.get()
    return result


@gluon.jit
def async_tma_kernel(input_desc, XBLOCK: ttgl.constexpr, FAILURE: ttgl.constexpr):
    smem = ttgl.allocate_shared_memory(ttgl.float32, [XBLOCK, XBLOCK], input_desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    mbarrier.expect(bar, XBLOCK * XBLOCK * ttgl.float32.primitive_bitwidth // 8)
    tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem)
    if FAILURE:
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem)
    mbarrier.wait(bar, 0)

    if not FAILURE:
        tma.async_copy_global_to_shared(input_desc, [0, 0], bar, smem)
        mbarrier.wait(bar, 0)

    mbarrier.invalidate(bar)

    tma.async_copy_shared_to_global(input_desc, [0, 0], smem)
    tma.store_wait(0)


def run_async_tma_kernel(FAILURE, device):
    knobs.compilation.enable_experimental_consan = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    input = torch.randn((1024, 1024), device=device, dtype=torch.float32)
    XBLOCK = 128
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)
    input_desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(input, [XBLOCK, XBLOCK], shared_layout)

    async_tma_kernel[(1, )](input_desc, XBLOCK, FAILURE=FAILURE, num_warps=1)
    getattr(torch, device).synchronize()


@pytest.mark.skipif(not is_cuda() or torch.cuda.get_device_capability()[0] < 9, reason="Requires hopper or newer")
@pytest.mark.parametrize("FAILURE", [True, False])
def test_async_tma_kernel(FAILURE, device):
    if FAILURE:
        result = run_in_process(run_async_tma_kernel, (FAILURE, device))
        assert "device-side assert" in str(result.exc)
        assert "TMA copy to buffer being read or written" in result.driver_stderr_output
    else:
        run_async_tma_kernel(FAILURE, device)
