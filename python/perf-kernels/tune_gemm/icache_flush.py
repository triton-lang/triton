import ctypes
import array
import random
import math

# the hip module can be installed as
# `python3 -m pip install -i https://test.pypi.org/simple hip-python~=$rocm_version`
# more information about hip-python is at: https://github.com/ROCm/hip-python
from hip import hip, hiprtc

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]

    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
        ):
        raise RuntimeError(str(err))

    return result

# S_ICACHE_INV Invalidate entire first level instruction cache.
# There must be 16 separate S_NOP instructions or a jump/branch instruction
# after this instruction to ensure the internal instruction buffers are also
# invalidated.
def gen_kernel():
    source = b"""\
        extern "C" __global__ void icache_flush_kernel() {
              asm __volatile__("s_icache_inv");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
              asm __volatile__("s_nop 0");
        }
    """

    # print(f"source = {source}")
    prog = hip_check(hiprtc.hiprtcCreateProgram(source, b"icache_flush_kernel", 0, [], []))
    progs = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(progs, 0))
    arch = progs.gcnArchName
    cflags = [b"--offload-arch="+arch]
    err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
    if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
        log = bytearray(log_size)
        hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
        print(f"log = {log.decode()}, err = {err}")
        raise RuntimeError(log.decode())

    code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
    code = bytearray(code_size)
    hip_check(hiprtc.hiprtcGetCode(prog, code))
    module = hip_check(hip.hipModuleLoadData(code))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"icache_flush_kernel"))

    return kernel

kernel = gen_kernel()
progs = hip.hipDeviceProp_t()
hip_check(hip.hipGetDeviceProperties(progs, 0))
cu_num = progs.multiProcessorCount

def icache_flush():
    block = hip.dim3(x=64)
    grid = hip.dim3(cu_num * 60)

    hip_check(hip.hipModuleLaunchKernel(
        kernel,
        *grid,
        *block,
        sharedMemBytes=0,
        stream=None,
        kernelParams=None,
        extra=()
        )
    )
