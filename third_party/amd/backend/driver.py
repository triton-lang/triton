import ctypes

import torch

from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
from triton.backends.amd import hip


def check(status):
    if status != 0:
        raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")


def get_pointer(a):
    if a is None:
        return None
    attributes = hip.hipPointerAttribute_t()
    data_ptr = hip.hipDeviceptr_t(a.data_ptr())
    check(hip.hipPointerGetAttributes(ctypes.byref(attributes), data_ptr))
    return hip.hipDeviceptr_t(attributes.devicePointer)


def ty_to_ctype(ty: str):
    if ty.startswith("*"):
        return hip.hipDeviceptr_t
    elif ty == "i1":
        return ctypes.c_int32
    elif ty == "i8":
        return ctypes.c_int8
    elif ty == "i16":
        return ctypes.c_int16
    elif ty == "i32":
        return ctypes.c_int32
    elif ty == "i64":
        return ctypes.c_int64
    elif ty == "u1":
        return ctypes.c_uint32
    elif ty == "u8":
        return ctypes.c_uint8
    elif ty == "u16":
        return ctypes.c_uint16
    elif ty == "u32":
        return ctypes.c_uint32
    elif ty == "u64":
        return ctypes.c_uint64
    elif ty == "fp32":
        return ctypes.c_float
    elif ty == "fp64":
        return ctypes.c_double
    # see https://github.com/llvm/llvm-project/blob/main/mlir/python/mlir/runtime/np_to_memref.py#L17-L43
    elif ty == "fp16":
        return ctypes.c_int16
    elif ty == "bf16":
        return ctypes.c_int16
    else:
        raise NotImplementedError(f"{ty=} not supported")


class HIPLauncher:

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        from triton.runtime.jit import TensorWrapper

        def launch(*args, **kwargs):
            assert not kwargs
            gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, *args = args
            num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ = kernel_metadata
            if launch_enter_hook or launch_exit_hook:
                raise NotImplementedError

            params = [(args[i], ty) for i, ty in signature.items() if i not in constants]
            for i, (p, ty) in enumerate(params):
                if isinstance(p, torch.Tensor):
                    params[i] = get_pointer(p)
                elif isinstance(p, TensorWrapper):
                    params[i] = get_pointer(p.data)
                elif isinstance(p, (int, float)):
                    params[i] = ty_to_ctype(ty)(p)
                else:
                    raise NotImplementedError(f"{p=} not supported with {ty=}")

            global_scratch = hip.hipDeviceptr_t()
            addresses = [ctypes.addressof(p) for p in params] + [ctypes.addressof(global_scratch)]
            c_args = (ctypes.c_void_p * len(addresses))(*addresses)
            function = ctypes.cast(function, hip.hipFunction_t)
            stream = ctypes.cast(stream, hip.hipStream_t)
            check(
                hip.hipModuleLaunchKernel(function, gridX, gridY, gridZ, metadata.warp_size * num_warps, 1, 1,
                                          shared_memory, stream, c_args, None))

        self.launch = launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class HIPUtils:

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HIPUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def get_device_properties(self, device):
        props = hip.hipDeviceProp_t()
        check(hip.hipGetDeviceProperties(ctypes.byref(props), device))
        return {
            "max_shared_mem": props.sharedMemPerBlock, "max_num_regs": props.regsPerBlock, "multiprocessor_count":
            props.multiProcessorCount, "sm_clock_rate": props.clockRate, "mem_clock_rate": props.memoryClockRate,
            "mem_bus_width": props.memoryBusWidth, "arch": props.gcnArchName.decode(), "warpSize": props.warpSize,
            "max_threads_per_sm": props.maxThreadsPerMultiProcessor
        }

    def load_binary(self, name, data, *_args):
        opt = [
            hip.hipJitOptionErrorLogBufferSizeBytes, hip.hipJitOptionErrorLogBuffer,
            hip.hipJitOptionInfoLogBufferSizeBytes, hip.hipJitOptionInfoLogBuffer, hip.hipJitOptionLogVerbose
        ]
        opt = (hip.hipJitOption * len(opt))(*opt)
        err_buf_size = 8192
        log_buf_size = 8192
        _err = ctypes.create_string_buffer(err_buf_size)
        _log = ctypes.create_string_buffer(log_buf_size)
        err_buf_size = ctypes.c_uint32(err_buf_size)
        log_buf_size = ctypes.c_uint32(log_buf_size)
        one = ctypes.c_uint32(1)

        optval = (ctypes.c_void_p * 5)(ctypes.addressof(err_buf_size), ctypes.addressof(_err),
                                       ctypes.addressof(log_buf_size), ctypes.addressof(_log), ctypes.addressof(one))

        mod = hip.hipModule_t()
        fun = hip.hipFunction_t()
        check(hip.hipModuleLoadDataEx(mod, data, 5, opt, optval))
        check(hip.hipModuleGetFunction(fun, mod, name.encode("utf-8")))

        n_regs = ctypes.c_int32(0)
        n_spills = ctypes.c_int32(0)
        check(hip.hipFuncGetAttribute(n_regs, hip.HIP_FUNC_ATTRIBUTE_NUM_REGS, fun))
        check(hip.hipFuncGetAttribute(n_spills, hip.HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun))

        return (mod, fun, n_regs.value, n_spills.value // 4)


class HIPDriver(GPUDriver):

    def __init__(self):
        super().__init__()
        self.utils = HIPUtils()
        self.launcher_cls = HIPLauncher

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        import torch
        return torch.version.hip is not None

    def get_current_target(self):
        device = self.get_current_device()
        device_properties = self.utils.get_device_properties(device)
        arch = device_properties['arch']
        warp_size = device_properties['warpSize']
        return GPUTarget("hip", arch.split(':')[0], warp_size)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # It's the same as the Nvidia backend.
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
