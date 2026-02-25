torch_import_attempts = []
_orig_import = __import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        torch_import_attempts.append(name)
        raise ImportError("torch import is forbidden in this test")
    return _orig_import(name, globals, locals, fromlist, level)


if isinstance(__builtins__, dict):
    __builtins__["__import__"] = _guarded_import
else:
    __builtins__.__import__ = _guarded_import

SIZE = 128


class DevicePtr:

    def __init__(self, ptr, dtype):
        self.ptr = ptr
        self.dtype = dtype

    def data_ptr(self):
        return self.ptr


def _check_no_torch():
    import sys

    if torch_import_attempts:
        raise RuntimeError(f"Unexpected torch import attempts: {torch_import_attempts!r}")
    if "torch" in sys.modules:
        raise RuntimeError("torch module was imported")


def main():
    import struct
    import triton
    import triton.language as tl
    from triton._C.libtriton import nvidia

    device = triton.runtime.driver.active.get_current_device()
    triton.runtime.driver.active.set_current_device(device)

    block_size = tl.constexpr(SIZE)

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr):
        offsets = tl.arange(0, block_size)
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        tl.store(out_ptr + offsets, x + y)

    n_bytes = SIZE * 4

    x = [float(i) for i in range(SIZE)]
    y = [float(2 * i + 1) for i in range(SIZE)]
    expected = tuple(xv + yv for xv, yv in zip(x, y))

    x_ptr = nvidia.device_malloc(n_bytes)
    y_ptr = nvidia.device_malloc(n_bytes)
    out_ptr = nvidia.device_malloc(n_bytes)

    nvidia.copy_host_to_device(x_ptr, struct.pack(f"{SIZE}f", *x))
    nvidia.copy_host_to_device(y_ptr, struct.pack(f"{SIZE}f", *y))

    x_arg = DevicePtr(x_ptr, tl.float32)
    y_arg = DevicePtr(y_ptr, tl.float32)
    out_arg = DevicePtr(out_ptr, tl.float32)

    add_kernel[(1, )](x_arg, y_arg, out_arg)
    nvidia.synchronize()

    out = struct.unpack(f"{SIZE}f", nvidia.copy_device_to_host(out_ptr, n_bytes))
    if out != expected:
        raise RuntimeError(f"Unexpected result: got {out!r}, expected {expected!r}")

    nvidia.device_free(out_ptr)
    nvidia.device_free(y_ptr)
    nvidia.device_free(x_ptr)

    _check_no_torch()


if __name__ == "__main__":
    main()
