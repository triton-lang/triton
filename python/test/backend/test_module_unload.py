import triton.compiler.compiler as compiler
from triton import knobs
from pathlib import Path
import triton
from triton._internal_testing import is_hip


def test_module_load_unload():
    def get_current_target():
        driver = triton.runtime.driver.active
        device = driver.get_current_device()
        backend = 'hip' if is_hip() else 'cuda'
        warp_size = device_properties['warpSize'] if is_hip() else 32
        if is_hip():
            device_properties = driver.utils.get_device_properties(device)
            arch = knobs.runtime.override_arch or device_properties['arch']
            capability = arch.split(':')[0]
        else:
            capability = driver.get_device_capability(device)
            capability = capability[0] * 10 + capability[1]

        return compiler.GPUTarget(backend, capability, warp_size)

    target = get_current_target()
    compiled_kernel = compiler.compile(str(Path(__file__).parent / "add_kernel.ttir"), target, None, None)
    compiled_kernel._init_handles()
    print(f"module = {compiled_kernel.module}")
    assert compiled_kernel.module is not None
    compiled_kernel.__del__()
    assert compiled_kernel.module is None

