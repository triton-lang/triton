import triton.compiler.compiler as compiler
from triton import knobs
from pathlib import Path
import triton


def test_module_load_unload():
    def get_current_target():
        driver = triton.runtime.driver.active
        device = driver.get_current_device()
        device_properties = driver.utils.get_device_properties(device)
        arch = knobs.runtime.override_arch or device_properties['arch']
        warp_size = device_properties['warpSize']
        return compiler.GPUTarget("hip", arch.split(':')[0], warp_size)

    target = get_current_target()
    compiled_kernel = compiler.compile(str(Path(__file__).parent / "add_kernel.ttir"), target, None, None)
    compiled_kernel._init_handles()
    assert compiled_kernel.module is not None
    compiled_kernel.__del__()
    assert compiled_kernel.module is None

