from triton.backends.compiler import GPUTarget
from triton.backends.driver import CPUDriverBase

# ------------------------
# Utils
# ------------------------


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    @staticmethod
    def get_device_properties(device):
        # This is just dummy for now. We will need to implement driver.c.
        return {
            "max_shared_mem": 0,
            "multiprocessor_count": 0,
            "sm_clock_rate": 0,
            "mem_clock_rate": 0,
            "mem_bus_width": 0,
        }

    @staticmethod
    def load_binary(name, kernel_asm, shared, device):
        # This is just dummy for now. We will need to implement driver.c.
        return (None, kernel_asm, 0, 0)


# ------------------------
# Launcher
# ------------------------


def make_launcher(constants, signature, ids):
    pass


class CPULauncher(object):

    def __init__(self, src, metadata):
        # TODO:
        self.launch = lambda *args, **kwargs: None

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDriver(CPUDriverBase):

    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        super().__init__()

    def get_current_target(self):
        # Capability and warp size are zeros for CPU.
        # TODO: GPUTarget naming isn't obviously good.
        return GPUTarget("cpu", 0, 0)

    @staticmethod
    def is_active():
        return True
