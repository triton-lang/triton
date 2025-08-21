import logging
import subprocess


def set_compute_frequency(backend: str, freq: int):
    """
    Sets the frequency of the device to the given value in MHz.
    """
    if backend == "nvidia":
        cmd = ["nvidia-smi", "--lock-gpu-clocks", str(freq)]
    elif backend == "amd":
        cmd = ["rocm-smi", "--setperfdeterminism", str(freq)]
    else:
        raise ValueError(f"Unsupported device type: {backend}")
    subprocess.run(cmd, check=True)


def set_memory_frequency(backend: str, freq: int):
    """
    Sets the frequency of the device to the given value in MHz.
    """
    if backend == "nvidia":
        cmd = ["nvidia-smi", "--lock-memory-clocks", str(freq)]
    elif backend == "amd":
        logging.warning("Setting memory frequency not supported for AMD devices.")
        return
        # cmd = ["rocm-smi", "--setmrange", str(freq), str(freq)]
    else:
        raise ValueError(f"Unsupported device type: {backend}")
    subprocess.run(cmd, check=True)


def reset_frequency(backend: str):
    """
    Resets the frequency of the device to the default value.
    """
    if backend == "nvidia":
        cmd = ["nvidia-smi", "--reset-gpu-clocks"]
    elif backend == "amd":
        cmd = ["sudo", "rocm-smi", "--resetperfdeterminism"]
    else:
        raise ValueError(f"Unsupported device type: {backend}")
    subprocess.run(cmd, check=True)
