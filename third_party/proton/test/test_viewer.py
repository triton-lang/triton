import subprocess
from triton.profiler.viewer import get_min_time_flops, get_min_time_bytes


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton-viewer", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


def test_min_time_flops():
    pass
