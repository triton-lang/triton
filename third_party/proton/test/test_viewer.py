import triton.profiler as proton
import subprocess


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton-viewer", "-h"])
    assert ret == 0
