import subprocess


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton-viewer", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0
