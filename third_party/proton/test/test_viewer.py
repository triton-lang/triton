import subprocess
from triton.profiler.viewer import get_min_time_flops, get_min_time_bytes, get_raw_metrics


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton-viewer", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


def test_min_time_flops():
    pass


def test_min_time_bytes():
    with open("example.json", "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_bytes(gf.dataframe, device_info)
        print(ret)
