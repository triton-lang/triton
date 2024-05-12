import subprocess
from triton.profiler.viewer import get_min_time_flops, get_min_time_bytes, get_raw_metrics
import numpy as np


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton-viewer", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


def test_min_time_flops():
    with open("example.json", "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_flops(gf.dataframe, device_info)
        device0_idx = gf.dataframe["DeviceId"] == "0"
        device1_idx = gf.dataframe["DeviceId"] == "1"
        # sm89
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[0.000025]], atol=1e-5)
        # sm90
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[0.000030]], atol=1e-5)


def test_min_time_bytes():
    with open("example.json", "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_bytes(gf.dataframe, device_info)
        device0_idx = gf.dataframe["DeviceId"] == "0"
        device1_idx = gf.dataframe["DeviceId"] == "1"
        # sm89
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[9.91969e-06]], atol=1e-6)
        # sm90
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[2.48584e-05]], atol=1e-6)
