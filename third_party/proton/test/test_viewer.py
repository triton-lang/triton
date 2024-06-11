import subprocess
from triton.profiler.viewer import get_min_time_flops, get_min_time_bytes, get_raw_metrics
import numpy as np

file_path = __file__
cuda_example_file = file_path.replace("test_viewer.py", "example_cuda.json")
hip_example_file = file_path.replace("test_viewer.py", "example_hip.json")


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton-viewer", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


def test_min_time_flops():
    with open(cuda_example_file, "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_flops(gf.dataframe, device_info)
        device0_idx = gf.dataframe["DeviceId"] == "0"
        device1_idx = gf.dataframe["DeviceId"] == "1"
        # sm89
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[0.000025]], atol=1e-5)
        # sm90
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[0.00005]], atol=1e-5)
    with open(hip_example_file, "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_flops(gf.dataframe, device_info)
        device0_idx = gf.dataframe["DeviceId"] == "0"
        device1_idx = gf.dataframe["DeviceId"] == "1"
        # MI200
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[0.000026]], atol=1e-5)
        # MI300
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[0.000038]], atol=1e-5)


def test_min_time_bytes():
    with open(cuda_example_file, "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_bytes(gf.dataframe, device_info)
        device0_idx = gf.dataframe["DeviceId"] == "0"
        device1_idx = gf.dataframe["DeviceId"] == "1"
        # sm89
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[9.91969e-06]], atol=1e-6)
        # sm90
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[2.48584e-05]], atol=1e-6)
    with open(hip_example_file, "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_bytes(gf.dataframe, device_info)
        device0_idx = gf.dataframe["DeviceId"] == "0"
        device1_idx = gf.dataframe["DeviceId"] == "1"
        # MI200
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[6.10351e-06]], atol=1e-6)
        # MI300
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[1.93378e-05]], atol=1e-6)
