import pytest
import subprocess
from triton.profiler.viewer import get_min_time_flops, get_min_time_bytes, get_raw_metrics, format_frames, derive_metrics
import numpy as np

file_path = __file__
cuda_example_file = file_path.replace("test_viewer.py", "example_cuda.json")
hip_example_file = file_path.replace("test_viewer.py", "example_hip.json")
frame_example_file = file_path.replace("test_viewer.py", "example_frame.json")


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton-viewer", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


@pytest.mark.parametrize("option", ["full", "file_function_line", "function_line", "file_function"])
def test_format_frames(option):
    with open(frame_example_file, "r") as f:
        gf, _, _ = get_raw_metrics(f)
        gf = format_frames(gf, option)
        if option == "full":
            idx = gf.dataframe["name"] == "/home/user/projects/example.py/test.py:foo@1"
        elif option == "file_function_line":
            idx = gf.dataframe["name"] == "test.py:foo@1"
        elif option == "function_line":
            idx = gf.dataframe["name"] == "foo@1"
        elif option == "file_function":
            idx = gf.dataframe["name"] == "test.py:foo"
        assert idx.sum() == 1


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


def test_avg_time_derivation():
    metrics = ["avg_time/s", "avg_time/ms", "avg_time/us", "avg_time/ns"]
    with open(cuda_example_file, "r") as f:
        expected_data = {
            'avg_time/s (inc)': [np.nan, 0.0000205, 0.000205], 'avg_time/ms (inc)': [np.nan, 0.02048, 0.2048],
            'avg_time/us (inc)': [np.nan, 20.48, 204.8], 'avg_time/ns (inc)': [np.nan, 20480.0, 204800.0]
        }
        gf, raw_metrics, device_info = get_raw_metrics(f)
        assert len(raw_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        derived_metrics = derive_metrics(gf, metrics, raw_metrics, device_info)
        for derived_metric in derived_metrics:
            np.testing.assert_allclose(gf.dataframe[derived_metric].to_numpy(), expected_data[derived_metric],
                                       atol=1e-6)


def test_util():
    metrics = ["util"]
    with open(cuda_example_file, "r") as f:
        gf, raw_metrics, device_info = get_raw_metrics(f)
        gf = format_frames(gf, format)
        assert len(raw_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        derived_metrics = derive_metrics(gf, metrics, raw_metrics, device_info)
        np.testing.assert_allclose(gf.dataframe[derived_metrics].to_numpy(), [[np.nan], [0.247044], [0.147830]],
                                   atol=1e-6)
