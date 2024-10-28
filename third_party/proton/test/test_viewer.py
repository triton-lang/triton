import pytest
import subprocess
from triton.profiler.viewer import get_min_time_flops, get_min_time_bytes, get_raw_metrics, format_frames, derive_metrics, filter_frames
import numpy as np

file_path = __file__
cuda_example_file = file_path.replace("test_viewer.py", "example_cuda.json")
hip_example_file = file_path.replace("test_viewer.py", "example_hip.json")
frame_example_file = file_path.replace("test_viewer.py", "example_frame.json")
leaf_example_file = file_path.replace("test_viewer.py", "example_leaf_nodes.json")


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton-viewer", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


def test_sort():
    with open(leaf_example_file, "r") as f:
        gf, raw_metrics, device_info = get_raw_metrics(f)
        gf = format_frames(gf, None)
        gf.update_inclusive_columns()
        metrics = ["time/s", "time/ms", "time/us", "time/ns"]
        metrics = derive_metrics(gf, metrics, raw_metrics, device_info)
        gf = filter_frames(gf, None, None, None, metrics[0])
        sorted_df = gf.dataframe.sort_values(by=[metrics[0]], ascending=False)
        actual = sorted_df.iloc[0:8]['name'].values
        expected = [
            'ROOT', 'matmul_1152_1152_1152', 'matmul_1024_1024_1024', 'matmul_896_896_896', 'matmul_768_768_768',
            'matmul_640_640_640', 'matmul_512_512_512', 'matmul_384_384_384'
        ]
        assert len(actual) == len(expected)
        assert all([a == b for a, b in zip(actual, expected)])


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


@pytest.mark.parametrize("option", ["include", "exclude"])
def test_filter_frames(option):
    include = ""
    exclude = ""
    with open(frame_example_file, "r") as f:
        gf, _, _ = get_raw_metrics(f)
        if option == "include":
            include = ".*test0.*"
        elif option == "exclude":
            exclude = ".*test1.*"
        gf = filter_frames(gf, include=include, exclude=exclude)
        idx = gf.dataframe["name"] == "test1"
        assert idx.sum() == 0
        idx = gf.dataframe["name"] == "test0"
        assert idx.sum() == 1


def test_min_time_flops():
    with open(cuda_example_file, "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_flops(gf.dataframe, device_info)
        device0_idx = gf.dataframe["device_id"] == "0"
        device1_idx = gf.dataframe["device_id"] == "1"
        # sm89
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[0.000025]], atol=1e-5)
        # sm90
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[0.00005]], atol=1e-5)
    with open(hip_example_file, "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_flops(gf.dataframe, device_info)
        device0_idx = gf.dataframe["device_id"] == "0"
        device1_idx = gf.dataframe["device_id"] == "1"
        # MI200
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[0.000026]], atol=1e-5)
        # MI300
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[0.000038]], atol=1e-5)


def test_min_time_bytes():
    with open(cuda_example_file, "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_bytes(gf.dataframe, device_info)
        device0_idx = gf.dataframe["device_id"] == "0"
        device1_idx = gf.dataframe["device_id"] == "1"
        # sm89
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[9.91969e-06]], atol=1e-6)
        # sm90
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[2.48584e-05]], atol=1e-6)
    with open(hip_example_file, "r") as f:
        gf, _, device_info = get_raw_metrics(f)
        ret = get_min_time_bytes(gf.dataframe, device_info)
        device0_idx = gf.dataframe["device_id"] == "0"
        device1_idx = gf.dataframe["device_id"] == "1"
        # MI200
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[6.10351e-06]], atol=1e-6)
        # MI300
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[1.93378e-05]], atol=1e-6)


def derivation_metrics_test(metrics, expected_data, sample_file, rtol=1e-7, atol=1e-6):
    with open(sample_file, "r") as f:
        gf, raw_metrics, device_info = get_raw_metrics(f)
        assert len(raw_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        derived_metrics = derive_metrics(gf, metrics, raw_metrics, device_info)
        for derived_metric in derived_metrics:
            np.testing.assert_allclose(gf.dataframe[derived_metric].to_numpy(), expected_data[derived_metric],
                                       rtol=rtol, atol=atol)


def test_avg_time_derivation():
    derivation_metrics_test(
        metrics=["avg_time/s", "avg_time/ms", "avg_time/us", "avg_time/ns"], expected_data={
            'avg_time/s (inc)': [np.nan, 0.0000205, 0.000205], 'avg_time/ms (inc)': [np.nan, 0.02048, 0.2048],
            'avg_time/us (inc)': [np.nan, 20.48, 204.8], 'avg_time/ns (inc)': [np.nan, 20480.0, 204800.0]
        }, sample_file=cuda_example_file)


def test_util():
    derivation_metrics_test(metrics=["util"], expected_data={
        'util (inc)': [np.nan, 0.247044, 0.147830],
    }, sample_file=cuda_example_file)


def test_time_derivation():
    derivation_metrics_test(
        metrics=["time/s", "time/ms", "time/us", "time/ns"], expected_data={
            'time/s (inc)': [0.0004096, 0.0002048, 0.0002048],
            'time/ms (inc)': [0.4096, 0.2048, 0.2048],
            'time/us (inc)': [409.6, 204.8, 204.8],
            'time/ns (inc)': [409600.0, 204800.0, 204800.0],
            'time/% (inc)': [100.0, 50.0, 50.0],
        }, sample_file=cuda_example_file)


def test_bytes_derivation():
    derivation_metrics_test(
        metrics=["byte/s", "gbyte/s", "tbyte/s"], expected_data={
            'byte/s (inc)': [2.68554687e+11, 4.88281250e+11, 4.88281250e+10], 'gbyte/s (inc)':
            [268.5546875, 488.28125, 48.828125], 'tbyte/s (inc)': [0.26855469, 0.48828125, 0.04882812]
        }, sample_file=cuda_example_file)


def test_flops_derivation():
    derivation_metrics_test(
        metrics=["flop8/s", "gflop8/s", "tflop8/s"],
        expected_data={
            'flop8/s (inc)': [2.68554687e+14, 4.88281250e+14, 4.88281250e+13], 'gflop8/s (inc)':
            [268554.6875, 488281.25, 48828.125], 'tflop8/s (inc)': [268.554687, 488.28125, 48.828125]
        },
        sample_file=cuda_example_file,
    )
