import pytest
import subprocess
import json
import os
import pathlib


def test_help():
    # Only check if the viewer can be invoked
    subprocess.check_call(["proton", "-h"], stdout=subprocess.DEVNULL)


@pytest.mark.parametrize("mode", ["script", "python", "pytest"])
def test_exec(mode, tmp_path: pathlib.Path):
    file_path = __file__
    helper_file = file_path.replace("test_cmd.py", "helper.py")
    temp_file = tmp_path / "test_exec.hatchet"
    name = str(temp_file.with_suffix(""))
    backend_args = ["-b", os.environ["PROTON_TEST_BACKEND"]] if os.environ.get("PROTON_TEST_BACKEND") else []
    if mode == "script":
        subprocess.check_call(["proton", "-n", name, *backend_args, helper_file, "test"], stdout=subprocess.DEVNULL)
    elif mode == "python":
        subprocess.check_call(
            ["python3", "-m", "triton.profiler.proton", "-n", name, *backend_args, helper_file, "test"],
            stdout=subprocess.DEVNULL)
    elif mode == "pytest":
        subprocess.check_call(["proton", "-n", name, *backend_args, "pytest", "-k", "test_main", helper_file],
                              stdout=subprocess.DEVNULL)
    with temp_file.open() as f:
        data = json.load(f, )
    kernels = data[0]["children"]
    assert len(kernels) == 2
    assert kernels[0]["frame"]["name"] == "test" or kernels[1]["frame"]["name"] == "test"
