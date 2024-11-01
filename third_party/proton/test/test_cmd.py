import pytest
import subprocess
import json


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


@pytest.mark.parametrize("mode", ["script", "python", "pytest"])
def test_exec(mode, tmp_path):
    file_path = __file__
    helper_file = file_path.replace("test_cmd.py", "helper.py")
    temp_file = tmp_path / "test_exec.hatchet"
    name = str(temp_file).split(".")[0]
    if mode == "script":
        ret = subprocess.check_call(["proton", "-n", name, helper_file, "test"], stdout=subprocess.DEVNULL)
    elif mode == "python":
        ret = subprocess.check_call(["python3", "-m", "triton.profiler.proton", "-n", name, helper_file, "test"],
                                    stdout=subprocess.DEVNULL)
    elif mode == "pytest":
        ret = subprocess.check_call(["proton", "-n", name, "pytest", "-k", "test_main", helper_file],
                                    stdout=subprocess.DEVNULL)
    assert ret == 0
    with open(name) as f:
        data = json.load(f, )
    kernels = data[0]["children"]
    assert len(kernels) == 2
    assert kernels[0]["frame"]["name"] == "test" or kernels[1]["frame"]["name"] == "test"
