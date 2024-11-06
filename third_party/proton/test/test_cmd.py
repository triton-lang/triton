import pytest
import subprocess
import json
import pathlib


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


def test_instrument_exec():

    test_stderr = '0     matmul_kernel     instrument.py:43:20     SHARED     STORE\n'\
                  '1     matmul_kernel     instrument.py:44:20     SHARED     STORE\n'\
                  '2     matmul_kernel     instrument.py:43:20     SHARED     LOAD\n'\
                  '3     matmul_kernel     instrument.py:44:20     SHARED     LOAD\n'

    out = subprocess.Popen(["proton", "--instrument=print-mem-spaces", "instrument.py"], stderr=subprocess.PIPE,
                           stdout=subprocess.PIPE)
    assert test_stderr == out.stderr.read().decode()


@pytest.mark.parametrize("mode", ["script", "python", "pytest"])
def test_exec(mode, tmp_path: pathlib.Path):
    file_path = __file__
    helper_file = file_path.replace("test_cmd.py", "helper.py")
    temp_file = tmp_path / "test_exec.hatchet"
    name = str(temp_file.with_suffix(""))
    if mode == "script":
        ret = subprocess.check_call(["proton", "-n", name, helper_file, "test"], stdout=subprocess.DEVNULL)
    elif mode == "python":
        ret = subprocess.check_call(["python3", "-m", "triton.profiler.proton", "-n", name, helper_file, "test"],
                                    stdout=subprocess.DEVNULL)
    elif mode == "pytest":
        ret = subprocess.check_call(["proton", "-n", name, "pytest", "-k", "test_main", helper_file],
                                    stdout=subprocess.DEVNULL)
    assert ret == 0
    with temp_file.open() as f:
        data = json.load(f, )
    kernels = data[0]["children"]
    assert len(kernels) == 2
    assert kernels[0]["frame"]["name"] == "test" or kernels[1]["frame"]["name"] == "test"
