import pytest
import subprocess
import json
import pathlib
import numpy as np


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


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


def test_instrument_exec():

    test_stderr = '0     matmul_kernel     instrument.py:32:20     SHARED     STORE\n'\
                  '1     matmul_kernel     instrument.py:33:20     SHARED     STORE\n'\
                  '2     matmul_kernel     instrument.py:32:20     SHARED     LOAD\n'\
                  '3     matmul_kernel     instrument.py:33:20     SHARED     LOAD\n'

    out = subprocess.Popen(["proton", "--instrument=print-mem-spaces", "instrument.py"], stderr=subprocess.PIPE,
                           stdout=subprocess.PIPE)
    result = []
    for line in str(out.stderr.read().decode()).split("\n"):
        if line:
            result.append(line.split())
    assert [row[0] for row in result] == ['0', '1', '2', '3']
    assert [row[1] for row in result] == ['matmul_kernel', 'matmul_kernel', 'matmul_kernel', 'matmul_kernel']
    assert [row[2] for row in result
            ] == ['instrument.py:32:20', 'instrument.py:33:20', 'instrument.py:32:20', 'instrument.py:33:20']
    assert [row[3] for row in result] == ['SHARED', 'SHARED', 'SHARED', 'SHARED']
    assert [row[4] for row in result] == ['STORE', 'STORE', 'LOAD', 'LOAD']
