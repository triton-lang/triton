import triton
import pytest
import subprocess
import json
import pathlib


def test_help():
    # Only check if the viewer can be invoked
    subprocess.check_call(["proton", "-h"], stdout=subprocess.DEVNULL)


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@pytest.mark.parametrize("mode", ["script", "python", "pytest"])
def test_exec(mode, tmp_path: pathlib.Path):
    file_path = __file__
    helper_file = file_path.replace("test_cmd.py", "helper.py")
    temp_file = tmp_path / "test_exec.hatchet"
    name = str(temp_file.with_suffix(""))
    if mode == "script":
        subprocess.check_call(["proton", "-n", name, helper_file, "test"], stdout=subprocess.DEVNULL)
    elif mode == "python":
        subprocess.check_call(["python3", "-m", "triton.profiler.proton", "-n", name, helper_file, "test"],
                              stdout=subprocess.DEVNULL)
    elif mode == "pytest":
        subprocess.check_call(["proton", "-n", name, "pytest", "-k", "test_main", helper_file],
                              stdout=subprocess.DEVNULL)
    with temp_file.open() as f:
        data = json.load(f, )
    kernels = data[0]["children"]
    assert len(kernels) == 2
    assert kernels[0]["frame"]["name"] == "test" or kernels[1]["frame"]["name"] == "test"


def test_instrument_exec():

    try:
        out = subprocess.Popen(["proton", "--instrument=print-mem-spaces", "instrument.py"],
                               cwd=pathlib.Path(__file__).parent, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    except Exception as e:
        print(f"An error occurred while executing proton: {e}")

    result = []
    for line in str(out.stderr.read().decode()).split("\n"):
        if line:
            result.append(line.split())

    if is_hip():
        assert [row[0] for row in result] == ['0', '1', '2', '3']
        assert [row[1] for row in result] == ['matmul_kernel', 'matmul_kernel', 'matmul_kernel', 'matmul_kernel']
        assert [row[2] for row in result
                ] == ['instrument.py:32:20', 'instrument.py:33:20', 'instrument.py:32:20', 'instrument.py:33:20']
        assert [row[3] for row in result] == ['SHARED', 'SHARED', 'SHARED', 'SHARED']
        assert [row[4] for row in result] == ['STORE', 'STORE', 'LOAD', 'LOAD']
    else:
        assert [row[0] for row in result] == ['0']
        assert [row[1] for row in result] == ['matmul_kernel']
        assert [row[2] for row in result] == ['instrument.py:42:21']
        assert [row[3] for row in result] == ['SHARED']
        assert [row[4] for row in result] == ['LOAD']
