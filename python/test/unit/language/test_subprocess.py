import os
import subprocess
import sys

import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))
printf_path = os.path.join(dir_path, "printf_helper.py")
assert_path = os.path.join(dir_path, "assert_helper.py")

# TODO: bfloat16 after LLVM-15
torch_types = ["int8", "uint8", "int16", "int32", "long", "float16", "float32", "float64"]


@pytest.mark.parametrize("data_type", torch_types)
def test_printf(data_type: str):
    proc = subprocess.Popen([sys.executable, printf_path, data_type], stdout=subprocess.PIPE, shell=False)
    outs, _ = proc.communicate()
    outs = outs.split()
    new_lines = set()
    for line in outs:
        try:
            value = int(float(line))
            new_lines.add(value)
        except Exception as e:
            print(e)
    for i in range(128):
        assert i in new_lines
    assert len(new_lines) == 128


def test_assert():
    os.environ["TRITON_ENABLE_DEVICE_ASSERT"] = "1"
    proc = subprocess.Popen([sys.executable, assert_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    _, errs = proc.communicate()
    errs = errs.splitlines()
    num_errs = 0
    for err in errs:
        if "x != 0" in err.decode("utf-8"):
            num_errs += 1
    os.environ["TRITON_ENABLE_DEVICE_ASSERT"] = "0"
    assert num_errs == 127
