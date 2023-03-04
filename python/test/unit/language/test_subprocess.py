import os
import subprocess
import sys

import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))
print_path = os.path.join(dir_path, "print_helper.py")
assert_path = os.path.join(dir_path, "assert_helper.py")

# TODO: bfloat16 after LLVM-15
func_types = ["device_assert", "assert", "static_assert"]
torch_types = ["int8", "uint8", "int16", "int32", "long", "float16", "float32", "float64"]


@pytest.mark.parametrize("func_type, data_type",
                         [("device_print", data_type) for data_type in torch_types] + [("print", "int32"), ("static_print", "int32")])
def test_print(func_type: str, data_type: str):
    proc = subprocess.Popen([sys.executable, print_path, func_type, data_type], stdout=subprocess.PIPE, shell=False)
    outs, _ = proc.communicate()
    outs = outs.split()
    new_lines = set()
    for line in outs:
        try:
            value = line
            if func_type != "static_print":
                value = int(float(line))
            new_lines.add(value)
        except Exception as e:
            print(e)
    if func_type != "static_print":
        for i in range(128):
            assert i in new_lines
        assert len(new_lines) == 128
    else:
        assert len(new_lines) == 1


@pytest.mark.parametrize("func_type", func_types)
def test_assert(func_type: str):
    os.environ["TRITON_DEBUG"] = "1"
    proc = subprocess.Popen([sys.executable, assert_path, func_type], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    _, errs = proc.communicate()
    errs = errs.splitlines()
    num_errs = 0
    for err in errs:
        if "x != 0" in err.decode("utf-8"):
            num_errs += 1
    os.environ["TRITON_DEBUG"] = "0"
    if func_type != "static_assert":
        assert num_errs == 127
    else:
        assert num_errs == 0
