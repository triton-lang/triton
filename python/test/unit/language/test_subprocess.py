import os
import subprocess
import sys

import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))
print_path = os.path.join(dir_path, "print_helper.py")
assert_path = os.path.join(dir_path, "assert_helper.py")

# TODO: bfloat16 after LLVM-15
assert_types = ["device_assert", "assert", "static_assert", "no_debug"]
nested_types = [(caller, callee) for caller in ["true", "false", "none"] for callee in ["true", "false", "none"]]
torch_types = ["int8", "uint8", "int16", "int32", "long", "float16", "float32", "float64"]


@pytest.mark.parametrize("func_type, data_type",
                         [("device_print", data_type) for data_type in torch_types] + [("print", "int32"), ("static_print", "int32"), ("no_arg_print", "int32")])
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
    if func_type != "static_print" and func_type != "no_arg_print":
        for i in range(128):
            assert i in new_lines
    else:
        assert len(new_lines) == 1


@pytest.mark.parametrize("func_type", assert_types)
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


@pytest.mark.parametrize("caller_type, callee_type", nested_types)
def test_assert_nested(caller_type, callee_type):
    proc = subprocess.Popen([sys.executable, assert_path, caller_type, callee_type], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    _, errs = proc.communicate()
    errs = errs.splitlines()
    num_errs = 0
    for err in errs:
        if "x != 0" in err.decode("utf-8"):
            num_errs += 1
    if caller_type == "none":
        if callee_type == "true":
            assert num_errs == 127
        else:
            assert num_errs == 0
    elif caller_type == "true":
        if callee_type == "false":
            assert num_errs == 0
        else:
            assert num_errs == 127
    elif caller_type == "false":
        if callee_type == "true":
            assert num_errs == 127
        else:
            assert num_errs == 0
