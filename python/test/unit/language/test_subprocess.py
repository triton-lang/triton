import itertools
import os
import subprocess
import sys
from collections import Counter

import triton
from triton._internal_testing import is_interpreter

import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))
print_path = os.path.join(dir_path, "print_helper.py")
torch_types = ["int8", "uint8", "int16", "int32", "long", "float16", "float32", "float64"]


@pytest.mark.interpreter
@pytest.mark.parametrize("func_type, data_type", [(fn, data_type)
                                                  for fn in ["device_print", "device_print_scalar"]
                                                  for data_type in torch_types] + [
                                                      ("print", "int32"),
                                                      ("static_print", "int32"),
                                                      ("no_arg_print", "int32"),
                                                      ("print_no_arg", "int32"),
                                                      ("device_print_large", "int32"),
                                                      ("print_multiple_args", "int32"),
                                                      ("device_print_multiple_args", "int32"),
                                                      ("device_print_hex", "int16"),
                                                      ("device_print_hex", "int32"),
                                                      ("device_print_hex", "int64"),
                                                      ("device_print_pointer", "int32"),
                                                      ("device_print_negative", "int32"),
                                                      ("device_print_uint", "uint32"),
                                                      ("device_print_uint_cast", "uint8"),
                                                      ("device_print_2d_tensor", "int32"),
                                                  ])
def test_print(func_type: str, data_type: str, device: str):
    proc = subprocess.run(
        [sys.executable, print_path, "test_print", func_type, data_type, device],
        capture_output=True,
    )
    assert proc.returncode == 0

    if is_interpreter() and func_type != "static_assert":
        # Interpreter uses a different format for device_print
        # Only check if there's no error
        assert proc.stderr == b''
        return

    outs = [line for line in proc.stdout.decode("UTF-8").splitlines() if line]
    # The total number of elements in the 1-D tensor to print.
    N = 128

    # Constant for testing the printing of scalar values
    SCALAR_VAL = 42

    # Format is
    #   pid (<x>, <y>, <z>) idx (<i1>, <i2>, ...) <prefix> (operand <n>) <elem>
    expected_lines = Counter()
    if func_type in ("print", "device_print", "device_print_uint", "device_print_uint_cast"):
        for i in range(N):
            offset = 0
            if func_type == "device_print_uint_cast":
                offset = 1 << 7
            elif func_type == "device_print_uint":
                offset = (1 << 31)
            line = f"pid (0, 0, 0) idx ({i:3}) x: {i + offset}"
            if data_type.startswith("float"):
                line += ".000000"
            expected_lines[line] = 1
    elif func_type == "device_print_scalar":
        line = f"pid (0, 0, 0) idx () x: {SCALAR_VAL}"
        if data_type.startswith("float"):
            line += ".000000"
        expected_lines[line] = N
    elif func_type == "device_print_negative":
        for i in range(N):
            line = f"pid (0, 0, 0) idx ({i:3}) x: {-i}"
            expected_lines[line] = 1
    elif func_type == "device_print_hex":
        for i in range(N):
            line = f"pid (0, 0, 0) idx ({i:3}) x: 0x"
            if data_type == "int16":
                line += f"{i:04x}"
            if data_type == "int32":
                line += f"{i:08x}"
            if data_type == "int64":
                line += f"{i:016x}"
            expected_lines[line] = 1
    elif func_type == "static_print":
        expected_lines[f" int32[constexpr[{N}]]"] = 1
    elif func_type == "no_arg_print":
        expected_lines["pid (0, 0, 0) idx (): 0"] = N
    elif func_type == "print_no_arg":
        expected_lines["pid (0, 0, 0) no arg"] = N
    elif func_type == "device_print_large":
        for i, j, k in itertools.product(range(2), range(64), range(N)):
            expected_lines[f"pid (0, {i}, 0) idx ({j:2}, {k:3}) x: 1"] = 1
    elif func_type == "print_multiple_args" or func_type == "device_print_multiple_args":
        for i in range(N):
            expected_lines[f"pid (0, 0, 0) idx ({i:3}): (operand 0) {i}"] = 1
            expected_lines[f"pid (0, 0, 0) idx ({i:3}): (operand 1) 1"] = 1
    elif func_type == "device_print_pointer":
        for i in range(N):
            expected_lines[f"pid (0, 0, 0) idx ({i:3}) ptr: 0x"] = 1
    elif func_type == "device_print_2d_tensor":
        warp_size = triton.runtime.driver.active.get_current_target().warp_size
        x_dim = N // warp_size
        y_dim = warp_size
        for x in range(x_dim):
            for y in range(y_dim):
                expected_lines[f"pid (0, 0, 0) idx ({x}, {y:2}): {(x * y_dim + y)}"] = 1

    actual_lines = Counter()
    for line in outs:
        # Trim the exact pointer address in the output--they can change per run.
        line = (line.split(':')[0] + ": 0x") if func_type == "device_print_pointer" else line
        actual_lines[line] += 1

    diff = Counter(actual_lines)
    diff.subtract(expected_lines)
    for line, delta in diff.items():
        if delta == 0:
            continue
        print(f'Expected line "{line}" {expected_lines[line]} time(s), but saw {actual_lines[line]} time(s)')
    assert all(delta == 0 for delta in diff.values())
