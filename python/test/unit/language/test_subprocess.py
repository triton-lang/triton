import itertools
import os
import subprocess
import sys
from collections import Counter

import pytest

import triton

dir_path = os.path.dirname(os.path.realpath(__file__))
print_path = os.path.join(dir_path, "print_helper.py")
torch_types = ["int8", "uint8", "int16", "int32", "long", "float16", "float32", "float64"]


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cpu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cpu"


# TODO: Print with multiple operands


@pytest.mark.cpu
@pytest.mark.interpreter
@pytest.mark.parametrize("func_type, data_type", [(fn, data_type)
                                                  for fn in ["device_print", "device_print_scalars"]
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
                                                  ])
def test_print(func_type: str, data_type: str, device: str):
    if is_cpu() and (data_type == "float16" or func_type in ["device_print_pointer", "device_print_large"]):
        pytest.skip("test_print for float16/pointer/large are not yet supported on CPU.")

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

    # TODO: Consider cases for signedness, overflow, and multiple pids (non-determinism).
    if is_cpu():
        _check_cpu_print(proc.stdout.decode("UTF-8"), func_type, data_type, N, SCALAR_VAL)
        return

    # Format is
    #   pid (<x>, <y>, <z>) idx (<i1>, <i2>, ...) <prefix> (operand <n>) <elem>
    expected_lines = Counter()
    if func_type in ("print", "device_print", "device_print_uint"):
        for i in range(N):
            offset = (1 << 31) if data_type == "uint32" else 0
            line = f"pid (0, 0, 0) idx ({i:3}) x: {i + offset}"
            if data_type.startswith("float"):
                line += ".000000"
            expected_lines[line] = 1
    elif func_type == "device_print_scalars":
        line = f"pid (0, 0, 0) idx () x: {SCALAR_VAL}"
        if data_type.startswith("float"):
            line += ".000000"
        expected_lines[line] = N
        line = f"pid (0, 0, 0) idx () int: {SCALAR_VAL}"
        expected_lines[line] = N
        line = "pid (0, 0, 0) idx () float: 3.140000"
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

    cpu_gpu_msg = "Both CPU and GPU backends are available. Using the GPU backend."
    actual_lines = Counter()
    for line in outs:
        if line == cpu_gpu_msg:
            continue
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


def _check_cpu_print(actual, func_type, data_type, N, SCALAR_VAL):
    # An example of a tensor printing is like:
    # (0, 0, 0) x: [  0,   1,   2,   3,   4,   5,   6,   7,
    #                 8,   9,  10,  11,  12,  13,  14,  15,
    #                 ...
    #               120, 121, 122, 123, 124, 125, 126, 127]
    PID_PREFIX = "(0, 0, 0)"
    NEWLINE_WITH_PADDING = "\n" + " " * (len(PID_PREFIX + " x: ["))
    if func_type in ("print", "device_print", "device_print_uint"):
        expected = PID_PREFIX + " x: ["
        for i in range(N):
            offset = (1 << 31) if data_type == "uint32" else 0
            expected += f"{i + offset:3}"
            if data_type.startswith("float"):
                expected += ".0000"
            if i == N - 1:
                continue
            expected += ","
            if i % 8 == 7:
                expected += NEWLINE_WITH_PADDING
            else:
                expected += " "
        expected += "]"
    elif func_type == "device_print_scalars":
        expected = f"{PID_PREFIX} x: {SCALAR_VAL}"
        if data_type.startswith("float"):
            expected += ".000000"
        expected += f"\n{PID_PREFIX} int: {SCALAR_VAL}"
        expected += f"\n{PID_PREFIX} float: 3.140000"
    elif func_type == "device_print_negative":
        expected = PID_PREFIX + " x: ["
        for i in range(N):
            expected += f"{-i:4}"
            if i == N - 1:
                continue
            expected += ","
            if i % 8 == 7:
                expected += NEWLINE_WITH_PADDING
            else:
                expected += " "
        expected += "]"
    elif func_type == "device_print_hex":
        expected = PID_PREFIX + " x: ["
        for i in range(N):
            if data_type.endswith("8"):
                expected += f"0x{i:02x}"
            elif data_type.endswith("16"):
                expected += f"0x{i:04x}"
            elif data_type.endswith("32"):
                expected += f"0x{i:08x}"
            elif data_type.endswith("64"):
                expected += f"0x{i:016x}"
            if i == N - 1:
                continue
            expected += ","
            if i % 8 == 7:
                expected += NEWLINE_WITH_PADDING
            else:
                expected += " "
        expected += "]"
    elif func_type == "static_print":
        expected = f" int32[constexpr[{N}]]"
    elif func_type == "no_arg_print":
        expected = f"{PID_PREFIX}: 0"
    elif func_type == "print_no_arg":
        expected = f"{PID_PREFIX} no arg"
    elif func_type == "print_multiple_args" or func_type == "device_print_multiple_args":
        expected = ""
        for k in range(2):
            expected += PID_PREFIX + ": ["
            for i in range(N):
                expected += f"{i:3}" if k == 0 else "1"
                if i == N - 1:
                    continue
                expected += ","
                if i % 8 == 7:
                    expected += "\n" + " " * (len(PID_PREFIX + ": ["))
                else:
                    expected += " "
            expected += "]"
            if k == 0:
                expected += "\n"

    # Ignore the trailing new line.
    assert actual[:-1] == expected
