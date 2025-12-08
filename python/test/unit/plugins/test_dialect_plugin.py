import os
import subprocess
import pathlib
import pytest

from triton._internal_testing import is_cuda, is_hip, is_hip_cdna2

pytestmark = pytest.mark.skipif(is_hip_cdna2(), reason="old AMD GPUs are not supported")


def test_override(tmp_path: pathlib.Path):
    if os.environ.get('LLVM_BUILD_SHARED_LIBS', '0') == '0':
        return
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Run once to get the file dumps
    first_env = os.environ.copy()
    first_env["TRITON_ALWAYS_COMPILE"] = "1"
    first_env["TRITON_KERNEL_DUMP"] = "1"
    first_env["TRITON_DUMP_DIR"] = str(tmp_path)

    subprocess.run(["python3", dir_path + "/override_helper.py", str(tmp_path)], env=first_env)

    ttir_files = list(tmp_path.rglob("*.ttir"))
    ttgir_files = list(tmp_path.rglob("*.ttgir"))
    llir_files = list(tmp_path.rglob("*.llir"))

    assert len(ttir_files) == 1
    assert len(ttgir_files) == 1
    assert len(llir_files) == 1

    os.remove(ttgir_files[0])
    os.remove(llir_files[0])

    if is_cuda():
        ptx_files = list(tmp_path.rglob("*.ptx"))
        cubin_files = list(tmp_path.rglob("*.cubin"))
        assert len(ptx_files) == 1
        assert len(cubin_files) == 1
        os.remove(ptx_files[0])
        os.remove(cubin_files[0])

    if is_hip():
        pytest.skip("plugin not supported/tested on AMD yet")

    filename = str(list(tmp_path.rglob("*.ttir"))[0])

    with open(filename, "r") as infile:
        file_str = infile.readlines()

    # # Add ttgir instrumentation
    with open(filename, "w") as outfile:
        for line in file_str:
            if "tt.get_program_id x" in line:
                line = '    %pid_base = arith.constant 0 : i32\n    %pid = plugin.magic %pid_base : i32\n'
            outfile.write(line)

    # # # Run again with kernel override
    second_env = os.environ.copy()
    second_env["TRITON_ALWAYS_COMPILE"] = "1"
    second_env["TRITON_KERNEL_OVERRIDE"] = "1"
    second_env["TRITON_OVERRIDE_DIR"] = str(tmp_path)
    second_env["TRITON_KERNEL_DUMP"] = "1"
    second_env["TRITON_DUMP_DIR"] = str(tmp_path)
    subprocess.run(["python3", dir_path + "/override_helper.py", str(tmp_path)], env=second_env)

    with open(ttir_files[0], 'r') as f:
        ttir = f.read()
    assert "plugin.magic" in ttir

    ttgir_files = list(tmp_path.rglob("*.ttgir"))

    with open(ttgir_files[0], 'r') as f:
        ttgir = f.read()
    assert "gpu.thread_id" in ttgir
