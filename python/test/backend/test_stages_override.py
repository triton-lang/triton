import os
import subprocess
import pathlib
import re

import triton


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def test_override(tmp_path: pathlib.Path):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Run once to get the file dumps
    first_env = os.environ.copy()
    first_env["TRITON_ALWAYS_COMPILE"] = "1"
    first_env["TRITON_ENABLE_COMPILER_INSPECTION"] = "1"
    first_env["TRITON_DUMP_PASS_STAGES"] = "1"
    first_env["TRITON_DUMP_DIR"] = str(tmp_path)

    subprocess.run(["python3", dir_path + "/override_helper.py"], env=first_env)
    filename = tmp_path / "compiler_override.py"

    with open(filename, "r") as infile:
        file_str = infile.readlines()

    with open(filename, "w") as outfile:
        for line in file_str:
            # turn off pre-fetching for test
            if "add_prefetch" in line:
                continue
            outfile.write(line)

    # # Run again with pipeline override
    second_env = os.environ.copy()
    second_env["TRITON_ALWAYS_COMPILE"] = "1"
    second_env["TRITON_ENABLE_COMPILER_INSPECTION"] = "1"
    second_env["TRITON_OVERRIDE_PASS_STAGES"] = "1"
    second_env["TRITON_REPRODUCER_PATH"] = str(tmp_path)
    second_env["TRITON_OVERRIDE_DIR"] = str(tmp_path)

    subprocess.run(["python3", dir_path + "/override_helper.py"], env=second_env)

    curr_repro_path = tmp_path / ("../test_override0." + "make_ttgir" + ".repro.mlir")
    repro = curr_repro_path.read_text()
    m = re.search(r"pipeline: \"(.*" + "convert-triton-to-tritongpu" + ".*)\"", repro)
    assert "tritongpu-prefetch" not in m.group(1)


def test_override_integrity(tmp_path: pathlib.Path):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Run once to get the clean repro dump
    golden_env = os.environ.copy()
    golden_env["TRITON_ALWAYS_COMPILE"] = "1"
    golden_env["TRITON_DUMP_DIR"] = str(tmp_path)
    golden_env["TRITON_REPRODUCER_PATH"] = str(tmp_path)
    minimal_kernel = "/override_helper.py"
    subprocess.run(["python3", dir_path + minimal_kernel], env=golden_env)

    curr_repro_path = tmp_path / ("../test_override_integrity0." + "make_ttir" + ".repro.mlir")
    golden_ttir_repro = curr_repro_path.read_text()
    curr_repro_path.unlink()
    curr_repro_path = tmp_path / ("../test_override_integrity0." + "make_ttgir" + ".repro.mlir")
    golden_ttgir_repro = curr_repro_path.read_text()
    curr_repro_path.unlink()

    # Run once to get the file dumps
    first_env = os.environ.copy()
    first_env["TRITON_ALWAYS_COMPILE"] = "1"
    first_env["TRITON_ENABLE_COMPILER_INSPECTION"] = "1"
    first_env["TRITON_DUMP_PASS_STAGES"] = "1"
    first_env["TRITON_DUMP_DIR"] = str(tmp_path)

    subprocess.run(["python3", dir_path + "/override_helper.py"], env=first_env)

    # # Run again with pipeline override
    second_env = os.environ.copy()
    second_env["TRITON_ALWAYS_COMPILE"] = "1"
    second_env["TRITON_ENABLE_COMPILER_INSPECTION"] = "1"
    second_env["TRITON_OVERRIDE_PASS_STAGES"] = "1"
    second_env["TRITON_REPRODUCER_PATH"] = str(tmp_path)
    second_env["TRITON_OVERRIDE_DIR"] = str(tmp_path)

    subprocess.run(["python3", dir_path + "/override_helper.py"], env=second_env)

    curr_repro_path = tmp_path / ("../test_override_integrity0." + "make_ttir" + ".repro.mlir")
    override_ttir_repro = curr_repro_path.read_text()
    curr_repro_path = tmp_path / ("../test_override_integrity0." + "make_ttgir" + ".repro.mlir")
    override_ttgir_repro = curr_repro_path.read_text()

    assert golden_ttir_repro == override_ttir_repro and golden_ttgir_repro == override_ttgir_repro
