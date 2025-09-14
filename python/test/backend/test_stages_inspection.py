import os
import subprocess
import pathlib

def test_override(tmp_path: pathlib.Path):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    test_env = os.environ.copy()
    test_env["TRITON_ALWAYS_COMPILE"] = "1"
    test_env["TRITON_MOCK_INSPECT_STAGES"] = "1"
    test_env["TRITON_DUMP_DIR"] = str(tmp_path)

    subprocess.run(["python3", dir_path + "/inspection_helper.py"], env=test_env)

    filename = tmp_path / "inspect_stages.py"
    with open(filename, "r") as infile:
        file_str = infile.readlines()

    assert file_str == ["# This is generated from Triton add_stages_inspection_hook"]


def test_override_integrity(tmp_path: pathlib.Path):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Run once to get the clean/golden repro dump
    golden_env = os.environ.copy()
    golden_env["TRITON_ALWAYS_COMPILE"] = "1"
    golden_env["TRITON_DUMP_DIR"] = str(tmp_path)
    golden_env["TRITON_REPRODUCER_PATH"] = str(tmp_path)
    minimal_kernel = "/inspection_helper.py"
    subprocess.run(["python3", dir_path + minimal_kernel], env=golden_env)

    curr_repro_path = tmp_path / ("../test_override_integrity0." + "make_ttir" + ".repro.mlir")
    golden_ttir_repro = curr_repro_path.read_text()
    curr_repro_path.unlink()
    curr_repro_path = tmp_path / ("../test_override_integrity0." + "make_ttgir" + ".repro.mlir")
    golden_ttgir_repro = curr_repro_path.read_text()
    curr_repro_path.unlink()

    # # Run again with pipeline override
    test_env = os.environ.copy()
    test_env["TRITON_ALWAYS_COMPILE"] = "1"
    test_env["TRITON_MOCK_OVERRIDE_STAGES"] = "1"
    test_env["TRITON_REPRODUCER_PATH"] = str(tmp_path)
    test_env["TRITON_DUMP_DIR"] = str(tmp_path)
    test_env["TRITON_OVERRIDE_DIR"] = str(tmp_path)

    subprocess.run(["python3", dir_path + "/inspection_helper.py"], env=test_env)

    curr_repro_path = tmp_path / ("../test_override_integrity0." + "make_ttir" + ".repro.mlir")
    override_ttir_repro = curr_repro_path.read_text()
    curr_repro_path = tmp_path / ("../test_override_integrity0." + "make_ttgir" + ".repro.mlir")
    override_ttgir_repro = curr_repro_path.read_text()

    assert golden_ttir_repro == override_ttir_repro and golden_ttgir_repro == override_ttgir_repro
