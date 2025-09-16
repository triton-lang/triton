import os
import subprocess
import pathlib

def test_inspection(tmp_path: pathlib.Path):
    _run_test = lambda env, dir_path, filename: subprocess.run(["python3", dir_path + "/" + filename],
                                                     env=env, capture_output=True, text=True)
    curr_repro_path = tmp_path / ("../test_inspection0." + "make_ttgir" + ".repro.mlir")
    dir_path = os.path.dirname(os.path.realpath(__file__))

    golden_env = os.environ.copy()
    golden_env["TRITON_ALWAYS_COMPILE"] = "1"
    golden_env["TRITON_DUMP_DIR"] = str(tmp_path)
    golden_env["TRITON_REPRODUCER_PATH"] = str(tmp_path)
    test_env = golden_env.copy()
    test_env["TRITON_OVERRIDE_DIR"] = str(tmp_path)

    # Run once to get the clean/golden repro dump
    assert "Called make_ttgir_wrapper" not in str(_run_test(golden_env, dir_path, "inspection_helper_golden.py").stdout)
    golden_ttgir_repro = curr_repro_path.read_text()
    curr_repro_path.unlink()

    # Run again with pipeline inspection
    assert "Called make_ttgir_wrapper" in str(_run_test(test_env, dir_path, "inspection_helper.py").stdout)
    assert golden_ttgir_repro == curr_repro_path.read_text()
