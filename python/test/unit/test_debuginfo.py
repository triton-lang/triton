import os
import subprocess
import pathlib

all_names = ["offsets", "pid", "block_start", "mask", "x", "y", "output"]

def checkDbgInfo(llir, hasDbgInfo):
    assert hasDbgInfo == ('dbg_value' in llir)
    for name in all_names:
        assert hasDbgInfo == ('!DILocalVariable(name: \"' + name + '\"' in llir)

def test_triton_debuginfo_on(tmp_path: pathlib.Path):
    _run_test = lambda test_env, dir_path: subprocess.run(["python3", dir_path + "/test_debuginfo_helper.py"],
                                                          env=test_env, capture_output=True, text=True)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    test_env = os.environ.copy()
    test_env["TRITON_ALWAYS_COMPILE"] = "1"
    test_env["TRITON_DUMP_DIR"] = str(tmp_path)

    checkDbgInfo(str(_run_test(test_env, dir_path).stdout), hasDbgInfo=False)

    test_env["LLVM_EXTRACT_DI_LOCAL_VARIABLES"] = "1"
    checkDbgInfo(str(_run_test(test_env, dir_path).stdout), hasDbgInfo=True)

    test_env["LLVM_EXTRACT_DI_LOCAL_VARIABLES"] = "0"
    checkDbgInfo(str(_run_test(test_env, dir_path).stdout), hasDbgInfo=False)
