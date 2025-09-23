import os
import subprocess

all_names = ["offsets", "pid", "block_start", "mask", "x", "y", "output"]


def checkDbgInfo(llir, hasDbgInfo):
    assert hasDbgInfo == ('dbg_value' in llir)
    for name in all_names:
        assert hasDbgInfo == ('!DILocalVariable(name: \"' + name + '\"' in llir)


def test_triton_debuginfo_on():
    lineInfoKey = "TRITON_DISABLE_LINE_INFO"
    diLocalVarKey = "LLVM_EXTRACT_DI_LOCAL_VARIABLES"

    isEnvSet = lambda env, str: env.get(str, None) is not None
    hasOrigLineInfo = (not isEnvSet(os.environ, lineInfoKey)
                       or os.environ[lineInfoKey].lower() not in ["on", "true", "1"])
    envs = [
        # expect no dbginfo if unset
        {lineInfoKey: None, diLocalVarKey: None, "hasDbgInfo": False},
        # expect dbginfo based on parent proccess' TRITON_DISABLE_LINE_INFO
        {lineInfoKey: None, diLocalVarKey: "1", "hasDbgInfo": hasOrigLineInfo},
        {lineInfoKey: "0", diLocalVarKey: "1", "hasDbgInfo": True},
        {lineInfoKey: "1", diLocalVarKey: "1", "hasDbgInfo": False},
        {lineInfoKey: "0", diLocalVarKey: "0", "hasDbgInfo": False},
        {lineInfoKey: "1", diLocalVarKey: "0", "hasDbgInfo": False},
    ]

    _run_test = lambda test_env: subprocess.run([
        "python3", os.path.dirname(os.path.realpath(__file__)) + "/test_debuginfo_helper.py"
    ], env=test_env, capture_output=True, text=True)
    for env in envs:
        test_env = os.environ.copy()
        test_env["TRITON_ALWAYS_COMPILE"] = "1"
        for entry in env:
            if not isEnvSet(env, entry): continue
            test_env[entry] = str(env[entry])
        checkDbgInfo(str(_run_test(test_env).stdout), hasDbgInfo=env["hasDbgInfo"])
