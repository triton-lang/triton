import os
import subprocess
import pathlib
import json

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
    first_env["TRITON_KERNEL_DUMP"] = "1"
    first_env["TRITON_DUMP_DIR"] = str(tmp_path)

    subprocess.run(["python3", dir_path + "/override_helper.py", str(tmp_path)], env=first_env)

    ttir_files = list(tmp_path.rglob("*.ttir"))
    ttgir_files = list(tmp_path.rglob("*.ttgir"))
    llir_files = list(tmp_path.rglob("*.llir"))

    assert len(ttir_files) == 1
    assert len(ttgir_files) == 1
    assert len(llir_files) == 1

    os.remove(ttir_files[0])
    os.remove(llir_files[0])

    if is_cuda():
        ptx_files = list(tmp_path.rglob("*.ptx"))
        cubin_files = list(tmp_path.rglob("*.cubin"))
        assert len(ptx_files) == 1
        assert len(cubin_files) == 1
        os.remove(ptx_files[0])
        os.remove(cubin_files[0])

    if is_hip():
        gcn_files = list(tmp_path.rglob("*.amdgcn"))
        hsaco_files = list(tmp_path.rglob("*.hsaco"))
        assert len(hsaco_files) == 1
        assert len(gcn_files) == 1
        os.remove(gcn_files[0])
        os.remove(hsaco_files[0])

    filename = str(list(tmp_path.rglob("*.ttgir"))[0])

    with open(filename, "r") as infile:
        file_str = infile.readlines()

    # Add ttgir instrumentation
    isFirstLoad = True
    with open(filename, "w") as outfile:
        for line in file_str:
            if "tt.get_program_id x" in line:
                #insert before the line
                line = '    proton.record start "kernel" loc(#loc)\n' + line
            elif "arith.cmpi slt" in line:
                #insert after the line
                line = line + '    proton.record start "load_ops" loc(#loc)\n'
                line = line + '    proton.record start "load_x" loc(#loc)\n'
            elif ("tt.load" in line and isFirstLoad) or ("amdgpu.buffer_load" in line and isFirstLoad):
                #insert after the line
                line = line + '    proton.record end "load_x" loc(#loc)\n'
                line = line + '    proton.record start "load_y" loc(#loc)\n'
                isFirstLoad = False
            elif ("tt.load" in line and not isFirstLoad) or ("amdgpu.buffer_load" in line and not isFirstLoad):
                #insert after the line
                line = line + '    proton.record end "load_y" loc(#loc)\n'
                line = line + '    proton.record end "load_ops" loc(#loc)\n'
            elif "tt.return" in line:
                #insert before the line
                line = '    proton.record end "kernel" loc(#loc)\n' + line
            outfile.write(line)

    # # Run again with kernel override
    second_env = os.environ.copy()
    second_env["TRITON_ALWAYS_COMPILE"] = "1"
    second_env["TRITON_KERNEL_OVERRIDE"] = "1"
    second_env["TRITON_OVERRIDE_DIR"] = str(tmp_path)
    subprocess.run(["python3", dir_path + "/override_helper.py", str(tmp_path)], env=second_env)

    temp_file = tmp_path / "test_override.hatchet"

    with open(temp_file, "rb") as f:
        data = json.load(f)
        kernel_frame = data[0]["children"][0]["children"][0]
        load_ops = kernel_frame["children"][0]
        assert "load_ops" in load_ops["frame"]["name"]
        assert ("load_x" in load_ops["children"][0]["frame"]["name"]
                or "load_x" in load_ops["children"][1]["frame"]["name"])
        assert ("load_y" in load_ops["children"][0]["frame"]["name"]
                or "load_y" in load_ops["children"][1]["frame"]["name"])
        assert load_ops["children"][0]["metrics"]["cycles"] > 0
        assert load_ops["children"][0]["metrics"]["normalized_cycles"] > 0
        assert load_ops["children"][1]["metrics"]["cycles"] > 0
        assert load_ops["children"][1]["metrics"]["normalized_cycles"] > 0
