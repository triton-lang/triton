import os
import subprocess
import shutil
import pathlib
import json

dir_path = os.path.dirname(os.path.realpath(__file__))

tmp_dir = os.getcwd() + '/tmp'

if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)

# Run once to get the file dumps
first_env = os.environ.copy()
first_env["TRITON_ALWAYS_COMPILE"] = "1"
first_env["TRITON_KERNEL_DUMP"]= "1"
first_env["TRITON_DUMP_DIR"]=tmp_dir

subprocess.run(["python", dir_path + "/01-vector-add.py", "off"], env=first_env)

path = pathlib.Path(tmp_dir)

ttir_files = list(path.rglob(f"*.ttir"))
ttgir_files = list(path.rglob(f"*.ttgir"))
llir_files = list(path.rglob(f"*.llir"))
ptx_files = list(path.rglob(f"*.ptx"))
cubin_files = list(path.rglob(f"*.cubin"))

assert len(ttir_files) == 1
assert len(ttgir_files) == 1
assert len(llir_files) == 1
assert len(ptx_files) == 1
assert len(cubin_files) == 1

os.remove(ttir_files[0])
os.remove(llir_files[0])
os.remove(ptx_files[0])
os.remove(cubin_files[0])

print(str(list(path.rglob(f"*.ttgir"))[0]))

filename = str(list(path.rglob(f"*.ttgir"))[0])

with open(filename, "r") as infile:
    file_str = infile.readlines()

# Add ttgir instrumentation
with open(filename, "w") as outfile:
    for line in file_str:
        if "tt.get_program_id x" in line and "loc(#loc2)" in line:
            #insert before the line
            line = '    proton.record start "kernel" loc(#loc1)\n' + line
        if "arith.cmpi slt" in line and "loc(#loc6)" in line:
            #insert after the line
            line = line + '    proton.record start "load_ops loc(#loc1)"\n'
            line = line + '    proton.record start "load_x loc(#loc1)"\n'
        if "tt.load" in line and "loc(#loc8)" in line:
            #insert after the line
            line = line + '    proton.record end "load_x loc(#loc1)"\n'
            line = line + '    proton.record start "load_y loc(#loc1)"\n'
        if "tt.load" in line and "loc(#loc10)" in line:
            #insert after the line
            line = line + '    proton.record end "load_y loc(#loc1)"\n'
            line = line + '    proton.record end "load_ops loc(#loc1)"\n'
        if "tt.return" in line and "loc(#loc14)" in line:
            #insert before the line
            line = '    proton.record end "kernel" loc(#loc1)\n' + line
        outfile.write(line)

# Run again with kernel override
second_env = os.environ.copy()
second_env["TRITON_ALWAYS_COMPILE"]="1"
second_env["TRITON_KERNEL_OVERRIDE"]="1"
second_env["TRITON_OVERRIDE_DIR"]=tmp_dir
subprocess.run(["python", dir_path + "/01-vector-add.py", "on"], env=second_env)

tmp_path = pathlib.Path(dir_path + '/tmp')
temp_file = tmp_path / "test_tree.hatchet"

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
    assert load_ops["children"][1]["metrics"]["cycles"] > 0

# Clean up
shutil.rmtree(tmp_dir)
