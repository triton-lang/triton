import os
import subprocess
import shutil
import pathlib

dir_path = os.path.dirname(os.path.realpath(__file__))

tmp_dir = dir_path + '/tmp'
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_DUMP_DIR"] = tmp_dir
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"
os.environ["TRITON_OVERRIDE_DIR"] = tmp_dir

if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)


# Run once to get the file dumps
subprocess.run(["python", dir_path + "/test_override.py"])
path = pathlib.Path(tmp_dir)
asm_files = list(path.rglob(f"*.ttgir"))
assert len(asm_files) == 1
print(str(list(path.rglob(f"*.ttgir"))[0]))

filename = str(list(path.rglob(f"*.ttgir"))[0])

with open(filename, "r") as infile:
    file_str = infile.readlines()

# Add ttgir instrumentation
with open(filename, "w") as outfile:
    for line in file_str:
        # if "tt.get_program_id x" in line and "loc(#loc2)" in line:
        #     line = '    proton.record start "kernel" loc(#loc1)\n' + line
        # if "arith.cmpi slt" in line and "loc(#loc6)" in line:
        #     line = '    proton.record start "load_ops loc(#loc1)"\n' + line
        #     line = '    proton.record start "load_x loc(#loc1)"\n' + line
        # if "tt.load" in line and "loc(#loc8)" in line:
        #     line = '    proton.record end "load_x loc(#loc1)"\n' + line
        #     line = '    proton.record start "load_y loc(#loc1)"\n' + line
        # if "tt.load" in line and "loc(#loc10)" in line:
        #     line = '    proton.record end "load_y loc(#loc1)"\n' + line
        #     line = '    proton.record end "load_ops loc(#loc1)"\n' + line
        # if "tt.return" in line and "loc(#loc14)" in line:
        #     line = '    proton.record end "kernel" loc(#loc1)\n' + line
        outfile.write(line)

# Run again with kernel override
subprocess.run(["python", dir_path + "/test_override.py"])

# temp_file = tmp_dir / "test_tree.hatchet"
# with open(temp_file, "rb") as f:
#     data = json.load(f)
#     if hook:
#         assert "add_1024" == data[0]["children"][0]["frame"]["name"]
#     kernel_frame = data[0]["children"][0]["children"][0]
#     load_ops = kernel_frame["children"][0]
#     assert "load_ops" in load_ops["frame"]["name"]
#     assert ("load_x" in load_ops["children"][0]["frame"]["name"]
#             or "load_x" in load_ops["children"][1]["frame"]["name"])
#     assert ("load_y" in load_ops["children"][0]["frame"]["name"]
#             or "load_y" in load_ops["children"][1]["frame"]["name"])
#     assert load_ops["children"][0]["metrics"]["cycles"] > 0
#     assert load_ops["children"][1]["metrics"]["cycles"] > 0
