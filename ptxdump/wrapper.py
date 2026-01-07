import sys
import os

ROOT_DIR = "/root/wkspace/triton"

sys.path.append(ROOT_DIR)
import patch_triton

if len(sys.argv) < 2:
    print("Usage: python wrapper.py <script_path> [args...]")
    sys.exit(1)

target_script = sys.argv[1]
sys.argv = sys.argv[1:]

script_name = os.path.basename(target_script).replace(".py", "")
os.environ["TRITON_PTX_DUMP_DIR"] = os.path.join(ROOT_DIR, "ptx_dump_all", script_name)
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
print(f"Wrapper: Dumping PTX to {os.environ['TRITON_PTX_DUMP_DIR']}")

# Set __file__ to the script path so inspect works
# We need to execute in a namespace that looks like __main__
import runpy

# We use runpy.run_path which handles __name__='__main__' and __file__ correctly
try:
    runpy.run_path(target_script, run_name='__main__')
except Exception as e:
    print(f"Error running {target_script}: {e}")
    # We don't exit with error because we want to continue dumping other kernels
