import os
import sys
import subprocess
import glob

ROOT_DIR = "/root/wkspace/triton"
PYTHON_DIR = os.path.join(ROOT_DIR, "python")
PTX_DUMP_DIR = os.path.join(ROOT_DIR, "ptxdump")
PATCH_FILE = os.path.join(PTX_DUMP_DIR, "patch_triton.py")

env = os.environ.copy()
env["PYTHONPATH"] = f"{PTX_DUMP_DIR}:{PYTHON_DIR}:{ROOT_DIR}:{env.get('PYTHONPATH', '')}"

def run_pytest(path, output_subdir):
    print(f"Running pytest on {path}...")
    
    current_env = env.copy()
    current_env["TRITON_PTX_DUMP_DIR"] = os.path.join(ROOT_DIR, "ptx_dump_all", output_subdir)
    current_env["TRITON_ALWAYS_COMPILE"] = "1"
    current_env["DISABLE_SUBPROCESS"] = "1"
    
    cmd = [
        sys.executable, "-m", "pytest", path, "-v" , "-k", "not test_core"
    ]
    
    try:
        subprocess.run(cmd, env=current_env, check=False) # Don't check return code, we expect failures
    except Exception as e:
        print(f"Error running pytest on {path}: {e}")

def run_script(path):
    print(f"Running script {path}...")
    cmd = [
        sys.executable, "wrapper.py", path
    ]
    try:
        subprocess.run(cmd, env=env, check=False)
    except Exception as e:
        print(f"Error running script {path}: {e}")

def run_pytest_on_directory(directory, output_subdir):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file.startswith("test_"):
                filepath = os.path.join(root, file)
                file_basename = os.path.basename(filepath)
                run_pytest(filepath, os.path.join(output_subdir, file_basename))

def main():
    tutorials_dir = os.path.join(PYTHON_DIR, "tutorials")
    tutorials = glob.glob(os.path.join(tutorials_dir, "*.py"))
    gluon_tutorials = glob.glob(os.path.join(tutorials_dir, "gluon", "*.py"))
    
    all_tutorials = tutorials + gluon_tutorials
    for tutorial in all_tutorials:
        run_script(tutorial)

    run_pytest_on_directory(os.path.join(PYTHON_DIR, "test/functional"), "test/functional")    # run_pytest_on_directory(os.path.join(PYTHON_DIR, "examples"), "examples")
    run_pytest_on_directory(os.path.join(PYTHON_DIR, "test/gluon"), "test/gluon")
    run_pytest_on_directory(os.path.join(PYTHON_DIR, "test/regression"), "test/regression")
    run_pytest_on_directory(os.path.join(PYTHON_DIR, "test/unit"), "test/unit")
    run_pytest_on_directory(os.path.join(PYTHON_DIR, "triton_kernels"), "triton_kernels")

if __name__ == "__main__":
    main()
