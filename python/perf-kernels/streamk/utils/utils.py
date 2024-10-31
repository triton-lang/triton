import torch
import triton
import triton.language as tl

import os
import subprocess
from datetime import datetime

TORCH_HAS_FP8E5B16 = hasattr(torch, 'float8_e5m2fnuz')
TORCH_HAS_FP8E4B8 = hasattr(torch, 'float8_e4m3fnuz')
tl_to_torch_types = {
    tl.float16: torch.float16,
    tl.bfloat16: torch.bfloat16,
    tl.float32: torch.float32,
    tl.int8: torch.int8,
    tl.int32: torch.int32,
}
if TORCH_HAS_FP8E5B16:
    tl_to_torch_types[tl.float8e5b16] = torch.float8_e5m2fnuz
if TORCH_HAS_FP8E4B8:
    tl_to_torch_types[tl.float8e4b8] = torch.float8_e4m3fnuz

name_to_tl_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8': tl.float8e4b8,
    'bf8': tl.float8e5b16,
}


def run_bash_command_wrapper(commandstring, capture=True):
    try:
        run_bash_command(commandstring, capture)
    except subprocess.CalledProcessError:
        if not capture:
            print(f"running {commandstring} one more time")
        run_bash_command(commandstring, capture)


def run_bash_command(commandstring, capture=True):
    if capture:
        proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
        return proc.stdout.splitlines()
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash')
    return None


def get_filename_myKernels():
    path = os.path.dirname(os.path.abspath(__file__))
    return f"{path}/../myKernels.py"


def get_filename_without_extension(file_path):
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    return file_name


def get_filename_compile_driver():
    path = os.path.dirname(os.path.abspath(__file__))
    return f"{path}/../compile_driver.py"


def get_filename_profile_driver(M, N, K, job_id):
    path = os.path.dirname(os.path.abspath(__file__))
    return f"{path}/../profile_driver_{M}x{N}x{K}_{job_id}.py"


def get_default_tuning_result_filename():
    git_branch_name = run_bash_command("git rev-parse --abbrev-ref HEAD")
    git_branch_name = git_branch_name[0].decode()
    # handle branch name of "xxx/xxx" format
    git_branch_name = git_branch_name.replace('/', '_')
    git_commit_hash = run_bash_command("git rev-parse --short HEAD")
    git_commit_hash = git_commit_hash[0].decode()

    dt_string = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")

    path = os.path.dirname(os.path.abspath(__file__))
    defaultName = f"{path}/../tuning_results_{git_branch_name}@{git_commit_hash}_{dt_string}.yaml"
    return defaultName


def patch_triton_compiler():
    device = triton.runtime.driver.active.get_current_device()
    stream = triton.runtime.driver.active.get_current_stream(device)
    target = triton.runtime.driver.active.get_current_target()

    triton_location_str = run_bash_command("pip show triton | grep Editable")
    if not triton_location_str:
        print("triton source not found from pip show triton")

    triton_dir = triton_location_str[0].split()[-1].decode('utf-8')

    jit_filename = os.path.join(triton_dir, "triton/runtime", "jit.py")

    run_bash_command(f"sed -i 's/driver.active.get_current_device()/{device}/g' {jit_filename}")
    run_bash_command(f"sed -i 's/driver.active.get_current_stream(device)/{stream}/g' {jit_filename}")

    hip_driver_filename = os.path.join(triton_dir, "../third_party/amd/backend/", "driver.py")
    cuda_driver_filename = os.path.join(triton_dir, "../third_party/nvidia/backend/", "driver.py")

    run_bash_command(f"sed -i 's/import torch/return True/g' {hip_driver_filename}")
    run_bash_command(
        f"sed -i 's/device = self.get_current_device()/return GPUTarget(\"hip\", \"{target.arch}\", 64)/g' {hip_driver_filename}"
    )
    run_bash_command(f"sed -i 's/import torch/return False/g' {cuda_driver_filename}")
