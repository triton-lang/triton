from typing import Sequence
import os
import subprocess
import shutil
from pathlib import Path

def libcuda_dirs():
    locs = subprocess.check_output(["whereis", "libcuda.so"]).decode().strip().split()[1:]
    return [os.path.dirname(loc) for loc in locs]

if not libcuda_dirs():
    print("No libcuda.so found")
    # exit(1)   

def default_cuda_dir():
    default_dir = Path("/usr/local/cuda")
    if not default_dir.exists():
      # On LambdaLabs
      default_dir = Path('/lib/cuda')
    return os.getenv("CUDA_HOME", default=str(default_dir))

if not os.path.exists(default_cuda_dir()):
    print("No cuda directory found, set CUDA_HOME env var")
    # exit(1)

def find_compiler():
  cc = os.environ.get("CC")
  if cc is None:
      # TODO: support more things here.
      clang = shutil.which("clang")
      gcc = shutil.which("gcc")
      cc = gcc if gcc is not None else clang
  return cc


def build_cuda_bin(out:str, *src_files, include_dirs: Sequence[str]=None, pic: bool = False):
  cuda_lib_dirs = libcuda_dirs()
  
  cuda_path = os.environ.get('CUDA_PATH', default_cuda_dir())
  cu_include_dir = os.path.join(cuda_path, "include")

  cc = find_compiler()
  if include_dirs is None:
    include_dirs = []

  includes = [f"-I{inc_dir}" for inc_dir in include_dirs]  
  
  options = []
  if pic:
    options.append("-fPIC")
  
  cc_cmd = [cc, *src_files, "-O3", *includes, f"-I{cu_include_dir}" , *options, "-lcuda", "-o", out]
  cc_cmd += [f"-L{dir}" for dir in cuda_lib_dirs]
  print(" ".join(cc_cmd))
#   ret = subprocess.check_call(cc_cmd)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--out", "-o", type=str)
    parser.add_argument("src", nargs="+")
    parser.add_argument("--include-dirs", "-I", nargs="+")

    args = parser.parse_args()

    build_cuda_bin(args.out, *args.src, include_dirs=args.include_dirs)