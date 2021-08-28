from dataclasses import dataclass
from pathlib import Path

import toml
import torch

import c_codegen as cc
from compilation_config import parse_compilation_config
from tt_bindings import TritonContext

# General TODOs
# TODO: proper logging


class CCompilationError(Exception):
    pass

class CCompilationError(Exception):
    pass

def find_cuda() -> cc.LibCudaConf:
    # TODO: turn to real function

    print("Warning!!! FAKE find CUDA witn my local hard-coded path")
    return  cc.LibCudaConf(
        include_dir="/usr/local/cuda-11.3/include/",
        link_dir="/usr/local/cuda-11.3/targets/x86_64-linux/lib/stubs/"
    )
    

def find_c_compiler():
    # TODO: turn to real function
    print("Warning!!! FAKE find compiler witn my local hard-coded paths")
    import subprocess
    
    cuda = find_cuda()

    def c_compile(output_dir: str, sources:cc.CSources):
        obj_files = []
        procs = []
        base = Path(output_dir)
        for idx, (header, soruce) in enumerate(sources.iter_header_and_sources()):
            obj_file = str(base / f"obj{idx}.o")
            cmd = [
                "gcc", "-Wall", "-c", "-fPIC", f"{soruce}", f"-I{cuda.include_dir}", f"-L{cuda.link_dir}", "-o", obj_file
            ]
            # TODO: catch C compilation errors
            try:
                procs.append(subprocess.Popen(cmd,shell=False,stdout=subprocess.PIPE))
            except Exception:
                raise CCompilationError(f"Command {' '.join(cmd)} failed")
            obj_files.append(obj_file)
            
            # TODO: make tmp dir for building
            procs.append(
                subprocess.Popen(["cp", header, str(base / Path(header).name)],shell=False,stdout=subprocess.PIPE)
            )
        
        # make sure all objects are ready before linking
        for p in procs:
            p.wait()

        cmd = ["gcc", "-Wall", *obj_files, "-shared", "-o", f"{output_dir}/lib{sources.lib_name}.so"]            
        # TODO: catch C compilation errors
        try:
            subprocess.Popen(cmd,shell=False,stdout=subprocess.PIPE)
        except Exception:
            raise CCompilationError(f"Command {' '.join(cmd)} failed")
        subprocess.Popen(["rm", *obj_files],shell=False,stdout=subprocess.PIPE)
    return c_compile


if __name__ == "__main__":

    # Arguments
    # TODO: add argparsing
    path_to_script = "test_data/_mod.py"
    path_to_config = "test_data/conf.toml"
    lib_name = "aot_kernels"

    # this one is needed to init Cuda context
    # TODO: (?) add light weight cuda init module to remove Torch deps.
    x = torch.empty(1, device="cuda")

    print(f"Building Triton context from {path_to_script}")
    # Exectue script to get Kernel objects
    tt_ctx = TritonContext.build_from_script(path_to_script)
    print(f"\t -> Done.")

    print(f"Parsing config from {path_to_config}")
    conf_dict = toml.loads(Path(path_to_config).read_text())
    conf = parse_compilation_config(conf_dict, tt_ctx.module_scope)
    kernels = tt_ctx.init_kernels()
    print(f"\t -> Done.")

    ccode = cc.SharedLibrarySource(lib_name, "test_data/build")
    
    print(f"Starting compilation of {len(kernels)} kernels")
    for idx, (ker_name, kernel) in enumerate(kernels.items()):
        kernel_conf = conf.get(ker_name)
        if kernel_conf is None:
            # TODO: (?) maybe should be an error
            msg = f"\n\t Skipping: Kernel defined in {path_to_script} but no config for it found"
            print(f"[{idx+1}/{len(kernels)}] {ker_name} {msg}")
            continue

        # Compile all argument permutation
        print(f"[{idx+1}/{len(kernels)}] {ker_name}")
        
        for abst_args in kernel_conf.signiture_iter():
            attr_sign, bin_ = kernel.aot_compile(
                *abst_args,
                num_warps=kernel_conf.compile_params.num_warps,
                num_stages=kernel_conf.compile_params.num_stages,
                force_nc_cache=kernel_conf.compile_params.force_nc_cache,
                **kernel_conf.meta,
            )

            new_ker_name = f"{ker_name}_{attr_sign.replace('<','').replace('>','_').replace(',','_')}"
            new_ker_name = new_ker_name[:-1]
            # TODO: cubin is probably better(?)
            ptx = bin_.asm("ptx").replace(ker_name, new_ker_name)

            ccode.add_kernel(
                new_ker_name,
                ptx,
                cc.CInputSigniture(
                    next(kernel_conf.signiture_iter()), kernels[ker_name].fn.arg_names
                ),
            )
        print(f"\t -> Done.")
    
    compiler = find_c_compiler()
    compiler("test_data/bin", ccode.gen_c_code())