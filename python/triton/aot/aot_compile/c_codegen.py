import binascii
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence, Tuple
from pathlib import Path

THREADS_PER_WARP = 32
# from triton.compiler import ty_to_cpp
from .compile_metadata import AOTKernelMetadata, CompileMetadata


@dataclass
class TemplateFields:
    kernel_name: str
    compiled_func_name: str
    """ compiler adds specialization info to kernel name. this is the full PTX name"""
    binary_arg_name: str
    """ the var names that stores the cubin/ptx in source"""
    bin_size: int
    """ lenght in bytes of kernels binary """
    binary_val: str 
    """ Binary as hex string """
    signature: str
    """ Kernels input args signature """
    arg_pointers: str
    """ Singatures arguments by reference (passed to cuLaunchKernel) """
    num_args: int
    """ number of arguments """
    kernel_docstring: str
    threads_per_warp: int = THREADS_PER_WARP


def metadata_to_template_strings(
    compile_meta: CompileMetadata,
    bin_: bytes,
    docstring: str = None,
):
    kernel_name = compile_meta.compiled_function_name 
    binary_arg_name = f"{kernel_name}_cubin"
    binary_val, bin_size = bytes_to_uchar_array(bin_)

    signature, arg_pointers = signature_tt_to_c_args(names=compile_meta.arg_names, tt_types=compile_meta.signature)
    num_args = len(compile_meta.arg_names)

    return TemplateFields(
        kernel_name=kernel_name,
        compiled_func_name=kernel_name,
        binary_arg_name=binary_arg_name,
        bin_size=bin_size,
        binary_val=binary_val,
        signature=signature,
        arg_pointers=arg_pointers,
        num_args=num_args,
        threads_per_warp=THREADS_PER_WARP,
        kernel_docstring=docstring or "",
    )


def py_str_to_uchar_array(txt: str) -> Tuple[str, int]:  # (c_code, array len)
    """Hexdump as string into a C array"""
    arr = bytes(txt, "utf-8")
    return bytes_to_uchar_array(arr)


def bytes_to_uchar_array(arr: bytes) -> Tuple[str, int]:
    hex_ = str(binascii.hexlify(arr))[2:-1]
    it = iter(hex_)
    data = ", ".join([f"0x{x}{next(it)}" for x in it])
    return data, len(hex_)


def signature_tt_to_c_args(
    names: Sequence[str], tt_types: Sequence[str]
) -> Tuple[str, str]:  # (args function signature, args array of pointers)
    """
    Generate signature code from arg names and Triton type annotations
    e.g.
        CUDeviceptr X, int32_t n_elem
        void *args[2] = { &X, &n_elem };
    :return:
        (args function signature, args array of pointers)
    """
    args_signature = ", ".join(
        [f"{ty_to_cpp(ty)} {name}" for name, ty in zip(names, tt_types)]
    )
    arg_pointers = ", ".join([f"&{arg}" for arg in names])
    # args_ptr_array = f"void *args[{len(tt_types)}] = {{ {arg_pointers} }};"
    return args_signature, arg_pointers


@dataclass
class CodegenTemplates:
    common_header: str
    kernel_header: str
    default_load: str
    """ Load cuda module and function using a global CUmodule & CUfunction pointers"""
    default_launch: str
    """ Launch kernel using the global CUfunction pointers """
    user_launch: str
    """ User handles kernel loading, and passes CUfunction as an argument """

@dataclass
class KernelCSource:
    """ Header and source strings """
    header: str
    source: str
    docstring: str
    name: str

    def dump_to_file(self, fname: str):
        _path = Path(fname)
        h, c =_path.with_suffix(".h"), _path.with_suffix(".c") 

        with h.open("w") as fp:
                fp.write(self.header)
        
        with c.open("w") as fp:
            fp.write(self.source)

        print(f"\tWritten {self.name} to [{', '.join([str(h), str(c)])}]")
        if self.docstring:
            print(f"\t{self.docstring}")
            print("")
        print("")
        return

class CodeGenerator:
    def __init__(self, template_path: str = None):
        if template_path is None:
            template_path = Path(__file__).parent / "kernel.c"
        self._template = Path(template_path).read_text()
        self._common_h = split_template(self._template).common_header
        self._gen_code = []

    def make_source(
        self,
        compile_meta: CompileMetadata,
        bin_: bytes,
        docstring: str = None,
    ) -> KernelCSource:

        template_fields = metadata_to_template_strings(
            compile_meta=compile_meta,
            bin_=bin_,
            docstring=docstring,
        )
        _fields_dict = asdict(template_fields)
        _code = self._template.format(**_fields_dict)
        code = split_template(_code)

        header = code.kernel_header
        source = [
            f'#include "{template_fields.kernel_name}.h"',
            code.default_load,
            code.default_launch,
            code.user_launch,
        ]
        src_str = "\n".join(source)

        return KernelCSource(source=src_str, header=header, docstring=docstring,name=template_fields.kernel_name)


    def gen_kernel(
        self,
        kernel_meta: AOTKernelMetadata,
        compile_meta: CompileMetadata,
        bin_: bytes,
        docstring: str = None,
    ):
        generated_code = self.make_source(kernel_meta=kernel_meta, compile_meta=compile_meta, bin_=bin_, docstring=docstring)
        self._gen_code.append(generated_code)
        file_name = generated_code.name


        self._headers[file_name] = generated_code.header 
        self._sources[file_name] = generated_code.source 
        self._summray_docstrings[file_name] = generated_code.docstring 

    def dump(self, out_dir: str):
        outdir = Path(out_dir)
        print(f"Summary of generated code in ({str(outdir)}):")

        with (out_dir /"common.h").open("w") as fp:
                fp.write(self._common_h)

        code: KernelCSource
        for code in self._gen_code:
            file_name = code.name
            code.dump_to_file(file_name)

def split_template(template_src: str):
    """
    Helper func to keep sanity when dealing with C code inside a python string
    """

    _is_sep = lambda line: line.startswith("/*[") and line.endswith("]*/")

    snippets = defaultdict(list)

    lines = template_src.split("\n")
    i = 0
    if not _is_sep(lines[0]):
        raise ValueError(f"need to start with separator line /*[ ]*/ found: {lines[0]}")

    while i < len(lines):
        line = lines[i]
        section_name = line.replace("/*[", "").replace("]*/", "").strip()
        i += 1
        while i < len(lines) and not _is_sep(lines[i]):
            snippets[section_name].append(lines[i])
            i += 1

    c_source_template = {k: "\n".join(v) for k, v in snippets.items()}
    c_source_template[
        "common_header"
    ] = """
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda.h>


typedef struct
{
    unsigned int gX;
    unsigned int gY;
    unsigned int gZ;
    unsigned int numWarps;
} GridWarps;
    """

    return CodegenTemplates(**c_source_template)
