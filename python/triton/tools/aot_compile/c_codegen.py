import binascii
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence, Tuple
from pathlib import Path

THREADS_PER_WARP = 32
from triton.compiler import ty_to_cpp
from .compile_metadata import AOTKernelMetadata, CompileMetadata

def _unzip(seq_of_pairs):
    A, B = [], []
    for (a, b) in seq_of_pairs:
        A.append(a)
        B.append(b)
    return A, B


@dataclass
class CodegenKernelData:
    kernel_name: str
    compiled_func_name: str
    """ compiler adds specialization info to kernel name. this is the full PTX name"""
    binary_arg_name: str
    """ the var names that stores the cubin/ptx in source"""
    bin_size: int
    """ lenght in bytes of kernels binary """
    binary_val: bytes
    signature: str
    """ Kernels input args signature """
    arg_pointers: str
    """ Singatures arguments by reference (passed to cuLaunchKernel) """
    num_args: int
    """ number of arguments """
    kernel_docstring: str
    threads_per_warp: int = THREADS_PER_WARP


def parse_aot_kerenl_metadata(
    kernel_meta: AOTKernelMetadata,
    compile_meta: CompileMetadata,
    bin_: bytes,
    docstring: str = None,
):
    kernel_name = kernel_meta.name
    compiled_func_name = compile_meta.compiled_function_name
    binary_arg_name = f"{kernel_name}_cubin"
    binary_val, bin_size = bytes_to_uchar_array(bin_)

    names, tt_types = _unzip((p.name, p.type_ann) for p in kernel_meta.params)
    signature, arg_pointers = signature_tt_to_c_args(names=names, tt_types=tt_types)

    return CodegenKernelData(
        kernel_name=kernel_name,
        compiled_func_name=compiled_func_name,
        binary_arg_name=binary_arg_name,
        bin_size=bin_size,
        binary_val=binary_val,
        signature=signature,
        arg_pointers=arg_pointers,
        num_args=len(names),
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
    default_launch: str
    user_launch: str


class CodeGenerator:
    def __init__(self, template_path: str = None):
        if template_path is None:
            template_path = Path(__file__).parent / "kernel.c"
        self._template = Path(template_path).read_text()
        common_h = parse_template(self._template).common_header
        self._headers = {"common": common_h}
        self._sources = {}

        self._summray_docstrings = {}

    def gen_kernel(
        self,
        kernel_meta: AOTKernelMetadata,
        compile_meta: CompileMetadata,
        bin_: bytes,
        docstring: str = None,
    ):
        codegen_data = parse_aot_kerenl_metadata(
            kernel_meta=kernel_meta,
            compile_meta=compile_meta,
            bin_=bin_,
            docstring=docstring,
        )
        _cgen_dict = asdict(codegen_data)
        _code = self._template.format(**_cgen_dict)
        code = parse_template(_code)

        file_name = codegen_data.kernel_name
        self._headers[file_name] = code.kernel_header
        source = [
            f'#include "{codegen_data.kernel_name}.h"',
            code.default_load,
            code.default_launch,
            code.user_launch,
        ]
        src_str = "\n".join(source)

        self._sources[file_name] = src_str

        self._summray_docstrings[file_name] = docstring

    def dump(self, out_dir: str):
        outdir = Path(out_dir)
        print(f"Summary of generated code in ({str(outdir)}):")
        for hname, txt in self._headers.items():
            _path = outdir / hname
            files = []
            with _path.with_suffix(".h").open("w") as fp:
                fp.write(txt)
                files.append(str(_path.with_suffix(".h").name))

            if hname in self._sources:
                src = self._sources[hname]
                with _path.with_suffix(".c").open("w") as fp:
                    fp.write(src)
                    files.append(str(_path.with_suffix(".c").name))

            print(f"\tFILES: {', '.join(files)}")
            if hname in self._summray_docstrings:
                print(f"\t{self._summray_docstrings[hname]}")
                print("")
            print("")


def parse_template(template_src: str):
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
