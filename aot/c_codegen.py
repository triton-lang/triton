from pathlib import Path
from dataclasses import dataclass
import binascii
from typing import Sequence, Tuple


from abstract_values import AbstractPtr, AbstractValue

@dataclass
class LibCudaConf:
    include_dir: str
    link_dir: str
    

@dataclass
class CSources:
    sources: Sequence[str]
    headers: Sequence[str]
    lib_name: str
    
    def iter_header_and_sources(self):
        for hdr, src in zip(self.headers, self.sources):
            yield hdr, src

@dataclass
class CInputSigniture:
    """Build kernel specific function signiture"""

    # TODO: make sure the mapping is correct
    _TRITON_TO_C_TYPE = {
        "I": "int32_t",
        "f": "float",
        "B": "bool",
        "f8": "float",
        "f16": "float",
        "bf16": "float",
        "f32": "float",
        "f64": "double",
        "i1": "bool",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
    }

    input_types: Sequence[AbstractValue]
    arg_names: Sequence[str]

    def __len__(self):
        return len(self.input_types)

    def signitue_c_code(self) -> str:
        sig = ""
        for arg, ty in zip(self.arg_names, self.input_types):
            if isinstance(ty, AbstractPtr):
                dtype = "CUdeviceptr"
            else:
                dtype = CInputSigniture._TRITON_TO_C_TYPE[ty.tt_dtype]
            sig = f"{sig}{dtype} {arg}, "
        if len(sig):
            sig = sig[:-2]
        return sig


def py_str_to_uchar_array(txt: str) -> Tuple[str, int]:  # (c_code, array len)
    """Hexdump as string into a C array"""
    hex_ = str(binascii.hexlify(bytes(txt, "utf-8")))[2:-1]
    it = iter(hex_)
    data = ", ".join([f"0x{x}{next(it)}" for x in it])
    return data, len(hex_)


# CUdeviceptr x, CUdeviceptr y, CUdeviceptr res, size_t size
def gen_kernel_code(name: str, ptx_src: str, data_sign: CInputSigniture) -> str:
    ptx_as_hex, ptx_size = py_str_to_uchar_array(ptx_src)

    ptx_def = f"unsigned char {name}_ptx[{ptx_size}]"
    kernel_fn_def = (
        f"CUresult {name}(CUstream stream, GridWarps g, {data_sign.signitue_c_code()})"
    )

    get_func_def = f"CUfunction {name}_fn()"

    # === Kernel header ===
    header_src = [f"{ptx_def};", f"{get_func_def};", f"{kernel_fn_def};"]

    # === PTX data ===
    ptx_data_src = [f"{ptx_def} = ", "{", f" {ptx_as_hex}", "};"]

    # === Functions ===
    _if_error = ["if (err != 0)", "{", "\treturn NULL;", "}"]

    def _func_wrapper(fn_def, fn_body):
        return [f"{fn_def}", "{", *fn_body, "}"]

    # ==== Load Kernel Func ====
    load_kernel_body = [
        "CUmodule mod_ptr;",
        "CUfunction func;",
        "CUresult err;",
        f"void *image = (void *)&{name}_ptx;",
        "err = cuModuleLoadData(&mod_ptr, image);",
        *_if_error,
        f'err = cuModuleGetFunction(&func, mod_ptr, "{name}");',
        *_if_error,
        "return func;",
    ]

    load_kernel_fn_src = _func_wrapper(get_func_def, load_kernel_body)

    # ==== Run Kernel Func ====
    args = "\n".join(["{", ", ".join([f"&{arg}" for arg in data_sign.arg_names]), "}"])

    kernel_fn_body = [
        f"CUfunction func = {name}_fn();",
        f"void *args[{len(data_sign)}] = {args};",
        "return cuLaunchKernel(func, g.gX, g.gY, g.gZ, g.numWarps * 32, 1, 1, 0, stream, args, NULL);",
    ]

    kernel_fn_src = _func_wrapper(kernel_fn_def, kernel_fn_body)

    return header_src, ptx_data_src, load_kernel_fn_src, kernel_fn_src


class SharedLibrarySource:
    def __init__(self, lib_name: str, output_dir=None) -> None:

        self._lib_name = lib_name
        self._output_dir = output_dir

        self.header_tmplt = [
            "#include <stdio.h>",
            "#include <stdint.h>",
            "#include <inttypes.h>",
            "#include <cuda.h>",
            "", "",
            "typedef struct",
            "{",
            "int gX;",
            "int gY;",
            "int gZ;",
            "int numWarps;",
            "} GridWarps;",
            "", "",
        ]

        self.src_ptx_data = []
        self.src_get_funcs = []
        self.src_launch_funcs = []

    def __enter__(self):
        return self

    def __exit__(self):
        return False

    def _wrapped_header(self):
        return (
            ["#ifndef TRITON_KERNELS_H", "#define TRITON_KERNEL_H"]
            + self.header_tmplt
            + ["#endif"]
        )

    def _wrapped_source(self):
        return (
            [
                "#include <stdio.h>",
                "#include <stdint.h>",
                "#include <inttypes.h>",
                f'#include "{self._lib_name}.h"',
            ]
            + self.src_ptx_data
            + self.src_get_funcs
            + self.src_launch_funcs
        )

    def add_kernel(self, name: str, ptx_src: str, data_sign: CInputSigniture):
        header_src, ptx_data_src, load_ker_fn, launch_ker_fn = gen_kernel_code(
            name, ptx_src, data_sign
        )
        self.header_tmplt += header_src + ["", ""]
        self.src_ptx_data += ptx_data_src
        self.src_get_funcs += load_ker_fn
        self.src_launch_funcs += launch_ker_fn

    def get_source(self) -> str:
        return "\n".join(self._wrapped_source())

    def get_header(self) -> str:
        return "\n".join(self._wrapped_header())

    def gen_c_code(self) -> CSources:
        base = Path(self._output_dir)
        header = base / f"{self._lib_name}.h"
        source = base / f"{self._lib_name}.c"
        srcs = CSources(
            headers=[str(header)],
            sources=[str(source)],
            lib_name=self._lib_name
        )
        
        # TODO: error handling
        try:
            header.write_text(self.get_header())
        except Exception:
            raise Exception(f" {srcs.headers[0]} header write failed")
        # TODO: error handling
        try:
            source.write_text(self.get_source())
        except Exception:
            raise Exception(f" {srcs.sources[0]} source write failed")

        return srcs
