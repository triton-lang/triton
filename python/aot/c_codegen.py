import binascii
from dataclasses import dataclass
from typing import Sequence, Tuple

from .abstract_values import TRITON_TO_C_TYPES
from .sig_annotation_dsl import SignatureTokens, tokenize_signature_annotation

THREADS_PER_WARP = 32


def py_str_to_uchar_array(txt: str) -> Tuple[str, int]:  # (c_code, array len)
    """Hexdump as string into a C array"""
    hex_ = str(binascii.hexlify(bytes(txt, "utf-8")))[2:-1]
    it = iter(hex_)
    data = ", ".join([f"0x{x}{next(it)}" for x in it])
    return data, len(hex_)


common_header = """
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda.h>

typedef struct
{
    int gX;
    int gY;
    int gZ;
    int numWarps;
} GridWarps;

"""


@dataclass
class KernelSignatureData:
    pointer_names: Sequence[str]
    pointer_types: Sequence[str]
    attribute_names: Sequence[str]
    attribute_types: Sequence[str]
    attr_size_vars: Sequence[str]

    @property
    def total_kernel_arguments(self):
        return len(self.pointer_names) + len(self.attribute_names)

    @property
    def all_argument_names(self):
        return self.pointer_names + self.attribute_names

    @classmethod
    def parse_sig_dsl(cls, sig_ann: str, arg_names: Sequence[str]):
        tokens: SignatureTokens  = tokenize_signature_annotation(sig_ann, arg_names)
        return KernelSignatureData(
            pointer_names=tokens.pointer_names,
            pointer_types=tokens.pointers,
            attribute_names=tokens.attribute_names,
            attribute_types=tokens.attributes,
            attr_size_vars=tokens.attribute_sizes
        )



# def make_kernel_signature(poiniter_types: Sequence[str], attribute_types: Sequence[str], arg_names: Sequence[str]):
def make_kernel_signature(sig_data: KernelSignatureData):
    """
    Assuming that kernel signature follow the following pattern (pointers..., attribure..., constexpr...)
    """
    pointers = [f"CUdeviceptr {arg}" for arg in sig_data.pointer_names]
    attributes = []
    for ty, arg in zip(sig_data.attribute_types, sig_data.attribute_names):
        dtype = TRITON_TO_C_TYPES[ty]
        attributes.append(f"{dtype} {arg}")

    return ", ".join(pointers + attributes)


# def _make_kernel_header_source(name: str, arg_names: Sequence[str], ptr_types: Sequence[str], attr_types: Sequence[str], hex_, bin_size: int, threads_per_warp: int) -> Tuple[str, str]:
def _make_kernel_header_source(
    name: str, sig_data: KernelSignatureData, hex_, bin_size: int, threads_per_warp: int
) -> Tuple[str, str]:
    kernel_signature = make_kernel_signature(sig_data)
    arg_pointers = ", ".join(
        [f"&{arg}" for arg in sig_data.pointer_names + sig_data.attribute_names]
    )
    num_args = sig_data.total_kernel_arguments

    header = """
unsigned char {name}_ptx[{bin_size}];
CUfunction load_{name}(void);
CUresult {name}(CUstream stream, GridWarps g, {kernel_signature});
"""

    source = """
unsigned char {name}_ptx[{bin_size}] = 
{{
    {hex_}
}};

CUfunction load_{name}(void)
{{
    CUmodule mod_ptr;
    CUfunction func;
    CUresult err;
    void *image = (void *)&{name}_ptx;
    err = cuModuleLoadData(&mod_ptr, image);
    if (err != 0) {{
        return NULL;
    }}
    err = cuModuleGetFunction(&func, mod_ptr, "{name}");
    if (err != 0) {{
        return NULL;
    }}
    return func;
}}

CUresult {name}(CUstream stream, GridWarps g, {kernel_signature})
{{
    CUfunction func = load_{name}();
    void *args[{num_args}] = {{ {arg_pointers} }};
    return cuLaunchKernel(func, g.gX, g.gY, g.gZ, g.numWarps * {threads_per_warp}, 1, 1, 0, stream, args, NULL);
}}
    """

    return header.format(**locals()), source.format(**locals())


def _make_dispatch_condition_expr(attr_names, attr_sizes: Sequence[int]):
    return " & ".join([f"{k} % {v} == 0" for k, v in zip(attr_names, attr_sizes)])


def make_kernel_dispatcher_header_source(
    name: str,
    binaries: Sequence[str],
    attr_sizes: Sequence[Sequence[int]],
    sig_data: KernelSignatureData,
    common_h: str = common_header
):
    kernel_names = []
    kernel_headers = []
    kernel_src = []
    dispatch_cond_exprs = []

    # collect all individual kernels that should be dispatched
    for idx, (binary, attr_size) in enumerate(zip(binaries, attr_sizes)):
        new_ker_name = f"{name}_" + "_".join(map(str, attr_size))
        kernel_names.append(new_ker_name)
        # TODO: this assumes we use PTX (we are not going to use that later on)
        new_bin = binary.replace(name, new_ker_name)
        hex_, bin_size = py_str_to_uchar_array(new_bin)

        # we are buildig a dispatcher function that checks input sizes and calls a kernel according to those sizes
        # this is the actual kernel call 
        dispatched_func = f"return {new_ker_name}(stream, g, {', '.join(sig_data.all_argument_names)});"  

        # there is an if condition to check the sizes of all kernels expect one
        # if we failed on all conditions, assume the last kerenel is the correct one
        # TODO: if none of the kernels are a good fit, we gonna suffer performace degradation. 
        # TODO: Maybe should raise an error or at least a warning that the sizes are wrong?
        if idx < len(binaries) - 1:
            dispatch_condition_expr = _make_dispatch_condition_expr(
                sig_data.attribute_names, attr_size
                )   
            dispatch = f"if ({dispatch_condition_expr}) \n\t {dispatched_func}"
        else:
            dispatch = dispatched_func 

        # This actually takes the binary generated by Triton and wraps a cuda call to it.
        # TODO: what if we use some other, non-cuda backend? this whole setup needs to change in that case
        # TODO: multiple backend support is not something that the AOT supports at this point.
        h, src = _make_kernel_header_source(
            new_ker_name, sig_data, hex_, bin_size, threads_per_warp=THREADS_PER_WARP
        )

        kernel_headers.append(h)
        kernel_src.append(src)
        dispatch_cond_exprs.append(dispatch)

    dispatcher_sig = f"CUresult {name}(CUstream stream, GridWarps g, {make_kernel_signature(sig_data)})"
    dispatcher_body = "\n".join(dispatch_cond_exprs)
    dispatcher_func = """
{dispatcher_sig} {{
{dispatcher_body}
}}
"""
    data = {            
        "name": name,
        "common": common_h,
        "kernel_headers": "\n\n".join(kernel_headers),
        "kernel_sources": "\n\n".join(kernel_src),
        "dispatcher_header": f"{dispatcher_sig};",
        "dispatcher_source": dispatcher_func.format(dispatcher_sig=dispatcher_sig, dispatcher_body=dispatcher_body),
        }

    header = "{common}\n{kernel_headers}\n{dispatcher_header}".format(**data)

    source = """
#include "{name}.h"

{kernel_sources}

{dispatcher_source}
""".format(**data)

    return header, source
