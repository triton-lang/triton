from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple
import regex as re

header = """
#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda.h>

#endif

unsigned char {binary_arg_name}[{bin_size}];
void load_{kernel_name}(void);
// tt-linker-name: {kernel_name}
// tt-linker-args: {signature}
CUresult {kernel_name}(CUstream stream, unsigned int gX,unsigned int gY,unsigned int gZ,unsigned int numWarps, {signature});
CUmodule {kernel_name}_mod;
CUfunction {kernel_name}_func;
"""

kernel_suffix_regex = re.compile("[0-9]+[c,d]")


def _name_and_specs(val):
    fname, specs = val.split("_")

    ty_to_size = {"d": 16, "c": 1}
    # assumes all kernel_suffix look like this <ARGNUM>d or <ARGNUM>c
    sizes = []
    for arg_ann in kernel_suffix_regex.findall(specs):
        argnum, arg_type = arg_ann[:-1], arg_ann[-1:]

        argnum = int(argnum)
        while len(sizes) < argnum:
            sizes.append(None)

        sizes.append(ty_to_size[arg_type])

    return fname, sizes


def _parse_args_and_sig(sig_args_str) -> Tuple[Sequence[str], str]:
    """
    Returns the arg names and signatrue
    """
    args = sig_args_str.split(",")
    names = []
    for arg in args:
        ty, name = arg.strip().split(" ")
        names.append(name.strip())

    return names, sig_args_str


def _make_dispatch_condition_expr(attr_names, attr_sizes: Sequence[int]):
    conds = []
    for arg, size_ in zip(attr_names, attr_sizes):

        if size_ is None:
            # non-specialized arg, no condition
            continue

        if size_ == 1:
            conds.append(f"{arg} == 1")
        else:
            conds.append(f"{arg} % {size_} == 0")

    return " & ".join(conds)


@dataclass
class LinkerInfo:
    sizes: Sequence[int]
    arg_names: Sequence[str]
    signature: str


def linker_info_from_header(header: str) -> Dict[str, Sequence[LinkerInfo]]:
    parsers = {
        "tt-linker-name": _name_and_specs,  # this takes kernel names like add_vec_0d1c2d3c and returns add_vec, [16,1,16,1]
        "tt-linker-args": _parse_args_and_sig,  # this takes sig string returns arg names and the sign string
    }

    metadata = defaultdict(list)

    # handle differnet linker meta-data comments
    for ln in header.split("\n"):
        if not ln.startswith("//"):
            continue
        line = ln.replace("//", "").strip()
        arg, val = line.split(":")
        val = val.strip()

        metadata[arg].append(parsers[arg](val))

    #  parse metadata into dict with kernel name as keys and meta as values
    # TODO: should I try and catch errors here? maybe as a start its better to let the C compiler catch those?
    kernels = defaultdict(list)
    for idx, (name, sizes) in enumerate(metadata["tt-linker-name"]):
        arg_names, c_signature_str = metadata["tt-linker-args"][idx]
        kernels[name].append(
            LinkerInfo(sizes=sizes, arg_names=arg_names, signature=c_signature_str)
        )

    return kernels

@dataclass
class CondAndCall:
    cond: str
    call_name: str

def parse_all_headers(headers: Sequence[str]):

    conds_per_kernel = defaultdict(list)
    signatures = {}
    arg_names = {}

    includes = []

    for header in headers:
        h_path = Path(header)
        h_str = h_path.read_text()
        includes.append(h_path.name)
        ker_info = linker_info_from_header(header=h_str)
        for k, v in ker_info.items():
            for s in v:
                signatures[k] = s.signature
                arg_names[k] = ",".join(s.arg_names)
                cond = _make_dispatch_condition_expr(s.arg_names, s.sizes)
                suffix = "".join(
                    [
                        f"{idx}{'d' if s == 16 else 'c'}"
                        for idx, s in enumerate(s.sizes)
                        if s is not None
                    ]
                )
            conds_per_kernel[k].append(
                CondAndCall(cond=cond, call_name=f"{k}_{suffix}")
            )  # condition and kernel name to call

    return includes, conds_per_kernel, signatures, arg_names


def make_dispatcher(name: str, includes: Sequence[str],signature: str, arg_names: str, conds_and_calls: Sequence[CondAndCall]):
    src = """


CUresult {kernel_name}(CUstream stream, unsigned int gX,unsigned int gY,unsigned int gZ,unsigned int numWarps, {signature}) {{
    {conds}
}} 
    """

    cond = """
    if ({cond}) {{
        return {call_name}(stream, gX, gY, gZ, {arg_names});
    }}
    """

    conds = []
    for cnc in conds_and_calls:
        conds.append(cond.format(cond=cnc.cond, call_name=cnc.call_name, arg_names=arg_names))

    conds = "\n".join(conds)

    

    dispatcher = src.format(kernel_name=name, conds=conds, signature=signature, includes=includes)
    return dispatcher


if __name__ == "__main__":
    import tempfile


    def dummy_header(name="add", suffix="0d1d2c3d"):
        binary_arg_name = "cubin"
        bin_size = 405
        binary_val = b"1234"
        kernel_name = f"{name}_{suffix}"
        signature = "CUdeviceptr X, CUdeviceptr Y, int32_t n, CUdeviceptr OUT"
        tdata = header.format(**locals())
        return tdata

    with tempfile.TemporaryDirectory() as tdir:
        (Path(tdir) / "dummy.h").write_text(dummy_header())
        (Path(tdir) / "mult.h").write_text(dummy_header(name="mult"))
        (Path(tdir) / "dummy2.h").write_text(dummy_header(suffix="0c3d"))
        (Path(tdir) / "mult2.h").write_text(dummy_header(name="mult", suffix="0c3d"))
        headers = Path(tdir).glob("*.h")

        inc, conds_calls, sigs, arg_names = parse_all_headers(headers)
        includes = '\n'.join([f'#include "{inc_}"' for inc_ in inc])

        disps = []
        for k, cncs in conds_calls.items():
            disps.append(make_dispatcher(name=k, includes=inc, signature=sigs[k], conds_and_calls=cncs, arg_names=arg_names[k]))

        fn = "\n\n".join(disps)
        print(f"{includes}\n\n{fn}")