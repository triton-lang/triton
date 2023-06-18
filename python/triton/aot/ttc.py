import binascii
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Tuple

from aot_compile.static_analysis import build_jit_stubs
from dataclasses import dataclass

import triton
from triton.compiler.code_generator import kernel_suffix
from triton.compiler.make_launcher import ty_to_cpp


def py_str_to_uchar_array(txt: str) -> Tuple[str, int]:  # (c_code, array len)
    """Hexdump as string into a C array"""
    arr = bytes(txt, "utf-8")
    return bytes_to_uchar_array(arr)


def bytes_to_uchar_array(arr: bytes) -> Tuple[str, int]:
    hex_ = str(binascii.hexlify(arr))[2:-1]
    it = iter(hex_)
    data = ", ".join([f"0x{x}{next(it)}" for x in it])
    return data, len(hex_)


@dataclass
class CodegenTemplates:
    # common_header: str
    kernel_header: str


@dataclass
class KernelCSource:
    """Header and source strings"""
    header: str
    source: str
    docstring: str
    name: str

    def dump_to_file(self, fname: str):
        _path = Path(fname)
        with _path.with_suffix(".h").open("w") as fp:
            fp.write(self.header)
        with _path.with_suffix(".c").open("w") as fp:
            fp.write(self.source)


class CodeGenerator:
    def __init__(self, template_path: str = None):
        if template_path is None:
            template_path = Path(__file__).parent / "aot_compile" / "template.c"
        self._template = Path(template_path).read_text()
        self._gen_code = []

    def make_source(
        self,
        signature,
        arg_names,
        func_name,
        docstr,
        shared,
        bin_: bytes, out_filename: str = None
    ) -> KernelCSource:

        if out_filename is None:
            out_filename = func_name

        binary_val, bin_size = bytes_to_uchar_array(bin_)
        signature = ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, signature.values())])
        arg_pointers = ", ".join([f"&{arg}" for arg in arg_names])
        num_args = len(arg_names)
        _fields_dict = {
            "kernel_name": func_name,
            "bin_size": bin_size,
            "binary_val": binary_val,
            "signature": signature,
            "arg_pointers": arg_pointers,
            "num_args": num_args,
            "kernel_docstring": docstr,
            "shared": shared,
        }

        _code = self._template.format(**_fields_dict)
        code = split_template(_code)

        header = code.kernel_header
        source = [
            f'#include "{out_filename}.h"',
        ]
        src_str = "\n".join(source)

        return KernelCSource(
            source=src_str,
            header=header,
            docstring=docstr,
            name=func_name,
        )

    def dump(self, out_dir: str):
        outdir = Path(out_dir)
        print(f"Summary of generated code in ({str(outdir)}):")

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

    return CodegenTemplates(**c_source_template)


if __name__ == "__main__":

    # command-line arguments
    parser = ArgumentParser(description="Triton Ahead of Time Compilation (AoT)")
    parser.add_argument(
        "path",
        help="Path to Python source that contains JITFunction in scope (note the source will be executed)",
    )
    parser.add_argument("--kernel-name", "-n", type=str, default="")
    parser.add_argument("--include", "-I", nargs="*", help="")
    parser.add_argument("--out-path", "-o", type=Path, help="Out filename")
    parser.add_argument("--out-name", "-on", type=str, default=None, help="Out name for the compiled kernel")
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel")
    args = parser.parse_args()

    # # execute python sources and extract functions wrapped in JITFunction
    src_files = [args.path]
    if args.include:
        src_files += args.include
    ast_gen_objects = build_jit_stubs(*src_files)
    kernel = ast_gen_objects[args.kernel_name]

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        return None
    hints = {i: constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constexprs = {i: constexpr(s) for i, s in enumerate(signature)}
    constexprs = {k: v for k, v in constexprs.items() if v is not None}
    signature = {i: s.split(":")[0] for i, s in enumerate(signature) if i not in constexprs}

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    divisible_by_16 = [i for i, h in hints.items() if h == 16]
    equal_to_1 = [i for i, h in hints.items() if h == 1]
    config = triton.compiler.instance_descriptor(divisible_by_16=divisible_by_16, equal_to_1=equal_to_1)
    ccinfo = triton.compile(kernel, signature=signature, constants=constexprs, config=config)
    arg_names = [kernel.arg_names[i] for i in signature.keys()]

    # dump C stub code
    out_filename = Path(args.out_path).stem
    func_name = '_'.join([kernel.__name__, kernel_suffix(signature.values(), config)])
    code_generator = CodeGenerator()
    code_generator.make_source(
        signature,
        arg_names,
        func_name,
        "",
        shared=ccinfo.shared,
        bin_=ccinfo.asm["cubin"],
        out_filename=out_filename,
    ).dump_to_file(args.out_path)
