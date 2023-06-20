import binascii
from argparse import ArgumentParser
from pathlib import Path

from static_analysis import build_jit_stubs

import triton
from triton.compiler.code_generator import kernel_suffix
from triton.compiler.make_launcher import ty_to_cpp

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

    # execute python sources and extract functions wrapped in JITFunction
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
    ccinfo = triton.compile(kernel, signature=signature, constants=constexprs, configs=[config], num_warps=1)
    arg_names = [kernel.arg_names[i] for i in signature.keys()]

    # dump C stub code
    suffix = kernel_suffix(signature.values(), config)
    func_name = '_'.join([kernel.__name__, suffix])
    hex_ = str(binascii.hexlify(ccinfo.asm["cubin"]))[2:-1]
    params = {
        "kernel_name": func_name,
        "bin_size": len(hex_),
        "bin_data": ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        "signature": ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, signature.values())]),
        "arg_pointers": ", ".join([f"&{arg}" for arg in arg_names]),
        "num_args": len(arg_names),
        "kernel_docstring": "",
        "shared": ccinfo.shared,
    }
    for ext in ['h', 'c']:
        template_path = Path(__file__).parent / f"template.{ext}"
        with args.out_path.with_suffix(f".{suffix}.{ext}").open("w") as fp:
            fp.write(Path(template_path).read_text().format(**params))
