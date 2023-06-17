from argparse import ArgumentParser

from aot_compile.c_codegen import CodeGenerator
from aot_compile.static_analysis import build_jit_stubs

import triton
from triton.compiler.code_generator import kernel_suffix

if __name__ == "__main__":
    from pathlib import Path

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
            print(ret)
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
