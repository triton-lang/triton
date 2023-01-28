from argparse import ArgumentParser, Namespace
from typing import Dict, Sequence, Tuple, Union

import triton
from aot_compile.c_codegen import CodeGenerator
from aot_compile.compile_metadata import (
    AOTKernelMetadata,
    ASTGeneratingObject,
    compilation_metadata_from_args,
    CompileMetadata,
)

from aot_compile.static_analysis import build_jit_stubs
from aot_compile.static_analysis import JITStub


ASMObject = Dict[str, Union[str, bytes]]
""" The `asm` attr from `CompiledKernel` """


def generate_asm(
    kernel: JITStub,
    meta: CompileMetadata,
) -> ASMObject:

    kwargs = {
        "signature": meta.signature,
        "configs": [meta.specializations],
        "constants": meta.constants,
    }
    print(f"Compiling with name: {kernel.__name__}")
    compiled = triton.compile(kernel, **kwargs)
    asm = compiled.asm

    return asm


def generate_c_code(kernel: JITStub, meta: CompileMetadata, out_path: str, out_format: str="cubin"):
    """
    Code generation goes like this:
        - each jitted function gets replicated according to the number of constexpr variants it has
        - each variant has its own header and source (with binary/ptx stored in source)
        - common header is generated for includes and utils
    """
    # C Code generation.
    codegen = CodeGenerator()

    asm = generate_asm(kernel=kernel, meta=meta)
    if out_format not in asm:
        raise ValueError(f"{out_format} is not a supported output format")

    out_filename = Path(out_path).name

    codegen.make_source(
        compile_meta=meta,
        bin_=asm[out_format],
        out_filename=out_filename,
    ).dump_to_file(out_path)


if __name__ == "__main__":
    from pathlib import Path

    parser = ArgumentParser(description="Triton Ahead of Time Compilation (AoT)")
    parser.add_argument(
        "path",
        help="Path to Python source that contains JITFunction in scope (note the source will be executed)",
    )
    parser.add_argument("--kernel-name", "-n", type=str, default="")
    parser.add_argument("--include", "-I", nargs="*", help="")

    parser.add_argument("--out-path", "-o", type=Path, help="Out filename")
    parser.add_argument(
        "--out-name",
        "-on",
        type=str,
        default=None,
        help="Out name for the compiled kernel",
    )
    parser.add_argument("--target", "-t", default="Csource", type=str)

    args, unknown = parser.parse_known_intermixed_args()

    # execute python sources and extract functions wrapped in JITFunction
    src_files = [args.path]
    if args.include:
        src_files += args.include

    ast_gen_objects = build_jit_stubs(*src_files)

    kernel_to_compile = ast_gen_objects[args.kernel_name]

    if args.out_name:
        print(args.out_name)
        kernel_to_compile.__name__ = args.out_name

    compile_meta = compilation_metadata_from_args(
        kernel_to_compile, kernel_args=unknown
    )

    # TODO: support different compilation stages (similar to aot.py)
    if args.target == "Csource":
        generate_c_code(
            kernel=kernel_to_compile,
            meta=compile_meta,
            out_path=args.out_path,
        )
