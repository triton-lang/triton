from argparse import ArgumentParser

from aot_compile.c_codegen import CodeGenerator
from aot_compile.compile_metadata import compilation_metadata_from_args
from aot_compile.static_analysis import build_jit_stubs

import triton

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
    parser.add_argument(
        "--out-name",
        "-on",
        type=str,
        default=None,
        help="Out name for the compiled kernel",
    )
    args, unknown = parser.parse_known_intermixed_args()

    # execute python sources and extract functions wrapped in JITFunction
    src_files = [args.path]
    if args.include:
        src_files += args.include

    ast_gen_objects = build_jit_stubs(*src_files)

    kernel = ast_gen_objects[args.kernel_name]

    if args.out_name:
        kernel.__name__ = args.out_name

    metadata = compilation_metadata_from_args(
        kernel, kernel_args=unknown
    )

    # generate and dump C source code
    kwargs = {name: getattr(metadata, name) for name in ["signature", "specializations", "constants"]}
    handle = triton.compile(kernel, **kwargs)
    out_filename = Path(args.out_path).stem
    code_generator = CodeGenerator()
    code_generator.make_source(
        compile_meta=metadata,
        bin_=handle.asm["cubin"],
        out_filename=out_filename,
    ).dump_to_file(args.out_path)
