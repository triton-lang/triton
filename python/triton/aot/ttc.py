from argparse import ArgumentParser
from typing import Callable, Dict, Sequence, Tuple, Union
import yaml
from aot_compile.compile_metadata import CompileMetadata

import triton
from aot_compile.c_codegen import CodeGenerator
from aot_compile.compile_metadata import (
    AOTKernelMetadata,
    ASTGeneratingObject,
    infer_triton_signature,
    make_compile_metadata,
    CompileMetadata,
)

from aot_compile.static_analysis import build_jit_stubs


def infer_config_dump_stdout(jitted: Dict[str, Callable]) -> str:
    sigs = {k: infer_triton_signature(v) for k, v in jitted.items()}
    print(yaml.safe_dump(sigs))


def parse_config_file(conf_file) -> Dict[str, Sequence[AOTKernelMetadata]]:
    conf = Path(conf_file)
    if not conf.exists():
        raise ValueError(f"Config File Not Found [{conf_file}]")

    sigs = yaml.safe_load(conf.open("r"))

    aot_meta = {
        name: AOTKernelMetadata.parse(name, meta) for name, meta in sigs.items()
    }

    return aot_meta


AOTMetadataWithAST = Dict[str, Tuple[Sequence[AOTKernelMetadata], ASTGeneratingObject]]
""" Mapping: 
    name -> ([`AOTKernelMetadata`...], `JITFunction`) """


def reconcile_metadata_with_ast(
    aot_meta: Dict[str, Sequence[AOTKernelMetadata]],
    jitted: Dict[str, ASTGeneratingObject],
) -> AOTMetadataWithAST:
    """

    Take metadata and jitted functions found and make sure those match.
    Kernels with missing jit function or metadata are skipped

    :return:
        Dict with function names (major names)
    """

    # TODO: add some checks to make sure signatures are the same
    valid_meta_func_paris = {}
    for conf_func in aot_meta:
        if conf_func not in jitted:
            print(
                f"[Skipping] {conf_func} -- is not a valid function in sources provided"
            )
            continue
        metas, fn = aot_meta[conf_func], jitted[conf_func]
        valid_meta_func_paris[conf_func] = (metas, fn)

    # check what functoin are imported and have no metadata specified
    for fn_name in set(jitted.keys()) - set(aot_meta.keys()):
        print(f"[Missing Meta]: {fn_name} is imported but has no metadata")
        # TODO: try and infer the missing funcitons?

    if not len(valid_meta_func_paris):
        raise ValueError(f"Config file has no configs for source kernels")

    return valid_meta_func_paris


ASMObject = Dict[str, Union[str, bytes]]
""" The `asm` attr from `CompiledKernel` """


def compile_func(
    meta: AOTKernelMetadata, fn: ASTGeneratingObject
) -> Tuple[ASMObject, CompileMetadata]:
    # NOTE: we ovveride the function name here. in case we have several variants in the kerenel
    #       (those will have same signature in C)
    old_name = fn.__name__
    fn.__name__ = meta.name

    compile_metadata = make_compile_metadata(meta)
    compiled = triton.compile(fn, **compile_metadata.kwargs)
    asm = compiled.asm

    fn.__name__ = old_name
    return asm, compile_metadata


def generate_c_code(meta_and_ast: AOTMetadataWithAST, outdir: str):
    """
    Code generation goes like this:
        - each jitted function gets replicated according to the number of constexpr variants it has
        - each variant has its own header and source (with binary/ptx stored in source)
        - common header is generated for includes and utils
    """
    # C Code generation.
    codegen = CodeGenerator()

    for fn_name, (metas, jit_fn) in meta_and_ast.items():
        for variant_idx, meta in enumerate(metas):
            asm, comple_meta = compile_func(meta, jit_fn)
            docstr = str(meta)
            if jit_fn.__doc__ is not None:
                docstr = f"{docstr}\n\t{jit_fn.__doc__}"
            codegen.gen_kernel(
                kernel_meta=meta,
                compile_meta=comple_meta,
                bin_=asm["cubin"],
                docstring=docstr,
            )

    codegen.dump(out_dir=outdir)


if __name__ == "__main__":
    from pathlib import Path

    parser = ArgumentParser(description="Triton Ahead of Time Compilation (AoT)")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Path to Python source that contains JITFunction in scope (note the source will be executed)",
    )
    parser.add_argument(
        "--infer-signature",
        "-infer",
        action="store_true",
        help="Output inferred compile annotations as an stdout YAML",
    )
    parser.add_argument("--config", default="dummy.yml")
    parser.add_argument("--out-dir", "-o", type=Path)
    parser.add_argument("--target", "-t", default="Csource", type=str)

    args = parser.parse_args()

    # execute python sources and extract functions wrapped in JITFunction
    ast_gen_objects = build_jit_stubs(*args.paths)

    # This generated a YAML file with needed configs for AOT (signature annotation, specialization values)
    # If you annotate your kernel inputs with type annotations  e.g. X: *fp32(16), config will automatically populate those
    # otherwise you'll need to edit the YAML file manually
    # You can also provide default values for constants e.g. BLOCK_SIZE = 32 and those will be populated
    # otherwise, you need to provide constant values manually
    # In the YAML file you can provide several values, to generate different kernel variants.
    # ( all constants must have same number of variants)
    if args.infer_signature:
        infer_config_dump_stdout(ast_gen_objects)
        exit(0)

    # When you have fully filled config YAML we use it to infer meta data needed for compilation
    aot_metadata = parse_config_file(args.config)
    matched_metas_with_fn = reconcile_metadata_with_ast(
        aot_meta=aot_metadata, jitted=ast_gen_objects
    )

    # TODO: support different compilation stages (similar to aot.py)
    if args.target == "Csource":
        generate_c_code(meta_and_ast=matched_metas_with_fn, outdir=args.out_dir)
