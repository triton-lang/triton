import ast
import os
from argparse import ArgumentParser
from types import ModuleType
from typing import Callable, Dict, Sequence, Tuple, Union
import yaml
from aot_compile.compile_metadata import CompileMetadata

import triton
from aot_compile.c_codegen import CodeGenerator
from aot_compile.compile_metadata import (AOTKernelMetadata, ASTGeneratingObject,
                                          infer_triton_signature, make_compile_metadata, CompileMetadata)


def filter_jitted_functions(*scopes):
    """
    Filter scopes for JITFunction objects
    """
    res = {}
    for scope in scopes:
        if isinstance(scope, ModuleType):
            scope = scope.__dict__
        all_jitted = {
            v.__name__: v for v in scope.values() if isinstance(v, ASTGeneratingObject)
        }
        res.update(all_jitted)
    return res



def _extract_kernel_src_from_ast(src: str, fpath: str):
    """
    Parse source and extract functions that are jit compiled

    Why is this here? 
    When executing modules from string with exec function, inspect has no access to the source code.
    This function extracts sources of functions and passes those to exec as global scope data.

    Assumption:
    - functions must be decorated with the tirton.jit (or anything that has `jit` in the name of the decorator)
    """

    lines = src.split(os.linesep)

    def _is_jit(n):
        if isinstance(n, ast.Attribute):
            return 'jit' in n.attr
        if isinstance(n, ast.Name):
            return 'jit' in n.id

        return False

    tree = ast.parse(src)
    kernels = {}
    if isinstance(tree, ast.Module):
        tree = tree.body
        for node in tree:
            if isinstance(node, ast.FunctionDef):
                is_jitted = any(_is_jit(dec) for dec in node.decorator_list)
                if is_jitted:
                    st = node.lineno - 1
                    en = node.end_lineno
                    kernels[node.name] = os.linesep.join(lines[st:en])

        if len(kernels):
            return kernels

    raise ValueError(f"AOT Compilation is supported for valid python modules.\n Failed on {fpath}:\n\n {src}")

def generate_triton_ast_from_src(*paths: Sequence[str]) -> Dict[str, ASTGeneratingObject]:
    """
        - Load python source files that define Jitted functions.
        - Execute sources  
        - Return dict of kernel name -> `ASTGeneratingObject` 
    """

    # TODO: maybe compile source file one by one and generate a single C source per python source?
    # TODO: (idea) conserve tree structure of a python source lib - people can use existing projects in a familiar structure
    scope = {}
    for fpath in paths:
        p = Path(fpath)
        if p.exists():
            src = p.read_text()
            scope["__AOT_COMPILE_src"] = _extract_kernel_src_from_ast(src, fpath)
            try:
                exec(src, scope)
            except Exception as e:
                raise 
            continue
        print(f"[Source Not Found]: {str(p)}")

        
    return filter_jitted_functions(scope)


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


def compile_func(meta: AOTKernelMetadata, fn: ASTGeneratingObject) -> Tuple[ASMObject, CompileMetadata]:
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
            codegen.gen_kernel(kernel_meta=meta, compile_meta=comple_meta, bin_=asm['cubin'], docstring=docstr)

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
    ast_gen_objects = generate_triton_ast_from_src(*args.paths)

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