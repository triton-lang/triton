import ast
import os
from argparse import ArgumentParser
from types import ModuleType
from typing import Callable, Dict, Sequence, Tuple

import triton
from aot_compile.c_codegen import CodeGenerator
from aot_compile.compile_metadata import (AOTKernelMetadata, ASTGeneratingObject,
                                          infer_triton_signature, make_compile_kwargs)


def find_all_jitted(*wheres):
    res = {}
    for where in wheres:
        if isinstance(where, ModuleType):
            where = where.__dict__
        all_jitted = {
            v.__name__: v for v in where.values() if isinstance(v, ASTGeneratingObject)
        }
        res.update(all_jitted)
    return res



def _extract_kernel_src_from_ast(src: str, fpath: str):
    """
    Parse source and extract functions that are jit compiled

    Why is this here? 
    When executing modules from string with exec function, inspect has no access to the source code.
    This function extracts sources of funcitons and passes those to exec as global scope data.

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

def generate_jiited(*paths):

    # TODO: maybe compile source file one by one and generate a single C source per python source?
    # TODO: (idea) conserve tree structure of a python source lib - people can use exisitng projects in a familiar structure
    sources = []
    for p in paths:
        p = Path(p)
        if p.exists():
            src = p.read_text()
            sources.append((src, str(p)))
            continue
        print(f"[Source Not Found]: {str(p)}")

    scope = {}
    for (src, fpath) in sources:
        scope["__AOT_COMPILE_src"] = _extract_kernel_src_from_ast(src, fpath)
        exec(src, scope)

    return find_all_jitted(scope)


def infer_config_dump_stdout(jitted: Dict[str, Callable]) -> str:
    sigs = {k: infer_triton_signature(v) for k, v in jitted.items()}
    print(yaml.safe_dump(sigs))


def parse_config_file(conf_file):
    conf = Path(conf_file)
    if not conf.exists():
        raise ValueError(f"Config File Not Found [{conf_file}]")

    sigs = yaml.safe_load(conf.open("r"))

    aot_meta = {
        name: AOTKernelMetadata.parse(name, meta) for name, meta in sigs.items()
    }

    return aot_meta


AOTMetadataWithAST = Dict[str, Tuple[Sequence[AOTKernelMetadata], ASTGeneratingObject]]


def config_matches_jitted_functions(
    aot_meta: Dict[str, Sequence[AOTKernelMetadata]],
    jitted: Dict[str, ASTGeneratingObject],
) -> AOTMetadataWithAST:
    """

    Take meta data and jitted functions found and make sure those are matching.
    Kernels with missing jit funciton or metadata are skipped

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


def compile_func(meta: AOTKernelMetadata, fn: ASTGeneratingObject) -> bytes:
    # NOTE: we ovveride the function name here. in case we have several variants in the kerenel
    #       (those will have same signature in C)
    old_name = fn.__name__
    fn.__name__ = meta.name

    kwargs = make_compile_kwargs(meta)
    compiled = triton.compile(fn, **kwargs)
    bin_ = compiled.asm['cubin']

    fn.__name__ = old_name
    return bin_


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
            bin_ = compile_func(meta, jit_fn)
            docstr = str(meta)
            if jit_fn.__doc__ is not None:
                docstr = f"{docstr}\n\t{jit_fn.__doc__}"
            codegen.gen_kernel(meta=meta, bin_=bin_, docstring=docstr)

    codegen.dump(out_dir=outdir)


if __name__ == "__main__":
    from pathlib import Path

    import yaml

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
        help="Output inferred compile annotaitons as an stdout YAML",
    )
    parser.add_argument("--config", default="dummy.yml")
    parser.add_argument("--out-dir", "-o", type=Path)
    parser.add_argument("--target", "-t", default="Csource", type=str)

    args = parser.parse_args()

    all_jitted = generate_jiited(*args.paths)

    if args.infer_signature:
        infer_config_dump_stdout(all_jitted)
        exit(0)

    aot_metas = parse_config_file(args.config)
    matched_metas_with_fn = config_matches_jitted_functions(
        aot_meta=aot_metas, jitted=all_jitted
    )

    generate_c_code(meta_and_ast=matched_metas_with_fn, outdir=args.out_dir)
