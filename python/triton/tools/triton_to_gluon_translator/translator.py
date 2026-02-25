import argparse
import ast
import builtins
import inspect
from dataclasses import dataclass, field
from types import FunctionType, ModuleType
from typing import Any, cast

import triton.language as tl  # type: ignore[import-untyped]
from triton.runtime.jit import JITCallable, JITFunction  # type: ignore[import-untyped]
from triton.tools.triton_to_gluon_translator.ordered_set import ordered_set
from triton.tools.triton_to_gluon_translator.scoped_dict import scoped_dict
from triton.tools.triton_to_gluon_translator.slice_kernel import (
    GlobalValue,
    ReferenceRewriter,
    RewriteFn,
    add_sugar_rewrites,
    find_references,
    get_base_value,
    is_submodule,
    mangle_reference_names,
    parse_expr,
)
from triton.tools.triton_to_gluon_translator.stable_toposort import stable_toposort


def one_to_one_rewrite(obj: Any) -> RewriteFn:
    def rewrite(global_value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if global_value.original_value is obj:
            return ast.Attribute(
                value=ast.Name(id="gl", ctx=ast.Load()), attr=obj.__name__, ctx=ast.Load()
            )
        return None

    return rewrite


def add_one_to_one_rewrites(rewrites: list[RewriteFn]) -> None:
    import triton.experimental.gluon.language as gl  # type: ignore[import-untyped]

    for value in vars(gl).values():
        module = inspect.getmodule(value)
        if module is None:
            continue
        if getattr(value, "__triton_builtin__", False) and is_submodule(
            module, ["triton.language"]
        ):
            tl_value = getattr(tl, value.__name__, None)
            if tl_value is None:
                tl_value = getattr(tl.core, value.__name__, None)
            if tl_value is None:
                continue
            rewrites.append(one_to_one_rewrite(tl_value))
        elif isinstance(value, JITFunction):
            tl_value = getattr(tl, value.fn.__name__, None)
            if tl_value is None:
                tl_value = getattr(tl.standard, value.fn.__name__, None)
            if tl_value is None:
                continue
            assert isinstance(tl_value, JITFunction) and value is not tl_value
            if value.fn is tl_value.fn:
                rewrites.append(one_to_one_rewrite(tl_value))


def translator_helper_rewrite(obj: Any, helper_name: str) -> RewriteFn:
    def rewrite(global_value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if global_value.original_value is obj:
            return ast.Attribute(
                value=ast.Name(id="helpers", ctx=ast.Load()), attr=helper_name, ctx=ast.Load()
            )
        return None

    return rewrite


def add_translator_helper_rewrites(rewrites: list[RewriteFn]) -> None:
    remap: list[tuple[Any, str]] = [
        (tl.arange, "tl_arange"),
        (tl.full, "tl_full"),
        (tl.trans, "tl_trans"),
        (tl.cat, "tl_cat"),
        (tl.dot, "tl_dot"),
        (tl.dot_scaled, "tl_dot_scaled"),
        (tl.make_tensor_descriptor, "tl_make_tensor_descriptor"),
        (tl.load_tensor_descriptor, "tl_load_tensor_descriptor"),
        (tl.store_tensor_descriptor, "tl_store_tensor_descriptor"),
        (tl.atomic_add, "tl_atomic_add"),
    ]
    if (tl_cuda := getattr(tl.extra, "cuda", None)) is not None:
        remap.append((tl_cuda.num_threads, "get_num_threads_per_program"))
    for value, helper_name in remap:
        rewrites.append(translator_helper_rewrite(value, helper_name))


def expr_rewrite(obj: Any, expr: str) -> RewriteFn:
    def rewrite(value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if value.original_value is obj:
            return parse_expr(expr)
        return None

    return rewrite


def add_expr_rewrites(rewrites: list[RewriteFn]) -> None:
    import triton
    import triton.language as tl

    rewrites.append(expr_rewrite(triton.jit, "gluon.jit"))
    rewrites.append(expr_rewrite(tl.debug_barrier, "gl.barrier"))


@dataclass
class Translator(ReferenceRewriter):
    tensor_member_match_fns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        import triton
        import triton.language as tl

        self.context.setdefault("tl", tl)
        self.context.setdefault("triton", triton)

        add_sugar_rewrites(self.rewrites, translate_to_gluon=True)
        add_translator_helper_rewrites(self.rewrites)
        add_one_to_one_rewrites(self.rewrites)
        add_expr_rewrites(self.rewrites)

        self.imports.add("import triton.experimental.gluon as gluon")
        self.imports.add("import triton.experimental.gluon.language as gl")
        self.imports.add("import triton.tools.triton_to_gluon_translator.translator_helpers as helpers")

        self.tensor_member_match_fns = ["reshape", "trans", "permute", "split", "reduce", "sum"]

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        new_node = super().visit_Attribute(node)
        if node.attr == "T":
            new_node = parse_expr(f"helpers.reset_to_default_layout({ast.unparse(new_node)})")
        return new_node

    def canonicalize_call(self, node: ast.Call) -> tuple[ast.Call, str | None]:
        if self.get_reference(node.func) is not None:
            return node, None
        if (
            not isinstance(node.func, ast.Attribute)
            or node.func.attr not in self.tensor_member_match_fns
        ):
            return node, None
        new_callable = parse_expr(f"tl.{node.func.attr}")
        new_call = ast.Call(
            func=new_callable, args=[node.func.value] + node.args, keywords=node.keywords
        )
        return new_call, node.func.attr

    def uncanonicalize_call(self, node: ast.Call, fn_name: str | None) -> ast.Call:
        if fn_name is None:
            return node
        value = node.args[0]
        new_callable = ast.Attribute(value, fn_name, ctx=ast.Load())
        return ast.Call(func=new_callable, args=node.args[1:], keywords=node.keywords)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node, canonicalized = self.canonicalize_call(node)
        ref = self.get_reference(node.func)
        if ref is None:
            assert canonicalized is None
            if isinstance(node.func, ast.Attribute) and node.func.attr in [
                "store",
                "load",
                "gather",
                "scatter",
            ]:
                new_callee = parse_expr(f"helpers.tl_obj_{node.func.attr}")
                node = ast.Call(
                    func=new_callee, args=[node.func.value] + node.args, keywords=node.keywords
                )
            return self.generic_visit(node)
        value, _, _ = ref
        if value in [tl.reshape, tl.ravel]:
            node.keywords = [kw for kw in node.keywords if kw.arg != "can_reorder"]
        elif value is tl.split:
            node.args[0] = parse_expr(f"helpers.set_split_src_layout({ast.unparse(node.args[0])})")
        elif value is tl.range:
            return ast.Call(
                func=ast.Name("range", ast.Load()),
                args=[cast(ast.expr, self.generic_visit(arg)) for arg in node.args],
                keywords=[],
            )

        node = self.uncanonicalize_call(node, canonicalized)
        new_node = self.generic_visit(node)
        if value in [tl.reshape, tl.trans, tl.permute, tl.join, tl.split, tl.reduce, tl.sum]:
            new_node = cast(
                ast.Call, parse_expr(f"helpers.reset_to_default_layout({ast.unparse(new_node)})")
            )
        return new_node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if not isinstance(node.slice, ast.Tuple):
            return self.generic_visit(node)

        expand_dims: list[int] = []
        for index, dim in enumerate(node.slice.elts):
            if isinstance(dim, ast.Constant) and dim.value is None:
                expand_dims.append(index)
            elif isinstance(dim, ast.Slice) and all(
                d is None for d in [dim.lower, dim.upper, dim.step]
            ):
                continue
            else:
                return self.generic_visit(node)
        value_expr = parse_expr(
            f"helpers.convert_to_expand_dims_layout({ast.unparse(node.value)}, {expand_dims})"
        )
        node = ast.Subscript(value=value_expr, slice=node.slice, ctx=node.ctx)
        return self.generic_visit(node)


def translate_kernels(kernels: list[GlobalValue]) -> str:
    def filter(value: ModuleType | GlobalValue) -> bool:
        if isinstance(value, ModuleType):
            return False
        if getattr(value.original_value, "__triton_builtin__", False):
            return True
        if isinstance(value.original_value, JITFunction):
            if getattr(tl.tensor, value.name, None) is value.original_value:
                return True
            if value.original_value.is_gluon():
                return True
            return False
        assert isinstance(value.original_value, object)
        if isinstance(value.original_value, type | FunctionType | JITCallable):
            return True
        if isinstance(value.original_value, int | float | tl.constexpr):
            return False
        return True

    references, graph = find_references(kernels, filter, value_remap={})
    mangle_reference_names(references, filter)

    ordered_ids = stable_toposort(graph)

    output = ""
    imports: ordered_set[str] = ordered_set()

    ordered_ids = stable_toposort(graph)
    for ref_id in reversed(ordered_ids):
        reference = references[ref_id]
        if filter(reference.value.module) or filter(reference.value):
            continue
        tree = reference.value.parse_ast()
        context = reference.value.get_contextual_defs()
        rewriter = Translator(
            reference.value.module,
            scoped_dict(context),
            references,
            imports,
            filter,
            value_remap={},
        )
        tree = rewriter.visit(tree)
        source = ast.unparse(tree)
        assert reference.mangled_name is not None
        source = reference.value.mangle_source(source, reference.mangled_name)
        output += source + "\n\n\n"
    output = "\n".join(imports) + "\n\n" + output
    return output


def translate_paths(kernel_paths: list[str]) -> str:
    kernels = [get_base_value(kernel_path) for kernel_path in kernel_paths]
    return translate_kernels(kernels)


def convert_triton_to_gluon(src: list[JITCallable]) -> str:
    kernels = [
        GlobalValue.wrap(
            kernel,
            getattr(getattr(kernel, "fn", kernel), "__name__", ""),
            lambda: builtins,
        )
        for kernel in src
    ]
    return translate_kernels(kernels)


def main(kernels: list[str], output_path: str) -> None:
    output = translate_paths(kernels)
    with open(output_path, "w") as f:
        f.write(output)


def _main_cli() -> None:
    parser = argparse.ArgumentParser(description="Translate Triton kernels to Gluon source.")
    parser.add_argument("kernels", nargs="+", help="Kernel symbols in module.path:object format.")
    parser.add_argument("--output-path", required=True, help="Path to write the translated source.")
    args = parser.parse_args()
    main(args.kernels, args.output_path)


if __name__ == "__main__":
    _main_cli()
