from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import logging
import re
import sys
import sysconfig
import tempfile
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, Callable, TypeAlias

from triton import language as tl  # type: ignore[import-untyped]
from triton.runtime.jit import JITCallable, JITFunction  # type: ignore[import-untyped]
from triton.tools.ragged_tma import create_ragged_descriptor  # type: ignore[import-untyped]
from triton.tools.tensor_descriptor import TensorDescriptor  # type: ignore[import-untyped]
from triton.tools.triton_to_gluon_translator.inline_helpers import defs as inline_helper_defs
from triton.tools.triton_to_gluon_translator.ordered_set import ordered_set
from triton.tools.triton_to_gluon_translator.scoped_dict import scoped_dict
from triton.tools.triton_to_gluon_translator.stable_toposort import stable_toposort

logger = logging.getLogger(__name__)


@dataclass
class GlobalVariable:
    name: str
    value: Any
    module: ModuleType


@dataclass
class GlobalValue:
    value: GlobalVariable | BuiltinFunctionType | FunctionType | type
    original_value: Any

    @staticmethod
    def wrap(value: Any, name: str, find_module: Callable[[], ModuleType]) -> "GlobalValue":
        assert not isinstance(value, GlobalValue), "value is already a GlobalValue"
        if isinstance(value, FunctionType) and hasattr(value, "__triton_builtin__"):
            return GlobalValue(value, value)
        if isinstance(value, FunctionType) and hasattr(value, "cls"):
            return GlobalValue(value, value)
        # Treat closure globals as global variables, not function definitions.
        if isinstance(value, FunctionType) and value.__closure__ is not None:
            return GlobalValue(GlobalVariable(name, value, find_module()), value)

        if isinstance(value, BuiltinFunctionType | FunctionType | type):
            return GlobalValue(value, value)
        if isinstance(value, JITCallable):
            assert isinstance(value.fn, FunctionType)
            return GlobalValue(value.fn, value)
        return GlobalValue(GlobalVariable(name, value, find_module()), value)

    @property
    def name(self) -> str:
        if isinstance(self.value, BuiltinFunctionType | FunctionType | type):
            return self.value.__name__
        assert isinstance(self.value, GlobalVariable)
        return self.value.name

    @property
    def module(self) -> ModuleType:
        if isinstance(self.value, BuiltinFunctionType | FunctionType | type):
            module = inspect.getmodule(self.value)
            assert module is not None, "value is missing module"
            return module
        assert isinstance(self.value, GlobalVariable)
        return self.value.module

    @property
    def id(self) -> int:
        return id(self.original_value)

    def get_contextual_defs(self) -> dict[str, Any]:
        assert not isinstance(self.value, BuiltinFunctionType), "builtin function cannot be scanned"
        if isinstance(self.value, FunctionType):
            # If the function is wrapped, retrieve the original globals to avoid polluting the namespace.
            value = self.value
            while (wrapped := getattr(value, "__wrapped__", None)) is not None:
                value = wrapped
            return value.__globals__
        return self.module.__dict__

    def parse_ast(self) -> ast.AST:
        assert not isinstance(self.value, BuiltinFunctionType), "builtin function cannot be parsed"
        if isinstance(self.value, type | FunctionType):
            return ast.parse(inspect.getsource(self.value))

        assert isinstance(self.value, GlobalVariable), "expected global variable"
        source = inspect.getsource(self.module)
        tree = ast.parse(source)
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign | ast.AnnAssign):
                continue
            target = get_assign_target(stmt)
            if target is not None and target.id == self.value.name:
                assert stmt.value is not None, "FIXME: global variable value is missing"
                return stmt.value
        raise ValueError(f"could not find definition of {self.value} in {self.module}")

    def mangle_source(self, source: str, mangled_name: str) -> str:
        # HACK: An AST rewrite would be more robust, but this works for now.
        if isinstance(self.value, FunctionType):
            return source.replace(f"def {self.name}(", f"def {mangled_name}(")
        elif isinstance(self.value, type):
            return source.replace(f"class {self.name}", f"class {mangled_name}")
        else:
            assert isinstance(self.value, GlobalVariable), "expected global variable"
            return f"{mangled_name} = {source}"


FilterFn = Callable[[ModuleType | GlobalValue], bool]
DecoratorMatcher: TypeAlias = Callable[[scoped_dict[str, Any], ModuleType, ast.expr], bool]


def get_assign_target(stmt: ast.Assign | ast.AnnAssign) -> ast.Name | None:
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            return None
        target = stmt.targets[0]
    else:
        target = stmt.target
    return target if isinstance(target, ast.Name) else None


def resolve_module_alias(stmt: ast.ImportFrom, cur_module: ModuleType) -> ModuleType:
    assert stmt.module is not None, "import statement with no module"
    assert cur_module.__file__ is not None
    if stmt.level > 0:
        parts = cur_module.__name__.split(".")
        parent_name = parts[:len(parts) - stmt.level + cur_module.__file__.endswith("__init__.py")]
        module_name = ".".join(parent_name) + "." + stmt.module
    else:
        module_name = stmt.module
    return sys.modules[module_name]


def get_name_ref_module(name: str, cur_module: ModuleType, filter: FilterFn) -> ModuleType:
    # Bottom out at the leaf modules.
    if filter(cur_module):
        return cur_module
    source = inspect.getsource(cur_module)
    tree = ast.parse(source)
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.asname == name or (alias.asname is None and alias.name == name):
                    next_module = resolve_module_alias(stmt, cur_module)
                    return get_name_ref_module(alias.name, next_module, filter)
        elif isinstance(stmt, ast.Assign | ast.AnnAssign):
            target = get_assign_target(stmt)
            if target is not None and target.id == name:
                return cur_module
    raise ValueError(f"could not find module for {name} in {cur_module.__name__}")


def find_module(context: scoped_dict[str, Any], node: ast.AST) -> ModuleType | None:
    if isinstance(node, ast.Name):
        if node.id in context:
            module = context[node.id]
            if isinstance(module, ModuleType):
                return module
    elif isinstance(node, ast.Attribute):
        module = find_module(context, node.value)
        if module is not None:
            module = getattr(module, node.attr)
            if isinstance(module, ModuleType):
                return module
    return None


def get_reference(context: scoped_dict[str, Any], cur_module: ModuleType,
                  node: ast.AST) -> tuple[Any, ModuleType, str] | None:
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load) or node.id not in context:
            return None
        return context[node.id], cur_module, node.id
    if isinstance(node, ast.Attribute):
        if not isinstance(node.ctx, ast.Load):
            return None
        rel_module = find_module(context, node.value)
        if rel_module is None:
            return None
        value = getattr(rel_module, node.attr)
        return value, rel_module, node.attr
    return None


def is_ignored_decorator(
    context: scoped_dict[str, Any],
    cur_module: ModuleType,
    decorator: ast.expr,
    ignored_decorator_matchers: Sequence[DecoratorMatcher],
) -> bool:
    return any(matcher(context, cur_module, decorator) for matcher in ignored_decorator_matchers)


@dataclass
class Reference:
    value: GlobalValue
    module: ModuleType
    edges: ordered_set[int] | None = None
    mangled_name: str | None = None


@dataclass(frozen=True)
class LocalMarker:
    pass


@dataclass
class ReferenceScanner(ast.NodeVisitor):
    cur_module: ModuleType
    context: scoped_dict[str, Any]
    references: OrderedDict[int, Reference]
    queue: list[GlobalValue]
    value_remap: dict[int, GlobalValue]
    filter: FilterFn
    ignored_decorator_matchers: Sequence[DecoratorMatcher] = field(default_factory=tuple)

    edges: ordered_set[int] = field(default_factory=ordered_set[int])

    def process_reference(self, node: ast.Name | ast.Attribute, name: str, value: Any, rel_module: ModuleType) -> None:
        if isinstance(value, ModuleType | LocalMarker):
            return self.generic_visit(node)
        global_value = self.value_remap.get(id(value), None) or GlobalValue.wrap(
            value, name, lambda: get_name_ref_module(name, rel_module, self.filter))

        ref_id = global_value.id
        module = global_value.module
        if ref_id not in self.references:
            self.references[ref_id] = Reference(global_value, module)
            logger.debug(f"Added reference: {global_value} {module}")
            self.queue.append(global_value)

        # reference = self.references[ref_id]
        # assert reference.module is module, f"inconsistent value reference {global_value}"
        self.edges.add(ref_id)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return self.generic_visit(node)
        if node.id not in self.context:
            return self.generic_visit(node)
        value = self.context[node.id]
        return self.process_reference(node, node.id, value, self.cur_module)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.ctx, ast.Load):
            return self.generic_visit(node)
        rel_module = find_module(self.context, node.value)
        if rel_module is None:
            return self.generic_visit(node)
        value = getattr(rel_module, node.attr)
        return self.process_reference(node, node.attr, value, rel_module)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        original_decorators = node.decorator_list
        node.decorator_list = [
            decorator for decorator in original_decorators if not is_ignored_decorator(
                self.context,
                self.cur_module,
                decorator,
                self.ignored_decorator_matchers,
            )
        ]
        args = node.args
        with self.context.scope():
            for arg in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
                if arg is not None:
                    self.context[arg.arg] = LocalMarker()
            return self.generic_visit(node)


def match_regex(path: str) -> bool:
    identifier = r"[a-zA-Z_][a-zA-Z0-9_]*"
    pattern = rf"^{identifier}(\.{identifier})*:{identifier}$"
    return re.match(pattern, path) is not None


def get_base_value(path: str) -> GlobalValue:
    if not match_regex(path):
        raise ValueError(f"invalid Python object format: {path}")
    module_str, value_name = path.split(":")
    module = importlib.import_module(module_str)
    return GlobalValue.wrap(getattr(module, value_name), value_name, lambda: module)


def is_submodule(module: ModuleType, leaf_modules: list[str]) -> bool:
    return any(module.__name__ == leaf_module or module.__name__.startswith(f"{leaf_module}.")
               for leaf_module in leaf_modules)


def mangle_name(name: str, module: ModuleType, reference_names: set[str]) -> str:
    if name not in reference_names:
        return name
    for part in reversed(module.__name__.split(".")):
        name = f"{part}_{name}"
        if name not in reference_names:
            return name
    raise ValueError(f"FIXME: failed to mangle a unique name for {name} in {module.__name__}")


def parse_expr(expr_str: str) -> ast.expr:
    expr: ast.stmt = ast.parse(expr_str).body[0]
    assert isinstance(expr, ast.Expr)
    return expr.value


RewriteFn = Callable[[GlobalValue, ordered_set[str]], ast.AST | None]


def sugar_rewrite(module: str, alias: str) -> RewriteFn:

    def rewrite(global_value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if module not in sys.modules:
            return None
        if not hasattr(sys.modules[module], global_value.name):
            return None
        if id(getattr(sys.modules[module], global_value.name)) == global_value.id:
            imports.add(f"import {module} as {alias}" if alias != module else f"import {module}")
            return ast.Attribute(value=ast.Name(id=alias, ctx=ast.Load()), attr=global_value.name, ctx=ast.Load())
        return None

    return rewrite


def add_sugar_rewrites(rewrites: list[RewriteFn], translate_to_gluon: bool) -> None:
    if translate_to_gluon:
        rewrites.append(sugar_rewrite("triton.experimental.gluon.language", "gl"))
        rewrites.append(sugar_rewrite("triton.experimental.gluon", "gluon"))
    else:
        rewrites.append(sugar_rewrite("triton.language", "tl"))
        rewrites.append(sugar_rewrite("triton", "triton"))

    def sugar_tensor_descriptor(global_value: GlobalValue, imports: ordered_set[str]) -> ast.AST | None:
        if global_value.original_value is TensorDescriptor:
            imports.add("from triton.tools.tensor_descriptor import TensorDescriptor")
            return ast.Name(id="TensorDescriptor", ctx=ast.Load())
        return None

    rewrites.append(sugar_tensor_descriptor)


@dataclass
class ReferenceRewriter(ast.NodeTransformer):
    cur_module: ModuleType
    context: scoped_dict[str, Any]
    references: OrderedDict[int, Reference]
    imports: ordered_set[str]
    filter: FilterFn
    value_remap: dict[int, GlobalValue]
    ignored_decorator_matchers: Sequence[DecoratorMatcher] = field(default_factory=tuple)

    rewrites: list[RewriteFn] = field(default_factory=list)

    def process_reference(self, node: ast.Name | ast.Attribute, name: str, value: Any,
                          rel_module: ModuleType) -> ast.AST:
        if isinstance(value, ModuleType | LocalMarker):
            return self.generic_visit(node)
        global_value = self.value_remap.get(id(value), None) or GlobalValue.wrap(
            value, name, lambda: get_name_ref_module(name, rel_module, self.filter))

        ref_id = global_value.id
        if ref_id not in self.references:
            return self.generic_visit(node)
        reference = self.references[ref_id]
        assert reference.mangled_name

        module = global_value.module
        name = global_value.name
        if self.filter(module) or self.filter(global_value):
            for rewrite in self.rewrites:
                result = rewrite(global_value, self.imports)
                if result is not None:
                    return result

            # Special rule for aliases to builtins.
            if module.__name__ == "builtins" and rel_module.__name__ != "builtins":
                self.imports.add(f"import {rel_module.__name__}")
                return node

            parts = module.__name__.split(".")
            reference_node: ast.Name | ast.Attribute = ast.Name(id=parts[0], ctx=ast.Load())
            for part in parts[1:]:
                reference_node = ast.Attribute(value=reference_node, attr=part, ctx=ast.Load())
            self.imports.add(f"import {module.__name__}")
            return ast.Attribute(value=reference_node, attr=name)

        return ast.Name(id=reference.mangled_name, ctx=node.ctx)

    def get_reference(self, node: ast.AST) -> tuple[Any, ModuleType, str] | None:
        return get_reference(self.context, self.cur_module, node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        ref = self.get_reference(node)
        if ref is None:
            return self.generic_visit(node)
        value, rel_module, name = ref
        return self.process_reference(node, name, value, rel_module)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        ref = self.get_reference(node)
        if ref is None:
            return self.generic_visit(node)
        value, rel_module, name = ref
        return self.process_reference(node, name, value, rel_module)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        args = node.args
        with self.context.scope():
            for arg in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
                if arg is not None:
                    self.context[arg.arg] = LocalMarker()
            return self.generic_visit(node)


@dataclass
class SliceRewriter(ReferenceRewriter):
    translate_to_gluon: bool = False
    inline_helpers: ordered_set[str] = field(default_factory=ordered_set[str])

    def __post_init__(self) -> None:
        # Special rules for sugaring imports.
        add_sugar_rewrites(self.rewrites, self.translate_to_gluon)

    def emit_reference_impl(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            ref = self.get_reference(node)
            if ref is None:
                return None
            value, _, _ = ref
            if isinstance(value, LocalMarker):
                return None
            return value
        elif isinstance(node, ast.Attribute):
            base = self.emit_reference_impl(node.value)
            if base is None:
                return None
            return getattr(base, node.attr)
        return None

    def emit_reference(self, node: ast.AST) -> Any:
        try:
            return self.emit_reference_impl(node)
        except RuntimeError as e:
            # HACK: Workaround triton.runtime.driver.active failing on CPU-only machines.
            if "0 active drivers" in str(e):
                return node
            raise e

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if not self.translate_to_gluon:
            return super().visit_Attribute(node)

        # Manual manipulation of JITFunctions.
        value = self.emit_reference(node)
        new_node = super().visit_Attribute(node)
        if value is JITFunction:
            self.imports.add("import triton.experimental.gluon._runtime as gluon_runtime")
            new_node = parse_expr("gluon_runtime.GluonJITFunction")
        elif value is tl.tensor_descriptor:
            self.imports.add("from triton.experimental.gluon.language.nvidia.hopper.tma import tensor_descriptor")
            new_node = ast.Name(id="tensor_descriptor", ctx=ast.Load())
        return new_node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not self.translate_to_gluon:
            return self.generic_visit(node)

        # Rewrite host code when translating to Gluon.
        callee = self.emit_reference(node.func)
        new_node = self.generic_visit(node)
        if callee in [TensorDescriptor, TensorDescriptor.from_tensor, create_ragged_descriptor]:
            self.inline_helpers.add("convert_host_descriptor")
            new_node = parse_expr(f"convert_host_descriptor({ast.unparse(new_node)})")
        return new_node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        new_decorators: list[ast.expr] = []
        for decorator in node.decorator_list:
            # Decorators are applied bottom to top, so decorators above the
            # matched downstream kernel decorator will be applied after.
            if is_ignored_decorator(
                    self.context,
                    self.cur_module,
                    decorator,
                    self.ignored_decorator_matchers,
            ):
                new_decorators = []
                continue
            new_decorators.append(decorator)
        node.decorator_list = new_decorators
        return super().visit_FunctionDef(node)


def is_stdlib_module(module: ModuleType) -> bool:
    modspec = module.__spec__
    # Native modules don't have a spec. Always treat them as leaf modules.
    if modspec is None:
        return True
    assert modspec is not None, f"module {module.__name__} is missing spec metadata"
    origin = modspec.origin
    assert origin is not None, "module is missing origin"
    if origin in ["built-in", "frozen"]:
        return True

    stdlib_path = Path(sysconfig.get_paths()["stdlib"])
    return Path(origin).is_relative_to(stdlib_path)


def find_references(
    base_values: list[GlobalValue],
    filter: FilterFn,
    value_remap: dict[int, GlobalValue],
    ignored_decorator_matchers: Sequence[DecoratorMatcher] | None = None,
) -> tuple[OrderedDict[int, Reference], dict[int, ordered_set[int]]]:
    references: OrderedDict[int, Reference] = OrderedDict()
    queue: list[GlobalValue] = []
    graph: dict[int, ordered_set[int]] = {}
    ignored_decorator_matchers = tuple(ignored_decorator_matchers or ())

    for base_value in base_values:
        base_value = value_remap.get(base_value.id, base_value)
        queue.append(base_value)
        references[base_value.id] = Reference(base_value, base_value.module)

    while len(queue):
        value = queue.pop(0)
        if filter(value.module) or filter(value):
            graph[value.id] = ordered_set()
            continue
        logger.debug(f"Processing: {value.name}")
        logger.debug(f"Value: {value}")
        scanner = ReferenceScanner(
            value.module,
            scoped_dict(value.get_contextual_defs()),
            references,
            queue,
            value_remap,
            filter,
            ignored_decorator_matchers=ignored_decorator_matchers,
        )
        tree = value.parse_ast()
        scanner.visit(tree)
        graph[value.id] = scanner.edges

    return references, graph


def mangle_reference_names(references: OrderedDict[int, Reference], filter: FilterFn) -> None:
    reference_names: set[str] = set()
    for reference in references.values():
        name = reference.value.name
        module = reference.value.module
        if filter(module) or filter(reference.value):
            mangled_name = f"{module.__name__}.{name}"
        else:
            mangled_name = mangle_name(name, module, reference_names)
        reference_names.add(mangled_name)
        reference.mangled_name = mangled_name
        logger.debug(f"Value: {reference.value}")
        logger.debug(f"Module: {reference.module}")
        logger.debug(f"Name: {name} -> {mangled_name}")


def find_jit_functions(
    base_values: list[GlobalValue],
    filter: FilterFn,
    ignored_decorator_matchers: Sequence[DecoratorMatcher] | None = None,
) -> list[GlobalValue]:

    def new_filter(value: ModuleType | GlobalValue) -> bool:
        if isinstance(value, GlobalValue) and isinstance(value.original_value, JITFunction):
            return True
        return filter(value)

    references, _ = find_references(
        base_values,
        new_filter,
        value_remap={},
        ignored_decorator_matchers=ignored_decorator_matchers,
    )
    return [
        reference.value for reference in references.values() if isinstance(reference.value.original_value, JITFunction)
    ]


def load_module_from_file(name: str, path: str | Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert module is not None
    spec.loader.exec_module(module)
    return module


def slice_kernel(
    root_paths: list[str],
    leaf_modules: list[str],
    include_below: list[str] | None = None,
    leaf_paths: list[str] | None = None,
    translate_to_gluon: bool = False,
    ignored_decorator_matchers: Sequence[DecoratorMatcher] | None = None,
) -> str:
    base_values: list[GlobalValue] = [get_base_value(root_path) for root_path in root_paths]
    base_value_ids: set[int] = set()
    for leaf_path in leaf_paths or []:
        base_value = get_base_value(leaf_path)
        base_value_ids.add(base_value.id)

    def filter(value: ModuleType | GlobalValue) -> bool:
        if isinstance(value, ModuleType):
            if is_stdlib_module(value):
                return True
            if is_submodule(value, leaf_modules):
                return not is_submodule(value, include_below or [])
            return False
        return value.id in base_value_ids

    value_remap: dict[int, GlobalValue] = {}
    if translate_to_gluon:
        # FIXME: Refactor code to avoid circular imports.
        from triton.tools.triton_to_gluon_translator.translator import translate_kernels

        jit_functions = find_jit_functions(
            base_values,
            filter,
            ignored_decorator_matchers=ignored_decorator_matchers,
        )
        converted_functions = translate_kernels(jit_functions)
        module_file = tempfile.NamedTemporaryFile(delete=False, prefix="translated_", suffix=".py")
        module_path = Path(module_file.name)
        module_path.write_text(converted_functions)
        module = load_module_from_file("converted_functions", module_path)
        for fn in jit_functions:
            gluon_fn = getattr(module, fn.name)
            assert isinstance(gluon_fn, JITFunction)
            value_remap[fn.id] = GlobalValue.wrap(gluon_fn, fn.name, lambda: module)

    references, graph = find_references(
        base_values,
        filter,
        value_remap,
        ignored_decorator_matchers=ignored_decorator_matchers,
    )
    mangle_reference_names(references, filter)

    output = ""
    imports: ordered_set[str] = ordered_set()
    inline_helpers: ordered_set[str] = ordered_set()

    # Use a stable toposort to order the references. This is because global
    # values in the same module can form reference cycles, but we need to
    # generate them in the same order as they are in the original source in case
    # they have to be resolved in that particular order.
    ordered_ids = stable_toposort(graph)
    for ref_id in reversed(ordered_ids):
        reference = references[ref_id]
        name = reference.value.name
        if reference.mangled_name != name:
            logger.debug(f"Name mangled: {name} -> {reference.mangled_name}")
        if filter(reference.value.module) or filter(reference.value):
            continue
        tree = reference.value.parse_ast()
        context = reference.value.get_contextual_defs()
        rewriter = SliceRewriter(
            reference.value.module,
            scoped_dict(context),
            references,
            imports,
            filter,
            value_remap,
            ignored_decorator_matchers=tuple(ignored_decorator_matchers or ()),
            translate_to_gluon=translate_to_gluon,
            inline_helpers=inline_helpers,
        )
        tree = rewriter.visit(tree)
        source = ast.unparse(tree)
        assert reference.mangled_name is not None
        source = reference.value.mangle_source(source, reference.mangled_name)
        output += source + "\n\n\n"
    output = "\n".join(imports) + "\n\n" + output

    if translate_to_gluon:
        # HACK: This updates the strings generated by `specialize`.
        output = output.replace("@triton.jit", "@gluon.jit")
        output = output.replace("tl.constexpr", "gl.constexpr")

    for helper in inline_helpers:
        output += inline_helper_defs[helper].strip() + "\n"

    return output


def slice_kernel_from_trace(
    kernel_path: str,
    trace: list[dict[str, list[str]]],
    translate_to_gluon: bool,
    extra_modules: dict[str, str],
    ignored_decorator_matchers: Sequence[DecoratorMatcher] | None = None,
) -> str:
    module_remap: dict[str, str] = {}
    for name, path in extra_modules.items():
        load_module_from_file(name, path)
        module_remap[name] = Path(path).with_suffix("").stem

    leaf_paths: set[str] = set()
    root_paths: set[str] = {kernel_path}
    for entry in trace:
        leaf_paths.update(entry["type_names"])
        root_paths.update(entry["jit_fn_names"])

    # Remove obvious leaf paths.
    for leaf_path in list(leaf_paths):
        base_module = leaf_path.split(":")[0].split(".")[0]
        if base_module in ["triton", "torch"] or is_stdlib_module(importlib.import_module(base_module)):
            leaf_paths.remove(leaf_path)

    sliced = slice_kernel(
        root_paths=sorted(root_paths),
        leaf_modules=["triton", "torch", "ki.spo"],
        leaf_paths=sorted(leaf_paths),
        translate_to_gluon=translate_to_gluon,
        ignored_decorator_matchers=ignored_decorator_matchers,
    )

    fn_name = lambda path: path.split(":")[1]
    if len(root_paths) > 1:
        jit_fns = list(root_paths - {kernel_path})
        remap_lines = "\n".join(f"    '{fn_name(fn)}': {fn_name(fn)}," for fn in jit_fns)
        sliced += (
            "\n"
            + f"""
{fn_name(kernel_path)}.__jit_fn_remap__ = {{
{remap_lines}
}}"""
        )

    return sliced


def main(
    root_paths: list[str],
    leaf_modules: list[str],
    include_below: list[str] | None = None,
    leaf_paths: list[str] | None = None,
    translate_to_gluon: bool = False,
    output_path: str = "/tmp/reference.py",
    ignored_decorator_matchers: Sequence[DecoratorMatcher] | None = None,
) -> None:
    output = slice_kernel(
        root_paths,
        leaf_modules,
        include_below,
        leaf_paths,
        translate_to_gluon,
        ignored_decorator_matchers,
    )
    with open(output_path, "w") as f:
        f.write(output)


def _main_cli() -> None:
    parser = argparse.ArgumentParser(description="Slice Triton Python kernels into a standalone file.")
    parser.add_argument("root_paths", nargs="+", help="Root symbols to keep, in module.path:object format.")
    parser.add_argument("--leaf-module", dest="leaf_modules", action="append", default=[],
                        help="Module roots to treat as leaves. May be repeated.")
    parser.add_argument("--include-below", action="append", default=[],
                        help="Leaf modules that should still be traversed below. May be repeated.")
    parser.add_argument("--leaf-path", dest="leaf_paths", action="append", default=[],
                        help="Specific symbols to treat as leaves. May be repeated.")
    parser.add_argument("--translate-to-gluon", action="store_true",
                        help="Translate Triton JIT callables to Gluon while slicing.")
    parser.add_argument("--output-path", default="/tmp/reference.py", help="Path to write the sliced output.")
    args = parser.parse_args()
    main(
        root_paths=args.root_paths,
        leaf_modules=args.leaf_modules,
        include_below=args.include_below or None,
        leaf_paths=args.leaf_paths or None,
        translate_to_gluon=args.translate_to_gluon,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    _main_cli()
