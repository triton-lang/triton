# Experimental Triton to Gluon AST translator.
# This file takes a Triton JIT entry point and generates a Gluon equivalent including all
# its dependencies. This generates highly inefficient Gluon code and is only used for
# functional testing.
#
import ast
from typing import Optional
import triton
import triton.language.core as tlc
import sys
import importlib
import importlib.util
import copy

GLUON_IMPORT_LINES = ("from triton.experimental import gluon\n"
                      "from triton.experimental.gluon import language as ttgl\n"
                      "from triton.tools.triton_to_gluon_translater.translator_helpers import *\n")


class TritonToGluonTransformer(ast.NodeTransformer):
    """Transforms Triton kernel source into a functionally equivalent Gluon source.

    This transformer rewrites builtins, dtype/tensor attributes, constexpr annotations,
    and records nested JIT callables to be converted and appended to the output.
    """

    def __init__(self, globals_map: dict, shared_jit_set: set, shared_queue: list, is_jit, constexpr_globals: dict):
        super().__init__()
        # Resolution scope (globals ∪ nonlocals)
        self.scope: dict = globals_map or {}
        # Track discovered JIT functions to inline/append later
        self.jit_functions: set = shared_jit_set
        self.queue: list = shared_queue
        self.is_jit = is_jit
        # Maps module_file -> {name: value} to pull constexpr globals from the original source code
        self.constexpr_globals: dict = constexpr_globals

    def is_triton_constexpr_annotation(self, ann: ast.expr) -> bool:
        # Resolve the annotation to a Python object and compare by identity
        obj = self.resolve_value(ann)
        return obj is tlc.constexpr

    def as_ttgl_constexpr(self) -> ast.expr:
        # Build ttgl.constexpr
        return self.ttgl_attr("constexpr")

    def maybe_rewrite_constexpr_annotation(self, ann: Optional[ast.expr]) -> Optional[ast.expr]:
        if ann is None:
            return None
        if self.is_triton_constexpr_annotation(ann):
            return self.as_ttgl_constexpr()
        return ann

    def ttgl_attr(self, name: str) -> ast.AST:
        return ast.Attribute(value=ast.Name(id="ttgl", ctx=ast.Load()), attr=name, ctx=ast.Load())

    def resolve_value(self, expr: ast.expr):
        if isinstance(expr, ast.Name):
            value = self.scope.get(expr.id) or sys.modules.get(expr.id)
            return value
        if isinstance(expr, ast.Attribute):
            base = self.resolve_value(expr.value)
            if base is None:
                return None
            return getattr(base, expr.attr, None)
        return None

    def forward_call(self, node: ast.Call, target_func: ast.expr, filter_keywords: list[str] = []) -> ast.Call:
        new_keywords = [kw for kw in node.keywords if kw.arg not in filter_keywords]
        return ast.Call(func=target_func, args=list(node.args), keywords=list(new_keywords))

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        resolved_callable = self.resolve_value(node.func)
        if resolved_callable is not None:
            resolved_callable = triton.language.core._unwrap_if_constexpr(resolved_callable)
            base_function = getattr(resolved_callable, "fn", resolved_callable)
            function_name = getattr(base_function, "__qualname__", getattr(base_function, "__name__",
                                                                           str(base_function)))
            if triton.language.core.is_builtin(resolved_callable):
                builtin_name = function_name.split(".")[-1]
                builtin_mapping: dict[str, ast.expr] = {
                    "arange": ast.Name(id="tl_arange", ctx=ast.Load()),
                    "program_id": self.ttgl_attr("program_id"),
                    "load": self.ttgl_attr("load"),
                    "store": self.ttgl_attr("store"),
                    "cdiv": self.ttgl_attr("cdiv"),
                    "static_print": self.ttgl_attr("static_print"),
                    "static_assert": self.ttgl_attr("static_assert"),
                    "device_assert": self.ttgl_attr("device_assert"),
                    "device_print": self.ttgl_attr("device_print"),
                    "max_contiguous": self.ttgl_attr("max_contiguous"),
                    "multiple_of": self.ttgl_attr("multiple_of"),
                    "assume": self.ttgl_attr("assume"),
                    "minimum": self.ttgl_attr("minimum"),
                    "maximum": self.ttgl_attr("maximum"),
                    "fma": self.ttgl_attr("fma"),
                    "where": self.ttgl_attr("where"),
                    "cast": self.ttgl_attr("cast"),
                    "reshape": self.ttgl_attr("reshape"),
                    "trans": self.ttgl_attr("trans"),
                    "split": self.ttgl_attr("split"),
                    "inline_asm_elementwise": self.ttgl_attr("inline_asm_elementwise"),
                    "join": self.ttgl_attr("join"),
                    "atomic_max": self.ttgl_attr("atomic_max"),
                    "atomic_min": self.ttgl_attr("atomic_min"),
                    "atomic_or": self.ttgl_attr("atomic_or"),
                    "atomic_xchg": self.ttgl_attr("atomic_xchg"),
                    "atomic_xor": self.ttgl_attr("atomic_xor"),
                    "atomic_add": self.ttgl_attr("atomic_add"),
                    "atomic_and": self.ttgl_attr("atomic_and"),
                    "atomic_cas": self.ttgl_attr("atomic_cas"),
                    "num_warps": self.ttgl_attr("num_warps"),
                    "reduce": self.ttgl_attr("reduce"),
                    "full": ast.Name(id="tl_full", ctx=ast.Load()),
                    "dot": ast.Name(id="tl_dot", ctx=ast.Load()),
                    "dot_scaled": ast.Name(id="tl_dot_scaled", ctx=ast.Load()),
                    "make_tensor_descriptor": ast.Name(id="tl_make_tensor_descriptor", ctx=ast.Load()),
                    "load_tensor_descriptor": ast.Name(id="tl_load_tensor_descriptor", ctx=ast.Load()),
                    "store_tensor_descriptor": ast.Name(id="tl_store_tensor_descriptor", ctx=ast.Load()),
                    "num_threads": ast.Name(id="get_num_threads_per_program", ctx=ast.Load()),
                }
                mapped_target = builtin_mapping.get(builtin_name)
                filter_keywords = []
                # for reshape drop the can_reorder keyword, it is just an optimization and doesn't help much in Gluon.
                if builtin_name == "reshape":
                    filter_keywords = ["can_reorder"]
                if mapped_target is not None:
                    node = self.forward_call(node, mapped_target, filter_keywords)
                    # For split, apply on the source argument rather than wrapping destination
                    if builtin_name == "split":
                        source_arg = node.args[0]
                        wrapped_src = ast.Call(func=ast.Name(id="set_split_src_layout", ctx=ast.Load()),
                                               args=[source_arg], keywords=[])
                        node.args[0] = ast.copy_location(wrapped_src, source_arg)
                    # For shape/layout changing ops, wrap to reset layout
                    if builtin_name in {"reshape", "trans", "join", "reduce", "split"}:
                        forwarded_call = self.forward_call(node, mapped_target, filter_keywords)
                        reset_layout_wrapped = ast.Call(func=ast.Name(id="reset_to_default_layout", ctx=ast.Load()),
                                                        args=[forwarded_call], keywords=[])
                        node = ast.copy_location(reset_layout_wrapped, node)
                    return node
            # Track JITFunction callees
            if isinstance(resolved_callable, triton.runtime.jit.JITCallable):
                if resolved_callable not in self.jit_functions:
                    self.jit_functions.add(resolved_callable)
                    self.queue.append(resolved_callable)
                # Strip namespace: rewrite to local function name
                return self.forward_call(node, ast.Name(id=getattr(base_function, "__name__", ""), ctx=ast.Load()))
            if resolved_callable is triton.language.core.range:
                # skip all keywords except arg1, arg2, and step and replace with range.
                allowed = {"arg1", "arg2", "step"}
                new_keywords = [kw for kw in node.keywords if kw.arg in allowed]
                new_args = list(node.args[:3])
                return ast.copy_location(
                    ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=new_args, keywords=new_keywords),
                    node,
                )
            if resolved_callable is triton.language.core.static_range:
                return self.forward_call(node, self.ttgl_attr("static_range"))
        else:
            if isinstance(node.func, ast.Attribute) and node.func.attr in ["store", "load", "gather"]:
                helper_name = "tl_obj_" + node.func.attr
                return ast.Call(
                    func=ast.Name(id=helper_name, ctx=ast.Load()),
                    args=[node.func.value] + list(node.args),
                    keywords=list(node.keywords),
                )
            if isinstance(node.func,
                          ast.Attribute) and node.func.attr in ["reshape", "trans", "split", "join", "reduce"]:
                if node.func.attr == "split":
                    receiver_expr = node.func.value
                    wrapped_receiver = ast.Call(func=ast.Name(id="set_split_src_layout", ctx=ast.Load()),
                                                args=[receiver_expr], keywords=[])
                    new_func = ast.Attribute(value=ast.copy_location(wrapped_receiver, receiver_expr),
                                             attr=node.func.attr, ctx=ast.Load())
                    node = ast.copy_location(
                        ast.Call(func=new_func, args=list(node.args), keywords=list(node.keywords)), node)
                wrapped = ast.Call(
                    func=ast.Name(id="reset_to_default_layout", ctx=ast.Load()),
                    args=[node],
                    keywords=[],
                )
                return ast.copy_location(wrapped, node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        last_part = node.attr
        # Only rewrite dtypes when the resolved object is a tl.dtype instance
        # or the tl.dtype class itself (e.g., tl.float16 or tl.dtype.float16 / tl.dtype)
        resolved_obj = self.resolve_value(node)
        if resolved_obj is not None:
            if isinstance(resolved_obj, tlc.dtype):
                return self.ttgl_attr(last_part)
            if resolved_obj is tlc.dtype and last_part == "dtype":
                return self.ttgl_attr("dtype")
            if resolved_obj is tlc.tensor and last_part == "tensor":
                return self.ttgl_attr("tensor")
            if resolved_obj is tlc.constexpr and last_part == "constexpr":
                return self.ttgl_attr("constexpr")
        if last_part == "tensor_descriptor":
            return self.ttgl_attr("nvidia.hopper.tma.tensor_descriptor")
        return node

    def visit_Name(self, node):
        node = self.generic_visit(node)
        resolved_obj = self.resolve_value(node)
        if resolved_obj is not None:
            # Track standalone references to JITCallable and normalize name
            if isinstance(resolved_obj, triton.runtime.jit.JITCallable):
                if resolved_obj not in self.jit_functions:
                    self.jit_functions.add(resolved_obj)
                    self.queue.append(resolved_obj)
                base_function = getattr(resolved_obj, "fn", resolved_obj)
                normalized_name = getattr(base_function, "__name__",
                                          getattr(base_function, "__qualname__", getattr(node, "id", "")))
                return ast.copy_location(ast.Name(id=normalized_name, ctx=node.ctx), node)
            if isinstance(resolved_obj, triton.language.core.constexpr):
                identifier = getattr(node, "id", None)
                if identifier is not None:
                    # Use the current capture scope's file for the defining module
                    module_file = self.scope.get("__file__")
                    if isinstance(module_file, str):
                        bucket = self.constexpr_globals.setdefault(module_file, {})
                        bucket[identifier] = resolved_obj
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self.generic_visit(node)
        # TODO: generalize to
        # For patterns like x[None, :] or x[:, None], ensure x has a SliceLayout along the expanded dim
        expanded_dim = None
        if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
            first, second = node.slice.elts
            if isinstance(first, ast.Constant) and first.value is None:
                expanded_dim = 0
            elif isinstance(second, ast.Constant) and second.value is None:
                expanded_dim = 1
        if expanded_dim is not None:
            value_expr = node.value
            # Construct a 2D parent shape with a dummy dimension of size 1 at the expanded dim
            # Use value.type.shape[0] as the vector length
            type_attr = ast.Attribute(value=value_expr, attr="type", ctx=ast.Load())
            shape_attr = ast.Attribute(value=type_attr, attr="shape", ctx=ast.Load())
            len_expr = ast.Subscript(value=shape_attr, slice=ast.Constant(value=0), ctx=ast.Load())
            if expanded_dim == 0:
                parent_shape = ast.List(elts=[len_expr, ast.Constant(value=1)], ctx=ast.Load())
            else:
                parent_shape = ast.List(elts=[ast.Constant(value=1), len_expr], ctx=ast.Load())
            # Build SliceLayout(dim, default_blocked_layout(parent_shape, ttgl.num_warps()))
            slice_layout = ast.Call(
                func=self.ttgl_attr("SliceLayout"),
                args=[
                    ast.Constant(value=expanded_dim),
                    ast.Call(
                        func=ast.Name(id="default_blocked_layout", ctx=ast.Load()),
                        args=[parent_shape,
                              ast.Call(func=self.ttgl_attr("num_warps"), args=[], keywords=[])],
                        keywords=[],
                    ),
                ],
                keywords=[],
            )
            converted_value = ast.Call(
                func=self.ttgl_attr("convert_layout"),
                args=[value_expr, slice_layout],
                keywords=[],
            )
            return ast.Subscript(value=converted_value, slice=node.slice, ctx=node.ctx)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Rewrite parameter annotations: triton.language.constexpr -> ttgl.constexpr
        # Positional-only and regular args
        for arg in list(getattr(node.args, "posonlyargs", [])) + list(node.args.args):
            arg.annotation = self.maybe_rewrite_constexpr_annotation(arg.annotation)
        # Vararg / kwarg
        if node.args.vararg is not None:
            node.args.vararg.annotation = self.maybe_rewrite_constexpr_annotation(node.args.vararg.annotation)
        if node.args.kwarg is not None:
            node.args.kwarg.annotation = self.maybe_rewrite_constexpr_annotation(node.args.kwarg.annotation)
        # Keyword-only args
        for arg in node.args.kwonlyargs:
            arg.annotation = self.maybe_rewrite_constexpr_annotation(arg.annotation)
        if self.is_jit:
            node.decorator_list.insert(
                0, ast.Attribute(value=ast.Name(id="gluon", ctx=ast.Load()), attr="jit", ctx=ast.Load()))
        else:
            node.decorator_list.insert(
                0, ast.Attribute(value=ast.Name(id="gluon", ctx=ast.Load()), attr="constexpr_function", ctx=ast.Load()))
        # Process body
        return self.generic_visit(node)


def unparse_original_assignments(constexpr_globals: dict) -> list[str]:
    """Reconstruct original assignments for captured constexpr globals.

    We parse each defining module once to extract assignments, and rewrite tl.constexpr
    calls to ttgl.constexpr so the generated code remains consistent.
    """

    # Build assignment strings for captured globals by parsing each module once.
    def collect_names(target_node, names_out):
        if isinstance(target_node, ast.Name):
            names_out.append(target_node.id)
        elif isinstance(target_node, (ast.Tuple, ast.List)):
            for element in target_node.elts:
                collect_names(element, names_out)

    def parse_assigns_and_imports(path: str) -> tuple[dict[str, ast.AST], dict[str, str]]:
        try:
            with open(path, "r") as f:
                module_ast = ast.parse(f.read())
        except Exception:
            return {}, {}
        assigns: dict[str, ast.AST] = {}
        imports: dict[str, str] = {}
        for stmt in getattr(module_ast, "body", []):
            if isinstance(stmt, ast.Assign):
                names: list[str] = []
                for target in stmt.targets:
                    collect_names(target, names)
                for identifier in names:
                    assigns[identifier] = stmt
            elif isinstance(stmt, ast.AnnAssign):
                names: list[str] = []
                collect_names(stmt.target, names)
                if stmt.value is not None:
                    for identifier in names:
                        assigns[identifier] = stmt
            elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and isinstance(stmt.module, str):
                for alias in stmt.names:
                    alias_name = alias.asname or alias.name.split(".")[-1]
                    imports[alias_name] = stmt.module
        return assigns, imports

    def rewrite_constexpr_to_ttgl(node: ast.AST) -> ast.AST:

        class ConstexprToTtglRewriter(ast.NodeTransformer):

            def visit_Call(self, call_node: ast.Call) -> ast.AST:
                call_node = self.generic_visit(call_node)
                if isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "constexpr":
                    call_node.func = ast.copy_location(
                        ast.Attribute(value=ast.Name(id="ttgl", ctx=ast.Load()), attr="constexpr", ctx=ast.Load()),
                        call_node.func)
                return call_node

        return ConstexprToTtglRewriter().visit(node)

    results: list[str] = []
    imported_cache: dict[str, dict[str, ast.AST]] = {}
    for mod_file, name_to_obj in constexpr_globals.items():
        assigns, imports = parse_assigns_and_imports(mod_file)
        for identifier in sorted(name_to_obj.keys()):
            node = assigns.get(identifier)
            if node is None:
                imported_module_name = imports.get(identifier)
                if imported_module_name:
                    try:
                        module_spec = importlib.util.find_spec(imported_module_name)
                        origin = getattr(module_spec, "origin", None) if module_spec is not None else None
                    except Exception:
                        origin = None
                    if origin:
                        assignment_map = imported_cache.get(origin)
                        if assignment_map is None:
                            assignment_map, _ = parse_assigns_and_imports(origin)
                            imported_cache[origin] = assignment_map
                        node = assignment_map.get(identifier)
            if node is not None:
                edited_node = rewrite_constexpr_to_ttgl(copy.deepcopy(node))
                ast.fix_missing_locations(edited_node)
                results.append(ast.unparse(edited_node))
            else:
                results.append(f"{identifier} = {repr(name_to_obj[identifier])}")
    return results


def convert_triton_to_gluon(src: triton.runtime.jit.JITCallable) -> str:
    """Convert a Triton JIT entry point into a Gluon source string."""
    shared_jit_set: set = set()
    function_queue: list = [src]
    constexpr_globals: dict = {}
    out = ""
    # Process discovered callee JITFunctions, converting and appending them
    while function_queue:
        callee = function_queue.pop(0)
        callee_src = callee._src
        callee_tree = ast.parse(callee_src)
        callee_scope = getattr(callee, "__globals__", {}) or {}
        jit = isinstance(callee, triton.runtime.JITFunction)
        callee_transformer = TritonToGluonTransformer(globals_map=callee_scope, shared_jit_set=shared_jit_set,
                                                      shared_queue=function_queue, is_jit=jit,
                                                      constexpr_globals=constexpr_globals)
        callee_new = callee_transformer.visit(callee_tree)
        ast.fix_missing_locations(callee_new)
        out += "\n\n" + ast.unparse(callee_new)

    out = "\n\n" + out

    # Pull constexpr globals from the original source code
    for line in unparse_original_assignments(constexpr_globals):
        out = line + "\n" + out

    # Prepend required Gluon imports
    out = GLUON_IMPORT_LINES + "\n\n" + out

    return out
