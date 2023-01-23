import ast
from pathlib import Path
from typing import Any, Dict, Sequence

from dataclasses import dataclass

from triton import JITFunction


class JITStub(JITFunction):
    def __init__(
        self,
        fn,
        arg_names: Sequence[str],
        src: str,
        globals: Dict[str, Any],
    ) -> None:
        self.fn = fn
        self.arg_names = arg_names
        self.src = src

        self.__globals__ = globals
        self.__name__ = fn.__name__

    def __repr__(self):
        return f"JITStub({self.fn.__name__})"


@dataclass
class KernelMeta:
    arg_names: Sequence[str]
    src: str
    consts: Sequence[int]
    stub: Sequence[ast.FunctionDef]
    dependencies: Sequence[str]


# Undefined names will be caught by IR Generator
# All I need to do is find a topological order of the calls
# That is -> Assume all known kernels are all the ones that are available
class CollectJITandGlobals(ast.NodeVisitor):
    def __init__(self) -> None:
        self._depth = 0

    def extract_kernel_and_globals(self, src_path):
        def _init_state():
            self.deps = set()
            self.globals = {}
            self.src_lines = ""
            self._src_path = ""

        p = Path(src_path)
        module_name = p.stem
        src = p.read_text()

        _init_state()

        self._src_path = src_path
        self.src_lines = src.split("\n")
        self.visit(ast.parse(src))

        globals_ = self.globals

        _init_state()
        return globals_

    def visit_arg(self, node: ast.arg):
        return node.arg, self.visit(node.annotation) in (
            "constexpr",
            "tl.constexpr",
            "triton.language.constexpr",
        )

    def visit_arguments(self, node: ast.arguments):
        consts = []
        arg_names = []
        for idx, arg in enumerate(node.args):
            arg_name, is_const = self.visit(arg)
            arg_names.append(arg_name)
            if is_const:
                consts.append(idx)

        return arg_names, consts

    def visit_Assign(self, node: ast.Assign):
        # assume global scope has constants and other jit funcitons
        names = [self.visit(t) for t in node.targets]
        assert len(names) == 1, "No tuple unpacking and stuff like that"
        names = names[0]
        vals = self.visit(node.value)
        if not isinstance(vals, Sequence):
            vals = (vals,)
            names = (names,)

        if self._depth > 0:
            return

        for name, val in zip(names, vals):
            if val is None:
                print(
                    self._err_msg(
                        f"[Skipping] Non-constant global assignment to {name}", node
                    )
                )
                continue
            self.globals[name] = val

        return

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_Call(self, node: ast.Call):
        name = self.visit(node.func)
        # self.function_calls.add(name)
        return

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load):
            self.deps.add(node.id)

        return node.id

    def visit_alias(self, node: ast.alias) -> Any:
        return node.name, node.asname

    def generic_visit(self, node):
        if node is None:
            return
        typename = type(node).__name__
        # print(f"[DEBUG] Unsupported node: {typename}")
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Module(self, node: ast.Module):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        attr = node.attr
        prefix = self.visit(node.value)
        return f"{prefix}.{attr}"

    def visit_Return(self, node: ast.Return) -> Any:
        return

    def _make_kernel_stub(self, func: ast.FunctionDef) -> str:
        import copy

        stub = copy.copy(func)
        for arg in stub.args.args:
            arg.annotation = None
        stub.decorator_list = []
        stub.body = [ast.Pass(lineno=0, col_offset=0)]
        return stub

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for dec in node.decorator_list:
            dname = self.visit(dec)
            if dname in ("triton.jit",):
                name = node.name
                st = node.lineno
                en = node.end_lineno
                src = "\n".join(self.src_lines[st - 1 : en])
                arg_names, consts = self.visit(node.args)

                self._depth += 1
                self.deps = set()
                ast.NodeVisitor.generic_visit(self, node)
                self._depth -= 1

                self.globals[name] = KernelMeta(
                    arg_names=arg_names,
                    src=src,
                    consts=consts,
                    stub=self._make_kernel_stub(node),
                    dependencies=self.deps,
                )
                self.deps = set()

                return

        msg = self._err_msg(f"[Skipping] Non-jitted function", node)
        print(msg)

    def _skip_global_visit(self, node):
        if self._depth == 0:
            return
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Try(self, node: ast.Try) -> Any:
        # avoid non-static globals
        self._skip_global_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        # avoid non-static globals
        self._skip_global_visit(node)

    def _err_msg(self, msg, node):
        src = self.src_lines[node.lineno - 1]
        return f"{msg} (in {self._src_path})\n\t {node.lineno} | {src} \n"


def build_stubs(func_stubs: Sequence[ast.FunctionDef]):
    module = ast.Module(body=func_stubs, type_ignores=[])
    scope = {}
    exec(compile(module, filename="<ast>", mode="exec"), scope)
    scope.pop("__builtins__")
    return scope


def _merge_conflict_report(main_, includes, main_path, inc_paths):
    """
    Looks for name conflicts compared with the main file.
    Reports any conflict it finds an fails.
    """
    name_conflicts = []
    first_appeared = {k: main_path for k in main_}
    for src_idx, mod in enumerate(includes):
        for k, v in mod.items():
            if k not in main_:
                first_appeared[k] = inc_paths[src_idx]
                main_[k] = v
                continue
            name_conflicts.append(
                f"[Conflict] `{k}` exists.\n\t- First defined in {first_appeared[k]}\n\t- Redefined in {inc_paths[src_idx]}"
            )

    if len(name_conflicts):
        raise NameError("\n".join(name_conflicts))
    return main_


def build_jit_stubs(*src_paths: Sequence[str]) -> Dict[str, JITStub]:
    """
    - Analyze AST: find global constants + triton.jit decorated functions
    - Function Stubs: Generate function stubs and wrap in a stub JITFunction
    - All include paths get added to the kernel function scope (kernel takes precedence) 
    """
    # Get globals and Jitted function from static source code
    generator = CollectJITandGlobals()

    globals_ = []
    for src_path in src_paths:
        _scope = generator.extract_kernel_and_globals(src_path)
        globals_.append(_scope)

    # Find conflicting names
    all_globals = _merge_conflict_report(
        globals_[0],
        includes=globals_[1:],
        main_path=src_paths[0],
        inc_paths=src_paths[1:],
    )

    # Build global scope for functions
    gscope = {}
    ker_name = []
    ker_vals = []
    stubs = []
    for k, v in all_globals.items():
        if isinstance(v, KernelMeta):
            stubs.append(v.stub)
            ker_name.append(k)
            ker_vals.append(v)
        else:
            gscope[k] = v

    # Create live function stubs for JITFunction stub
    stub_scope = build_stubs(stubs)

    # Update globals as we generate JITStub
    jit_stubs = {}
    for k, meta in zip(ker_name, ker_vals):
        fn = stub_scope[k]
        jit_stubs[k] = JITStub(
            fn=fn, arg_names=meta.arg_names, src=meta.src, globals=gscope
        )
        gscope.update(jit_stubs)

    return jit_stubs


if __name__ == "__main__":
    jit_stubs = build_jit_stubs(
        "python/examples/vector_addition.py",
        "python/examples/copy_strided.py",
        "python/examples/layer-norm.py",
    )
