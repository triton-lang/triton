import ast
import inspect
import re
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from .. import language
from .._C.libtriton.triton import ir
from ..language import constexpr, tensor
# ideally we wouldn't need any runtime component
from ..runtime import JITFunction
from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)


def mangle_ty(ty):
    if ty.is_ptr():
        return 'P' + mangle_ty(ty.element_ty)
    if ty.is_int():
        SIGNED = language.dtype.SIGNEDNESS.SIGNED
        prefix = 'i' if ty.int_signedness == SIGNED else 'u'
        return prefix + str(ty.int_bitwidth)
    if ty.is_floating():
        return str(ty)
    if ty.is_block():
        elt = mangle_ty(ty.scalar)
        shape = '_'.join(map(str, ty.shape))
        return f'{elt}S{shape}S'
    if ty.is_void():
        return 'V'
    assert False, "Unsupported type"


def mangle_fn(name, arg_tys, constants):
    # doesn't mangle ret type, which must be a function of arg tys
    mangled_arg_names = '_'.join([mangle_ty(ty) for ty in arg_tys])
    mangled_constants = '_'.join([f'{i}c{repr(constants[i])}' for i in sorted(constants)])
    mangled_constants = mangled_constants.replace('.', '_d_')
    mangled_constants = mangled_constants.replace("'", '_sq_')
    # [ and ] are not allowed in LLVM identifiers
    mangled_constants = mangled_constants.replace('[', '_').replace(']', '_')
    ret = f'{name}__{mangled_arg_names}__{mangled_constants}'
    return ret


def _is_triton_tensor(o: Any) -> bool:
    return isinstance(o, tensor)


def _is_constexpr(o: Any) -> bool:
    return isinstance(o, constexpr)


def _is_triton_scalar(o: Any) -> bool:
    return _is_triton_tensor(o) and (not o.type.is_block() or o.type.numel == 1)


def _is_list_like(o: Any) -> bool:
    return isinstance(o, (list, tuple))


def _unwrap_if_constexpr(o: Any):
    return o.value if isinstance(o, constexpr) else o


def _check_fn_args(node, fn, args):
    if fn.noinline:
        for idx, arg in enumerate(args):
            if not _is_constexpr(arg) and not _is_triton_scalar(arg):
                raise UnsupportedLanguageConstruct(
                    fn.src, node,
                    f'Function {fn.__name__} is marked noinline, but was called with non-scalar argument {fn.arg_names[idx]}:{arg}'
                )


def _get_fn_file_line(fn):
    base_fn = fn
    while not isinstance(base_fn, JITFunction):
        base_fn = base_fn.fn
    file_name = base_fn.fn.__code__.co_filename
    lines, begin_line = inspect.getsourcelines(base_fn.fn)
    # Match the following pattern:
    # @triton.autotune(...) <- foo.__code__.co_firstlineno
    # @triton.heuristics(...)
    # @triton.jit
    # def foo(...): <- this line is the first line
    for idx, line in enumerate(lines):
        if line.strip().startswith("def "):
            begin_line += idx
            break
    return file_name, begin_line


_condition_types = {bool, int, type(None)}  # Python types accepted for conditionals inside kernels


class enter_sub_region:

    def __init__(self, generator):
        self.generator = generator

    def __enter__(self):
        # record lscope & local_defs in the parent scope
        self.liveins = self.generator.lscope.copy()
        self.prev_defs = self.generator.local_defs.copy()
        self.generator.local_defs = {}
        self.insert_block = self.generator.builder.get_insertion_block()
        self.insert_point = self.generator.builder.get_insertion_point()
        return self.liveins, self.insert_block

    def __exit__(self, *args, **kwargs):
        self.generator.builder.restore_insertion_point(self.insert_point)
        self.generator.lscope = self.liveins
        self.generator.local_defs = self.prev_defs


# Check if the given syntax node has an "early" return
class ContainsReturnChecker(ast.NodeVisitor):

    def __init__(self, gscope):
        self.gscope = gscope

    def _visit_stmts(self, body) -> bool:
        for s in body:
            if self.visit(s):
                return True
        return False

    def _visit_function(self, fn) -> bool:
        # Currently we only support JITFunctions defined in the global scope
        if isinstance(fn, JITFunction) and not fn.noinline:
            fn_node = fn.parse()
            return ContainsReturnChecker(self.gscope).visit(fn_node)
        return False

    def generic_visit(self, node) -> bool:
        ret = False
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        ret = ret or self.visit(item)
            elif isinstance(value, ast.AST):
                ret = ret or self.visit(value)
        return ret

    def visit_Attribute(self, node: ast.Attribute) -> bool:
        # If the left part is a name, it's possible that
        # we call triton native function or a jit function from another module.
        # If the left part is not a name, it must return a tensor or a constexpr
        # whose methods do not contain return statements
        # e.g., (tl.load(x)).to(y)
        # So we only check if the expressions within value have return or not
        if isinstance(node.value, ast.Name):
            if node.value.id in self.gscope:
                value = self.gscope[node.value.id]
                fn = getattr(value, node.attr)
                return self._visit_function(fn)
            return False
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> bool:
        if type(node.ctx) == ast.Store:
            return False
        if node.id in self.gscope:
            fn = self.gscope[node.id]
            return self._visit_function(fn)
        return False

    def visit_Return(self, node: ast.Return) -> bool:
        return True

    def visit_Assign(self, node: ast.Assign) -> bool:
        # There couldn't be an early return
        # x = ...
        return False

    def visit_AugAssign(self, node: ast.AugAssign) -> bool:
        # There couldn't be an early return
        # x += ...
        return False

    def visit_Module(self, node: ast.Module) -> bool:
        return self._visit_stmts(node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> bool:
        return self._visit_stmts(node.body)

    def visit_If(self, node: ast.If) -> bool:
        # TODO: optimize the following case in which we actually don't have
        # a return when static_cond is false:
        # if dynamic_cond
        #   if static_cond
        #     func_with_return
        #   else
        #     func_without_return
        ret = self._visit_stmts(node.body)
        if node.orelse:
            ret = ret or self._visit_stmts(node.orelse)
        return ret

    def visit_IfExp(self, node: ast.IfExp) -> bool:
        return self.visit(node.body) or self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> bool:
        return self.visit(node.func)


class CodeGenerator(ast.NodeVisitor):

    def __init__(self, context, prototype, gscope, attributes, constants, function_name, target, module=None,
                 is_kernel=False, function_types: Optional[Dict] = None, debug=False, noinline=False,
                 file_name: Optional[str] = None, begin_line=0):
        self.context = context
        self.builder = ir.builder(context)
        self.file_name = file_name
        # node.lineno starts from 1, so we need to subtract 1
        self.begin_line = begin_line - 1
        self.builder.set_loc(file_name, begin_line, 0)
        self.builder.target = target
        self.module = self.builder.create_module() if module is None else module
        self.function_ret_types = {} if function_types is None else function_types
        self.prototype = prototype
        self.gscope = gscope
        self.lscope = dict()
        self.attributes = attributes
        self.constants = constants
        self.function_name = function_name
        self.is_kernel = is_kernel
        self.last_node = None
        self.debug = debug
        self.noinline = noinline
        self.scf_stack = []
        self.last_ret_type = None
        # SSA-construction
        # name => language.tensor
        self.local_defs: Dict[str, tensor] = {}
        self.global_uses: Dict[str, tensor] = {}
        self.dereference_name: Callable[[str], Any] = self._define_name_lookup()
        self.fn = None

    builtin_namespace: Dict[str, Any] = {_.__name__: _ for _ in (range, float, int, isinstance, getattr)}
    builtin_namespace.update((
        ('print', language.core.device_print),
        ('min', language.minimum),
    ))

    def _define_name_lookup(self):

        def local_lookup(name: str, absent):
            # this needs to be re-fetched from `self` every time, because it gets switched occasionally
            value = self.lscope.get(name, absent)
            if value is not absent and name not in self.local_defs:
                self.global_uses[name] = value
            return value

        absent_marker = object()

        def name_lookup(name: str) -> Any:
            absent = absent_marker
            for lookup_function in local_lookup, self.gscope.get, self.builtin_namespace.get:
                value = lookup_function(name, absent)
                if value is not absent:
                    return value
            raise NameError(f'{name} is not defined')

        return name_lookup

    def set_value(self, name: str, value: Union[tensor, constexpr]) -> None:
        ''' This function:
            called by visit_Assign() & visit_FunctionDef() to store left value (lvalue)
        1. record local defined name (FIXME: should consider control flow)
        2. store tensor in self.lvalue
        '''
        self.lscope[name] = value
        self.local_defs[name] = value

    def _get_insertion_point_and_loc(self):
        # XXX: this is a hack to get the location of the insertion point.
        # The insertion point's location could be invalid sometimes,
        # so we need to explicitly set the location
        loc = self.builder.get_loc()
        ip = self.builder.get_insertion_point()
        return ip, loc

    def _set_insertion_point_and_loc(self, ip, loc):
        self.builder.restore_insertion_point(ip)
        self.builder.set_loc(loc)

    #
    # AST visitor
    #
    def visit_compound_statement(self, stmts):
        # Ensure that stmts is iterable
        if not _is_list_like(stmts):
            stmts = [stmts]
        for stmt in stmts:
            ret_type = self.visit(stmt)
            if ret_type is not None and isinstance(stmt, ast.Return):
                self.last_ret_type = ret_type

    def visit_Module(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_List(self, node):
        ctx = self.visit(node.ctx)
        assert ctx is None
        elts = [self.visit(elt) for elt in node.elts]
        return elts

    # By design, only non-kernel functions can return
    def visit_Return(self, node):
        ret_value = self.visit(node.value)
        # ret_block = self.builder.create_block()
        # post_ret_block = self.builder.create_block()
        # self.builder.create_branch(ret_block)
        # self.builder.set_insertion_point_to_end(ret_block)
        if ret_value is None:
            self.builder.ret([])
            ret_ty = None
        elif isinstance(ret_value, tuple):
            ret_values = [language.core._to_tensor(v, self.builder) for v in ret_value]
            ret_types = [v.type for v in ret_values]
            self.builder.ret([v.handle for v in ret_values])
            ret_ty = tuple(ret_types)
        else:
            ret = language.core._to_tensor(ret_value, self.builder)
            self.builder.ret([ret.handle])
            ret_ty = ret.type
        # self.builder.create_branch(post_ret_block)
        # self.builder.set_insertion_point_to_end(post_ret_block)
        return ret_ty

    def visit_FunctionDef(self, node):
        arg_names, kwarg_names = self.visit(node.args)
        if self.fn:
            raise UnsupportedLanguageConstruct(None, node, "nested function definition is not supported.")
        # initialize defaults
        for i, default_value in enumerate(node.args.defaults):
            arg_node = node.args.args[-i - 1]
            annotation = arg_node.annotation
            name = arg_node.arg
            st_target = ast.Name(id=name, ctx=ast.Store())
            if annotation is None:
                init_node = ast.Assign(targets=[st_target], value=default_value)
            else:
                init_node = ast.AnnAssign(target=st_target, value=default_value, annotation=annotation)
            self.visit(init_node)
        # initialize function
        visibility = "public" if self.is_kernel else "private"
        self.fn = self.builder.get_or_insert_function(self.module, self.function_name,
                                                      self.prototype.to_ir(self.builder), visibility, self.noinline)
        self.module.push_back(self.fn)
        entry = self.fn.add_entry_block()
        arg_values = []
        idx = 0
        for i, arg_name in enumerate(arg_names):
            if i in self.constants:
                cst = self.constants[i]
                if not _is_constexpr(cst):
                    cst = constexpr(self.constants[i])
                arg_values.append(cst)
                continue
            else:
                if i in self.attributes:
                    for name, value in self.attributes[i]:
                        self.fn.set_arg_attr(idx, name, value)
                arg_values.append(tensor(self.fn.args(idx), self.prototype.param_types[idx]))
                idx += 1

        insert_pt = self.builder.get_insertion_block()
        for arg_name, arg_value in zip(arg_names, arg_values):
            self.set_value(arg_name, arg_value)
        self.builder.set_insertion_point_to_start(entry)
        # visit function body
        self.visit_compound_statement(node.body)
        # finalize function
        if self.last_ret_type is None:
            self.builder.ret([])
        else:
            # update return type
            if isinstance(self.last_ret_type, tuple):
                self.prototype.ret_types = list(self.last_ret_type)
                self.fn.reset_type(self.prototype.to_ir(self.builder))
            else:
                self.prototype.ret_types = [self.last_ret_type]
                self.fn.reset_type(self.prototype.to_ir(self.builder))
        if insert_pt:
            self.builder.set_insertion_point_to_end(insert_pt)
        # Remove dead code
        self.fn.finalize()

    def visit_arguments(self, node):
        arg_names = []
        for arg in node.args:
            arg_names += [self.visit(arg)]
        kwarg_names = self.visit(node.kwarg)
        return arg_names, kwarg_names

    def visit_arg(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        return node.arg

    def visit_AnnAssign(self, node):
        # extract attributes
        annotation = self.visit(node.annotation)
        target = self.visit(node.target)
        value = self.visit(node.value)
        # constexpr
        if annotation == constexpr:
            if target in self.lscope:
                raise ValueError(f'{target} is already defined.'
                                 f' constexpr cannot be reassigned.')
            if not _is_constexpr(value):
                value = constexpr(value)
            self.lscope[target] = value
            return self.lscope[target]
        # default: call visit_Assign
        return self.visit_Assign(node)

    def visit_Assign(self, node):
        _names = []
        for target in node.targets:
            _names += [self.visit(target)]
        if len(_names) > 1:
            raise UnsupportedLanguageConstruct(None, node, "simultaneous multiple assignment is not supported.")
        names = _names[0]
        values = self.visit(node.value)
        if not _is_list_like(names):
            names = [names]
        if not _is_list_like(values):
            values = [values]
        native_nontensor_types = (language.dtype, )
        for name, value in zip(names, values):
            # by default, constexpr are assigned into python variable
            value = _unwrap_if_constexpr(value)
            if value is not None and \
               not _is_triton_tensor(value) and \
               not isinstance(value, native_nontensor_types):
                value = language.core._to_tensor(value, self.builder)
            self.set_value(name, value)

    def visit_AugAssign(self, node):
        name = node.target.id
        lhs = ast.Name(id=name, ctx=ast.Load())
        rhs = ast.BinOp(lhs, node.op, node.value)
        assign = ast.Assign(targets=[node.target], value=rhs)
        self.visit(assign)
        return self.dereference_name(name)

    def visit_Name(self, node):
        if type(node.ctx) == ast.Store:
            return node.id
        return self.dereference_name(node.id)

    def visit_Store(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Load(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Tuple(self, node):
        args = [self.visit(x) for x in node.elts]
        return tuple(args)

    def _apply_binary_method(self, method_name, lhs, rhs):
        # TODO: raise something meaningful if getattr fails below, esp for reverse method
        if _is_triton_tensor(lhs):
            return getattr(lhs, method_name)(rhs, _builder=self.builder)
        if _is_triton_tensor(rhs):
            reverse_method_name = re.sub(r"__(.*)__", r"__r\1__", method_name)
            return getattr(rhs, reverse_method_name)(lhs, _builder=self.builder)
        return getattr(lhs, method_name)(rhs)

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        method_name = self._method_name_for_bin_op.get(type(node.op))
        if method_name is None:
            raise UnsupportedLanguageConstruct(
                None, node, "AST binary operator '{}' is not (currently) implemented.".format(node.op.__name__))
        return self._apply_binary_method(method_name, lhs, rhs)

    _method_name_for_bin_op: Dict[Type[ast.operator], str] = {
        ast.Add: '__add__',
        ast.Sub: '__sub__',
        ast.Mult: '__mul__',
        ast.Div: '__truediv__',
        ast.FloorDiv: '__floordiv__',
        ast.Mod: '__mod__',
        ast.Pow: '__pow__',
        ast.LShift: '__lshift__',
        ast.RShift: '__rshift__',
        ast.BitAnd: '__and__',
        ast.BitOr: '__or__',
        ast.BitXor: '__xor__',
    }

    def visit_then_else_blocks(self, node, liveins, then_block, else_block):
        # then block
        self.builder.set_insertion_point_to_start(then_block)
        self.visit_compound_statement(node.body)
        then_block = self.builder.get_insertion_block()
        then_defs = self.local_defs.copy()
        # else block
        else_defs = {}
        if node.orelse:
            self.builder.set_insertion_point_to_start(else_block)
            self.lscope = liveins.copy()
            self.local_defs = {}
            self.visit_compound_statement(node.orelse)
            else_defs = self.local_defs.copy()
            else_block = self.builder.get_insertion_block()

        # update block arguments
        names = []
        ret_types = []
        ir_ret_types = []
        # variables in livein whose value is updated in `if`
        for name in liveins:
            # check type
            for defs, block_name in [(then_defs, 'then'), (else_defs, 'else')]:
                if name in defs:
                    assert defs[name].type == liveins[name].type, \
                        f'initial value for `{name}` is of type {liveins[name].type}, '\
                        f'but the {block_name} block redefines it as {defs[name].type}'
            if name in then_defs or name in else_defs:
                names.append(name)
                ret_types.append(then_defs[name].type if name in then_defs else else_defs[name].type)
                ir_ret_types.append(then_defs[name].handle.get_type() if name in
                                    then_defs else else_defs[name].handle.get_type())
            # variable defined in then but not in else
            if name in then_defs and name not in else_defs:
                else_defs[name] = liveins[name]
            # variable defined in else but not in then
            if name in else_defs and name not in then_defs:
                then_defs[name] = liveins[name]
        # variables that are both in then and else but not in liveins
        # TODO: could probably be cleaned up
        for name in then_defs.keys() & else_defs.keys():
            if name in names:
                continue
            then_ty = then_defs[name].type
            else_ty = else_defs[name].type
            assert then_ty == else_ty, \
                f'mismatched type for {name} between then block ({then_ty}) '\
                f'and else block ({else_ty})'
            names.append(name)
            ret_types.append(then_ty)
            ir_ret_types.append(then_defs[name].handle.get_type())

        return then_defs, else_defs, then_block, else_block, names, ret_types, ir_ret_types

    def visit_if_top_level(self, cond, node):
        has_endif_block = True
        with enter_sub_region(self) as sr:
            liveins, ip_block = sr
            then_block = self.builder.create_block()
            else_block = self.builder.create_block()
            # create basic-block after conditional
            endif_block = self.builder.create_block()
            # create branch
            self.builder.set_insertion_point_to_end(ip_block)
            self.builder.create_cond_branch(cond.handle, then_block, else_block)
            # visit then and else blocks
            then_defs, else_defs, then_block, else_block, names, ret_types, ir_ret_types = \
                self.visit_then_else_blocks(node, liveins, then_block, else_block)
            # then terminator
            self.builder.set_insertion_point_to_end(then_block)
            if then_block.has_return() and else_block.has_return():
                has_endif_block = False
                endif_block.erase()
            if not then_block.has_terminator() and has_endif_block:
                self.builder.create_branch(endif_block, [then_defs[n].handle for n in names])
            # else terminator
            self.builder.set_insertion_point_to_end(else_block)
            if not else_block.has_terminator() and has_endif_block:
                self.builder.create_branch(endif_block, [else_defs[n].handle for n in names])
            if has_endif_block:
                for ty in ir_ret_types:
                    endif_block.add_argument(ty)
        if has_endif_block:
            # change block
            self.builder.set_insertion_point_to_start(endif_block)
            # update value
            for i, name in enumerate(names):
                new_tensor = language.core.tensor(endif_block.arg(i), ret_types[i])
                self.set_value(name, new_tensor)

    # TODO: refactor
    def visit_if_scf(self, cond, node):
        with enter_sub_region(self) as sr:
            liveins, _ = sr
            ip, last_loc = self._get_insertion_point_and_loc()
            then_block = self.builder.create_block()
            else_block = self.builder.create_block() if node.orelse else None
            then_defs, else_defs, then_block, else_block, names, ret_types, _ = \
                self.visit_then_else_blocks(node, liveins, then_block, else_block)
            # create if op
            self._set_insertion_point_and_loc(ip, last_loc)
            if_op = self.builder.create_if_op([ty.to_ir(self.builder) for ty in ret_types], cond.handle, True)
            then_block.merge_block_before(if_op.get_then_block())
            self.builder.set_insertion_point_to_end(if_op.get_then_block())
            if len(names) > 0:
                self.builder.create_yield_op([then_defs[n].handle for n in names])
            if not node.orelse:
                else_block = if_op.get_else_block()
            else:
                else_block.merge_block_before(if_op.get_else_block())
            self.builder.set_insertion_point_to_end(if_op.get_else_block())
            if len(names) > 0:
                self.builder.create_yield_op([else_defs[n].handle for n in names])
        # update values
        for i, name in enumerate(names):
            new_tensor = language.core.tensor(if_op.get_result(i), ret_types[i])
            self.set_value(name, new_tensor)

    def visit_If(self, node):
        cond = self.visit(node.test)
        if _is_triton_tensor(cond):
            cond = cond.to(language.int1, _builder=self.builder)
            contains_return = ContainsReturnChecker(self.gscope).visit(node)
            if self.scf_stack and contains_return:
                raise UnsupportedLanguageConstruct(
                    None, node, "Cannot have `return` statements inside `while` or `for` statements in triton "
                    "(note that this also applies to `return` statements that are inside functions "
                    "transitively called from within `while`/`for` statements)")
            elif self.scf_stack or not contains_return:
                self.visit_if_scf(cond, node)
            else:
                self.visit_if_top_level(cond, node)
        else:
            cond = _unwrap_if_constexpr(cond)
            # not isinstance - we insist the real thing, no subclasses and no ducks
            if type(cond) not in _condition_types:
                raise UnsupportedLanguageConstruct(
                    None, node,
                    "`if` conditionals can only accept values of type {{{}}}, not objects of type {}".format(
                        ', '.join(_.__name__ for _ in _condition_types),
                        type(cond).__name__))
            if cond:
                self.visit_compound_statement(node.body)
            else:
                self.visit_compound_statement(node.orelse)

    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if _is_triton_tensor(cond):
            cond = cond.to(language.int1, _builder=self.builder)
            # TODO: Deal w/ more complicated return types (e.g tuple)
            with enter_sub_region(self):
                ip, last_loc = self._get_insertion_point_and_loc()

                then_block = self.builder.create_block()
                self.builder.set_insertion_point_to_start(then_block)
                then_val = language.core._to_tensor(self.visit(node.body), self.builder)
                then_block = self.builder.get_insertion_block()

                else_block = self.builder.create_block()
                self.builder.set_insertion_point_to_start(else_block)
                # do not need to reset lscope since
                # ternary expressions cannot define new variables
                else_val = language.core._to_tensor(self.visit(node.orelse), self.builder)
                else_block = self.builder.get_insertion_block()

                self._set_insertion_point_and_loc(ip, last_loc)

                assert then_val.type == else_val.type, \
                    f'ternary expression with dynamic condition has inconsistent types {then_val.type} and {else_val.type}'
                ret_type = then_val.type

                ret_type_ir = [ret_type.to_ir(self.builder)] if ret_type != language.void else []
                if_op = self.builder.create_if_op(ret_type_ir, cond.handle, True)
                then_block.merge_block_before(if_op.get_then_block())
                if ret_type_ir:
                    self.builder.set_insertion_point_to_end(if_op.get_then_block())
                    self.builder.create_yield_op([then_val.handle])

                self.builder.set_insertion_point_to_end(if_op.get_then_block())
                else_block.merge_block_before(if_op.get_else_block())
                if ret_type_ir:
                    self.builder.set_insertion_point_to_end(if_op.get_else_block())
                    self.builder.create_yield_op([else_val.handle])
                return language.core.tensor(if_op.get_result(0), ret_type) if ret_type_ir else None
        else:
            cond = _unwrap_if_constexpr(cond)

            # not isinstance - we insist the real thing, no subclasses and no ducks
            if type(cond) not in _condition_types:
                raise UnsupportedLanguageConstruct(
                    None, node,
                    "`if` conditionals can only accept values of type {{{}}}, not objects of type {}".format(
                        ', '.join(_.__name__ for _ in _condition_types),
                        type(cond).__name__))
            if cond:
                return self.visit(node.body)
            else:
                return self.visit(node.orelse)

    def visit_Pass(self, node):
        pass

    def visit_Compare(self, node):
        if not (len(node.comparators) == 1 and len(node.ops) == 1):
            raise UnsupportedLanguageConstruct(None, node, "simultaneous multiple comparison is not supported")
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        lhs_value = _unwrap_if_constexpr(lhs)
        rhs_value = _unwrap_if_constexpr(rhs)
        if type(node.ops[0]) == ast.Is:
            return constexpr(lhs_value is rhs_value)
        if type(node.ops[0]) == ast.IsNot:
            return constexpr(lhs_value is not rhs_value)
        method_name = self._method_name_for_comp_op.get(type(node.ops[0]))
        if method_name is None:
            raise UnsupportedLanguageConstruct(
                None, node, "AST comparison operator '{}' is not (currently) implemented.".format(node.ops[0].__name__))
        return self._apply_binary_method(method_name, lhs, rhs)

    _method_name_for_comp_op: Dict[Type[ast.cmpop], str] = {
        ast.Eq: '__eq__', ast.NotEq: '__ne__', ast.Lt: '__lt__', ast.LtE: '__le__', ast.Gt: '__gt__', ast.GtE: '__ge__'
    }

    def visit_UnaryOp(self, node):
        op = self.visit(node.operand)
        fn = self._method_name_for_unary_op.get(type(node.op))
        if fn is None:
            raise UnsupportedLanguageConstruct(
                None, node, "AST unary operator '{}' is not (currently) implemented.".format(node.op.__name__))
        if _is_triton_tensor(op):
            return getattr(op, fn)(_builder=self.builder)
        return getattr(op, fn)()

    _method_name_for_unary_op: Dict[Type[ast.unaryop], str] = {
        ast.USub: '__neg__', ast.UAdd: '__pos__', ast.Not: '__not__', ast.Invert: '__invert__'
    }

    def visit_While(self, node):
        with enter_sub_region(self) as sr:
            liveins, insert_block = sr
            ip, last_loc = self._get_insertion_point_and_loc()

            # loop body (the after region)
            # loop_block = self.builder.create_block()
            dummy = self.builder.create_block()
            self.builder.set_insertion_point_to_start(dummy)
            self.scf_stack.append(node)
            self.visit_compound_statement(node.body)
            self.scf_stack.pop()
            loop_defs = self.local_defs
            dummy.erase()

            # collect loop-carried values
            names = []
            ret_types = []
            init_args = []
            for name in loop_defs:
                if name in liveins:
                    # We should not def new constexpr
                    assert _is_triton_tensor(loop_defs[name]), f'cannoe reassign constxpr {name} in the loop'
                    assert _is_triton_tensor(liveins[name]), f'cannot reasign constexpr {name} in the loop'
                    assert loop_defs[name].type == liveins[name].type, \
                        f'Loop-carried variable {name} has initial type {liveins[name].type} '\
                        f'but is re-assigned to {loop_defs[name].type} in loop! '\
                        f'Please make sure that the type stays consistent.'

                    # these are loop-carried values
                    names.append(name)
                    ret_types.append(loop_defs[name].type)
                    init_args.append(liveins[name])

            self._set_insertion_point_and_loc(ip, last_loc)
            while_op = self.builder.create_while_op([ty.to_ir(self.builder) for ty in ret_types],
                                                    [arg.handle for arg in init_args])
            # merge the condition region
            before_block = self.builder.create_block_with_parent(while_op.get_before(),
                                                                 [ty.to_ir(self.builder) for ty in ret_types])
            self.builder.set_insertion_point_to_start(before_block)
            for i, name in enumerate(names):
                self.lscope[name] = language.core.tensor(before_block.arg(i), ret_types[i])
                self.local_defs[name] = self.lscope[name]
            cond = self.visit(node.test)
            self.builder.set_insertion_point_to_end(before_block)
            # create ConditionOp: e.g., scf.condition(%cond) %arg0, %arg1, ...
            self.builder.create_condition_op(cond.handle, [before_block.arg(i) for i in range(len(init_args))])
            # merge the loop body
            after_block = self.builder.create_block_with_parent(while_op.get_after(),
                                                                [ty.to_ir(self.builder) for ty in ret_types])

            # generate loop body
            self.builder.set_insertion_point_to_start(after_block)
            for i, name in enumerate(names):
                self.lscope[name] = language.core.tensor(after_block.arg(i), ret_types[i])
                self.local_defs[name] = self.lscope[name]
            self.scf_stack.append(node)
            self.visit_compound_statement(node.body)
            self.scf_stack.pop()
            loop_defs = self.local_defs
            yields = []
            for name in loop_defs:
                if name in liveins:
                    yields.append(loop_defs[name])
            self.builder.create_yield_op([y.handle for y in yields])

        # WhileOp defines new values, update the symbol table (lscope, local_defs)
        for i, name in enumerate(names):
            new_def = language.core.tensor(while_op.get_result(i), ret_types[i])
            self.lscope[name] = new_def
            self.local_defs[name] = new_def

        for stmt in node.orelse:
            assert False, "Not implemented"
            ast.NodeVisitor.generic_visit(self, stmt)

    def visit_Subscript(self, node):
        assert node.ctx.__class__.__name__ == "Load"
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        if _is_triton_tensor(lhs):
            return lhs.__getitem__(slices, _builder=self.builder)
        return lhs[slices]

    def visit_ExtSlice(self, node):
        return [self.visit(dim) for dim in node.dims]

    def visit_For(self, node):
        IteratorClass = self.visit(node.iter.func)
        iter_args = [self.visit(arg) for arg in node.iter.args]
        if IteratorClass == language.static_range:
            iterator = IteratorClass(*iter_args)
            static_range = range(iterator.start.value, iterator.end.value, iterator.step.value)
            for i in static_range:
                self.lscope[node.target.id] = constexpr(i)
                self.visit_compound_statement(node.body)
                for stmt in node.orelse:
                    ast.NodeVisitor.generic_visit(self, stmt)
            return

        if IteratorClass is not range:
            raise RuntimeError('Only `range` and `static_range` iterators are currently supported')

        # visit iterator arguments
        # note: only `range` iterator is supported now
        # collect lower bound (lb), upper bound (ub), and step
        lb = iter_args[0] if len(iter_args) > 1 else self.visit(ast.Num(0))
        ub = iter_args[1] if len(iter_args) > 1 else self.visit(node.iter.args[0])
        step = iter_args[2] if len(iter_args) > 2 else self.visit(ast.Num(1))
        # handle negative constant step (not supported by scf.for in MLIR)
        negative_step = False
        if _is_constexpr(step) and step.value < 0:
            step = constexpr(-step.value)
            negative_step = True
            lb, ub = ub, lb
        lb = language.core._to_tensor(lb, self.builder)
        ub = language.core._to_tensor(ub, self.builder)
        step = language.core._to_tensor(step, self.builder)
        # induction variable type
        if not lb.dtype.is_int() or not ub.dtype.is_int() or not step.dtype.is_int():
            raise TypeError(f"For loop bounds and step must all be ints, are ({lb.dtype}, {ub.dtype}, {step.dtype})")
        iv_type = language.semantic.integer_promote_impl(lb.dtype, ub.dtype)
        iv_type = language.semantic.integer_promote_impl(iv_type, step.dtype)
        iv_ir_type = iv_type.to_ir(self.builder)
        iv_is_signed = iv_type.int_signedness == language.core.dtype.SIGNEDNESS.SIGNED
        # lb/ub/step might be constexpr, we need to cast them to tensor
        lb = lb.handle
        ub = ub.handle
        step = step.handle
        # ForOp can only accept IndexType as lb/ub/step. Cast integer to Index
        lb = self.builder.create_int_cast(lb, iv_ir_type, iv_is_signed)
        ub = self.builder.create_int_cast(ub, iv_ir_type, iv_is_signed)
        step = self.builder.create_int_cast(step, iv_ir_type, iv_is_signed)
        # Create placeholder for the loop induction variable
        iv = self.builder.create_undef(iv_ir_type)
        self.set_value(node.target.id, language.core.tensor(iv, iv_type))

        with enter_sub_region(self) as sr:
            liveins, insert_block = sr
            ip, last_loc = self._get_insertion_point_and_loc()

            # create loop body block
            block = self.builder.create_block()
            self.builder.set_insertion_point_to_start(block)
            # dry visit loop body
            self.scf_stack.append(node)
            self.visit_compound_statement(node.body)
            self.scf_stack.pop()
            block.erase()

            # If a variable (name) is defined in both its parent & itself, then it's
            # a loop-carried variable. (They must be of the same type)
            init_args = []
            yields = []
            names = []
            for name in self.local_defs:
                if name in liveins:
                    assert _is_triton_tensor(self.local_defs[name]), f'{name} is not tensor'
                    assert _is_triton_tensor(liveins[name])
                    assert self.local_defs[name].type == liveins[name].type, \
                        f'Loop-carried variable {name} has initial type {liveins[name].type} '\
                        f'but is re-assigned to {self.local_defs[name].type} in loop! '\
                        f'Please make sure that the type stays consistent.'

                    names.append(name)
                    init_args.append(language.core._to_tensor(liveins[name], self.builder))
                    yields.append(language.core._to_tensor(self.local_defs[name], self.builder))

            # create ForOp
            self._set_insertion_point_and_loc(ip, last_loc)
            for_op = self.builder.create_for_op(lb, ub, step, [arg.handle for arg in init_args])

            self.scf_stack.append(node)
            self.builder.set_insertion_point_to_start(for_op.get_body(0))
            for i, name in enumerate(names):
                self.set_value(name, language.core.tensor(for_op.get_body(0).arg(i + 1), yields[i].type))
            self.visit_compound_statement(node.body)
            self.scf_stack.pop()
            yields = []
            for name in self.local_defs:
                if name in liveins:
                    yields.append(language.core._to_tensor(self.local_defs[name], self.builder))

            # create YieldOp
            if len(yields) > 0:
                self.builder.create_yield_op([y.handle for y in yields])
            for_op_region = for_op.get_body(0).get_parent()
            assert for_op_region.size() == 1, "We use SCF, so the loop body should only have one block"

            # update induction variable with actual value, and replace all uses
            self.builder.set_insertion_point_to_start(for_op.get_body(0))
            iv = for_op.get_induction_var()
            if negative_step:
                iv = self.builder.create_sub(ub, iv)
                iv = self.builder.create_add(iv, lb)
            self.lscope[node.target.id].handle.replace_all_uses_with(iv)
            self.set_value(node.target.id, language.core.tensor(iv, iv_type))

        # update lscope & local_defs (ForOp defines new values)
        for i, name in enumerate(names):
            self.set_value(name, language.core.tensor(for_op.get_result(i), yields[i].type))

        for stmt in node.orelse:
            assert False, "Don't know what to do with else after for"
            ast.NodeVisitor.generic_visit(self, stmt)

    def visit_Slice(self, node):
        lower = self.visit(node.lower)
        upper = self.visit(node.upper)
        step = self.visit(node.step)
        return slice(lower, upper, step)

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_keyword(self, node) -> Tuple[str, Any]:
        return node.arg, self.visit(node.value)

    def visit_Assert(self, node) -> Any:
        if not self.debug:
            return
        test = self.visit(node.test)
        msg = self.visit(node.msg)
        # Convert assert to triton's device_assert which happens on the device
        return language.core.device_assert(test, msg, _builder=self.builder)

    def call_JitFunction(self, fn: JITFunction, args, kwargs):
        args = inspect.getcallargs(fn.fn, *args, **kwargs)
        args = [args[name] for name in fn.arg_names]
        args = [arg if _is_triton_tensor(arg) else constexpr(arg) for arg in args]
        # generate function def
        attributes = dict()
        constexprs = [i for i, arg in enumerate(args) if _is_constexpr(arg)]
        constants = {i: args[i] for i in constexprs}
        # generate call
        args = [None if i in constexprs else arg for i, arg in enumerate(args)]
        arg_vals = [arg.handle for arg in args if arg is not None]
        arg_types = [arg.type for arg in args if arg is not None]
        fn_name = mangle_fn(fn.__name__, arg_types, constants)
        # generate function def if necessary
        if not self.module.has_function(fn_name):
            prototype = language.function_type([], arg_types)
            gscope = sys.modules[fn.fn.__module__].__dict__
            # If the callee is not set, we use the same debug setting as the caller
            debug = self.debug if fn.debug is None else fn.debug
            file_name, begin_line = _get_fn_file_line(fn)
            generator = CodeGenerator(self.context, prototype, gscope, attributes, constants, module=self.module,
                                      function_name=fn_name, function_types=self.function_ret_types, debug=debug,
                                      noinline=fn.noinline, file_name=file_name, begin_line=begin_line,
                                      target=self.builder.target)
            generator.visit(fn.parse())
            callee_ret_type = generator.last_ret_type
            self.function_ret_types[fn_name] = callee_ret_type
        else:
            callee_ret_type = self.function_ret_types[fn_name]
        symbol = self.module.get_function(fn_name)
        call_op = self.builder.call(symbol, arg_vals)
        if call_op.get_num_results() == 0 or callee_ret_type is None:
            return None
        elif call_op.get_num_results() == 1:
            return tensor(call_op.get_result(0), callee_ret_type)
        else:
            # should return a tuple of tl.tensor
            results = []
            for i in range(call_op.get_num_results()):
                results.append(tensor(call_op.get_result(i), callee_ret_type[i]))
            return tuple(results)

    def visit_Call(self, node):
        fn = _unwrap_if_constexpr(self.visit(node.func))

        static_implementation = self.statically_implemented_functions.get(fn)
        if static_implementation is not None:
            return static_implementation(self, node)

        kws = dict(self.visit(keyword) for keyword in node.keywords)
        args = [self.visit(arg) for arg in node.args]
        if fn is language.core.device_assert:  # TODO: this should not be so hardcoded
            if not self.debug:
                return
        if isinstance(fn, JITFunction):
            _check_fn_args(node, fn, args)
            return self.call_JitFunction(fn, args, kws)
        if (hasattr(fn, '__self__') and _is_triton_tensor(fn.__self__)) or language.core.is_builtin(fn):
            extra_kwargs = dict(_builder=self.builder)
            sig = inspect.signature(fn)
            if '_generator' in sig.parameters:
                extra_kwargs['_generator'] = self
            return fn(*args, **extra_kwargs, **kws)
        if fn in self.builtin_namespace.values():
            args = map(_unwrap_if_constexpr, args)
        return fn(*args, **kws)

    def visit_Constant(self, node):
        return constexpr(node.value)

    def visit_BoolOp(self, node: ast.BoolOp):
        if len(node.values) != 2:
            raise UnsupportedLanguageConstruct(
                None, node,
                "chained boolean operators (A or B or C) are not supported; use parentheses to split the chain.")
        lhs = self.visit(node.values[0])
        rhs = self.visit(node.values[1])
        method_name = self._method_name_for_bool_op.get(type(node.op))
        if method_name is None:
            raise UnsupportedLanguageConstruct(
                None, node, "AST boolean operator '{}' is not (currently) implemented.".format(node.op.__name__))
        return self._apply_binary_method(method_name, lhs, rhs)

    _method_name_for_bool_op: Dict[Type[ast.boolop], str] = {ast.And: 'logical_and', ast.Or: 'logical_or'}

    if sys.version_info < (3, 8):

        def visit_NameConstant(self, node):
            return constexpr(node.value)

        def visit_Num(self, node):
            return constexpr(node.n)

        def visit_Str(self, node):
            return constexpr(ast.literal_eval(node))

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        if _is_triton_tensor(lhs):
            if node.attr == "T":
                return language.semantic.trans(lhs, builder=self.builder)
        return getattr(lhs, node.attr)

    def visit_Expr(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_NoneType(self, node):
        return None

    def visit_JoinedStr(self, node):
        values = list(node.values)
        for i, value in enumerate(values):
            if isinstance(value, ast.Constant):
                values[i] = str(value.value)
            elif isinstance(value, ast.FormattedValue):
                conversion_code = value.conversion
                evaluated = self.visit(value.value)
                if not _is_constexpr(evaluated):
                    raise UnsupportedLanguageConstruct(
                        None, node,
                        "Cannot evaluate f-string containing non-constexpr conversion values, found conversion of type "
                        + str(type(evaluated)))
                values[i] = ("{}" if conversion_code < 0 else "{!" + chr(conversion_code) + "}").format(evaluated.value)
            else:
                raise AssertionError("encountered unexpected node of type {} in a JoinedStr node".format(type(value)))
        return ''.join(values)

    def visit(self, node):
        if node is None:
            return
        with warnings.catch_warnings():
            # The ast library added visit_Constant and deprecated some other
            # methods but we can't move to that without breaking Python 3.6 and 3.7.
            warnings.simplefilter("ignore", DeprecationWarning)  # python 3.9
            warnings.simplefilter("ignore", PendingDeprecationWarning)  # python 3.8
            self.last_node = node
            last_loc = self.builder.get_loc()
            if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                self.builder.set_loc(self.file_name, self.begin_line + node.lineno, node.col_offset)
                last_loc = self.builder.get_loc()
            ret = super().visit(node)
            # Reset the location to the last one before the visit
            if last_loc:
                self.builder.set_loc(last_loc)
            return ret

    def generic_visit(self, node):
        raise UnsupportedLanguageConstruct(None, node, "unsupported AST node type: {}".format(type(node).__name__))

    def execute_static_print(self, node: ast.Call) -> None:
        # TODO: too simplistic? Perhaps do something else with non-constexpr

        kws = {name: _unwrap_if_constexpr(value) for name, value in (self.visit(keyword) for keyword in node.keywords)}
        args = [_unwrap_if_constexpr(self.visit(arg)) for arg in node.args]
        print(*args, **kws)

    def execute_static_assert(self, node: ast.Call) -> None:
        arg_count = len(node.args)
        if not (0 < arg_count <= 2) or len(node.keywords):
            raise TypeError("`static_assert` requires one or two positional arguments only")

        passed = _unwrap_if_constexpr(self.visit(node.args[0]))
        if not isinstance(passed, bool):
            raise NotImplementedError(
                "Assertion condition could not be determined at compile-time. Make sure that it depends only on `constexpr` values"
            )
        if not passed:
            if arg_count == 1:
                message = ""
            else:
                try:
                    message = self.visit(node.args[1])
                except Exception as e:
                    message = "<failed to evaluate assertion message: " + repr(e) + ">"

            raise CompileTimeAssertionFailure(None, node, _unwrap_if_constexpr(message))
        return None

    statically_implemented_functions: Dict[object, Callable[[ast.Call], Any]] = {
        language.core.static_assert: execute_static_assert,
        language.core.static_print: execute_static_print,
    }


def str_to_ty(name):
    if name[0] == "*":
        ty = str_to_ty(name[1:])
        return language.pointer_type(ty)
    tys = {
        "fp8e4nv": language.float8e4nv,
        "fp8e5": language.float8e5,
        "fp8e4b15": language.float8e4b15,
        "fp8e4b15x4": language.float8e4b15x4,
        "fp16": language.float16,
        "bf16": language.bfloat16,
        "fp32": language.float32,
        "fp64": language.float64,
        "i1": language.int1,
        "i8": language.int8,
        "i16": language.int16,
        "i32": language.int32,
        "i64": language.int64,
        "u8": language.uint8,
        "u16": language.uint16,
        "u32": language.uint32,
        "u64": language.uint64,
        "B": language.int1,
    }
    return tys[name]


def kernel_suffix(signature, specialization):
    # suffix format:
    # <argid><'c' if equal to 1><'d' if divisible by 16><'e' if divisible by 8>
    suffix = ''
    for i, _ in enumerate(signature):
        suffix += str(i)
        if i in specialization.equal_to_1:
            suffix += 'c'
        if i in specialization.divisible_by_16:
            suffix += 'd'
        if i in specialization.divisible_by_8:
            suffix += 'e'
    return suffix


def ast_to_ttir(fn, signature, specialization, constants, debug, target):
    # canonicalize signature
    if isinstance(signature, str):
        signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
    context = ir.context()
    context.load_triton()
    # create kernel prototype
    cst_key = lambda i: fn.arg_names.index(i) if isinstance(i, str) else i
    constants = {cst_key(key): value for key, value in constants.items()}
    # visit kernel AST
    gscope = fn.__globals__.copy()
    function_name = '_'.join([fn.__name__, kernel_suffix(signature.values(), specialization)])
    tys = list(signature.values())
    new_constants = {k: True if k in tys and tys[k] == "i1" else 1 for k in specialization.equal_to_1}
    new_attrs = {k: [("tt.divisibility", 16)] for k in specialization.divisible_by_16}

    # Note: Here we defines 'max_divisibility' for later TMA usage.
    # fp16 requires 'max_divisibility >= 8' and fp8 requires 'max_divisibility >= 16'.
    # Since we only need to support TMA for fp16 and fp8 now, 'max_divisibility' is either 8 or 16.
    for k in specialization.divisible_by_8:
        attr = new_attrs[k] if k in new_attrs else []
        if k in specialization.divisible_by_16:
            attr.append(("tt.max_divisibility", 16))
        else:
            attr.append(("tt.max_divisibility", 8))
        new_attrs[k] = attr

    all_constants = constants.copy()
    all_constants.update(new_constants)
    arg_types = [str_to_ty(v) for k, v in signature.items() if k not in constants]
    file_name, begin_line = _get_fn_file_line(fn)

    prototype = language.function_type([], arg_types)
    generator = CodeGenerator(context, prototype, gscope=gscope, constants=all_constants, function_name=function_name,
                              attributes=new_attrs, is_kernel=True, debug=debug, file_name=file_name,
                              begin_line=begin_line, target=target)
    try:
        generator.visit(fn.parse())
    except CompilationError as e:
        if e.src is None:
            e.set_source_code(fn.src)
        raise
    except Exception as e:
        node = generator.last_node
        if node is None:
            raise
        raise CompilationError(fn.src, node, repr(e)) from e
    ret = generator.module
    # module takes ownership of the context
    ret.context = context
    return ret
