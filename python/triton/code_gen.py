from __future__ import annotations

import ast
import builtins
import functools
import hashlib
import inspect
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import warnings
from typing import Dict, Set, Tuple, Union

import torch
from filelock import FileLock

import triton
import triton._C.libtriton.triton as _triton
from .tools.disasm import extract

try:
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
except ImportError:
    get_cuda_stream = lambda dev_idx: torch.cuda.current_stream(dev_idx).cuda_stream


def current_cuda_stream(device_idx=0):
    # Torch's torch.cuda.current_stream() is slow. We provide this
    # function to give the user an opportunity to monkey-patch their
    # own faster current stream lookup.
    return get_cuda_stream(device_idx)


def mangle_ty(ty):
    if ty.is_ptr():
        return 'P' + mangle_ty(ty.element_ty)
    if ty.is_int():
        return 'i' + str(ty.int_bitwidth)
    if ty.is_fp8():
        return 'fp8'
    if ty.is_fp16():
        return 'fp16'
    if ty.is_bf16():
        return 'bf16'
    if ty.is_fp32():
        return 'fp32'
    if ty.is_fp64():
        return 'fp64'
    if ty.is_void():
        return 'V'
    if ty.is_block():
        elt = mangle_ty(ty.scalar)
        shape = '_'.join(map(str, ty.shape))
        return f'{elt}S{shape}S'
    assert False, "Unsupport type"


def mangle_fn(name, arg_tys, constants):
    # doesn't mangle ret type, which must be a function of arg tys
    mangled_arg_names = '_'.join([mangle_ty(ty) for ty in arg_tys])
    key = lambda x: x.__name__ if isinstance(x, JITFunction) else repr(x)
    mangled_constants = '_'.join([f'{i}c{key(constants[i])}' for i in sorted(constants)])
    mangled_constants = mangled_constants.replace('.', '_d_')
    mangled_constants = mangled_constants.replace("'", '_sq_')
    mangled_constants = mangled_constants.replace("e-", '_em_')
    ret = f'{name}__{mangled_arg_names}__{mangled_constants}'
    return ret


def is_triton_tensor(value):
    return isinstance(value, triton.language.tensor)


class ValueConstructor:
    def __init__(self, module, builder, gscope) -> None:
        self.gscope = gscope
        self.lscope = dict()
        self.builder = builder
        self.module = module
        # [name, bb] => triton.language.tensor
        self.lvalues: Dict[Tuple[str, _triton.ir.basic_block], triton.language.tensor] = {}
        # bb => {name => phi}
        self.incomplete_phis = {}
        self.sealed_blocks: Set[_triton.ir.basic_block] = set()
        #
        self.builtins = {
            'range': range,
            'min': triton.language.minimum,
            'float': float,
            'int': int,
            'print': print,
            'isinstance': isinstance,
            'getattr': getattr,
        }

    def get_value(self, name):
        ''' This function:
        1. make sure `name` is defined
        2. if `name` is triton.language.tensor, get stored tensor by calling
           `self._get_tensor()`
        '''
        # search node.id in local scope
        ret = None
        if name in self.lscope:
            ret = self.lscope[name]
        # search node.id in global scope
        elif name in self.gscope:
            ret = self.gscope[name]
        # search node.id in builtins
        elif name in self.builtins:
            ret = self.builtins[name]
        else:
            raise ValueError(f'{name} is not defined')
        if is_triton_tensor(ret):
            return self._get_tensor(name, self.builder.get_insert_block())
        return ret

    def set_value(self, name: str,
                  value: Union[triton.language.tensor, triton.language.constexpr]) -> None:
        ''' This function:
          called by visit_Assign() & visit_FuncDef() to store left value (lvalue)
        1. record local defined name (FIXME: should consider control flow)
        2. store tensor in self.lvalue
        '''
        self.lscope[name] = value
        if isinstance(value, triton.language.tensor):
            self._set_value(name, self.builder.get_insert_block(), value)

    #
    # SSA-construction
    #
    def _get_tensor(self, name: str, bb: _triton.ir.basic_block) -> triton.language.tensor:
        # local value numbering
        if (name, bb) in self.lvalues:
            return self.lvalues[(name, bb)]
        # global value numbering
        saved_insert_point = self.builder.get_insert_point()
        result = self._get_tensor_recursive(name, bb)
        self.builder.set_insert_point(saved_insert_point)
        return result

    def _get_tensor_recursive(self, name: str, bb: _triton.ir.basic_block) -> triton.language.tensor:
        preds = bb.get_predecessors()
        type = self.lscope[name].type
        # some preds haven't been filled, create a phi as a proxy of the value
        if bb not in self.sealed_blocks:
            result = self._make_phi(type, len(preds), bb)
            if bb in self.incomplete_phis:
                self.incomplete_phis[bb][name] = result
            else:
                self.incomplete_phis[bb] = {name: result}
        elif len(preds) == 1:
            # one predecessor: no phi needed, try get value from pred
            result = self._get_tensor(name, preds[0])
        elif len(preds) == 0:
            result = self._get_tensor(name, None)
        else:  # multiple preds
            phi = self._make_phi(type, len(preds), bb)
            self._set_value(name, bb, phi)
            result = self._add_phi_operands(name, phi)
        self._set_value(name, bb, result)
        return result

    # returns a new phi tensor, which encausulate an ir.phi_node
    def _make_phi(self,
                  type: triton.language.dtype,
                  num_values: int,
                  bb: _triton.ir.basic_block) -> triton.language.tensor:
        instr = bb.get_first_non_phi()
        self.builder.set_insert_point((bb, instr))
        ir_phi = self.builder.create_phi(type.to_ir(self.builder), num_values)
        if instr:
            self.builder.set_insert_block(bb)
        return triton.language.tensor(ir_phi, type)

    # complete a phi node. (TODO: rename this as _complete_phis?)
    # Note: since we try to remove tryival phi, the return tensor might not be a phi
    def _add_phi_operands(self, name: str,
                          phi: triton.language.tensor) -> triton.language.tensor:
        bb = phi.handle.get_parent()
        for pred in bb.get_predecessors():
            v = self._get_tensor(name, pred)
            phi.handle.add_incoming(v.handle, pred)
        phi = self._try_remove_trivial_phi(phi)
        return phi

    def _set_value(self, name: str, bb: _triton.ir.basic_block, value: triton.language.tensor) -> None:
        self.lvalues[(name, bb)] = value
        # TODO: why we need this?
        self.module.set_instr_metadata(name, value.handle)

    def _seal_block(self, bb: _triton.ir.basic_block):
        # complete all incomplete phis
        if bb in self.incomplete_phis:
            for name, phi in self.incomplete_phis[bb].items():
                result = self._add_phi_operands(name, phi)
                # it's possible that this phi is trivial
                if self._get_tensor(name, bb).handle == phi.handle:
                    self._set_value(name, bb, result)
            del self.incomplete_phis[bb]
        self.sealed_blocks.add(bb)

    def _try_remove_trivial_phi(self, phi: triton.language.tensor) -> triton.language.tensor:
        unique_handles = {op for op in phi.handle.ops() if op != phi.handle}
        if len(unique_handles) != 1:  # non-trivial phi
            return phi
        v = unique_handles.pop()
        phi.handle.replace_all_uses_with(v)
        # phi.handle.erase_from_parent()
        # TODO: remove trivial phis recursively
        return triton.language.tensor(v, phi.type)


class CodeGenerator(ast.NodeVisitor):

    def __init__(self, context, prototype, gscope, attributes, constants, prototypes=None, module=None, is_kernel=False):
        self.prototypes = dict() if prototypes is None else prototypes
        self.builder = _triton.ir.builder(context)
        self.module = _triton.ir.module('', self.builder) if module is None else module
        self.prototype = prototype
        self.attributes = attributes
        self.constants = constants
        self.last_node = None
        self.is_kernel = is_kernel

        self.value_constructor = ValueConstructor(self.module, self.builder, gscope)

    #
    # AST visitor
    #

    def visit_compound_statement(self, stmts):
        for stmt in stmts:
            self.last_ret = self.visit(stmt)
            if isinstance(stmt, ast.Return):
                break
        return stmts and isinstance(stmt, ast.Return)

    def visit_Module(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_List(self, node):
        ctx = self.visit(node.ctx)
        assert ctx is None
        elts = [self.visit(elt) for elt in node.elts]
        return elts

    # By design, only non-kernel functions can return
    def visit_Return(self, node):
        ret = self.visit(node.value)
        if ret is None:
            return triton.language.tensor(self.builder.ret_void(), triton.language.void)
        ret = triton.language.core._to_tensor(ret, self.builder)
        ret = triton.language.tensor(self.builder.ret(ret.handle), ret.type)
        return ret

    def visit_FunctionDef(self, node):
        arg_names, kwarg_names = self.visit(node.args)
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
        fn_name = mangle_fn(node.name, self.prototype.param_types, self.constants)
        self.prototypes[fn_name] = self.prototype
        fn = self.module.get_or_insert_function(fn_name, self.prototype.to_ir(self.builder))
        fn.set_is_kernel(self.is_kernel)
        arg_values = []
        idx = 0
        for i, arg_name in enumerate(arg_names):
            if i in self.constants:
                cst = self.constants[i]
                if not isinstance(cst, triton.language.constexpr):
                    cst = triton.language.constexpr(self.constants[i])
                arg_values.append(cst)
            else:
                if i in self.attributes:
                    is_ptr = fn.args[idx].type.is_ptr()
                    attr = 'aligned' if is_ptr else 'multiple_of'
                    attr = getattr(_triton.ir.attribute_kind, attr)
                    attr = _triton.ir.attribute(attr, self.attributes[i])
                    fn.add_attr(idx + 1, attr)
                fn.args[idx].name = arg_name
                arg_values.append(triton.language.tensor(fn.args[idx], self.prototype.param_types[idx]))
                idx += 1

        insert_pt = self.builder.get_insert_block()
        entry = _triton.ir.basic_block.create(self.builder.context, "entry", fn)
        self.builder.set_insert_block(entry)
        self.value_constructor._seal_block(entry)
        for arg_name, arg_value in zip(arg_names, arg_values):
            self.value_constructor.set_value(arg_name, arg_value)
        # visit function body
        has_ret = self.visit_compound_statement(node.body)
        # finalize
        if not has_ret:
            self.builder.ret_void()
        else:
            # a bit hacky: we only know the return type at the last moment so we update type info here
            self.module.reset_ret_ty(fn_name, self.last_ret.type.to_ir(self.builder))
            self.prototype.ret_type = self.last_ret.type
        self.builder.set_insert_block(insert_pt)

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
        if annotation == triton.language.constexpr:
            if target in self.value_constructor.lscope:
                raise ValueError(f'{target} is already defined.'
                                 f' constexpr cannot be reassigned.')
            if not isinstance(value, triton.language.constexpr):
                value = triton.language.constexpr(value)
            self.value_constructor.lscope[target] = value
            return self.value_constructor.lscope[target]
        # default: call visit_Assign
        return self.visit_Assign(node)

    def visit_Assign(self, node):
        _names = []
        for target in node.targets:
            _names += [self.visit(target)]
        assert len(_names) == 1
        names = _names[0]
        values = self.visit(node.value)
        if not isinstance(names, tuple):
            names = [names]
        if not isinstance(values, tuple):
            values = [values]
        if isinstance(values[0], triton.language.tensor) \
                and isinstance(values[0].type, triton.language.tuple_type):
            struct = values[0].handle
            tys = values[0].type.element_types
            values = [self.builder.extract_value(struct, i) for i in range(len(tys))]
            values = [triton.language.tensor(v, ty) for v, ty in zip(values, tys)]
        assert len(values) == len(names)
        for name, value in zip(names, values):
            # TODO: can we store constexpr here to support constant folding?
            # by default, constexpr are assigned into python variable
            if isinstance(value, triton.language.constexpr):
                value = value.value
            if value is None:
                raise ValueError(f'Cannot assign None to non-constexpr `{name}`. Please annotate as `: tl.constexpr`')
            if not isinstance(value, triton.language.tensor):
                value = triton.language.core._to_tensor(value, self.builder)
            self.value_constructor.set_value(name, value)

    def visit_AugAssign(self, node):
        name = node.target.id
        lhs = ast.Name(id=name, ctx=ast.Load())
        rhs = ast.BinOp(lhs, node.op, node.value)
        assign = ast.Assign(targets=[node.target], value=rhs)
        self.visit(assign)
        return self.value_constructor.get_value(name)

    def visit_Name(self, node):
        if type(node.ctx) == ast.Store:
            return node.id
        return self.value_constructor.get_value(node.id)

    def visit_Store(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Load(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Tuple(self, node):
        args = [self.visit(x) for x in node.elts]
        mode = type(args[0])
        # tuple of values -- create a struct
        if len(args) > 1 and mode == triton.language.tensor\
                and all([type(arg) == mode for arg in args]):
            tuple_ty = triton.language.tuple_type([arg.type for arg in args])
            ret = _triton.ir.undef.get(tuple_ty.to_ir(self.builder))
            for i, arg in enumerate(args):
                ret = self.builder.insert_value(ret, arg.handle, i)
            ret = triton.language.tensor(ret, tuple_ty)
            return ret
        return tuple(args)

    def visit_BinOp(self, node):
        # visit operand
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        is_lhs_constexpr = isinstance(lhs, triton.language.constexpr)
        is_rhs_constexpr = isinstance(rhs, triton.language.constexpr)
        lhs = lhs.value if is_lhs_constexpr else lhs
        rhs = rhs.value if is_rhs_constexpr else rhs
        # get function name
        fn = {
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
        }[type(node.op)]
        # return a new constexpr if both arg are constexprs
        if is_lhs_constexpr and is_rhs_constexpr:
            return triton.language.constexpr(getattr(lhs, fn)(rhs))
        # call operator
        if is_triton_tensor(lhs):
            return getattr(lhs, fn)(rhs, _builder=self.builder)
        elif is_triton_tensor(rhs):
            fn = fn[:2] + 'r' + fn[2:]
            return getattr(rhs, fn)(lhs, _builder=self.builder)
        else:
            return getattr(lhs, fn)(rhs)

    def visit_If(self, node):
        cond = self.visit(node.test)
        if isinstance(cond, triton.language.tensor):
            cond = cond.to(triton.language.int1, _builder=self.builder)
            current_bb = self.builder.get_insert_block()
            then_bb = _triton.ir.basic_block.create(self.builder.context, "then", current_bb.parent)
            else_bb = _triton.ir.basic_block.create(self.builder.context, "else", current_bb.parent) if node.orelse else None
            endif_bb = _triton.ir.basic_block.create(self.builder.context, "endif", current_bb.parent)
            self.value_constructor._seal_block(then_bb)
            if else_bb:
                self.value_constructor._seal_block(else_bb)
                self.builder.cond_br(cond.handle, then_bb, else_bb)
            else:
                self.builder.cond_br(cond.handle, then_bb, endif_bb)
            self.builder.set_insert_block(then_bb)
            is_terminator = self.visit_compound_statement(node.body)
            # TODO: last statement is a terminator?
            if not is_terminator:
                self.builder.br(endif_bb)
            if else_bb:
                self.builder.set_insert_block(else_bb)
                is_terminator = self.visit_compound_statement(node.orelse)
                # TODO: last statement is a terminator?
                if not is_terminator:
                    self.builder.br(endif_bb)
            self.value_constructor._seal_block(endif_bb)
            self.builder.set_insert_block(endif_bb)
        else:
            if isinstance(cond, triton.language.constexpr):
                cond = cond.value
            if cond:
                self.visit_compound_statement(node.body)
            else:
                self.visit_compound_statement(node.orelse)

    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if cond.value:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_Pass(self, node):
        pass

    def visit_Compare(self, node):
        assert len(node.comparators) == 1
        assert len(node.ops) == 1
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        is_lhs_constexpr = isinstance(lhs, triton.language.constexpr)
        is_rhs_constexpr = isinstance(rhs, triton.language.constexpr)
        lhs = lhs.value if is_lhs_constexpr else lhs
        rhs = rhs.value if is_rhs_constexpr else rhs
        # handle `is`` and `is not``
        if type(node.ops[0]) == ast.Is:
            return triton.language.constexpr(lhs is rhs)
        if type(node.ops[0]) == ast.IsNot:
            return triton.language.constexpr(lhs is not rhs)
        # function name
        fn = {
            ast.Eq: '__eq__',
            ast.NotEq: '__ne__',
            ast.Lt: '__lt__',
            ast.LtE: '__le__',
            ast.Gt: '__gt__',
            ast.GtE: '__ge__',
        }[type(node.ops[0])]
        # return a new constexpr if both arg are constexprs
        if is_lhs_constexpr and is_rhs_constexpr:
            return triton.language.constexpr(getattr(lhs, fn)(rhs))
        # call operator
        if is_triton_tensor(lhs):
            return getattr(lhs, fn)(rhs, _builder=self.builder)
        elif is_triton_tensor(rhs):
            fn = fn[:2] + 'r' + fn[2:]
            return getattr(rhs, fn)(lhs, _builder=self.builder)
        else:
            assert False

    def visit_UnaryOp(self, node):
        op = self.visit(node.operand)
        if type(node.op) == ast.Not:
            assert isinstance(op, triton.language.constexpr), "`not` only supported for constexpr at the moment"
            return triton.language.constexpr(not op)
        fn = {
            ast.USub: '__neg__',
            ast.UAdd: '__pos__',
            ast.Invert: '__invert__',
        }[type(node.op)]
        if isinstance(op, triton.language.constexpr):
            return triton.language.constexpr(getattr(op.value, fn)())
        assert is_triton_tensor(op)
        return getattr(op, fn)(_builder=self.builder)

    def visit_While(self, node):
        current_bb = self.builder.get_insert_block()
        loop_bb = _triton.ir.basic_block.create(self.builder.context, "loop", current_bb.parent)
        next_bb = _triton.ir.basic_block.create(self.builder.context, "postloop", current_bb.parent)

        def continue_fn():
            cond = self.visit(node.test)
            return self.builder.cond_br(cond.handle, loop_bb, next_bb)

        continue_fn()
        self.builder.set_insert_block(loop_bb)
        self.visit_compound_statement(node.body)
        continue_fn()
        stop_bb = self.builder.get_insert_block()
        self.value_constructor._seal_block(stop_bb)
        self.value_constructor._seal_block(loop_bb)
        self.value_constructor._seal_block(next_bb)
        self.builder.set_insert_block(next_bb)

        for stmt in node.orelse:
            ast.NodeVisitor.generic_visit(self, stmt)

    def visit_Subscript(self, node):
        assert node.ctx.__class__.__name__ == "Load"
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        if is_triton_tensor(lhs):
            return lhs.__getitem__(slices, _builder=self.builder)
        return lhs[slices]

    def visit_ExtSlice(self, node):
        return [self.visit(dim) for dim in node.dims]

    def visit_For(self, node):
        iterator = self.visit(node.iter.func)
        if iterator != self.value_constructor.builtins['range']:
            raise RuntimeError('Only `range` iterator currently supported')
        # static for loops: all iterator arguments are constexpr
        iter_args = [self.visit(arg) for arg in node.iter.args]
        is_static = all([isinstance(x, triton.language.constexpr) for x in iter_args])
        if is_static:
            st_target = ast.Name(id=node.target.id, ctx=ast.Store())
            iter_args = [arg.value for arg in iter_args]
            range = iterator(*iter_args)
            if len(range) <= 10:
                for i in iterator(*iter_args):
                    self.value_constructor.lscope[node.target.id] = triton.language.constexpr(i)
                    self.visit_compound_statement(node.body)
                    for stmt in node.orelse:
                        ast.NodeVisitor.generic_visit(self, stmt)
                return

        # create nodes
        st_target = ast.Name(id=node.target.id, ctx=ast.Store())
        ld_target = ast.Name(id=node.target.id, ctx=ast.Load())
        arg_0 = node.iter.args[0] if len(node.iter.args) > 1 else ast.Num(0)
        arg_1 = node.iter.args[1] if len(node.iter.args) > 1 else node.iter.args[0]
        arg_2 = node.iter.args[2] if len(node.iter.args) > 2 else ast.Num(1)
        # init node
        init_node = ast.Assign(targets=[st_target], value=arg_0)

        # step node
        pos_cond_node = ast.Compare(ld_target, [ast.Lt()], [arg_1])
        neg_cond_node = ast.Compare(ld_target, [ast.Gt()], [arg_1])
        pos_step_node = ast.Compare(arg_2, [ast.Gt()], [ast.Num(0)])
        build_cond = lambda: triton.language.where(self.visit(pos_step_node),
                                                   self.visit(pos_cond_node),
                                                   self.visit(neg_cond_node),
                                                   _builder=self.builder)
        # cond_node = neg_cond_node
        step_node = ast.AugAssign(target=st_target, op=ast.Add(), value=arg_2)
        # code generation
        current_bb = self.builder.get_insert_block()
        loop_bb = _triton.ir.basic_block.create(self.builder.context, "loop", current_bb.parent)
        next_bb = _triton.ir.basic_block.create(self.builder.context, "postloop", current_bb.parent)

        def continue_fn():
            self.visit(step_node)
            cond = build_cond()
            return self.builder.cond_br(cond.handle, loop_bb, next_bb)

        # init loop induction variable
        self.visit(init_node)
        # promote it to right type
        init_val = self.value_constructor.get_value(node.target.id)
        promote = lambda a, b: triton.language.semantic.computation_type_impl(a, b, False)
        start_ty = triton.language.core._to_tensor(iter_args[0], self.builder).type
        stop_ty = triton.language.core._to_tensor(iter_args[1], self.builder).type if len(iter_args) > 1 else None
        ty = promote(start_ty, stop_ty) if len(iter_args) > 1 else start_ty
        casted = triton.language.semantic.cast(init_val, ty, self.builder)
        self.value_constructor.set_value(node.target.id, casted)
        # create cond
        cond = build_cond()
        self.builder.cond_br(cond.handle, loop_bb, next_bb)
        self.builder.set_insert_block(loop_bb)
        self.visit_compound_statement(node.body)
        # TODO: handle case where body breaks control flow
        continue_fn()
        stop_bb = self.builder.get_insert_block()
        self.value_constructor._seal_block(stop_bb)
        self.value_constructor._seal_block(loop_bb)
        self.value_constructor._seal_block(next_bb)
        self.builder.set_insert_block(next_bb)

        for stmt in node.orelse:
            ast.NodeVisitor.generic_visit(self, stmt)

    def visit_Slice(self, node):
        lower = self.visit(node.lower)
        upper = self.visit(node.upper)
        step = self.visit(node.step)
        return slice(lower, upper, step)

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_keyword(self, node):
        return {node.arg: self.visit(node.value)}

    def visit_Call(self, node):
        fn = self.visit(node.func)
        if isinstance(fn, triton.language.constexpr):
            fn = fn.value
        kws = dict()
        for keyword in node.keywords:
            kws.update(self.visit(keyword))
        args = [self.visit(arg) for arg in node.args]

        if isinstance(fn, JITFunction):
            from inspect import getcallargs
            args = getcallargs(fn.fn, *args, **kws)
            args = [args[name] for name in fn.arg_names]
            args = [arg if isinstance(arg, triton.language.tensor)
                    else triton.language.constexpr(arg) for arg in args]
            # generate function def
            attributes = dict()
            constexprs = [i for i, arg in enumerate(args) if isinstance(arg, triton.language.constexpr)]
            constants = {i: args[i] for i in constexprs}
            # generate call
            args = [None if i in constexprs else arg for i, arg in enumerate(args)]
            arg_vals = [arg.handle for arg in args if arg is not None]
            arg_types = [arg.type for arg in args if arg is not None]
            fn_name = mangle_fn(fn.__name__, arg_types, constants)
            # generate function def if necessary
            if not self.module.has_function(fn_name):
                ret_type = triton.language.void
                prototype = triton.language.function_type(ret_type, arg_types)
                gscope = sys.modules[fn.fn.__module__].__dict__
                generator = CodeGenerator(self.builder.context, prototype, gscope, attributes, constants, prototypes=self.prototypes, module=self.module)
                generator.visit(fn.parse())
            symbol = self.module.get_function(fn_name)
            ret = self.builder.call(symbol, arg_vals)
            if not ret.type.is_void():
                ret = triton.language.tensor(ret, self.prototypes[fn_name].ret_type)
            return ret
        # built-in function
        if sys.modules[fn.__module__] is triton.language.core or isinstance(fn, triton.language.extern.ExternalFunction):
            ret = fn(*args, _builder=self.builder, **kws)
        if fn in self.value_constructor.builtins.values():
            args = [arg.value if isinstance(arg, triton.language.constexpr) else arg
                    for arg in args]
            ret = fn(*args, **kws)
            if isinstance(ret, (bool, int, float)):
                ret = triton.language.core.constexpr(ret)
            else:
                ret = triton.language.core._to_tensor(ret, self.builder)
        # special case: dynamic parallelism
        # in this case the core primitive returns a proxy
        # if isinstance(ret, triton.language.core.LaunchProxy):
        #     ret_type  = _triton.ir.type.get_void(self.builder.context)
        #     arg_tys = [x.type for x in ret.args]
        #     prototype = _triton.ir.type.make_function(ret_type, arg_tys)
        #     gscope = sys.modules[ret.fn.fn.__module__].__dict__
        #     constants = ret.constants
        #     fn_name = mangle_fn(ret.fn.__name__, arg_tys, ret.constants)
        #     # TODO: clean-up attributes handling in function
        #     if not self.module.has_function(fn_name):
        #         attributes = {i: list(arg.parent.get_attrs(arg))[0].value for i, arg in enumerate(ret.args) \
        #                 if isinstance(arg, _triton.ir.argument) and arg.parent.has_attr(i + 1) }
        #         generator = CodeGenerator(self.builder.context, prototype, gscope, attributes, constants, module=self.module, is_kernel=True)
        #         generator.visit(ret.fn.parse())
        #     symbol = self.module.get_function(fn_name)
        #     # TODO: should ret.args not include any constants ?
        #     ret = self.builder.launch(symbol, ret.args, ret.grid, ret.num_warps)
        return ret
        # return fn(*args, **kws)

    def visit_Constant(self, node):
        return triton.language.constexpr(node.value)

    if sys.version_info < (3, 8):
        def visit_NameConstant(self, node):
            return triton.language.constexpr(node.value)

        def visit_Num(self, node):
            return triton.language.constexpr(node.n)

        def visit_Str(self, node):
            return triton.language.constexpr(ast.literal_eval(node))

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        return getattr(lhs, node.attr)

    def visit_Expr(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_NoneType(self, node):
        return None

    def visit(self, node):
        if node is not None:
            self.last_node = node
        with warnings.catch_warnings():
            # The ast library added visit_Constant and deprecated some other
            # methods but we can't move to that without breaking Python 3.6 and 3.7.
            warnings.simplefilter("ignore", DeprecationWarning)  # python 3.9
            warnings.simplefilter("ignore", PendingDeprecationWarning)  # python 3.8
            return super().visit(node)

    def generic_visit(self, node):
        typename = type(node).__name__
        raise NotImplementedError("Unsupported node: {}".format(typename))


class Binary:
    def __init__(self, backend, name, asm, shared_mem, num_warps):
        self.backend = backend
        self.name = name
        self.asm = asm
        self.shared_mem = shared_mem
        self.num_warps = num_warps


class LoadedBinary:
    def __init__(self, device: int, bin: Binary):
        module, kernel, n_regs, n_spills = _triton.code_gen.load_binary(bin.backend,
                                                                        bin.name,
                                                                        bin.asm,
                                                                        bin.shared_mem,
                                                                        device)
        self.bin = bin
        self.asm = bin.asm
        self.sass = ''
        self.module = module
        self.kernel = kernel
        self.n_regs = n_regs
        self.n_spills = n_spills
        self.device = device
        self.shared_mem = bin.shared_mem

    def __call__(self, stream, args, grid_0, grid_1=1, grid_2=1):
        _triton.runtime.enqueue(self.bin.backend, stream, self.kernel,
                                grid_0, grid_1, grid_2,
                                self.bin.num_warps * 32, 1, 1,
                                args, self.bin.shared_mem)

    def get_sass(self, fun=None):
        if self.sass:
            return self.sass
        fd, path = tempfile.mkstemp()
        try:
            with open(fd, 'wb') as cubin:
                cubin.write(self.asm['cubin'])
            self.sass = extract(path, fun)
        finally:
            os.remove(path)
        self.asm['sass'] = self.sass
        return self.sass


class CompilationError(Exception):
    def __init__(self, src, node):
        self.message = f'at {node.lineno}:{node.col_offset}:\n'
        self.message += '\n'.join(src.split('\n')[:node.lineno])
        self.message += '\n' + ' ' * node.col_offset + '^'
        self.src = src
        self.node = node
        super().__init__(self.message)

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.src, self.node))


class OutOfResources(Exception):
    def __init__(self, required, limit, name):
        self.message = f'out of resource: {name}, '\
                       f'Required: {required}, '\
                       f'Hardware limit: {limit}'
        self.required = required
        self.limit = limit
        self.name = name
        super().__init__(self.message)

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.required, self.limit, self.name))


class Kernel:

    @staticmethod
    def _type_name(obj):
        type_names = {
            triton.language.float8: 'f8',
            torch.bfloat16: 'bf16',
            torch.float16: 'f16',
            torch.float32: 'f32',
            torch.float64: 'f64',
            torch.bool: 'i1',
            torch.uint8: 'u8',
            torch.int8: 'i8',
            torch.int16: 'i16',
            torch.int32: 'i32',
            torch.int64: 'i64',
            triton.language.uint8: 'u8',
            triton.language.uint16: 'u16',
            triton.language.uint32: 'u32',
            triton.language.uint64: 'u64',
        }
        if hasattr(obj, 'data_ptr'):
            return type_names[obj.dtype]
        if isinstance(obj, triton.language.constexpr):
            obj = obj.value
        if isinstance(obj, int):
            if -2**31 <= obj < 2**31:
                return 'i32'
            elif 2**31 <= obj < 2**32:
                return 'u32'
            elif -2**63 <= obj < 2**63:
                return 'i64'
            elif 2**63 <= obj < 2**64:
                return 'u64'
            else:
                raise ValueError(f'integer overflow representing {obj}')
        if isinstance(obj, float):
            return 'f'
        if isinstance(obj, bool):
            return 'B'
        if isinstance(obj, str):
            return 'str'
        raise NotImplementedError(f'could not compute type name for {obj}')

    @staticmethod
    def _to_python_ir(obj):
        # convert torch.Tensor to Triton IR pointers
        if hasattr(obj, 'data_ptr'):
            name = Kernel._type_name(obj)
            return 'ptr', name
        # default path returns triton.ir.type directly
        name = Kernel._type_name(obj)
        return 'scalar', name

    @staticmethod
    def _to_triton_ir(obj):
        which, name = obj
        type_map = {
            'I': triton.language.int32,
            'L': triton.language.int64,
            'f': triton.language.float32,
            'B': triton.language.int1,
            'f8': triton.language.float8,
            'f16': triton.language.float16,
            'bf16': triton.language.bfloat16,
            'f32': triton.language.float32,
            'f64': triton.language.float64,
            'i1': triton.language.int1,
            'i8': triton.language.int8,
            'i16': triton.language.int16,
            'i32': triton.language.int32,
            'i64': triton.language.int64,
            'u8': triton.language.uint8,
            'u16': triton.language.uint16,
            'u32': triton.language.uint32,
            'u64': triton.language.uint64,
        }
        # convert torch.Tensor to Triton IR pointers
        if which == 'ptr':
            elt_ty = type_map[name]
            return triton.language.pointer_type(elt_ty, 1)
        # default path returns triton.ir.type directly
        return type_map[name]

    @staticmethod
    def pow2_divisor(N):
        if N % 16 == 0:
            return 16
        if N % 8 == 0:
            return 8
        if N % 4 == 0:
            return 4
        if N % 2 == 0:
            return 2
        return 1

    def __init__(self, fn):
        self.fn = fn
        self.cache_key = {}

    def add_to_cache(self, key, wargs, device_idx, num_warps, num_stages, extern_libs):
        tensor_idxs = [i for i, arg in enumerate(wargs) if hasattr(arg, 'data_ptr')]

        # attributes
        attributes = dict()
        for i, arg in enumerate(wargs):
            if i in self.fn.do_not_specialize:
                continue
            if isinstance(arg, int):
                attributes[i] = Kernel.pow2_divisor(arg)
            elif i in tensor_idxs:
                addr = arg.data_ptr()
                range_size = _triton.runtime.get_pointer_range_size(addr)
                attributes[i] = min(Kernel.pow2_divisor(addr),
                                    Kernel.pow2_divisor(range_size))
        # transforms ints whose value is one into constants for just-in-time compilation
        constants = {i: arg for i, arg in enumerate(wargs) if isinstance(arg, int) and arg == 1 and i not in self.fn.do_not_specialize}
        constants.update({i: arg.value for i, arg in enumerate(wargs) if isinstance(arg, triton.language.constexpr)})
        constants.update({i: None for i, arg in enumerate(wargs) if arg is None})
        arg_types = [Kernel._to_python_ir(arg) for i, arg in enumerate(wargs) if i not in constants]
        return self.fn._warmup(key, arg_types=arg_types, device=device_idx, attributes=attributes, constants=constants, num_warps=num_warps, num_stages=num_stages,
                               extern_libs=extern_libs, is_manual_warmup=False)

    def __call__(self, *wargs, grid, num_warps=4, num_stages=2, extern_libs={}, **kwargs):
        assert num_warps != 0 and (num_warps & (num_warps - 1)) == 0, f"num_warps={num_warps} must be a power of 2."
        # handle arguments passed by name
        kwargs = {self.fn.arg_names.index(name): value for name, value in kwargs.items()}
        wargs = list(wargs)
        for i, pos in enumerate(sorted(kwargs)):
            wargs.insert(pos + i, kwargs[pos])
        if len(wargs) != len(self.fn.arg_names):
            raise TypeError(f"Function takes {len(self.fn.arg_names)} positional arguments but {len(wargs)} were given")
        # handle annotations
        for pos, _type in self.fn.annotations.items():
            assert _type == triton.language.constexpr, "only constexpr annotations are supported for now"
            wargs[pos] = _type(wargs[pos])
        # check that tensors are on GPU.
        # for arg in wargs:
        #     if hasattr(arg, 'data_ptr'):
        #         assert arg.is_cuda, "All tensors must be on GPU!"
        # set device (i.e., make sure torch has the context initialized)
        device = torch.cuda.current_device()
        # torch creates new thread for backward pass that may have uninitlialized context
        # no way to know if this function should or shouldn't initialize the cuda context
        # so we're being conservative here
        torch.cuda.set_device(device)
        if device not in self.cache_key:
            cc = torch.cuda.get_device_capability(device)
            cc = str(cc[0]) + '-' + str(cc[1])
            self.cache_key[device] = self.fn.cache_key + cc
        cache_key = self.cache_key[device]
        stream = current_cuda_stream(device)
        return _triton.runtime.launch(wargs, self.fn.do_not_specialize, cache_key, self.fn.arg_names,
                                      device, stream, self.fn.bin_cache, num_warps, num_stages, extern_libs, self.add_to_cache,
                                      grid)


class Launcher:
    def __init__(self, kernel, grid):
        self.kernel = kernel
        self.grid = grid

    def __call__(self, *wargs, **kwargs):
        return self.kernel(*wargs, **kwargs, grid=self.grid)


class Autotuner:
    def __init__(self, kernel, arg_names, configs, key, reset_to_zero, prune_configs_by: Dict = None):
        '''
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It take configs:List[Config] as its input, and returns pruned configs.
        '''
        if not configs:
            self.configs = [Config(dict(), num_warps=4, num_stages=2)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.cache = dict()
        self.kernel = kernel
        # hook to reset all required tensor to zeros before relaunching a kernel
        self.hook = lambda args: 0
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]

            def _hook(args):
                for i in self.reset_idx:
                    args[i].zero_()
            self.hook = _hook
        self.arg_names = arg_names
        # prune configs
        if prune_configs_by:
            perf_model, top_k = prune_configs_by['perf_model'], prune_configs_by['top_k']
            if 'early_config_prune' in prune_configs_by:
                early_config_prune = prune_configs_by['early_config_prune']
        else:
            perf_model, top_k, early_config_prune = None, None, None
        self.perf_model, self.configs_top_k = perf_model, top_k
        self.early_config_prune = early_config_prune

    def _bench(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.kwargs)

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(self.nargs)
            self.hook(args)
            self.kernel(*args, num_warps=config.num_warps, num_stages=config.num_stages, **current)
        return triton.testing.do_bench(kernel_call)

    def __call__(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        if len(self.configs) > 1:
            key = tuple([args[i] for i in self.key_idx])
            if key not in self.cache:
                # prune configs
                pruned_configs = self.configs
                if self.early_config_prune:
                    pruned_configs = self.early_config_prune(self.configs, self.nargs)
                if self.perf_model:
                    top_k = self.configs_top_k
                    if isinstance(top_k, float) and top_k <= 1.0:
                        top_k = int(len(self.configs) * top_k)
                    if len(pruned_configs) > top_k:
                        est_timing = {config: self.perf_model(**self.nargs, **kwargs, **config.kwargs, num_stages=config.num_stages, num_warps=config.num_warps) for config in pruned_configs}
                        pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
                bench_start = time.time()
                timings = {config: self._bench(*args, config=config, **kwargs)
                           for config in pruned_configs}
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.hook(args)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if config.pre_hook is not None:
            config.pre_hook(self.nargs)
        return self.kernel(*args, num_warps=config.num_warps, num_stages=config.num_stages, **kwargs, **config.kwargs)


_version_key_lock = threading.Lock()
_version_key = None


def version_key():
    global _version_key

    if _version_key is not None:
        return _version_key

    with _version_key_lock:
        if _version_key is not None:
            return _version_key

        import pkgutil
        contents = []
        # frontend
        with open(triton.code_gen.__file__, "rb") as f:
            contents += [hashlib.md5(f.read()).hexdigest()]
        # backend
        with open(triton._C.libtriton.__file__, "rb") as f:
            contents += [hashlib.md5(f.read()).hexdigest()]
        # language
        language_path = os.path.join(*triton.__path__, 'language')
        for lib in pkgutil.iter_modules([language_path]):
            with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
                contents += [hashlib.md5(f.read()).hexdigest()]
        # ptxas version
        try:
            ptxas_version = hashlib.md5(subprocess.check_output(["ptxas", "--version"])).hexdigest()
        except Exception:
            ptxas_version = ''
        _version_key = '-'.join(triton.__version__) + '-' + ptxas_version + '-' + '-'.join(contents)
        return _version_key


class DependenciesFinder(ast.NodeVisitor):

    def __init__(self, globals, src) -> None:
        super().__init__()
        self.ret = hashlib.md5(src.encode("utf-8")).hexdigest()
        self.globals = globals

    def visit_Name(self, node):
        return self.globals.get(node.id, None)

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        while isinstance(lhs, ast.Attribute):
            lhs = self.visit(lhs.value)
        if lhs is None or lhs is triton:
            return None
        return getattr(lhs, node.attr)

    def visit_Call(self, node):
        func = self.visit(node.func)
        if func is None:
            return
        if inspect.isbuiltin(func):
            return
        if func.__module__ and func.__module__.startswith('triton.'):
            return
        assert isinstance(func, triton.JITFunction)
        if func.hash is None:
            tree = ast.parse(func.src)
            finder = DependenciesFinder(func.__globals__, func.src)
            finder.visit(tree)
            func.hash = finder.ret
        self.ret = (self.ret + func.hash).encode("utf-8")
        self.ret = hashlib.md5(self.ret).hexdigest()


def default_cache_dir():
    return os.path.join(os.environ["HOME"], ".triton", "cache")


class JITFunction:

    cache_hook = None

    def __init__(self, fn, version=None, inline=True, do_not_specialize=None):
        # information of wrapped function
        self.fn = fn
        self.module = fn.__module__
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]
        self.arg_defaults = [v.default for v in signature.parameters.values()]

        self.version = version
        self.inline = inline
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def"):]
        self.do_not_specialize = [] if do_not_specialize is None else do_not_specialize
        self.do_not_specialize = [self.arg_names.index(arg) if isinstance(arg, str) else arg for arg in self.do_not_specialize]
        # cache for callable driver objects (e.g. CUkernel)
        self.bin_cache = dict()
        self.hash = None
        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel_decorators = []
        self.kernel = None
        # annotations
        self.annotations = {self.arg_names.index(name): ty for name, ty in fn.__annotations__.items()}
        self.__annotations__ = fn.__annotations__
        # constexprs
        self.constexprs = [self.arg_names.index(ann) for ann in self.__annotations__.keys()]
        # forward docs
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

    @property
    @functools.lru_cache()
    def cache_key(self):
        # TODO : hash should be attribute of `self`
        if self.hash is None:
            dependencies_finder = DependenciesFinder(globals=self.__globals__, src=self.src)
            dependencies_finder.visit(self.parse())
            self.hash = dependencies_finder.ret + version_key()
        return self.hash

    # we do not parse `src` in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Some unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel.")

    # - when `.src` attribute is set, cache path needs
    #   to be reinitialized
    # - when kernel decorators change, cached kernel
    #   needs to be cleared
    def __setattr__(self, name, value):
        if name == 'kernel_decorators':
            self.kernel = None
        super(JITFunction, self).__setattr__(name, value)
        if name == 'src':
            self.hash = None
            JITFunction.cache_key.fget.cache_clear()

    def _init_kernel(self):
        if self.kernel is None:
            self.kernel = Kernel(self)
            for decorator in reversed(self.kernel_decorators):
                self.kernel = decorator(self.kernel)
        return self.kernel

    def warmup(self, compile):
        return self._warmup(**compile, is_manual_warmup=True)

    def _warmup(self, key, arg_types, device, attributes, constants, num_warps, num_stages, extern_libs, is_manual_warmup):
        hashed_key = hashlib.md5(key.encode("utf-8")).hexdigest()

        # create cache directory
        cache_dir = os.environ.get('TRITON_CACHE_DIR', default_cache_dir())
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        if cache_dir:
            bin_cache_path = os.path.join(cache_dir, hashed_key)
            bin_lock_path = bin_cache_path + ".lock"
        else:
            bin_cache_path = None
            bin_lock_path = None

        binary = None
        if bin_cache_path and os.path.exists(bin_cache_path):
            assert bin_lock_path is not None
            with FileLock(bin_lock_path):
                with open(bin_cache_path, 'rb') as f:
                    binary = pickle.load(f)["binary"]

        compile = dict(arg_types=arg_types, device=device, attributes=attributes, constants=constants, num_warps=num_warps, num_stages=num_stages, extern_libs=extern_libs)
        if JITFunction.cache_hook is not None:
            name = self.__name__
            info = key.split('-')[-3:]
            num_warps, num_stages, sig = info[0], info[1], info[2].split('_')[1:]
            # make signature human-readable
            arg_reprs = []
            for arg_name, arg_sig in zip(self.arg_names, sig):
                arg_reprs.append(f'{arg_name}: {arg_sig}')
            # assemble the repr
            arg_reprs = ", ".join(arg_reprs)
            repr = f"{name}[num_warps={num_warps}, num_stages={num_stages}]({arg_reprs})"
            noop = JITFunction.cache_hook(key=key, repr=repr, fn=self, compile={"key": key, **compile}, is_manual_warmup=is_manual_warmup, already_compiled=binary is not None)
            if noop:
                return True

        if binary is None:
            binary = self._compile(**compile)

        if bin_cache_path:
            assert bin_lock_path is not None
            with FileLock(bin_lock_path):
                with open(bin_cache_path + ".tmp", "wb") as f:
                    pickle.dump({"binary": binary, "key": key}, f)
                os.rename(bin_cache_path + ".tmp", bin_cache_path)

        self.bin_cache[key] = LoadedBinary(device, binary)
        return False

    def _compile(self, arg_types, device, attributes, constants, num_warps, num_stages, extern_libs):
        # create IR module
        context = _triton.ir.context()
        # get just-in-time proto-type of kernel
        arg_types = [Kernel._to_triton_ir(arg) for arg in arg_types]
        ret_type = triton.language.void
        prototype = triton.language.function_type(ret_type, arg_types)
        # generate Triton-IR
        # export symbols visible from self into code-generator object
        gscope = self.__globals__
        generator = CodeGenerator(context, prototype, gscope=gscope, attributes=attributes, constants=constants, is_kernel=True)
        try:
            generator.visit(self.parse())
        except Exception as e:
            node = generator.last_node
            if node is None or isinstance(e, (NotImplementedError, CompilationError)):
                raise e
            raise CompilationError(self.src, node) from e
        # Compile to machine code
        if torch.version.hip is None:
            backend = _triton.runtime.backend.CUDA
        else:
            backend = _triton.runtime.backend.ROCM
        name, asm, shared_mem = _triton.code_gen.compile_ttir(backend, generator.module, device, num_warps, num_stages, extern_libs)
        max_shared_memory = _triton.runtime.max_shared_memory(backend, device)
        if shared_mem > max_shared_memory:
            raise OutOfResources(shared_mem, max_shared_memory, "shared memory")
        return Binary(backend, name, asm, shared_mem, num_warps)

    def __getitem__(self, grid):
        return Launcher(self._init_kernel(), grid)

    def __repr__(self):
        return f"JITFunction({self.module}:{self.fn.__name__})"


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar meta: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type meta: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_stages: int
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    """

    def __init__(self, kwargs, num_warps=4, num_stages=2, pre_hook=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.pre_hook = pre_hook

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f'{k}: {v}')
        res.append(f'num_warps: {self.num_warps}')
        res.append(f'num_stages: {self.num_stages}')
        return ', '.join(res)


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(meta={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(meta={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']

    :note: When all the configurations are evaluated, the kernel will run multiple time.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           reset the value of the provided tensor to `zero` before running any configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It take configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    """
    def decorator(fn):
        def wrapper(kernel):
            return Autotuner(kernel, fn.arg_names, configs, key, reset_to_zero, prune_configs_by)

        fn.kernel_decorators.append(wrapper)
        return fn

    return decorator


def heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitevely expensive, or just not applicable.

    .. highlight:: python
    .. code-block:: python

        @triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size


    .param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a list of positional arguments as input.
    .type values: dict[str, Callable[[list[Any]], Any]]
    """
    def decorator(fn):
        def wrapper(kernel):
            def fun(*args, **meta):
                for v, heur in values.items():
                    assert v not in meta
                    meta[v] = heur({**dict(zip(fn.arg_names, args)), **meta})
                return kernel(*args, **meta)
            return fun

        fn.kernel_decorators.append(wrapper)
        return fn

    return decorator


def jit(*args, **kwargs):
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, :code:`torch.tensor` arguments are implicitly converted to pointers using the :code:`.data_ptr()` method.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * objects within the triton.language package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """
    if args:
        assert len(args) == 1
        assert callable(args[0])
        return JITFunction(args[0], **kwargs)
    else:
        def decorator(fn):
            return JITFunction(fn, **kwargs)
        return decorator

######

# class ForwardDeclaration:

#     def __init__(self, name, ret_ty, arg_tys) -> None:
#         self.name = name
#         self.ret_ty = ret_ty
#         self.arg_tys = arg_tys

# def forward_declare(name, ret_ty, arg_tys):
#     return ForwardDeclaration(name, ret_ty, arg_tys)

######


def cdiv(x, y):
    return (x + y - 1) // y


def next_power_of_2(n):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

######


class TensorWrapper:
    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.is_cuda = base.is_cuda
        self.device = base.device

    def data_ptr(self):
        return self.base.data_ptr()

    def __str__(self) -> str:
        return f'TensorWrapper[{self.dtype}]({self.base})'


def reinterpret(tensor, dtype):
    if isinstance(tensor, TensorWrapper):
        if dtype == tensor.base.dtype:
            # Reinterpreting to the original interpretation; return the base.
            return tensor.base
        else:
            # Reinterpreting a wrapped tensor to a different type.
            return TensorWrapper(tensor.base, dtype)
    elif isinstance(tensor, torch.Tensor):
        # A new wrapper is needed around an unwrapped tensor.
        return TensorWrapper(tensor, dtype)
    else:
        raise TypeError(f'Cannot reinterpret a {type(tensor)}.')
