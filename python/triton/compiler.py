from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import warnings
from sysconfig import get_paths
from typing import Any, Dict, Set, Tuple, Union

import setuptools
import torch
from filelock import FileLock

import triton
import triton._C.libtriton.triton as _triton
from .tools.disasm import extract


def str_to_ty(name):
    if name[0] == "*":
        ty = str_to_ty(name[1:])
        return triton.language.pointer_type(ty)
    tys = {
        "i1": triton.language.int1,
        "fp8": triton.language.float8,
        "fp16": triton.language.float16,
        "bf16": triton.language.bfloat16,
        "fp32": triton.language.float32,
        "fp64": triton.language.float64,
        "i8": triton.language.int8,
        "i16": triton.language.int16,
        "i32": triton.language.int32,
        "i64": triton.language.int64,
        "u8": triton.language.uint8,
        "u16": triton.language.uint16,
        "u32": triton.language.uint32,
        "u64": triton.language.uint64,
        "B": triton.language.int1,
    }
    return tys[name]


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
    assert False, "Unsupported type"


def mangle_fn(name, arg_tys, constants):
    # doesn't mangle ret type, which must be a function of arg tys
    mangled_arg_names = '_'.join([mangle_ty(ty) for ty in arg_tys])
    key = lambda x: x.__name__ if isinstance(x, triton.runtime.JITFunction) else repr(x)
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

    def __init__(self, context, prototype, gscope, attributes, constants, function_name, spec_to_1=None, prototypes=None, module=None, is_kernel=False):
        self.spec_to_1 = set() if spec_to_1 is None else spec_to_1
        self.prototypes = dict() if prototypes is None else prototypes
        self.builder = _triton.ir.builder(context)
        self.module = _triton.ir.module('', self.builder) if module is None else module
        self.prototype = prototype
        self.attributes = attributes
        self.constants = constants
        self.last_node = None
        self.function_name = function_name
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
        arg_names, arg_annotations, kwarg_names = self.visit(node.args)
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
        self.prototypes[self.function_name] = self.prototype
        fn = self.module.get_or_insert_function(self.function_name, self.prototype.to_ir(self.builder))
        fn.set_is_kernel(self.is_kernel)
        arg_values = []
        idx = 0
        for i, (arg_name, annotation) in enumerate(zip(arg_names, arg_annotations)):
            if i in self.constants:
                cst = self.constants[i]
                if not isinstance(cst, triton.language.constexpr):
                    cst = triton.language.constexpr(self.constants[i])
                arg_values.append(cst)
                continue
            if i in self.attributes:
                is_ptr = fn.args[idx].type.is_ptr()
                attr = 'aligned' if is_ptr else 'multiple_of'
                attr = getattr(_triton.ir.attribute_kind, attr)
                attr = _triton.ir.attribute(attr, self.attributes[i][1])
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
            self.module.reset_ret_ty(self.function_name, self.last_ret.type.to_ir(self.builder))
            self.prototype.ret_type = self.last_ret.type
        self.builder.set_insert_block(insert_pt)

    def visit_arguments(self, node):
        arg_names = []
        arg_annotations = []
        for arg in node.args:
            curr = self.visit(arg)
            arg_names += [curr[0]]
            arg_annotations += [curr[1]]
        kwarg_names = self.visit(node.kwarg)
        return arg_names, arg_annotations, kwarg_names

    def visit_arg(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        return node.arg, node.annotation

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

        if isinstance(fn, triton.runtime.JITFunction):
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
                generator = CodeGenerator(self.builder.context, prototype, gscope, attributes, constants, function_name=fn_name, prototypes=self.prototypes, module=self.module)
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


def kernel_suffix(signature, specialization):
    # suffix format:
    # <argid><'c' if equal to 1><'d' if divisible by 16>
    suffix = ''
    for i, _ in enumerate(signature):
        suffix += str(i)
        if i in specialization.equal_to_1:
            suffix += 'c'
        if i in specialization.divisible_by_16:
            suffix += 'd'
    return suffix


def make_triton_ir(fn, signature, specialization, constants):
    context = _triton.ir.context()
    # create kernel prototype
    cst_key = lambda i: fn.arg_names.index(i) if isinstance(i, str) else i
    constants = {cst_key(key): value for key, value in constants.items()}
    # visit kernel AST
    gscope = fn.__globals__.copy()
    function_name = '_'.join([fn.__name__, kernel_suffix(signature.values(), specialization)])
    tys = list(signature.values())
    new_constants = {k: True if k in tys and tys[k] == "i1" else 1 for k in specialization.equal_to_1}
    new_attrs = {k: ("multiple_of", 16) for k in specialization.divisible_by_16}
    all_constants = constants.copy()
    all_constants.update(new_constants)
    arg_types = [str_to_ty(v) for k, v in signature.items() if k not in constants]

    prototype = triton.language.function_type(triton.language.void, arg_types)
    generator = CodeGenerator(context, prototype, gscope=gscope, constants=all_constants, function_name=function_name, attributes=new_attrs, is_kernel=True)
    try:
        generator.visit(fn.parse())
    except Exception as e:
        node = generator.last_node
        if node is None or isinstance(e, (NotImplementedError, CompilationError)):
            raise e
        raise CompilationError(fn.src, node) from e
    ret = generator.module
    # module takes ownership of the context
    ret.context = context
    return ret, generator


def make_ptx(mod: Any, device: int) -> Tuple[str, int]:
    '''
    Translate TritonGPU module to PTX code.
    :param mod: a TritonGPU dialect module
    :return:
        - PTX code
        - shared memory alloaction size
    '''
    return _triton.translate_triton_gpu_to_ptx(mod, device)


def make_cubin(ptx, device):
    '''
    Compile TritonGPU module to cubin.
    :param ptx: ptx code
    :param device: CUDA device
    :return: str
    '''
    return _triton.compile_ptx_to_cubin(ptx, device)


def ptx_get_kernel_name(ptx: str) -> str:
    '''
    Get kernel name from PTX code.
    This Kernel name is required when launching the kernel.
    '''
    # There is a name mangling in PTX codegen, so the original kernel names in Triton IR are not available in PTX/cubin.
    assert ptx
    for line in ptx.split('\n'):
        line = line.strip()
        if line.startswith('// .globl'):
            return line.split()[-1]


def _compile(fn, signature: str, device: int = -1, constants=dict(),
             specialization=_triton.code_gen.instance_descriptor(),
             num_warps: int = 4, num_stages: int = 3, extern_libs=None,
             output: str = "ttgir", cc=0) -> Tuple[str, int, str]:
    valid_outputs = ("ttir", "ttgir", "ptx", "cubin")
    assert output in valid_outputs, "output should be one of [%s], but get \"%s\"" % (','.join(valid_outputs), output)

    # triton-ir
    module, _ = make_triton_ir(fn, signature, specialization, constants)
    if output == "ttir":
        return module

    assert output == "cubin"
    assert torch.version.hip is None
    backend = _triton.runtime.backend.CUDA
    if extern_libs is None:
        extern_libs = dict()
    name, asm, shared_mem = _triton.code_gen.compile_ttir(backend, module, device, num_warps, num_stages, extern_libs, cc)
    return asm, shared_mem, name


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "CUdeviceptr"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp32": "float",
    }[ty]


def generate_name_initializer(signature):
    src = "int i = 0;\n"
    tys = signature.split(',')
    for i, ty in enumerate(tys):
        src


def binary_name_to_header_name(name):
    if len(name) > 128:
        # avoid filename too long errors (filename limit is 255)
        name = "kernel_" + hashlib.sha256(name.encode("utf-8")).hexdigest()
    return f"{name}.h"


def generate_launcher(identifier, constants, signature):
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp32': 'float',
            'fp64': 'double',
        }[ty]

    def format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    format = "iiiiiKKOOO" + ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])

    # generate glue code
    src = f"""
#include \"cuda.h\"
#include <Python.h>

static inline void gpuAssert(CUresult code, const char *file, int line)
{{
   if (code != CUDA_SUCCESS)
   {{
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyErr_SetString(PyExc_RuntimeError, err);
   }}
}}
#define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}


void _launch(int gridX, int gridY, int gridZ, int num_warps, int shared_memory, CUstream stream, CUfunction function, {arg_decls}) {{
  void *params[] = {{ {', '.join(f"&arg{i}" for i in signature.keys() if i not in constants)} }};
  if(gridX*gridY*gridZ > 0){{
    CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0));
  }}
}}


static inline CUdeviceptr getPointer(PyObject *obj, int idx) {{
  if (PyLong_Check(obj)) {{
    return (CUdeviceptr)PyLong_AsUnsignedLongLong(obj);
  }}
  if (obj == Py_None) {{
    return (CUdeviceptr)0;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    }}
    return (CUdeviceptr)PyLong_AsUnsignedLongLong(ret);
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return (CUdeviceptr)0;
}}


static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  PyObject *hook_ret = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None) {{
    PyObject *new_args = PyTuple_Pack(1, compiled_kernel);
    hook_ret = PyObject_CallObject(launch_enter_hook, new_args);
    Py_DECREF(new_args);
  }}

  _launch(gridX, gridY, gridZ, num_warps, shared_memory, (CUstream)_stream, (CUfunction)_function, {', '.join(f"getPointer(_arg{i},{i})" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (launch_exit_hook != Py_None) {{
    PyObject *new_args = NULL;
    if (hook_ret) {{
        new_args = PyTuple_Pack(2, compiled_kernel, hook_ret);
    }} else {{
        new_args = PyTuple_Pack(1, compiled_kernel);
    }}
    hook_ret = PyObject_CallObject(launch_exit_hook, new_args);
    Py_DECREF(new_args);
  }}

  if (hook_ret) {{
      Py_DECREF(hook_ret);
  }}
  if(PyErr_Occurred()) {{
    return NULL;
  }}
  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""

    return src


def default_cache_dir():
    return os.path.join(os.environ["HOME"], ".triton", "cache")


class CacheManager:

    def __init__(self, key):
        self.key = key
        self.lock_path = None
        # create cache directory if it doesn't exist
        self.cache_dir = os.environ.get('TRITON_CACHE_DIR', default_cache_dir())
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)

    def _make_path(self, filename):
        return os.path.join(self.cache_dir, filename)

    def has_file(self, filename):
        if not self.cache_dir:
            return False
        return os.path.exists(self._make_path(filename))

    def put(self, data, filename, binary=True):
        if not self.cache_dir:
            return
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        with FileLock(self.lock_path):
            # use tempfile to be robust against program interruptions
            mode = "wb" if binary else "w"
            with open(filepath + ".tmp", mode) as f:
                f.write(data)
            os.rename(filepath + ".tmp", filepath)


# utilties for generating and compiling C wrappers


@functools.lru_cache()
def libcuda_dirs():
    locs = subprocess.check_output(["whereis", "libcuda.so"]).decode().strip().split()[1:]
    return [os.path.dirname(loc) for loc in locs]


@functools.lru_cache()
def cuda_home_dirs():
    default_dir = "/usr/local/cuda"
    return os.getenv("CUDA_HOME", default=default_dir)


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _build(name, src, srcdir):
    cuda_lib_dirs = libcuda_dirs()
    cu_include_dir = os.path.join(cuda_home_dirs(), "include")
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
    py_include_dir = get_paths()["include"]
    cc_cmd = [cc, src, "-O3", f"-I{cu_include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", "-lcuda", "-o", so]
    cc_cmd += [f"-L{dir}" for dir in cuda_lib_dirs]
    ret = subprocess.check_call(cc_cmd)
    if ret == 0:
        return so
    # fallback on setuptools
    extra_compile_args = []
    library_dirs = cuda_lib_dirs
    include_dirs = [srcdir, cu_include_dir]
    libraries = ['cuda']
    # extra arguments
    extra_link_args = []
    # create extension module
    ext = setuptools.Extension(
        name=name,
        language='c',
        sources=[src],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ['-O3'],
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libraries,
    )
    # build extension module
    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
    args = dict(
        name=name,
        ext_modules=[ext],
        script_args=args,
    )
    with quiet():
        setuptools.setup(**args)
    return so


def make_so_cache_key(version_hash, signature, constants):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{version_hash}-{''.join(signature.values())}{constants}"
    key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return key


def make_fn_cache_key(fn_hash, signature, configs, constants, num_warps, num_stages):
    # Get unique key for the compiled code
    get_conf_key = lambda conf: (sorted(conf.divisible_by_16), sorted(conf.equal_to_1))
    configs_key = [get_conf_key(conf) for conf in configs]
    key = f"{fn_hash}-{''.join(signature.values())}-{configs_key}-{constants}-{num_warps}-{num_stages}"
    key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return key


def compile(fn, signature: str, device: int = -1, constants=dict(), num_warps: int = 4,
            num_stages: int = 3, extern_libs=None, configs=None, cc=0, warm_cache_only=False):
    # we get the kernel, i.e. the first function generated in the module
    assert len(configs) == 1
    # cache manager
    name = fn.__name__
    # name of files that are cached
    so_cache_key = make_so_cache_key(triton.runtime.jit.version_key(), signature, constants)
    so_cache_manager = CacheManager(so_cache_key)
    so_name = f"{name}.so"
    # retrieve stub from cache if it exists
    if not so_cache_manager.has_file(so_name):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = generate_launcher(name, constants, signature)
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(fn.__name__, src_path, tmpdir)
            with open(so, "rb") as f:
                so_cache_manager.put(f.read(), so_name, binary=True)

    # retrieve cached shared object if it exists
    fn_cache_key = make_fn_cache_key(fn.cache_key, signature, configs, constants, num_warps, num_stages)
    fn_cache_manager = CacheManager(fn_cache_key)
    ptx_name = f"{name}.ptx"
    cubin_name = f"{name}.cubin"
    data_name = f"{name}.json"
    ttir_name = f"{name}.ttir"
    llir_name = f"{name}.llir"
    if not fn_cache_manager.has_file(cubin_name) or \
       not fn_cache_manager.has_file(data_name) or \
       not fn_cache_manager.has_file(ptx_name) or \
       not fn_cache_manager.has_file(ttir_name) or \
       not fn_cache_manager.has_file(llir_name):
        asm, shared, kernel_name = _compile(fn, signature, device, constants, configs[0], num_warps, num_stages,
                                            extern_libs, "cubin", cc)
        metadata = {"name": kernel_name, "shared": shared, "num_warps": num_warps, "num_stages": num_stages}
        fn_cache_manager.put(asm["cubin"], cubin_name)
        fn_cache_manager.put(asm["ptx"], ptx_name, binary=False)
        fn_cache_manager.put(asm["ttir"], ttir_name, binary=False)
        fn_cache_manager.put(asm["llir"], llir_name, binary=False)
        fn_cache_manager.put(json.dumps(metadata), data_name, binary=False)

    if warm_cache_only:
        return  # load_binary() requires a valid cuda context

    return CompiledKernel(name, so_cache_manager._make_path(so_name), fn_cache_manager.cache_dir, device)


class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    launch_enter_hook = None
    launch_exit_hook = None

    def __init__(self, fn_name, so_path, cache_dir, device):
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.c_wrapper = getattr(mod, "launch")
        # initialize metadata
        with open(os.path.join(cache_dir, f"{fn_name}.json")) as f:
            metadata = json.load(f)
        self.shared = metadata["shared"]
        self.num_warps = metadata["num_warps"]
        self.num_stages = metadata["num_stages"]
        # initialize asm dict
        self.asm = dict()
        with open(os.path.join(cache_dir, f"{fn_name}.cubin"), "rb") as f:
            self.asm["cubin"] = f.read()
        with open(os.path.join(cache_dir, f"{fn_name}.ptx"), "r") as f:
            self.asm["ptx"] = f.read()
        with open(os.path.join(cache_dir, f"{fn_name}.llir"), "r") as f:
            self.asm["llir"] = f.read()
        with open(os.path.join(cache_dir, f"{fn_name}.ttir"), "r") as f:
            self.asm["ttir"] = f.read()

        mod, func, n_regs, n_spills = _triton.code_gen.load_binary(metadata["name"], self.asm["cubin"], self.shared, device)
        self.fn_name = fn_name
        self.cu_module = mod
        self.cu_function = func
        self.n_regs = n_regs
        self.n_spills = n_spills

    def __getitem__(self, grid):
        def runner(*args, stream=None):
            if stream is None:
                stream = torch.cuda.current_stream().cuda_stream
            self.c_wrapper(grid[0], grid[1], grid[2], self.num_warps, self.shared, stream, self.cu_function,
                           CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, self, *args)
        return runner

    def get_sass(self, fun=None):
        if 'sass' in self.asm:
            return self.asm['sass']
        fd, path = tempfile.mkstemp()
        try:
            with open(fd, 'wb') as cubin:
                cubin.write(self.asm['cubin'])
            self.sass = extract(path, fun)
        finally:
            os.remove(path)
        self.asm['sass'] = self.sass
        return self.sass
