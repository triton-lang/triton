from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import warnings
from collections import namedtuple
from pathlib import Path
from sysconfig import get_paths
from typing import Any, Callable, Dict, Tuple, Union

import setuptools
import torch
from filelock import FileLock

import triton
import triton._C.libtriton.triton as _triton
from . import impl
from .tools.disasm import extract


def str_to_ty(name):
    if name[0] == "*":
        ty = str_to_ty(name[1:])
        return triton.language.pointer_type(ty)
    tys = {
        "fp8": triton.language.float8,
        "fp16": triton.language.float16,
        "bf16": triton.language.bfloat16,
        "fp32": triton.language.float32,
        "fp64": triton.language.float64,
        "i1": triton.language.int1,
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
    ret = f'{name}__{mangled_arg_names}__{mangled_constants}'
    return ret


class enter_sub_region:
    def __init__(self, generator: CodeGenerator):
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


class CodeGenerator(ast.NodeVisitor):
    def __init__(self, context, prototype, gscope, attributes, constants, function_name, module=None, is_kernel=False, function_types=dict()):
        self.builder = _triton.ir.builder(context)
        self.module = self.builder.create_module() if module is None else module
        self.function_ret_types = function_types
        self.prototype = prototype
        self.gscope = gscope
        self.lscope = dict()
        self.attributes = attributes
        self.constants = constants
        self.function_name = function_name
        self.is_kernel = is_kernel
        self.last_node = None
        self.builtins = {
            'range': range,
            'min': triton.language.minimum,
            'float': float,
            'int': int,
            'print': print,
            'isinstance': isinstance,
            'getattr': getattr,
        }
        self.scf_stack = []
        # SSA-construction
        # name => triton.language.tensor
        self.local_defs: Dict[str, triton.language.tensor] = {}
        self.global_uses: Dict[str, triton.language.tensor] = {}

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
            if name not in self.local_defs:
                self.global_uses[name] = ret
        # search node.id in global scope
        elif name in self.gscope:
            ret = self.gscope[name]
        # search node.id in builtins
        elif name in self.builtins:
            ret = self.builtins[name]
        else:
            raise ValueError(f'{name} is not defined')
        return ret

    def set_value(self, name: str,
                  value: Union[triton.language.tensor, triton.language.constexpr]) -> None:
        ''' This function:
          called by visit_Assign() & visit_FuncDef() to store left value (lvalue)
        1. record local defined name (FIXME: should consider control flow)
        2. store tensor in self.lvalue
        '''
        self.lscope[name] = value
        self.local_defs[name] = value

    def is_triton_tensor(self, value):
        return isinstance(value, triton.language.tensor)

    #
    # AST visitor
    #
    def visit_compound_statement(self, stmts):
        for stmt in stmts:
            self.last_ret_type = self.visit(stmt)
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
        ret_value = self.visit(node.value)
        # ret_block = self.builder.create_block()
        # post_ret_block = self.builder.create_block()
        # self.builder.create_branch(ret_block)
        # self.builder.set_insertion_point_to_end(ret_block)
        if ret_value is None:
            self.builder.ret([])
            ret_ty = None
        elif isinstance(ret_value, tuple):
            ret_values = [triton.language.core._to_tensor(v, self.builder) for v in ret_value]
            ret_types = [v.type for v in ret_values]
            self.builder.ret([v.handle for v in ret_values])
            ret_ty = tuple(ret_types)
        else:
            ret = triton.language.core._to_tensor(ret_value, self.builder)
            self.builder.ret([ret.handle])
            ret_ty = ret.type
        # self.builder.create_branch(post_ret_block)
        # self.builder.set_insertion_point_to_end(post_ret_block)
        return ret_ty

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
        visibility = "public" if self.is_kernel else "private"
        fn = self.builder.get_or_insert_function(self.module, self.function_name, self.prototype.to_ir(self.builder), visibility)
        self.module.push_back(fn)
        entry = fn.add_entry_block()
        arg_values = []
        idx = 0
        for i, arg_name in enumerate(arg_names):
            if i in self.constants:
                cst = self.constants[i]
                if not isinstance(cst, triton.language.constexpr):
                    cst = triton.language.constexpr(self.constants[i])
                arg_values.append(cst)
                continue
            else:
                if i in self.attributes:
                    fn.set_arg_attr(idx, "tt.divisibility", self.attributes[i][1])
                arg_values.append(triton.language.tensor(fn.args(idx), self.prototype.param_types[idx]))
                idx += 1

        insert_pt = self.builder.get_insertion_block()
        for arg_name, arg_value in zip(arg_names, arg_values):
            self.set_value(arg_name, arg_value)
        self.builder.set_insertion_point_to_start(entry)
        # visit function body
        has_ret = self.visit_compound_statement(node.body)
        # finalize function
        if not has_ret:
            self.builder.ret([])
        else:
            # update return type
            if isinstance(self.last_ret_type, tuple):
                self.prototype.ret_types = list(self.last_ret_type)
                fn.reset_type(self.prototype.to_ir(self.builder))
            else:
                self.prototype.ret_types = [self.last_ret_type]
                fn.reset_type(self.prototype.to_ir(self.builder))
        if insert_pt:
            self.builder.set_insertion_point_to_end(insert_pt)

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
            if target in self.lscope:
                raise ValueError(f'{target} is already defined.'
                                 f' constexpr cannot be reassigned.')
            if not isinstance(value, triton.language.constexpr):
                value = triton.language.constexpr(value)
            self.lscope[target] = value
            return self.lscope[target]
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
        for name, value in zip(names, values):
            # by default, constexpr are assigned into python variable
            if isinstance(value, triton.language.constexpr):
                value = value.value
            if not isinstance(value, triton.language.tensor):
                value = triton.language.core._to_tensor(value, self.builder)
            self.set_value(name, value)

    def visit_AugAssign(self, node):
        name = node.target.id
        lhs = ast.Name(id=name, ctx=ast.Load())
        rhs = ast.BinOp(lhs, node.op, node.value)
        assign = ast.Assign(targets=[node.target], value=rhs)
        self.visit(assign)
        return self.get_value(name)

    def visit_Name(self, node):
        if type(node.ctx) == ast.Store:
            return node.id
        return self.get_value(node.id)

    def visit_Store(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Load(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Tuple(self, node):
        args = [self.visit(x) for x in node.elts]
        return tuple(args)

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
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
        if self.is_triton_tensor(lhs):
            return getattr(lhs, fn)(rhs, _builder=self.builder)
        elif self.is_triton_tensor(rhs):
            fn = fn[:2] + 'r' + fn[2:]
            return getattr(rhs, fn)(lhs, _builder=self.builder)
        else:
            return getattr(lhs, fn)(rhs)

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
                    assert defs[name].type == liveins[name].type,\
                        f'initial value for `{name}` is of type {liveins[name].type}, '\
                        f'but the {block_name} block redefines it as {defs[name].type}'
            if name in then_defs or name in else_defs:
                names.append(name)
                ret_types.append(then_defs[name].type if name in then_defs else else_defs[name].type)
                ir_ret_types.append(then_defs[name].handle.get_type() if name in then_defs else else_defs[name].handle.get_type())
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
            assert then_ty == else_ty,\
                f'mismatched type for {name} between then block ({then_ty}) '\
                f'and else block ({else_ty})'
            names.append(name)
            ret_types.append(then_ty)
            ir_ret_types.append(then_defs[name].handle.get_type())

        return then_defs, else_defs, then_block, else_block, names, ret_types, ir_ret_types

    def visit_if_top_level(self, cond, node):
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
            if not then_block.has_terminator():
                self.builder.create_branch(endif_block, [then_defs[n].handle for n in names])
            # else terminator
            self.builder.set_insertion_point_to_end(else_block)
            if not else_block.has_terminator():
                self.builder.create_branch(endif_block, [else_defs[n].handle for n in names])
            for ty in ir_ret_types:
                endif_block.add_argument(ty)
        # change block
        self.builder.set_insertion_point_to_start(endif_block)
        # update value
        for i, name in enumerate(names):
            new_tensor = triton.language.core.tensor(endif_block.arg(i), ret_types[i])
            self.set_value(name, new_tensor)

    # TODO: refactor
    def visit_if_scf(self, cond, node):
        with enter_sub_region(self) as sr:
            liveins, _ = sr
            ip = self.builder.get_insertion_point()
            then_block = self.builder.create_block()
            else_block = self.builder.create_block() if node.orelse else None
            then_defs, else_defs, then_block, else_block, names, ret_types, _ = \
                self.visit_then_else_blocks(node, liveins, then_block, else_block)
            # create if op
            self.builder.restore_insertion_point(ip)
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
            new_tensor = triton.language.core.tensor(if_op.get_result(i), ret_types[i])
            self.set_value(name, new_tensor)

    def visit_If(self, node):
        cond = self.visit(node.test)
        if isinstance(cond, triton.language.tensor):
            cond = cond.to(triton.language.int1, _builder=self.builder)
            if self.scf_stack:
                self.visit_if_scf(cond, node)
            else:
                self.visit_if_top_level(cond, node)
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
        if isinstance(lhs, triton.language.constexpr):
            lhs = lhs.value
        if isinstance(rhs, triton.language.constexpr):
            rhs = rhs.value
        if type(node.ops[0]) == ast.Is:
            return triton.language.constexpr(lhs is rhs)
        if type(node.ops[0]) == ast.IsNot:
            return triton.language.constexpr(lhs is not rhs)
        fn = {
            ast.Eq: '__eq__',
            ast.NotEq: '__ne__',
            ast.Lt: '__lt__',
            ast.LtE: '__le__',
            ast.Gt: '__gt__',
            ast.GtE: '__ge__',
        }[type(node.ops[0])]
        if self.is_triton_tensor(lhs):
            return getattr(lhs, fn)(rhs, _builder=self.builder)
        elif self.is_triton_tensor(rhs):
            fn = fn[:2] + 'r' + fn[2:]
            return getattr(rhs, fn)(lhs, _builder=self.builder)
        else:
            return getattr(lhs, fn)(rhs)

    def visit_UnaryOp(self, node):
        op = self.visit(node.operand)
        fn = {
            ast.USub: '__neg__',
            ast.UAdd: '__pos__',
            ast.Not: '__not__',
            ast.Invert: '__invert__',
        }[type(node.op)]
        if self.is_triton_tensor(op):
            return getattr(op, fn)(_builder=self.builder)
        return getattr(op, fn)()

    def visit_While(self, node):
        with enter_sub_region(self) as sr:
            liveins, insert_block = sr

            # loop body (the after region)
            # loop_block = self.builder.create_block()
            dummy = self.builder.create_block()
            self.builder.set_insertion_point_to_start(dummy)
            self.scf_stack.append(node)
            self.visit_compound_statement(node.body)
            self.scf_stack.pop()
            loop_defs = self.local_defs

            # collect loop-carried values
            names = []
            ret_types = []
            init_args = []
            for name in loop_defs:
                if name in liveins:
                    # We should not def new constexpr
                    assert self.is_triton_tensor(loop_defs[name])
                    assert self.is_triton_tensor(liveins[name])
                    assert loop_defs[name].type == liveins[name].type
                    # these are loop-carried values
                    names.append(name)
                    ret_types.append(loop_defs[name].type)
                    init_args.append(liveins[name])

            self.builder.set_insertion_point_to_end(insert_block)
            while_op = self.builder.create_while_op([ty.to_ir(self.builder) for ty in ret_types],
                                                    [arg.handle for arg in init_args])
            # merge the condition region
            before_block = self.builder.create_block_with_parent(while_op.get_before(),
                                                                 [ty.to_ir(self.builder) for ty in ret_types])
            self.builder.set_insertion_point_to_start(before_block)
            for i, name in enumerate(names):
                self.lscope[name] = triton.language.core.tensor(before_block.arg(i), ret_types[i])
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
                self.lscope[name] = triton.language.core.tensor(after_block.arg(i), ret_types[i])
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

        # update global uses in while_op
        for i, name in enumerate(names):
            after_block.replace_use_in_block_with(init_args[i].handle, after_block.arg(i))

        # WhileOp defines new values, update the symbol table (lscope, local_defs)
        for i, name in enumerate(names):
            new_def = triton.language.core.tensor(while_op.get_result(i), ret_types[i])
            self.lscope[name] = new_def
            self.local_defs[name] = new_def

        for stmt in node.orelse:
            assert False, "Not implemented"
            ast.NodeVisitor.generic_visit(self, stmt)

    def visit_Subscript(self, node):
        assert node.ctx.__class__.__name__ == "Load"
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        if self.is_triton_tensor(lhs):
            return lhs.__getitem__(slices, _builder=self.builder)
        return lhs[slices]

    def visit_ExtSlice(self, node):
        return [self.visit(dim) for dim in node.dims]

    def visit_For(self, node):
        IteratorClass = self.visit(node.iter.func)
        iter_args = [self.visit(arg) for arg in node.iter.args]
        if IteratorClass == triton.language.static_range:
            iterator = IteratorClass(*iter_args)
            static_range = range(iterator.start.value,
                                 iterator.end.value,
                                 iterator.step.value)
            for i in static_range:
                self.lscope[node.target.id] = triton.language.constexpr(i)
                self.visit_compound_statement(node.body)
                for stmt in node.orelse:
                    ast.NodeVisitor.generic_visit(self, stmt)
            return

        if IteratorClass != self.builtins['range']:
            raise RuntimeError('Only `range` and `static_range` iterators are currently supported')

        # visit iterator arguments
        # note: only `range` iterator is supported now
        # collect lower bound (lb), upper bound (ub), and step
        lb = iter_args[0] if len(iter_args) > 1 else self.visit(ast.Num(0))
        ub = iter_args[1] if len(iter_args) > 1 else self.visit(node.iter.args[0])
        step = iter_args[2] if len(iter_args) > 2 else self.visit(ast.Num(1))
        # handle negative constant step (not supported by scf.for in MLIR)
        negative_step = False
        if isinstance(step, triton.language.constexpr) and step.value < 0:
            step = triton.language.constexpr(-step.value)
            negative_step = True
            lb, ub = ub, lb
        # lb/ub/step might be constexpr, we need to cast them to tensor
        lb = triton.language.core._to_tensor(lb, self.builder).handle
        ub = triton.language.core._to_tensor(ub, self.builder).handle
        step = triton.language.core._to_tensor(step, self.builder).handle
        # ForOp can only accept IndexType as lb/ub/step. Cast integer to Index
        lb = self.builder.create_to_index(lb)
        ub = self.builder.create_to_index(ub)
        step = self.builder.create_to_index(step)
        # Create placeholder for the loop induction variable
        iv = self.builder.create_undef(self.builder.get_int32_ty())
        self.set_value(node.target.id, triton.language.core.tensor(iv, triton.language.core.int32))

        with enter_sub_region(self) as sr:
            liveins, insert_block = sr
            ip = self.builder.get_insertion_point()

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
                    assert self.is_triton_tensor(self.local_defs[name]), f'{name} is not tensor'
                    assert self.is_triton_tensor(liveins[name])
                    assert self.local_defs[name].type == liveins[name].type,\
                        f'Loop-carried variable {name} has initial type {liveins[name].type} '\
                        f'but is re-assigned to {self.local_defs[name].type} in loop! '\
                        f'Please make sure that the type stays consistent.'

                    names.append(name)
                    init_args.append(triton.language.core._to_tensor(liveins[name], self.builder))
                    yields.append(triton.language.core._to_tensor(self.local_defs[name], self.builder))

            # create ForOp
            self.builder.restore_insertion_point(ip)
            for_op = self.builder.create_for_op(lb, ub, step, [arg.handle for arg in init_args])

            self.scf_stack.append(node)
            self.builder.set_insertion_point_to_start(for_op.get_body(0))
            for i, name in enumerate(names):
                self.set_value(name, triton.language.core.tensor(for_op.get_body(0).arg(i + 1), yields[i].type))
            self.visit_compound_statement(node.body)
            self.scf_stack.pop()
            yields = []
            for name in self.local_defs:
                if name in liveins:
                    yields.append(triton.language.core._to_tensor(self.local_defs[name], self.builder))

            # create YieldOp
            if len(yields) > 0:
                self.builder.create_yield_op([y.handle for y in yields])
            for_op_region = for_op.get_body(0).get_parent()
            assert for_op_region.size() == 1, "We use SCF, so the loop body should only have one block"

            # update induction variable with actual value, and replace all uses
            self.builder.set_insertion_point_to_start(for_op.get_body(0))
            iv = self.builder.create_index_to_si(for_op.get_induction_var())
            if negative_step:
                ub_si = self.builder.create_index_to_si(ub)
                iv = self.builder.create_sub(ub_si, iv)
            self.lscope[node.target.id].handle.replace_all_uses_with(iv)
            self.set_value(node.target.id, triton.language.core.tensor(iv, triton.language.core.int32))

        # update lscope & local_defs (ForOp defines new values)
        for i, name in enumerate(names):
            self.set_value(name, triton.language.core.tensor(for_op.get_result(i), yields[i].type))

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
                prototype = triton.language.function_type([], arg_types)
                gscope = sys.modules[fn.fn.__module__].__dict__
                generator = CodeGenerator(self.builder.context, prototype, gscope, attributes, constants, module=self.module, function_name=fn_name, function_types=self.function_ret_types)
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
                return triton.language.tensor(call_op.get_result(0), callee_ret_type)
            else:
                # should return a tuple of tl.tensor
                results = []
                for i in range(call_op.get_num_results()):
                    results.append(triton.language.tensor(call_op.get_result(i), callee_ret_type[i]))
                return tuple(results)
        if (hasattr(fn, '__self__') and self.is_triton_tensor(fn.__self__)) \
                or impl.is_builtin(fn):
            return fn(*args, _builder=self.builder, **kws)
        if fn in self.builtins.values():
            args = [arg.value if isinstance(arg, triton.language.constexpr) else arg
                    for arg in args]
        return fn(*args, **kws)

    def visit_Constant(self, node):
        return triton.language.constexpr(node.value)

    def visit_BoolOp(self, node: ast.BoolOp):
        assert len(node.values) == 2
        lhs = self.visit(node.values[0])
        rhs = self.visit(node.values[1])

        fn = {
            ast.And: 'logical_and',
            ast.Or: 'logical_or',
        }[type(node.op)]

        if self.is_triton_tensor(lhs):
            return getattr(lhs, fn)(rhs, _builder=self.builder)
        elif self.is_triton_tensor(rhs):
            fn = fn[:2] + 'r' + fn[2:]
            return getattr(rhs, fn)(lhs, _builder=self.builder)
        else:
            return getattr(lhs, fn)(rhs)

    if sys.version_info < (3, 8):
        def visit_NameConstant(self, node):
            return triton.language.constexpr(node.value)

        def visit_Num(self, node):
            return triton.language.constexpr(node.n)

        def visit_Str(self, node):
            return triton.language.constexpr(ast.literal_eval(node))

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        if isinstance(lhs, triton.language.tensor):
            if node.attr == "T":
                return triton.language.semantic.trans(lhs, builder=self.builder)
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
        self.message += '. Reducing block sizes or `num_stages` may help.'
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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def parse_mlir_module(path, context):
    module = _triton.ir.parse_mlir_module(path, context)
    # module takes ownership of the context
    module.context = context
    return module


def build_triton_ir(fn, signature, specialization, constants):
    # canonicalize signature
    if isinstance(signature, str):
        signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
    context = _triton.ir.context()
    context.load_triton()
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

    prototype = triton.language.function_type([], arg_types)
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


def optimize_triton_ir(mod):
    pm = _triton.ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_inliner_pass()
    pm.add_triton_combine_pass()
    pm.add_canonicalizer_pass()
    pm.add_cse_pass()
    pm.add_licm_pass()
    pm.run(mod)
    return mod


def ast_to_ttir(fn, signature, specialization, constants):
    mod, _ = build_triton_ir(fn, signature, specialization, constants)
    return optimize_triton_ir(mod)


def ttir_to_ttgir(mod, num_warps, num_stages, compute_capability):
    pm = _triton.ir.pass_manager(mod.context)
    pm.add_convert_triton_to_tritongpu_pass(num_warps)
    pm.enable_debug()
    pm.add_coalesce_pass()
    # The combine pass converts blocked layout to mma layout
    # for dot ops so that pipeline can get shared memory swizzled correctly.
    pm.add_tritongpu_combine_pass(compute_capability)
    pm.add_tritongpu_pipeline_pass(num_stages)
    # Prefetch must be done after pipeline pass because pipeline pass
    # extracts slices from the original tensor.
    pm.add_tritongpu_prefetch_pass()
    pm.add_canonicalizer_pass()
    pm.add_cse_pass()
    pm.add_tritongpu_combine_pass(compute_capability)
    pm.add_licm_pass()
    pm.add_tritongpu_combine_pass(compute_capability)
    pm.add_cse_pass()
    pm.add_tritongpu_decompose_conversions_pass()
    if compute_capability // 10 == 7:
        # The update_mma_for_volta pass helps to compute some information for MMA encoding specifically for MMAv1
        # NOTE this pass should be placed after all the passes those modifies mma layout
        pm.add_tritongpu_update_mma_for_volta_pass()
    pm.add_cse_pass()
    pm.add_symbol_dce_pass()
    pm.add_tritongpu_reorder_instructions_pass()
    pm.run(mod)
    return mod


def add_external_libs(mod, libs):
    for name, path in libs.items():
        if len(name) == 0 or len(path) == 0:
            return
    _triton.add_external_libs(mod, list(libs.keys()), list(libs.values()))


def ttgir_to_llir(mod, extern_libs, compute_capability):
    if extern_libs:
        add_external_libs(mod, extern_libs)
    return _triton.translate_triton_gpu_to_llvmir(mod, compute_capability)


def llir_to_ptx(mod: Any, compute_capability: int, ptx_version: int = None) -> Tuple[str, int]:
    '''
    Translate TritonGPU module to PTX code.
    :param mod: a TritonGPU dialect module
    :return:
        - PTX code
        - shared memory allocation size
    '''
    if ptx_version is None:
        _, cuda_version = path_to_ptxas()
        ptx_version = ptx_get_version(cuda_version)
    return _triton.translate_llvmir_to_ptx(mod, compute_capability, ptx_version)


def ptx_to_cubin(ptx: str, compute_capability: int):
    '''
    Compile TritonGPU module to cubin.
    :param ptx: ptx code
    :param compute_capability: compute capability
    :return: str
    '''
    ptxas, _ = path_to_ptxas()
    return _triton.compile_ptx_to_cubin(ptx, ptxas, compute_capability)


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


@functools.lru_cache
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        return 80 + minor
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher")


def path_to_ptxas():
    base_dir = os.path.dirname(__file__)
    paths = [
        os.environ.get("TRITON_PTXAS_PATH", ""),
        os.path.join(base_dir, "third_party", "cuda", "bin", "ptxas")
    ]

    for ptxas in paths:
        if os.path.exists(ptxas) and os.path.isfile(ptxas):
            result = subprocess.check_output([ptxas, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return ptxas, version.group(1)
    raise RuntimeError("Cannot find ptxas")


instance_descriptor = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"], defaults=[set(), set()])


# ------------------------------------------------------------------------------
# compiler
# ------------------------------------------------------------------------------


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
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
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


def generate_launcher(constants, signature):
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
            'fp16': 'float',
            'bf16': 'float',
            'fp32': 'float',
            'f32': 'float',
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
#include <stdbool.h>
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

typedef struct _DevicePtrInfo {{
    CUdeviceptr dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
    unsigned attr;
    CUresult status =
        cuPointerGetAttribute(&attr, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr_info.dev_ptr);
    if (ptr_info.dev_ptr &&
        (!(attr == CU_MEMORYTYPE_DEVICE || attr == CU_MEMORYTYPE_UNIFIED) ||
         !(status == CUDA_SUCCESS))) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
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


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_warps, shared_memory, (CUstream)_stream, (CUfunction)_function, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

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
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
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


def default_cuda_dir():
    default_dir = "/usr/local/cuda"
    return os.getenv("CUDA_HOME", default=default_dir)


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
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        with FileLock(self.lock_path):
            # use tempfile to be robust against program interruptions
            mode = "wb" if binary else "w"
            with open(filepath + ".tmp", mode) as f:
                f.write(data)
            os.rename(filepath + ".tmp", filepath)


# Utilities for generating and compiling C wrappers


@functools.lru_cache()
def libcuda_dirs():
    locs = subprocess.check_output(["whereis", "libcuda.so"]).decode().strip().split()[1:]
    return [os.path.dirname(loc) for loc in locs]


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
    cuda_path = os.environ.get('CUDA_PATH', default_cuda_dir())
    cu_include_dir = os.path.join(cuda_path, "include")
    base_dir = os.path.dirname(__file__)
    triton_include_dir = os.path.join(base_dir, "third_party/cuda/include")
    cuda_header = os.path.join(cu_include_dir, "cuda.h")
    triton_cuda_header = os.path.join(triton_include_dir, "cuda.h")
    if not os.path.exists(cuda_header) and os.path.exists(triton_cuda_header):
        cu_include_dir = triton_include_dir
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
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


def read_or_execute(cache_manager, force_compile, file_name, metadata,
                    run_if_found: Callable[[str], bytes] = None,
                    run_if_not_found: Callable = None):
    suffix = file_name.split(".")[1]
    if not force_compile and cache_manager.has_file(file_name):
        module = run_if_found(cache_manager._make_path(file_name))
        data = module if isinstance(module, bytes) else str(module).encode("utf-8")
        md5 = hashlib.md5(data).hexdigest()
        has_changed = metadata and md5 != metadata["md5"][suffix]
        return module, md5, has_changed, True
    module = run_if_not_found()
    data = module if isinstance(module, bytes) else str(module).encode("utf-8")
    md5 = hashlib.md5(data).hexdigest()
    cache_manager.put(data, file_name, True if isinstance(data, bytes) else data)
    return module, md5, True, False

#


def make_stub(name, signature, constants):
    # name of files that are cached
    so_cache_key = make_so_cache_key(triton.runtime.jit.version_key(), signature, constants)
    so_cache_manager = CacheManager(so_cache_key)
    so_name = f"{name}.so"
    # retrieve stub from cache if it exists
    if not so_cache_manager.has_file(so_name):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = generate_launcher(constants, signature)
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir)
            with open(so, "rb") as f:
                so_cache_manager.put(f.read(), so_name, binary=True)
    return so_cache_manager._make_path(so_name)


def convert_type_repr(x):
    match = re.search(r'!tt\.ptr<(.*)>', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


def make_hash(fn, **kwargs):
    if isinstance(fn, triton.runtime.JITFunction):
        configs = kwargs["configs"]
        signature = kwargs["signature"]
        constants = kwargs.get("constants", dict())
        num_warps = kwargs.get("num_warps", 4)
        num_stages = kwargs.get("num_stages", 3)
        # Get unique key for the compiled code
        get_conf_key = lambda conf: (sorted(conf.divisible_by_16), sorted(conf.equal_to_1))
        configs_key = [get_conf_key(conf) for conf in configs]
        key = f"{fn.cache_key}-{''.join(signature.values())}-{configs_key}-{constants}-{num_warps}-{num_stages}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    assert isinstance(fn, str)
    return hashlib.md5((Path(fn).read_text() + triton.runtime.jit.version_key()).encode("utf-8")).hexdigest()


# - ^\s*func\s+ : match the start of the string, any leading whitespace, the keyword func,
#    and any following whitespace
# - (public\s+)? : optionally match the keyword public and any following whitespace
# - (@\w+) : match an @ symbol followed by one or more word characters
#   (letters, digits, or underscores), and capture it as group 1 (the function name)
# - (\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\)) : match a pair of parentheses enclosing
#   zero or more arguments separated by commas, and capture it as group 2 (the argument list)
mlir_prototype_pattern = r'^\s*func\s+(?:public\s+)?(@\w+)(\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\))\s*\{\s*$'
ptx_prototype_pattern = r"\.(?:visible|extern)\s+\.(?:entry|func)\s+(\w+)\s*\(([^)]*)\)"
prototype_pattern = {
    "ttir": mlir_prototype_pattern,
    "ttgir": mlir_prototype_pattern,
    "ptx": ptx_prototype_pattern,
}

mlir_arg_type_pattern = r'%\w+: ([^,^\)\s]+)(?: \{\S+ = \S+ : \S+\})?,?'
ptx_arg_type_pattern = r"\.param\s+\.(\w+)"
arg_type_pattern = {
    "ttir": mlir_arg_type_pattern,
    "ttgir": mlir_arg_type_pattern,
    "ptx": ptx_arg_type_pattern,
}


# def compile(fn, signature: str, device: int = -1, constants=dict(), num_warps: int = 4, num_stages: int = 3, extern_libs=None, configs=None):
def compile(fn, **kwargs):
    capability = kwargs.get("cc", None)
    if capability is None:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
    # we get the kernel, i.e. the first function generated in the module
    # if fn is not a JITFunction, then it
    # has to be a path to a file
    context = _triton.ir.context()
    asm = dict()
    constants = kwargs.get("constants", dict())
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 3 if capability >= 75 else 2)
    extern_libs = kwargs.get("extern_libs", dict())
    # build compilation stages
    stages = {
        "ast": (lambda path: fn, None),
        "ttir": (lambda path: parse_mlir_module(path, context),
                 lambda src: ast_to_ttir(src, signature, configs[0], constants)),
        "ttgir": (lambda path: parse_mlir_module(path, context),
                  lambda src: ttir_to_ttgir(src, num_warps, num_stages, capability)),
        "llir": (lambda path: Path(path).read_text(),
                 lambda src: ttgir_to_llir(src, extern_libs, capability)),
        "ptx": (lambda path: Path(path).read_text(),
                lambda src: llir_to_ptx(src, capability)),
        "cubin": (lambda path: Path(path).read_bytes(),
                  lambda src: ptx_to_cubin(src, capability))
    }
    # find out the signature of the function
    if isinstance(fn, triton.runtime.JITFunction):
        configs = kwargs.get("configs", None)
        signature = kwargs["signature"]
        if configs is None:
            configs = [instance_descriptor()]
        assert len(configs) == 1
        kwargs["configs"] = configs
        name = fn.__name__
        first_stage = 0
        if isinstance(signature, str):
            signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
        kwargs["signature"] = signature
    else:
        assert isinstance(fn, str)
        _, ir = os.path.basename(fn).split(".")
        src = Path(fn).read_text()
        import re
        match = re.search(prototype_pattern[ir], src, re.MULTILINE)
        name, signature = match.group(1), match.group(2)
        # print(name, signature)
        types = re.findall(arg_type_pattern[ir], signature)
        # print(types)
        param_tys = [convert_type_repr(ty) for ty in types]
        signature = {k: v for k, v in enumerate(param_tys)}
        first_stage = list(stages.keys()).index(ir)

    # cache manager
    so_path = make_stub(name, signature, constants)
    # create cache manager
    fn_cache_manager = CacheManager(make_hash(fn, **kwargs))
    # determine name and extension type of provided function
    if isinstance(fn, triton.runtime.JITFunction):
        name, ext = fn.__name__, "ast"
    else:
        name, ext = os.path.basename(fn).split(".")

    # load metadata if any
    metadata = None
    if fn_cache_manager.has_file(f'{name}.json'):
        with open(fn_cache_manager._make_path(f"{name}.json")) as f:
            metadata = json.load(f)
    else:
        metadata = {"num_warps": num_warps, "num_stages": num_stages, "ctime": dict()}
        if ext == "ptx":
            assert "shared" in kwargs, "ptx compilation must provide shared memory size"
            metadata["shared"] = kwargs["shared"]

    first_stage = list(stages.keys()).index(ext)
    asm = dict()
    module = fn
    # run compilation pipeline  and populate metadata
    for ir, (parse, compile) in list(stages.items())[first_stage:]:
        path = fn_cache_manager._make_path(f"{name}.{ir}")
        if ir == ext:
            next_module = parse(fn)
        elif os.path.exists(path) and\
                ir in metadata["ctime"] and\
                os.path.getctime(path) == metadata["ctime"][ir]:
            next_module = parse(path)
        else:
            next_module = compile(module)
            fn_cache_manager.put(next_module, f"{name}.{ir}")
        if os.path.exists(path):
            metadata["ctime"][ir] = os.path.getctime(path)
        asm[ir] = next_module if ir == "cubin" else str(next_module)
        if ir == "llir" and "shared" not in metadata:
            metadata["shared"] = _triton.get_shared_memory_size(module)
        if ir == "ptx":
            metadata["name"] = ptx_get_kernel_name(next_module)
        module = next_module
    # write-back metadata
    fn_cache_manager.put(json.dumps(metadata), f"{name}.json", binary=False)
    # return handle to compiled kernel
    return CompiledKernel(so_path, metadata, asm)


class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    launch_enter_hook = None
    launch_exit_hook = None

    def __init__(self, so_path, metadata, asm):
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.c_wrapper = getattr(mod, "launch")
        # initialize metadata
        self.shared = metadata["shared"]
        self.num_warps = metadata["num_warps"]
        self.num_stages = metadata["num_stages"]
        # initialize asm dict
        self.asm = asm
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.metadata = metadata
        self.cu_module = None
        self.cu_function = None

    def _init_handles(self):
        if self.cu_module is not None:
            return
        device = torch.cuda.current_device()
        global cuda_utils
        init_cuda_utils()
        max_shared = cuda_utils.get_device_properties(device)["max_shared_mem"]
        if self.shared > max_shared:
            raise OutOfResources(self.shared, max_shared, "shared memory")
        mod, func, n_regs, n_spills = cuda_utils.load_binary(self.metadata["name"], self.asm["cubin"], self.shared, device)
        # print(self.shared, n_regs, n_spills)
        self.cu_module = mod
        self.cu_function = func

    def __getattribute__(self, name):
        if name == 'c_wrapper':
            self._init_handles()
        return super().__getattribute__(name)

    def __getitem__(self, grid):
        self._init_handles()

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


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    @staticmethod
    def _generate_src():
        return """
        #include <cuda.h>

        #include \"cuda.h\"
        #define PY_SSIZE_T_CLEAN
        #include <Python.h>

        static inline void gpuAssert(CUresult code, const char *file, int line)
        {
           if (code != CUDA_SUCCESS)
           {
              const char* prefix = "Triton Error [CUDA]: ";
              const char* str;
              cuGetErrorString(code, &str);
              char err[1024] = {0};
              strcat(err, prefix);
              strcat(err, str);
              PyErr_SetString(PyExc_RuntimeError, err);
           }
        }

        #define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); if(PyErr_Occurred()) return NULL; }

        static PyObject* getDeviceProperties(PyObject* self, PyObject* args){
            int device_id;
            if(!PyArg_ParseTuple(args, "i", &device_id))
                return NULL;
            // Get device handle
            CUdevice device;
            cuDeviceGet(&device, device_id);

            // create a struct to hold device properties
            int max_shared_mem;
            int multiprocessor_count;
            int sm_clock_rate;
            int mem_clock_rate;
            int mem_bus_width;
            CUDA_CHECK(cuDeviceGetAttribute(&max_shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
            CUDA_CHECK(cuDeviceGetAttribute(&multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
            CUDA_CHECK(cuDeviceGetAttribute(&sm_clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
            CUDA_CHECK(cuDeviceGetAttribute(&mem_clock_rate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
            CUDA_CHECK(cuDeviceGetAttribute(&mem_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));


            return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i}", "max_shared_mem", max_shared_mem,
                                       "multiprocessor_count", multiprocessor_count,
                                       "sm_clock_rate", sm_clock_rate,
                                       "mem_clock_rate", mem_clock_rate,
                                       "mem_bus_width", mem_bus_width);
        }

        static PyObject* loadBinary(PyObject* self, PyObject* args) {
            const char* name;
            const char* data;
            Py_ssize_t data_size;
            int shared;
            int device;
            if(!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared, &device)) {
                return NULL;
            }
            CUfunction fun;
            CUmodule mod;
            int32_t n_regs = 0;
            int32_t n_spills = 0;
            // create driver handles
            CUDA_CHECK(cuModuleLoadData(&mod, data));
            CUDA_CHECK(cuModuleGetFunction(&fun, mod, name));
            // get allocated registers and spilled registers from the function
            CUDA_CHECK(cuFuncGetAttribute(&n_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun));
            CUDA_CHECK(cuFuncGetAttribute(&n_spills, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun));
            n_spills /= 4;
            // set dynamic shared memory if necessary
            int shared_optin;
            CUDA_CHECK(cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
            if (shared > 49152 && shared_optin > 49152) {
              CUDA_CHECK(cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
              int shared_total, shared_static;
              CUDA_CHECK(cuDeviceGetAttribute(&shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device));
              CUDA_CHECK(cuFuncGetAttribute(&shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
              CUDA_CHECK(cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin - shared_static));
            }

            if(PyErr_Occurred()) {
              return NULL;
            }
            return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs, n_spills);
        }

        static PyMethodDef ModuleMethods[] = {
          {"load_binary", loadBinary, METH_VARARGS, "Load provided cubin into CUDA driver"},
          {"get_device_properties", getDeviceProperties, METH_VARARGS, "Get the properties for a given device"},
          {NULL, NULL, 0, NULL} // sentinel
        };

        static struct PyModuleDef ModuleDef = {
          PyModuleDef_HEAD_INIT,
          \"cuda_utils\",
          NULL, //documentation
          -1, //size
          ModuleMethods
        };

        PyMODINIT_FUNC PyInit_cuda_utils(void) {
          PyObject *m = PyModule_Create(&ModuleDef);
          if(m == NULL) {
            return NULL;
          }
          PyModule_AddFunctions(m, ModuleMethods);
          return m;
        }
        """

    def __init__(self):
        src = self._generate_src()
        key = hashlib.md5(src.encode("utf-8")).hexdigest()
        cache = CacheManager(key)
        fname = "cuda_utils.so"
        if not cache.has_file(fname):
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build("cuda_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("cuda_utils", cache._make_path(fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


def init_cuda_utils():
    global cuda_utils
    if cuda_utils is None:
        cuda_utils = CudaUtils()


cuda_utils = None
