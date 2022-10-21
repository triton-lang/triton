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
from sysconfig import get_paths
from typing import Any, Dict, Tuple, Union

import setuptools
import torch
from filelock import FileLock

import triton
import triton._C.libtriton.triton as _triton


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
        return self.liveins, self.insert_block

    def __exit__(self, *args, **kwargs):
        self.generator.builder.set_insertion_point_to_end(self.insert_block)
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
        if ret_value is None:
            self.builder.ret([])
            return None
        if isinstance(ret_value, tuple):
            ret_values = [triton.language.core._to_tensor(v, self.builder) for v in ret_value]
            ret_types = [v.type for v in ret_values]
            self.builder.ret([v.handle for v in ret_values])
            return tuple(ret_types)
        else:
            ret = triton.language.core._to_tensor(ret_value, self.builder)
            self.builder.ret([ret_value.handle])
            return ret.type

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
        if isinstance(lhs, triton.language.constexpr):
            lhs = lhs.value
        if isinstance(rhs, triton.language.constexpr):
            rhs = rhs.value
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

    def visit_If(self, node):
        cond = self.visit(node.test)
        if isinstance(cond, triton.language.tensor):
            cond = cond.to(triton.language.int1, _builder=self.builder)
            with enter_sub_region(self) as sr:
                liveins, ip_block = sr

                then_block = self.builder.create_block()
                self.builder.set_insertion_point_to_start(then_block)
                self.visit_compound_statement(node.body)
                then_defs = self.local_defs.copy()

                # when need an else block when:
                # 1. we have an orelse node
                #   or
                # 2. the then block defines new variable
                if then_defs or node.orelse:
                    if node.orelse:
                        self.lscope = liveins
                        self.local_defs = {}
                        else_block = self.builder.create_block()
                        self.builder.set_insertion_point_to_end(else_block)
                        self.visit_compound_statement(node.orelse)
                        else_defs = self.local_defs.copy()
                    else:
                        # collect else_defs
                        else_defs = {}
                        for name in then_defs:
                            if name in liveins:
                                assert self.is_triton_tensor(then_defs[name])
                                assert self.is_triton_tensor(liveins[name])
                                else_defs[name] = liveins[name]
                # collect yields
                names = []
                ret_types = []
                for then_name in then_defs:
                    for else_name in else_defs:
                        if then_name == else_name:
                            if then_defs[then_name].type == else_defs[else_name].type:
                                names.append(then_name)
                                ret_types.append(then_defs[then_name].type)

                self.builder.set_insertion_point_to_end(ip_block)

                if then_defs or node.orelse:  # with else block
                    if_op = self.builder.create_if_op([ty.to_ir(self.builder) for ty in ret_types], cond.handle, True)
                    then_block.merge_block_before(if_op.get_then_block())
                    self.builder.set_insertion_point_to_end(if_op.get_then_block())
                    self.builder.create_yield_op([then_defs[n].handle for n in names])
                    if not node.orelse:
                        else_block = if_op.get_else_block()
                    else:
                        else_block.merge_block_before(if_op.get_else_block())
                    self.builder.set_insertion_point_to_end(if_op.get_else_block())
                    self.builder.create_yield_op([else_defs[n].handle for n in names])
                else:  # no else block
                    if_op = self.builder.create_if_op([ty.to_ir(self.builder) for ty in ret_types], cond.handle, False)
                    then_block.merge_block_before(if_op.get_then_block())

            # update values yielded by IfOp
            for i, name in enumerate(names):
                new_tensor = triton.language.core.tensor(if_op.get_result(i), ret_types[i])
                self.lscope[name] = new_tensor
                self.local_defs[name] = new_tensor

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
        if type(node.op) == ast.Not:
            assert isinstance(op, triton.language.constexpr), "`not` only supported for constexpr at the moment"
            return triton.language.constexpr(not op)
        if isinstance(op, triton.language.constexpr):
            op = op.value
        fn = {
            ast.USub: '__neg__',
            ast.UAdd: '__pos__',
            ast.Invert: '__invert__',
        }[type(node.op)]
        if self.is_triton_tensor(op):
            return getattr(op, fn)(_builder=self.builder)
        return getattr(op, fn)()

    def visit_While(self, node):
        with enter_sub_region(self) as sr:
            liveins, insert_block = sr

            # condition (the before region)
            cond_block = self.builder.create_block()
            self.builder.set_insertion_point_to_start(cond_block)
            cond = self.visit(node.test)

            # loop body (the after region)
            loop_block = self.builder.create_block()
            self.builder.set_insertion_point_to_start(loop_block)
            self.visit_compound_statement(node.body)
            loop_defs = self.local_defs

            # collect loop-carried values
            names = []
            ret_types = []
            init_args = []
            yields = []
            for name in loop_defs:
                if name in liveins:
                    # We should not def new constexpr
                    assert self.is_triton_tensor(loop_defs[name])
                    assert self.is_triton_tensor(liveins[name])
                    if loop_defs[name].type == liveins[name].type:
                        # these are loop-carried values
                        names.append(name)
                        ret_types.append(loop_defs[name].type)
                        init_args.append(liveins[name])
                        yields.append(loop_defs[name])

            self.builder.set_insertion_point_to_end(insert_block)
            while_op = self.builder.create_while_op([ty.to_ir(self.builder) for ty in ret_types],
                                                    [arg.handle for arg in init_args])
            # merge the condition region
            before_block = self.builder.create_block_with_parent(while_op.get_before(),
                                                                 [ty.to_ir(self.builder) for ty in ret_types])
            cond_block.merge_block_before(before_block)
            self.builder.set_insertion_point_to_end(before_block)
            # create CondtionOp: e.g., scf.condition(%cond) %arg0, %arg1, ...
            self.builder.create_condtion_op(cond.handle, [before_block.arg(i) for i in range(len(init_args))])
            # merge the loop body
            after_block = self.builder.create_block_with_parent(while_op.get_after(),
                                                                [ty.to_ir(self.builder) for ty in ret_types])
            loop_block.merge_block_before(after_block)
            self.builder.set_insertion_point_to_end(after_block)
            self.builder.create_yield_op([y.handle for y in yields])

        # update global uses in while_op
        for i, name in enumerate(names):
            before_block.replace_use_in_block_with(init_args[i].handle, before_block.arg(i))
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
        iterator = self.visit(node.iter.func)
        if iterator != self.builtins['range']:
            raise RuntimeError('Only `range` iterator currently supported')
        # static for loops: all iterator arguments are constexpr
        iter_args = [self.visit(arg) for arg in node.iter.args]
        static_unrolling = os.environ.get('TRITON_STATIC_LOOP_UNROLLING', False)
        is_static = False
        if static_unrolling:
            is_static = all([isinstance(x, triton.language.constexpr) for x in iter_args])
        if is_static:
            iter_args = [arg.value for arg in iter_args]
            range = iterator(*iter_args)
            if len(range) <= 10:
                for i in iterator(*iter_args):
                    self.lscope[node.target.id] = triton.language.constexpr(i)
                    self.visit_compound_statement(node.body)
                    for stmt in node.orelse:
                        ast.NodeVisitor.generic_visit(self, stmt)
                return

        # collect lower bound (lb), upper bound (ub), and step
        lb = self.visit(node.iter.args[0] if len(node.iter.args) > 1 else ast.Num(0))
        ub = self.visit(node.iter.args[1] if len(node.iter.args) > 1 else node.iter.args[0])
        step = self.visit(node.iter.args[2] if len(node.iter.args) > 2 else ast.Num(1))
        # lb/ub/step might be constexpr, we need to cast them to tensor
        lb = triton.language.core._to_tensor(lb, self.builder).handle
        ub = triton.language.core._to_tensor(ub, self.builder).handle
        step = triton.language.core._to_tensor(step, self.builder).handle
        # ForOp can only accept IndexType as lb/ub/step. Cast integer to Index
        lb = self.builder.create_to_index(lb)
        ub = self.builder.create_to_index(ub)
        step = self.builder.create_to_index(step)
        # Create placeholder for the loop induction variable
        # We can use any value because the variable isn't a constexpr
        # but use a distinctive value (of the right type) to ease debugging
        st_target = ast.Name(id=node.target.id, ctx=ast.Store())
        init_node = ast.Assign(targets=[st_target], value=ast.Num(value=0xBADF00D))
        self.visit(init_node)

        with enter_sub_region(self) as sr:
            liveins, insert_block = sr

            # create loop body block
            block = self.builder.create_block()
            self.builder.set_insertion_point_to_start(block)

            # visit loop body
            self.visit_compound_statement(node.body)

            # If a variable (name) is defined in both its parent & itself, then it's
            # a loop-carried variable. (They must be of the same type)
            init_args = []
            yields = []
            names = []
            for name in self.local_defs:
                if name in liveins:
                    assert self.is_triton_tensor(self.local_defs[name]), f'{name} is not tensor'
                    assert self.is_triton_tensor(liveins[name])
                    if self.local_defs[name].type == liveins[name].type:
                        names.append(name)
                        init_args.append(triton.language.core._to_tensor(liveins[name], self.builder))
                        yields.append(triton.language.core._to_tensor(self.local_defs[name], self.builder))

            # create ForOp
            self.builder.set_insertion_point_to_end(insert_block)
            for_op = self.builder.create_for_op(lb, ub, step, [arg.handle for arg in init_args])
            block.merge_block_before(for_op.get_body(0))

            # update induction variable with actual value, and replace all uses
            self.builder.set_insertion_point_to_start(for_op.get_body(0))
            iv = self.builder.create_index_to_si(for_op.get_induction_var())
            self.lscope[node.target.id].handle.replace_all_uses_with(iv)
            self.set_value(name, triton.language.core.tensor(iv, triton.language.core.int32))

            # create YieldOp
            self.builder.set_insertion_point_to_end(for_op.get_body(0))
            if len(yields) > 0:
                self.builder.create_yield_op([y.handle for y in yields])
            for_op_region = for_op.get_body(0).get_parent()
            assert for_op_region.size() == 1, "We use SCF, so the loop body should only have one block"
            # replace global uses with block arguments
            for i, name in enumerate(names):
                # arg0 is the induction variable
                for_op.get_body(0).replace_use_in_block_with(init_args[i].handle, for_op.get_body(0).arg(i + 1))

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
                ret_type = triton.language.void
                prototype = triton.language.function_type([ret_type], arg_types)
                gscope = sys.modules[fn.fn.__module__].__dict__
                generator = CodeGenerator(self.builder.context, prototype, gscope, attributes, constants, module=self.module, function_name=fn_name, function_types=self.function_ret_types)
                generator.visit(fn.parse())
                callee_ret_type = generator.last_ret_type
                self.function_ret_types[fn_name] = callee_ret_type
            else:
                callee_ret_type = self.function_ret_types[fn_name]
            symbol = self.module.get_function(fn_name)
            call_op = self.builder.call(symbol, arg_vals)
            if call_op.get_num_results() == 0:
                return None
            elif call_op.get_num_results() == 1:
                return triton.language.tensor(call_op.get_result(0), callee_ret_type)
            else:
                # should return a tuple of tl.tensor
                results = []
                for i in range(call_op.get_num_results()):
                    results.append(triton.language.tensor(call_op.get_result(i), callee_ret_type[i]))
                return tuple(results)
        if hasattr(fn, '__self__') and self.is_triton_tensor(fn.__self__) or \
                sys.modules[fn.__module__] is triton.language.core or \
                isinstance(fn, triton.language.extern.ExternalFunction):
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
        if isinstance(lhs, triton.language.constexpr):
            lhs = lhs.value
        if isinstance(rhs, triton.language.constexpr):
            rhs = rhs.value

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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def make_triton_ir(fn, signature, specialization, constants):
    context = _triton.ir.context()
    context.load_triton()
    # create kernel prototype
    cst_key = lambda i: fn.arg_names.index(i) if isinstance(i, str) else i
    constants = {cst_key(key): value for key, value in constants.items()}
    # visit kernel AST
    gscope = fn.__globals__.copy()
    function_name = '_'.join([fn.__name__, kernel_suffix(signature.values(), specialization)])
    tys = list(signature.values())
    new_constants = {k: True if tys[k] == "i1" else 1 for k in specialization.equal_to_1}
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


def make_tritongpu_ir(mod, num_warps):
    pm = _triton.ir.pass_manager(mod.context)
    pm.add_convert_triton_to_tritongpu_pass(num_warps)
    pm.run(mod)
    return mod


def optimize_tritongpu_ir(mod, num_stages):
    pm = _triton.ir.pass_manager(mod.context)
    pm.enable_debug()
    pm.add_tritongpu_pipeline_pass(num_stages)
    pm.add_canonicalizer_pass()
    pm.add_cse_pass()
    pm.add_coalesce_pass()
    pm.add_triton_gpu_combine_pass()
    pm.add_licm_pass()
    pm.add_triton_gpu_swizzle_pass()
    pm.add_triton_gpu_combine_pass()
    pm.add_cse_pass()
    pm.run(mod)
    return mod


def make_llvm_ir(mod):
    return _triton.translate_triton_gpu_to_llvmir(mod)


def make_ptx(mod: Any, compute_capability: int, ptx_version: int) -> Tuple[str, int]:
    '''
    Translate TritonGPU module to PTX code.
    :param mod: a TritonGPU dialect module
    :return:
        - PTX code
        - shared memory alloaction size
    '''
    return _triton.translate_llvmir_to_ptx(mod, compute_capability, ptx_version)


def make_cubin(ptx: str, ptxas: str, compute_capability: int):
    '''
    Compile TritonGPU module to cubin.
    :param ptx: ptx code
    :param device: CUDA device
    :return: str
    '''
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
    version = major * 1000 + minor * 10
    if version >= 11040:
        return 74
    if version >= 11030:
        return 73
    if version >= 11020:
        return 72
    if version >= 11010:
        return 71
    if version >= 11000:
        return 70
    if version >= 10020:
        return 65
    if version >= 10010:
        return 64
    if version >= 10000:
        return 63
    raise RuntimeError("Triton only support CUDA 10.0 or higher")


def path_to_ptxas():
    prefixes = [os.environ.get("TRITON_PTXAS_PATH", ""), "", os.environ.get('CUDA_PATH', default_cuda_dir())]
    for prefix in prefixes:
        ptxas = os.path.join(prefix, "bin", "ptxas")
        if os.path.exists(ptxas):
            result = subprocess.check_output([ptxas, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return ptxas, version.group(1)
    raise RuntimeError("Cannot find ptxas")


instance_descriptor = namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"], defaults=[set(), set()])


def _compile(fn, signature: str, device: int = -1, constants=dict(), specialization=instance_descriptor(), num_warps: int = 4, num_stages: int = 3, extern_libs=None, output: str = "ttgir") -> Tuple[str, int, str]:
    if isinstance(signature, str):
        signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
    valid_outputs = ("ttir", "ttgir", "ptx", "cubin")
    assert output in valid_outputs, "output should be one of [%s], but get \"%s\"" % (','.join(valid_outputs), output)

    # triton-ir
    module, _ = make_triton_ir(fn, signature, specialization, constants)
    module = optimize_triton_ir(module)
    if output == "ttir":
        return module.str()

    # tritongpu-ir
    module = make_tritongpu_ir(module, num_warps)
    module = optimize_tritongpu_ir(module, num_stages)
    if output == "ttgir":
        return module.str()

    # llvm-ir
    llvm_ir = make_llvm_ir(module)

    assert device >= 0, "device should be provided."
    ptxas, cuda_version = path_to_ptxas()
    compute_capability = torch.cuda.get_device_capability(device)
    compute_capability = compute_capability[0] * 10 + compute_capability[1]
    ptx_version = ptx_get_version(cuda_version)
    ptx = make_ptx(llvm_ir, compute_capability, ptx_version)
    shem_size = _triton.get_shared_memory_size(module)
    kernel_name = ptx_get_kernel_name(ptx)
    if output == "ptx":
        return ptx, shem_size, kernel_name

    cubin = make_cubin(ptx, ptxas, compute_capability)
    if output == "cubin":
        return cubin, ptx, shem_size, kernel_name

    assert False


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

    format = "iiiiiKK" + ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])

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
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &shared_memory, &_stream, &_function, {', '.join(f"&_arg{i}" for i, ty in signature.items())})) {{
    return NULL;
  }}
  _launch(gridX, gridY, gridZ, num_warps, shared_memory, (CUstream)_stream, (CUfunction)_function, {', '.join(f"getPointer(_arg{i},{i})" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});
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


def default_cuda_dir():
    return os.path.join("/usr", "local", "cuda")


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
def libcuda_dir():
    loc = subprocess.check_output(["whereis", "libcuda.so"]).decode().strip().split()[-1]
    return os.path.dirname(loc)


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _build(name, src, srcdir):
    cuda_lib_dir = libcuda_dir()
    cuda_path = os.environ.get('CUDA_PATH', default_cuda_dir())
    cu_include_dir = os.path.join(cuda_path, "include")
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
    ret = subprocess.check_call([cc, src, "-O3", f"-I{cu_include_dir}", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", f"-L{cuda_lib_dir}", "-lcuda", "-o", so])
    if ret == 0:
        return so
    # fallback on setuptools
    extra_compile_args = []
    library_dirs = [cuda_lib_dir]
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


def make_so_cache_key(signature, constants):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{''.join(signature.values())}{constants}"
    key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return key


def make_fn_cache_key(fn_hash, signature, configs, constants, num_warps, num_stages):
    # Get unique key for the compiled code
    get_conf_key = lambda conf: (sorted(conf.divisible_by_16), sorted(conf.equal_to_1))
    configs_key = [get_conf_key(conf) for conf in configs]
    key = f"{fn_hash}-{''.join(signature.values())}-{configs_key}-{constants}-{num_warps}-{num_stages}"
    key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return key


def compile(fn, signature: str, device: int = -1, constants=dict(), num_warps: int = 4, num_stages: int = 3, extern_libs=None, configs=None):
    if isinstance(signature, str):
        signature = {k: v.strip() for k, v in enumerate(signature.split(","))}
    # we get the kernel, i.e. the first function generated in the module
    if configs is None:
        configs = [instance_descriptor()]
    assert len(configs) == 1
    # cache manager
    name = fn.__name__
    # name of files that are cached
    so_cache_key = make_so_cache_key(signature, constants)
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
    if not fn_cache_manager.has_file(cubin_name) or \
       not fn_cache_manager.has_file(data_name) or \
       not fn_cache_manager.has_file(ptx_name):
        cubin, ptx, shared, kernel_name = _compile(fn, signature, device, constants, configs[0], num_warps, num_stages, extern_libs, "cubin")
        metadata = {"name": kernel_name, "shared": shared, "num_warps": num_warps, "num_stages": num_stages}
        fn_cache_manager.put(cubin, cubin_name)
        fn_cache_manager.put(ptx, ptx_name, binary=False)
        fn_cache_manager.put(json.dumps(metadata), data_name, binary=False)

    return CompiledKernel(name, so_cache_manager._make_path(so_name), fn_cache_manager.cache_dir)


class CompiledKernel:

    def __init__(self, fn_name, so_path, cache_dir):

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

        device = torch.cuda.current_device()
        global cuda_utils
        if cuda_utils is None:
            cuda_utils = CudaUtils()
        mod, func, n_regs, n_spills = cuda_utils.load_binary(metadata["name"], self.asm["cubin"], self.shared, device)
        self.cu_module = mod
        self.cu_function = func

    def __getitem__(self, grid):
        def runner(*args, stream=None):
            if stream is None:
                stream = torch.cuda.current_stream().cuda_stream
            self.c_wrapper(grid[0], grid[1], grid[2], self.num_warps, self.shared, stream, self.cu_function, *args)
        return


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def _generate_src(self):
        return """
        #include <cuda.h>

        #include \"cuda.h\"
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

        #define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

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
            Py_BEGIN_ALLOW_THREADS;
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
            Py_END_ALLOW_THREADS;

            if(PyErr_Occurred()) {
              return NULL;
            }
            return Py_BuildValue("(KKii)", (uint64_t)mod, (uint64_t)fun, n_regs, n_spills);
        }

        static PyMethodDef ModuleMethods[] = {
          {"load_binary", loadBinary, METH_VARARGS, "Load provided cubin into CUDA driver"},
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


cuda_utils = None
