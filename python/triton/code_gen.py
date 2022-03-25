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
import time
import warnings
from typing import Dict

import torch
from filelock import FileLock

import triton
import triton._C.libtriton.triton as _triton
from .tools.disasm import extract


class CodeGenerator(ast.NodeVisitor):
    def get_value(self, name):
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
        if isinstance(ret, triton.language.block):
            handle = self.module.get_value(name)
            return triton.language.block(handle)
        return ret

    def set_value(self, name, value):
        if isinstance(value, _triton.ir.value):
            value = triton.language.block(value)
        if isinstance(value, triton.language.block):
            self.module.set_value(name, value.handle)
            self.module.set_type(name, value.handle.type)
        self.lscope[name] = value

    def is_triton_object(self, value):
        return isinstance(value, triton.language.block)

    def visit_compound_statement(self, stmts):
        for stmt in stmts:
            self.last_ret = self.visit(stmt)
            if isinstance(stmt, ast.Return):
                break
        return stmts and isinstance(stmt, ast.Return)

    def __init__(self, context, prototype, gscope, attributes, constants, kwargs):
        self.builder = _triton.ir.builder(context)
        self.module = _triton.ir.module('', self.builder)
        self.prototype = prototype
        self.gscope = gscope
        self.lscope = dict()
        self.attributes = attributes
        self.constants = constants
        self.kwargs = kwargs
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
            return self.builder.ret_void()
        return ret

    def visit_FunctionDef(self, node, inline=False, arg_values=None):
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
        # store keyword arguments in local scope
        self.lscope[kwarg_names] = self.kwargs
        # initialize function
        if inline:
            pass
        else:
            fn = self.module.get_or_insert_function(node.name, self.prototype)
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
                    arg_values.append(fn.args[idx])
                    idx += 1

        for arg_name, arg_value in zip(arg_names, arg_values):
            self.set_value(arg_name, arg_value)
        if inline:
            self.visit_compound_statement(node.body)
            return self.last_ret
        else:
            entry = _triton.ir.basic_block.create(self.builder.context, "entry", fn)
            self.module.seal_block(entry)
            self.builder.set_insert_block(entry)
            # visit function body
            self.visit_compound_statement(node.body)
            # finalize function
            self.builder.ret_void()

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
            if not isinstance(value, triton.language.block):
                value = triton.language.core._to_ir(value, self.builder)
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
        if isinstance(lhs, triton.language.core.constexpr):
            lhs = lhs.value
        if isinstance(rhs, triton.language.core.constexpr):
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
        if self.is_triton_object(lhs):
            return getattr(lhs, fn)(rhs, _builder=self.builder)
        elif self.is_triton_object(rhs):
            fn = fn[:2] + 'r' + fn[2:]
            return getattr(rhs, fn)(lhs, _builder=self.builder)
        else:
            return getattr(lhs, fn)(rhs)

    def visit_If(self, node):
        cond = self.visit(node.test)
        if isinstance(cond, triton.language.block):
            cond = cond.to(triton.language.int1, _builder=self.builder)
            current_bb = self.builder.get_insert_block()
            then_bb = _triton.ir.basic_block.create(self.builder.context, "then", current_bb.parent)
            else_bb = _triton.ir.basic_block.create(self.builder.context, "else", current_bb.parent) if node.orelse else None
            endif_bb = _triton.ir.basic_block.create(self.builder.context, "endif", current_bb.parent)
            self.module.seal_block(then_bb)
            if else_bb:
                self.module.seal_block(else_bb)
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
            self.module.seal_block(endif_bb)
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
        if isinstance(lhs, triton.language.core.constexpr):
            lhs = lhs.value
        if isinstance(rhs, triton.language.core.constexpr):
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
        if self.is_triton_object(lhs):
            return getattr(lhs, fn)(rhs, _builder=self.builder)
        elif self.is_triton_object(rhs):
            fn = fn[:2] + 'r' + fn[2:]
            return getattr(rhs, fn)(lhs, _builder=self.builder)
        else:
            return getattr(lhs, fn)(rhs)

    def visit_UnaryOp(self, node):
        op = self.visit(node.operand)
        if type(node.op) == ast.Not:
            assert isinstance(op, triton.language.constexpr), "`not` only supported for constexpr at the moment"
            return triton.language.constexpr(not op)
        if isinstance(op, triton.language.core.constexpr):
            op = op.value
        fn = {
            ast.USub: '__neg__',
            ast.UAdd: '__pos__',
            ast.Invert: '__invert__',
        }[type(node.op)]
        if self.is_triton_object(op):
            return getattr(op, fn)(_builder=self.builder)
        return getattr(op, fn)()

    def visit_While(self, node):
        current_bb = self.builder.get_insert_block()
        loop_bb = _triton.ir.basic_block.create(self.module.builder.context, "loop", current_bb.parent)
        next_bb = _triton.ir.basic_block.create(self.module.builder.context, "postloop", current_bb.parent)

        def continue_fn():
            cond = self.visit(node.test)
            return self.builder.cond_br(cond.handle, loop_bb, next_bb)

        continue_fn()
        self.builder.set_insert_block(loop_bb)
        self.visit_compound_statement(node.body)
        continue_fn()
        stop_bb = self.builder.get_insert_block()
        self.module.seal_block(stop_bb)
        self.module.seal_block(loop_bb)
        self.module.seal_block(next_bb)
        self.builder.set_insert_block(next_bb)

        for stmt in node.orelse:
            ast.NodeVisitor.generic_visit(self, stmt)

    def visit_Subscript(self, node):
        assert node.ctx.__class__.__name__ == "Load"
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        if self.is_triton_object(lhs):
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
        is_static = all([isinstance(x, triton.language.constexpr) for x in iter_args])
        if is_static:
            st_target = ast.Name(id=node.target.id, ctx=ast.Store())
            iter_args = [arg.value for arg in iter_args]
            range = iterator(*iter_args)
            if len(range) <= 10:
                for i in iterator(*iter_args):
                    self.lscope[node.target.id] = triton.language.constexpr(i)
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
        init_node = ast.Assign(targets=[st_target], value=arg_0)
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
        loop_bb = _triton.ir.basic_block.create(self.module.builder.context, "loop", current_bb.parent)
        next_bb = _triton.ir.basic_block.create(self.module.builder.context, "postloop", current_bb.parent)

        def continue_fn():
            self.visit(step_node)
            cond = build_cond()
            return self.builder.cond_br(cond.handle, loop_bb, next_bb)

        self.visit(init_node)
        cond = build_cond()
        self.builder.cond_br(cond.handle, loop_bb, next_bb)
        self.builder.set_insert_block(loop_bb)
        self.visit_compound_statement(node.body)
        # TODO: handle case where body breaks control flow
        continue_fn()
        stop_bb = self.builder.get_insert_block()
        self.module.seal_block(stop_bb)
        self.module.seal_block(loop_bb)
        self.module.seal_block(next_bb)
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
            return fn(*args, generator=self, **kws)
        if hasattr(fn, '__self__') and self.is_triton_object(fn.__self__) or \
                sys.modules[fn.__module__] is triton.language.core:
            return fn(*args, _builder=self.builder, **kws)
        if fn in self.builtins.values():
            args = [arg.value if isinstance(arg, triton.language.constexpr) else arg
                    for arg in args]
        return fn(*args, **kws)

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
        module, kernel = _triton.code_gen.load_binary(bin.backend,
                                                      bin.name,
                                                      bin.asm,
                                                      bin.shared_mem,
                                                      device)
        self.bin = bin
        self.asm = bin.asm
        self.sass = ''
        self.module = module
        self.kernel = kernel
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
        if isinstance(obj, triton.language.core.constexpr):
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
    def _to_triton_ir(context, obj):
        which, name = obj
        type_map = {
            'I': _triton.ir.type.get_int32,
            'L': _triton.ir.type.get_int64,
            'f': _triton.ir.type.get_fp32,
            'B': _triton.ir.type.get_int1,
            'f8': _triton.ir.type.get_fp8,
            'f16': _triton.ir.type.get_fp16,
            'bf16': _triton.ir.type.get_bf16,
            'f32': _triton.ir.type.get_fp32,
            'f64': _triton.ir.type.get_fp64,
            'i1': _triton.ir.type.get_int1,
            'i8': _triton.ir.type.get_int8,
            'i16': _triton.ir.type.get_int16,
            'i32': _triton.ir.type.get_int32,
            'i64': _triton.ir.type.get_int64,
            'u8': _triton.ir.type.get_uint8,
            'u16': _triton.ir.type.get_uint16,
            'u32': _triton.ir.type.get_uint32,
            'u64': _triton.ir.type.get_uint64,
        }
        # convert torch.Tensor to Triton IR pointers
        if which == 'ptr':
            elt_ty = type_map[name](context)
            return _triton.ir.type.make_ptr(elt_ty, 1)
        # default path returns triton.ir.type directly
        return type_map[name](context)

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

    def add_to_cache(self, key, wargs, device_idx, num_warps, num_stages):
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
        return self.fn._warmup(key, arg_types=arg_types, device=device_idx, attributes=attributes, constants=constants, num_warps=num_warps, num_stages=num_stages, is_manual_warmup=False)

    def __call__(self, *wargs, grid, num_warps=4, num_stages=2, **kwargs):
        # handle arguments passed by name
        kwargs = {self.fn.arg_names.index(name): value for name, value in kwargs.items()}
        wargs = list(wargs)
        for i, pos in enumerate(sorted(kwargs)):
            wargs.insert(pos + i, kwargs[pos])
        if len(wargs) != len(self.fn.arg_names):
            raise TypeError(f"Function takes {len(self.fn.arg_names)} positional arguments but {len(wargs)} were given")
        # handle annotations
        for pos, _type in self.fn.annotations.items():
            wargs[pos] = _type(wargs[pos])
        # check that tensors are on GPU.
        for arg in wargs:
            if hasattr(arg, 'data_ptr'):
                assert arg.is_cuda, "All tensors must be on GPU!"
        # query device index and cuda stream
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        cc = torch.cuda.get_device_capability(device)
        cc = str(cc[0]) + '-' + str(cc[1])
        # # query stream
        # # this is hacky but much faster than `torch.cuda.current_stream(device).cuda_stream`
        # # https://github.com/pytorch/pytorch/blob/master/c10/core/Stream.h#L154
        # # building a C wrapper to re-use the unpack function would add a build-time torch dependency
        # # and require different wheels for different torch versions -- undesirable!
        # bits = torch._C._cuda_getCurrentStream(device)
        # mask = 1 << 47
        # stream = ((bits & 0xFFFFFFFFFFFF) ^ mask) - mask
        stream = torch.cuda.current_stream(device).cuda_stream
        # make key for cache
        return _triton.runtime.launch(wargs, self.fn.do_not_specialize, self.fn.cache_key + cc, self.fn.arg_names, device, stream,
                                      self.fn.bin_cache, num_warps, num_stages, self.add_to_cache, grid)


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


@functools.lru_cache()
def version_key():
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
    return '-'.join(triton.__version__) + '-' + ptxas_version + '-' + '-'.join(contents)


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


class JITFunction:

    cache_hook = None

    def __init__(self, fn, version=None, do_not_specialize=None):
        # information of wrapped function
        self.fn = fn
        self.module = fn.__module__
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]
        self.arg_defaults = [v.default for v in signature.parameters.values()]

        self.version = version
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

    def __call__(self, *args, generator: CodeGenerator, **kwargs):
        try:
            from inspect import getcallargs
            arg_values = getcallargs(self.fn, *args, **kwargs)
            arg_values = [arg_values[name] for name in self.arg_names]
            arg_values = [arg if isinstance(arg, triton.language.block)
                          else triton.language.constexpr(arg) for arg in arg_values]

            gscope = generator.gscope.copy()
            lscope = generator.lscope.copy()
            values = generator.module.get_values().copy()
            types = generator.module.get_types().copy()
            generator.gscope = sys.modules[self.fn.__module__].__dict__
            generator.lscope = dict()
            ret = generator.visit_FunctionDef(self.parse().body[0], inline=True, arg_values=arg_values)
            generator.gscope = gscope
            generator.lscope = lscope
            generator.module.set_values(values)
            generator.module.set_types(types)
            return ret
        except Exception as e:
            node = generator.last_node
            if node is None or isinstance(e, (NotImplementedError, CompilationError)):
                raise e
            raise CompilationError(self.src, node) from e

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

    def _warmup(self, key, arg_types, device, attributes, constants, num_warps, num_stages, is_manual_warmup):
        hashed_key = hashlib.md5(key.encode("utf-8")).hexdigest()

        # create cache directory
        cache_dir = os.environ.get('TRITON_CACHE_DIR', '/tmp/triton/')
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

        compile = dict(arg_types=arg_types, device=device, attributes=attributes, constants=constants, num_warps=num_warps, num_stages=num_stages)
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

    def _compile(self, arg_types, device, attributes, constants, num_warps, num_stages):
        # create IR module
        context = _triton.ir.context()
        # get just-in-time proto-type of kernel
        arg_types = [Kernel._to_triton_ir(context, arg) for arg in arg_types]
        ret_type = _triton.ir.type.get_void(context)
        prototype = _triton.ir.type.make_function(ret_type, arg_types)
        # generate Triton-IR
        # export symbols visible from self into code-generator object
        gscope = self.__globals__
        generator = CodeGenerator(context, prototype, gscope=gscope, attributes=attributes, constants=constants, kwargs=dict())
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
        name, asm, shared_mem = _triton.code_gen.compile_ttir(backend, generator.module, device, num_warps, num_stages)
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
