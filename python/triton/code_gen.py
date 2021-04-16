import inspect
import struct
import enum
import types
import torch
import ast
import builtins
import triton._C.libtriton.triton as _triton
import triton
import sys
import textwrap
from abc import ABC, abstractmethod


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
        if isinstance(ret, triton.block):
            handle = self.module.get_value(name)
            return triton.block(handle)
        return ret

    def set_value(self, name, value):
        if isinstance(value, _triton.ir.value):
            value = triton.block(value)
        if isinstance(value, triton.block):
            self.module.set_value(name, value.handle)
            self.module.scope.set_type(name, value.type)
        self.lscope[name] = value

    def is_triton_object(self, value):
        return isinstance(value, triton.block)

    def visit_compound_statement(self, stmts, add_scope=False):
        if add_scope:
            self.module.add_new_scope()
        for stmt in stmts:
            self.last_ret = ast.NodeVisitor.visit(self, stmt)
            if isinstance(stmt, ast.Return):
                break
        if add_scope:
            self.module.pop_scope()
        return self.last_ret

    def __init__(self, context, prototype, gscope, attributes, constants, kwargs):
        self.builder = _triton.ir.builder(context)
        self.module = _triton.ir.module('', self.builder)
        self.prototype = prototype
        self.gscope = gscope
        self.lscope = dict()
        self.attributes = attributes
        self.constants = constants
        self.kwargs = kwargs
        self.builtins = {'range': range, 'min': triton.minimum, 'float': float, 'int': int, 'getattr': getattr}

    def visit_Module(self, node):
        self.module.add_new_scope()
        ast.NodeVisitor.generic_visit(self, node)
        self.module.pop_scope()

    # By design, only non-kernel functions can return
    def visit_Return(self, node):
        return ast.NodeVisitor.visit(self, node.value)

    def visit_FunctionDef(self, node, inline=False, arg_values=None):
        arg_names, kwarg_names = ast.NodeVisitor.visit(self, node.args)
        # store keyword arguments in local scope
        self.lscope[kwarg_names] = self.kwargs
        # initialize function
        if inline:
            pass
        else:
            fn = self.module.get_or_insert_function(node.name, self.prototype)
            arg_values = []
            for i, arg_name in enumerate(arg_names):
                if i in self.constants:
                    arg_values.append(self.constants[i])
                else:
                    if i in self.attributes:
                        is_ptr = fn.args[i].type.is_ptr()
                        attr = 'aligned' if is_ptr else 'multiple_of'
                        attr = getattr(_triton.ir.attribute_kind, attr)
                        attr = _triton.ir.attribute(attr, self.attributes[i])
                        fn.add_attr(i + 1, attr)
                    fn.args[i].name = arg_name
                    arg_values.append(fn.args[i])
        for arg_name, arg_value in zip(arg_names, arg_values):
            self.set_value(arg_name, arg_value)
        if inline:
            return self.visit_compound_statement(node.body, add_scope=True)
        else:
            entry = _triton.ir.basic_block.create(self.builder.context, "entry", fn)
            self.module.seal_block(entry)
            self.builder.set_insert_block(entry)
            # visit function body
            self.visit_compound_statement(node.body, add_scope=True)
            # finalize function
            self.builder.ret_void()

    def visit_arguments(self, node):
        arg_names = []
        for arg in node.args:
            arg_names += [ast.NodeVisitor.visit(self, arg)]
        kwarg_names = ast.NodeVisitor.visit(self, node.kwarg)
        return arg_names, kwarg_names

    def visit_arg(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        return node.arg

    def visit_Assign(self, node):
        names = []
        for target in node.targets:
            names += [ast.NodeVisitor.visit(self, target)]
        assert len(names) == 1
        name = names[0]
        value = ast.NodeVisitor.visit(self, node.value)
        self.set_value(names[0], value)

    def visit_AugAssign(self, node):
        name = node.target.id
        lhs = ast.Name(id=name, ctx=ast.Load())
        rhs = ast.BinOp(lhs, node.op, node.value)
        assign = ast.Assign(targets=[node.target], value=rhs)
        ast.NodeVisitor.visit(self, assign)
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
        args = [ast.NodeVisitor.visit(self, x) for x in node.elts]
        return tuple(args)

    def visit_BinOp(self, node):
        lhs = ast.NodeVisitor.visit(self, node.left)
        rhs = ast.NodeVisitor.visit(self, node.right)
        fn = {
            ast.Add: '__add__',
            ast.Sub: '__sub__',
            ast.Mult: '__mul__',
            ast.MatMult: '__mul__',
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
        kws = dict()
        if self.is_triton_object(lhs) or self.is_triton_object(rhs):
            lhs, rhs = triton.broadcast(lhs, rhs, builder=self.builder)
            kws['builder'] = self.builder
        ret = getattr(lhs, fn)(rhs, **kws)
        if ret is NotImplemented:
            fn = fn[:2] + 'r' + fn[2:]
            ret = getattr(rhs, fn)(lhs, **kws)
        return ret

    def visit_If(self, node):
        cond = ast.NodeVisitor.visit(self, node.test)
        if self.is_triton_object(cond):
            current_bb = self.builder.get_insert_block()
            then_bb = _triton.ir.basic_block.create(self.builder.context, "then", current_bb.parent)
            else_bb = _triton.ir.basic_block.create(self.builder.context, "else", current_bb.parent) if node.orelse else None
            endif_bb = _triton.ir.basic_block.create(self.builder.context, "endif", current_bb.parent)
            self.module.seal_block(then_bb)
            if else_bb:
                self.module.seal_block(else_bb)
                self.builder.cond_br(cond, then_bb, else_bb)
            else:
                self.builder.cond_br(cond, then_bb, endif_bb)
            self.builder.set_insert_block(then_bb)
            self.visit_compound_statement(node.body, add_scope=True)
            # TODO: last statement is a terminator?
            self.builder.br(endif_bb)
            if else_bb:
                self.builder.set_insert_block(else_bb)
                self.visit_compound_statement(node.orelse, add_scope=True)
                #TODO: last statement is a terminator?
                self.builder.br(endif_bb)
            self.module.seal_block(endif_bb)
            self.builder.set_insert_block(endif_bb)
        else:
            if cond:
                self.visit_compound_statement(node.body)
            else:
                self.visit_compound_statement(node.orelse)

    def visit_IfExp(self, node):
        cond = ast.NodeVisitor.visit(self, node.test)
        if cond:
            return ast.NodeVisitor.visit(self, node.body)
        else:
            return ast.NodeVisitor.visit(self, node.orelse)

    def visit_Pass(self, node):
        pass

    def visit_Compare(self, node):
        assert len(node.comparators) == 1
        assert len(node.ops) == 1
        lhs = ast.NodeVisitor.visit(self, node.left)
        rhs = ast.NodeVisitor.visit(self, node.comparators[0])
        fn = {
            ast.Eq: '__eq__',
            ast.NotEq: '__ne__',
            ast.Lt: '__lt__',
            ast.LtE: '__le__',
            ast.Gt: '__gt__',
            ast.GtE: '__ge__',
            ast.Is: '__eq__',
            ast.IsNot: '__ne__',
        }[type(node.ops[0])]
        if self.is_triton_object(lhs) or self.is_triton_object(rhs):
            lhs, rhs = triton.broadcast(lhs, rhs, builder=self.builder)
            return getattr(lhs, fn)(rhs, builder=self.builder)
        return getattr(lhs, fn)(rhs)

    def visit_UnaryOp(self, node):
        # Handle minus operator
        if type(node.op) == ast.USub:
            tmp = ast.BinOp(ast.Num(0), ast.Sub(), node.operand)
            return ast.NodeVisitor.visit(self, tmp)
        raise NotImplementedError(f"Unsupported op: {node.op}")

    def visit_While(self, node):
        current_bb = self.builder.get_insert_block()
        loop_bb = _triton.ir.basic_block.create(self.module.builder.context, "loop", current_bb.parent)
        next_bb = _triton.ir.basic_block.create(self.module.builder.context, "postloop", current_bb.parent)

        def continue_fn():
            cond = ast.NodeVisitor.visit(self, node.test)
            return self.builder.cond_br(cond, loop_bb, next_bb)

        continue_fn()
        self.builder.set_insert_block(loop_bb)
        self.visit_compound_statement(node.body, add_scope=True)
        continue_fn()
        stop_bb = self.builder.get_insert_block()
        self.module.seal_block(stop_bb)
        self.module.seal_block(loop_bb)
        self.module.seal_block(next_bb)
        self.builder.set_insert_block(next_bb)

        for stmt in node.orelse:
            ast.NodeVisitor.generic_visit(self, stmt)

    def visit_Str(self, node):
        return ast.literal_eval(node)

    def visit_Subscript(self, node):
        assert node.ctx.__class__.__name__ == "Load"
        lhs = ast.NodeVisitor.visit(self, node.value)
        slices = ast.NodeVisitor.visit(self, node.slice)
        if self.is_triton_object(lhs):
            return lhs.__getitem__(slices, builder=self.builder)
        return lhs[slices]

    def visit_ExtSlice(self, node):
        return [ast.NodeVisitor.visit(self, dim) for dim in node.dims]

    def visit_For(self, node):
        iterator = ast.NodeVisitor.visit(self, node.iter.func)
        assert iterator == self.builtins['range']
        # create nodes
        st_target = ast.Name(id=node.target.id, ctx=ast.Store())
        ld_target = ast.Name(id=node.target.id, ctx=ast.Load())
        init_node = ast.Assign(targets=[st_target], value=node.iter.args[0])
        pos_cond_node = ast.Compare(ld_target, [ast.Lt()], [node.iter.args[1]])
        neg_cond_node = ast.Compare(ld_target, [ast.Gt()], [node.iter.args[1]])
        pos_step_node = ast.Compare(node.iter.args[2], [ast.Gt()], [ast.Num(0)])
        build_cond = lambda: triton.where(ast.NodeVisitor.visit(self, pos_step_node),\
                                    ast.NodeVisitor.visit(self, pos_cond_node),\
                                    ast.NodeVisitor.visit(self, neg_cond_node),\
                                    builder=self.builder)
        #cond_node = neg_cond_node
        step_node = ast.AugAssign(target=st_target, op=ast.Add(), value=node.iter.args[2])
        # code generation
        current_bb = self.builder.get_insert_block()
        loop_bb = _triton.ir.basic_block.create(self.module.builder.context, "loop", current_bb.parent)
        next_bb = _triton.ir.basic_block.create(self.module.builder.context, "postloop", current_bb.parent)

        def continue_fn():
            ast.NodeVisitor.visit(self, step_node)
            cond = build_cond()
            return self.builder.cond_br(cond, loop_bb, next_bb)

        ast.NodeVisitor.visit(self, init_node)
        cond = build_cond()
        self.builder.cond_br(cond, loop_bb, next_bb)
        self.builder.set_insert_block(loop_bb)
        self.visit_compound_statement(node.body, add_scope=True)
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
        lower = ast.NodeVisitor.visit(self, node.lower)
        upper = ast.NodeVisitor.visit(self, node.upper)
        step = ast.NodeVisitor.visit(self, node.step)
        return (lower, upper, step)

    def visit_Index(self, node):
        return ast.NodeVisitor.visit(self, node.value)

    def visit_NameConstant(self, node):
        return ast.NodeVisitor.visit(self, node.value)

    def visit_keyword(self, node):
        return {node.arg: ast.NodeVisitor.visit(self, node.value)}

    def visit_Call(self, node):
        fn = ast.NodeVisitor.visit(self, node.func)
        kws = dict()
        for keyword in node.keywords:
            kws.update(ast.NodeVisitor.visit(self, keyword))
        args = [ast.NodeVisitor.visit(self, arg) for arg in node.args]
        if isinstance(fn, JITFunction):
            return fn(*args, generator=self, **kws)
        if hasattr(fn, '__self__') and self.is_triton_object(fn.__self__) or \
            sys.modules[fn.__module__] is triton.core:
            return fn(*args, builder=self.builder, **kws)
        return fn(*args, **kws)

    def visit_Num(self, node):
        return node.n

    def visit_Attribute(self, node):
        lhs = ast.NodeVisitor.visit(self, node.value)
        return getattr(lhs, node.attr)

    def visit_Expr(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_NoneType(self, node):
        return None

    def generic_visit(self, node):
        typename = type(node).__name__
        raise NotImplementedError("Unsupported node: {}".format(typename))


class Binary:
    def __init__(self, module, kernel, num_warps, shared_mem):
        self.module = module
        self.kernel = kernel
        self.shared_mem = shared_mem
        self.num_warps = num_warps

    def __call__(self, stream, args, grid_0, grid_1=1, grid_2=1):
        stream.enqueue(self.kernel, grid_0, grid_1, grid_2, self.num_warps * 32, 1, 1, args, self.shared_mem)


class Kernel:

    type_names = {
        int: 'I',
        float: 'f',
        bool: 'B',
        torch.float16: 'f16',
        torch.float32: 'f32',
        torch.float64: 'f64',
        torch.bool: 'i1',
        torch.int8: 'i8',
        torch.int16: 'i16',
        torch.int32: 'i32',
        torch.int64: 'i64',
    }

    @staticmethod
    def _to_triton_ir(context, obj):
        type_map = {
            'I': _triton.ir.type.get_int32,
            'f': _triton.ir.type.get_fp32,
            'B': _triton.ir.type.get_int1,
            'f16': _triton.ir.type.get_fp16,
            'f32': _triton.ir.type.get_fp32,
            'f64': _triton.ir.type.get_fp64,
            'i1': _triton.ir.type.get_int1,
            'i8': _triton.ir.type.get_int8,
            'i16': _triton.ir.type.get_int16,
            'i32': _triton.ir.type.get_int32,
            'i64': _triton.ir.type.get_int64,
        }
        # convert torch.Tensor to Triton IR pointers
        if isinstance(obj, torch.Tensor):
            name = Kernel.type_names[obj.dtype]
            elt_ty = type_map[name](context)
            return _triton.ir.type.make_ptr(elt_ty, 1)
        # default path returns triton.ir.type directly
        name = Kernel.type_names[obj.__class__]
        return type_map[name](context)

    @staticmethod
    def _types_key(*wargs, tensor_idxs):
        # type inference
        types_key = [None] * len(wargs)
        for i, arg in enumerate(wargs):
            prefix = 'P' if i in tensor_idxs else ''
            suffix = Kernel.type_names[arg.dtype] if i in tensor_idxs else Kernel.type_names[arg.__class__]
            types_key[i] = prefix + suffix
        return tuple(types_key)

    @staticmethod
    def pow2_divisor(N):
        if N % 16 == 0: return 16
        if N % 8 == 0: return 8
        if N % 4 == 0: return 4
        if N % 2 == 0: return 2
        return 1

    def __init__(self, fn, grid):
        self.fn = fn
        self.grid = grid

    def _compile(self, *wargs, device, attributes, constants, num_warps, **meta):
        # create IR module
        context = _triton.ir.context()
        # get just-in-time proto-type of kernel
        arg_types = [Kernel._to_triton_ir(context, arg) for arg in wargs]
        ret_type = _triton.ir.type.get_void(context)
        prototype = _triton.ir.type.make_function(ret_type, arg_types)
        # generate Triton-IR
        # export symbols visible from self.fn into code-generator object
        gscope = sys.modules[self.fn.module].__dict__
        generator = CodeGenerator(context, prototype, gscope=gscope, attributes=attributes, constants=constants, kwargs=meta)
        generator.visit(self.fn.parse())
        tt_device = _triton.driver.cu_device(device.index, False)
        # Compile to machine code
        mod, ker, shared_mem = _triton.code_gen.add_passes_to_emit_bin(generator.module, tt_device, num_warps)
        return Binary(mod, ker, num_warps, shared_mem)

    def __call__(self, *wargs, num_warps, **meta):
        # device inference
        tensor_idxs = [i for i, arg in enumerate(wargs) if isinstance(arg, torch.Tensor)]
        if len(tensor_idxs) == 0:
            raise ValueError("No Tensor argument found.")
        device = wargs[tensor_idxs[0]].device
        # attributes
        args = [arg.data_ptr() if i in tensor_idxs else arg for i, arg in enumerate(wargs)]
        attributes = {i: Kernel.pow2_divisor(a) for i, a in enumerate(args) if isinstance(a, int)}
        # transforms ints whose value is one into constants for just-in-time compilation
        constants = {i: arg for i, arg in enumerate(wargs) if isinstance(arg, int) and arg == 1}
        # determine if we need to re-compile
        types_key = Kernel._types_key(*wargs, tensor_idxs=tensor_idxs)
        attr_key = frozenset(attributes.items())
        meta_key = frozenset(meta.items())
        const_key = frozenset(constants.items())
        key = (device.type, device.index, types_key, attr_key, num_warps, meta_key, const_key)
        cache = self.fn.cache
        if key not in cache:
            # compile and cache configuration if necessary
            cache[key] = self._compile(
                *wargs, device=device, attributes=attributes, num_warps=num_warps, constants=constants, **meta
            )
        # pack arguments
        fmt = ''.join(['P' if i in tensor_idxs else Kernel.type_names[arg.__class__] for i, arg in enumerate(wargs)])
        params = struct.pack(fmt, *args)
        # enqueue cached function into stream
        binary = cache[key]
        cu_stream = torch.cuda.current_stream(device.index).cuda_stream
        stream = _triton.driver.cu_stream(cu_stream, False)
        grid = self.grid(meta) if hasattr(self.grid, '__call__') else self.grid
        binary(stream, params, *grid)


class Autotuner:
    def __init__(self, kernel, arg_names, configs, key):
        if not configs:
            self.configs = [Config(dict(), num_warps=4)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.cache = dict()
        self.kernel = kernel

    def _bench(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.meta.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.meta)
        kernel_call = lambda: self.kernel(*args, num_warps=config.num_warps, **current)
        return triton.testing.do_bench(kernel_call)

    def __call__(self, *args, **meta):
        if len(self.configs) > 1:
            key = tuple([args[i] for i in self.key_idx])
            if key not in self.cache:
                timings = {config: self._bench(*args, config=config, **meta) \
                        for config in self.configs}
                self.cache[key] = builtins.min(timings, key=timings.get)
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.kernel(*args, num_warps=config.num_warps, **meta, **config.meta)


class JITFunction:
    def __init__(self, fn):
        self.module = fn.__module__
        self.arg_names = inspect.getfullargspec(fn).args
        self.cache = dict()
        self.kernel_decorators = []
        self.src = textwrap.dedent(inspect.getsource(fn))

    # we do not parse in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Some unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def __call__(self, *args, generator: CodeGenerator, **meta):
        return generator.visit_FunctionDef(self.parse().body[0], inline=True, arg_values=args)

    def __getitem__(self, grid_fn):
        kernel = Kernel(self, grid_fn)
        for decorator in reversed(self.kernel_decorators):
            kernel = decorator(kernel)
        return kernel


class Config:
    def __init__(self, meta, num_warps=4):
        self.meta = meta
        self.num_warps = num_warps


def autotune(configs, key):
    def decorator(fn):
        def wrapper(kernel):
            return Autotuner(kernel, fn.arg_names, configs, key)

        fn.kernel_decorators.append(wrapper)
        return fn

    return decorator


def heuristics(values):
    def decorator(fn):
        def wrapper(kernel):
            def fun(*args, **meta):
                for v, heur in values.items():
                    assert v not in meta
                    meta[v] = heur(*args, **meta)
                return kernel(*args, **meta)

            return fun

        fn.kernel_decorators.append(wrapper)
        return fn

    return decorator


def jit(fn):
    return JITFunction(fn)
