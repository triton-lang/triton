import ast
import builtins
import functools
import inspect
import struct
import sys
import textwrap
import hashlib
import os
import pickle
import subprocess
import os
from .tools.disasm import extract
import torch
import triton
import triton._C.libtriton.triton as _triton
from filelock import FileLock
import dbm


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
                    cst = triton.language.core._to_ir(self.constants[i], self.builder)
                    arg_values.append(cst)
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
        kws = dict()

        if self.is_triton_object(lhs):
            kws['_builder'] = self.builder
        ret = getattr(lhs, fn)(rhs, **kws)
        if ret is NotImplemented:
            if self.is_triton_object(rhs):
                kws['_builder'] = self.builder
            fn = fn[:2] + 'r' + fn[2:]
            ret = getattr(rhs, fn)(lhs, **kws)
        return ret

    def visit_If(self, node):
        cond = self.visit(node.test)
        if self.is_triton_object(cond):
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
                #TODO: last statement is a terminator?
                if not is_terminator:
                    self.builder.br(endif_bb)
            self.module.seal_block(endif_bb)
            self.builder.set_insert_block(endif_bb)
        else:
            if cond:
                self.visit_compound_statement(node.body)
            else:
                self.visit_compound_statement(node.orelse)

    def visit_IfExp(self, node):
        cond = self.visit(node.test)
        if cond:
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
        if self.is_triton_object(lhs):
            return getattr(lhs, fn)(rhs, _builder=self.builder)
        elif self.is_triton_object(rhs):
            fn = fn[:2] + 'r' + fn[2:]
            return getattr(rhs, fn)(lhs, _builder=self.builder)
        else:
            return getattr(lhs, fn)(rhs)

    def visit_UnaryOp(self, node):
        op = self.visit(node.operand)
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

    def visit_Str(self, node):
        return ast.literal_eval(node)

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
        build_cond = lambda: triton.language.where(self.visit(pos_step_node),\
                                    self.visit(pos_cond_node),\
                                    self.visit(neg_cond_node),\
                                    _builder=self.builder)
        #cond_node = neg_cond_node
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

    def visit_NameConstant(self, node):
        return node.value

    def visit_keyword(self, node):
        return {node.arg: self.visit(node.value)}

    def visit_Call(self, node):
        fn = self.visit(node.func)
        kws = dict()
        for keyword in node.keywords:
            kws.update(self.visit(keyword))
        args = [self.visit(arg) for arg in node.args]
        if isinstance(fn, JITFunction):
            return fn(*args, generator=self, **kws)
        if hasattr(fn, '__self__') and self.is_triton_object(fn.__self__) or \
            sys.modules[fn.__module__] is triton.language.core:
            return fn(*args, _builder=self.builder, **kws)
        return fn(*args, **kws)

    def visit_Num(self, node):
        return node.n

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
        self.module = module
        self.kernel = kernel
        self.device = device

    def __call__(self, stream, args, grid_0, grid_1=1, grid_2=1):
        _triton.runtime.enqueue(self.bin.backend, stream, self.kernel,
                                grid_0, grid_1, grid_2, 
                                self.bin.num_warps * 32, 1, 1, 
                                args, self.bin.shared_mem)


class CompilationError(Exception):
    def __init__(self, src, node, err):
        self.message = '\n'.join(src.split('\n')[:node.lineno])
        self.message += '\n' + ' ' * node.col_offset + '^'
        self.message += '\n Error: ' + str(err)
        super().__init__(self.message)


class OutOfResources(Exception):
    def __init__(self, required, limit, name):
        self.message = f'out of resource: {name}'\
                       f'Required: {required}'\
                       f'Hardware limit: {limit}'
        super().__init__(self.message)


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
        }
        if hasattr(obj, 'data_ptr'):
            return type_names[obj.dtype]
        if isinstance(obj, int):
            if abs(obj) <= 0xffffffff:
                return 'I'
            return 'L'
        if isinstance(obj, float):
            return 'f'
        if isinstance(obj, bool):
            return 'B'
        assert False

        

    @staticmethod
    def _to_triton_ir(context, obj):
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
        }
        # convert torch.Tensor to Triton IR pointers
        if hasattr(obj, 'data_ptr'):
            name = Kernel._type_name(obj)
            elt_ty = type_map[name](context)
            return _triton.ir.type.make_ptr(elt_ty, 1)
        # default path returns triton.ir.type directly
        name = Kernel._type_name(obj)
        return type_map[name](context)

    @staticmethod
    def _types_key(*wargs, tensor_idxs):
        # type inference
        types_key = [None] * len(wargs)
        for i, arg in enumerate(wargs):
            prefix = 'P' if i in tensor_idxs else ''
            suffix = Kernel._type_name(arg) if i in tensor_idxs else Kernel._type_name(arg)
            types_key[i] = prefix + suffix
        return tuple(types_key)

    @staticmethod
    def pow2_divisor(N):
        if N % 16 == 0: return 16
        if N % 8 == 0: return 8
        if N % 4 == 0: return 4
        if N % 2 == 0: return 2
        return 1

    def __init__(self, fn):
        self.fn = fn

    def _compile(self, *wargs, device, attributes, constants, num_warps, num_stages, force_nc_cache, **meta):
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
        try:
            generator.visit(self.fn.parse())
        except Exception as e:
            node = generator.last_node
            if node is None or isinstance(e, (NotImplementedError, CompilationError)):
                raise e
            raise CompilationError(self.fn.src, node, e)
        # Compile to machine code
        if torch.version.hip is None:
            backend = _triton.runtime.backend.CUDA
        else:
            backend = _triton.runtime.backend.ROCM
        name, asm, shared_mem = _triton.code_gen.compile_ttir(backend, generator.module, device, num_warps, num_stages, force_nc_cache)
        max_shared_memory = _triton.runtime.max_shared_memory(backend, device)
        if shared_mem > max_shared_memory:
            raise OutOfResources(shared_mem, max_shared_memory, "shared memory")
        return Binary(backend, name, asm, shared_mem, num_warps)

    def __call__(self, *wargs, grid, num_warps=4, num_stages=2, force_nc_cache=False, **meta):
        # device inference
        tensor_idxs = [i for i, arg in enumerate(wargs) if hasattr(arg, 'data_ptr')]
        if len(tensor_idxs) == 0:
            raise ValueError("No Tensor argument found.")
        invalid_args = []
        device_ids = []
        for idx in tensor_idxs:
            curr = wargs[idx]
            if not curr.is_cuda:
                invalid_args.append(idx)
            else:
                device_ids.append(curr.device.index)
        if invalid_args:
            raise ValueError("Arguments at index {invalid_args} are on the wrong device.".format(invalid_args=invalid_args) +
                             " Only CUDA is supported at the moment")

        device = torch.device('cuda', torch.cuda.current_device())
        device_idx = device.index
        # if len(set(device_ids)) != 1 or device_ids[0] != device_idx:
        #     # try to enable P2P communication
        #     for arg_idx, dst_idx in zip(tensor_idxs, device_ids):
        #         if dst_idx != device_idx:
        #             try:
        #                 _triton.runtime.enable_peer_access(self.backend, wargs[arg_idx].data_ptr())
        #             except RuntimeError as e:
        #                 raise RuntimeError("Cannot enable P2P access from device {} to device {}: {}"
        #                                    .format(device_idx, dst_idx, str(e)))

        # enqueue kernel on the current device
        torch.cuda.set_device(device_idx)
        # attributes
        args = [arg.data_ptr() if i in tensor_idxs else arg for i, arg in enumerate(wargs)]
        attributes = {i: Kernel.pow2_divisor(a) for i, a in enumerate(args) \
                      if isinstance(a, int) and i not in self.fn.do_not_specialize}
        # transforms ints whose value is one into constants for just-in-time compilation
        constants = {i: arg for i, arg in enumerate(wargs) if isinstance(arg, int) and arg == 1}
        # compute hash for caching this kernel
        types_key = Kernel._types_key(*wargs, tensor_idxs=tensor_idxs)
        attr_key = tuple(attributes.items())
        meta_key = tuple(sorted(meta.items()))
        const_key = tuple(constants.items())
        compute_capability = torch.cuda.get_device_capability(device)

        key = (
            self.fn.cache_key, version_key(), compute_capability,
            types_key, attr_key, num_warps, num_stages, meta_key, const_key
        )
        key = repr(key)

        # get cached binary
        drv_cache = self.fn.drv_cache

        if key not in drv_cache:
            hashed_key = hashlib.md5(key.encode("utf-8")).hexdigest()

            # create cache directory
            cache_dir = os.environ.get('TRITON_CACHE_DIR', '/tmp/triton/')
            if cache_dir and not os.path.exists(cache_dir):
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
            if binary is None:
                binary = self._compile(
                    *wargs, device=device_idx, attributes=attributes,
                    num_warps=num_warps, num_stages=num_stages, force_nc_cache=force_nc_cache, 
                    constants=constants, **meta
                )
                if bin_cache_path:
                    assert bin_lock_path is not None
                    with FileLock(bin_lock_path):
                        with open(bin_cache_path + ".tmp", "wb") as f:
                            pickle.dump({"binary": binary, "key": key}, f)
                        os.rename(bin_cache_path + ".tmp", bin_cache_path)
                        if JITFunction.cache_hook is not None:
                            JITFunction.cache_hook(key=key, binary=binary)
 
            drv_cache[key] = LoadedBinary(device_idx, binary)
        # pack arguments
        fmt = ''.join(['P' if i in tensor_idxs else Kernel._type_name(arg) for i, arg in enumerate(wargs)])
        params = struct.pack(fmt, *args)
        # enqueue cached function into stream
        callable = drv_cache[key]
        stream = torch.cuda.current_stream(device_idx).cuda_stream
        grid = grid(meta) if hasattr(grid, '__call__') else grid
        callable(stream, params, *grid)
        return callable


class Launcher:
    def __init__(self, kernel, grid):
        self.kernel = kernel
        self.grid = grid

    def __call__(self, *wargs, **kwargs):
        return self.kernel(*wargs, **kwargs, grid=self.grid)


class Autotuner:
    def __init__(self, kernel, arg_names, configs, key, reset_to_zero):
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
        def kernel_call():
            self.hook(args)
            self.kernel(*args, num_warps=config.num_warps, num_stages=config.num_stages, **current)
        return triton.testing.do_bench(kernel_call)

    def __call__(self, *args, **meta):
        if len(self.configs) > 1:
            key = tuple([args[i] for i in self.key_idx])
            if key not in self.cache:
                timings = {config: self._bench(*args, config=config, **meta) \
                        for config in self.configs}
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.hook(args)
            config = self.cache[key]
        else:
            config = self.configs[0]
        return self.kernel(*args, num_warps=config.num_warps, num_stages=config.num_stages, **meta, **config.meta)


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
    for lib in pkgutil.iter_modules(triton.language.__path__):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.md5(f.read()).hexdigest()]
    # ptxas version
    try:
        ptxas_version = hashlib.md5(subprocess.check_output(["ptxas", "--version"])).hexdigest()
    except Exception:
        ptxas_version = None
    return (triton.__version__, ptxas_version) + tuple(contents)

class JITFunction:
    
    cache_hook = None

    def _set_cache_key(self):
        self.cache_key = (hashlib.md5(self.src.encode("utf-8")).hexdigest(), self.version)

    def __init__(self, fn, version=None, do_not_specialize=None):
        # information of wrapped function
        self.fn = fn
        self.module = fn.__module__
        self.arg_names = inspect.getfullargspec(fn).args
        self.version = version
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.do_not_specialize = [] if do_not_specialize is None else\
                                 [self.arg_names.index(arg) for arg in do_not_specialize]
        # cache for callable driver objects (e.g. CUkernel)
        self.drv_cache = dict()
        # cache for binaries (on-disk)
        self._set_cache_key()
        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel_decorators = []
        self.kernel = None
        # forward docs
        self.__doc__ = fn.__doc__


    # we do not parse `src` in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Some unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def __call__(self, *args, generator: CodeGenerator, **meta):
        try:
            gscope = generator.gscope.copy()
            lscope = generator.lscope.copy()
            values = generator.module.get_values().copy()
            generator.gscope = sys.modules[self.fn.__module__].__dict__
            ret = generator.visit_FunctionDef(self.parse().body[0], inline=True, arg_values=args)
            generator.gscope = gscope
            generator.lscope = lscope
            generator.module.set_values(values)
            return ret
        except Exception as e:
            node = generator.last_node
            if node is None or isinstance(e, (NotImplementedError, CompilationError)):
                raise e
            raise CompilationError(self.src, node, e)

    # - when `.src` attribute is set, cache path needs
    #   to be reinitialized
    # - when kernel decorators change, cached kernel
    #   needs to be cleared
    def __setattr__(self, name, value):
        if name == 'kernel_decorators':
            self.kernel = None
        super(JITFunction, self).__setattr__(name, value)
        if name == 'src':
            self._set_cache_key()

    def _init_kernel(self):
        if self.kernel is None:
            self.kernel = Kernel(self)
            for decorator in reversed(self.kernel_decorators):
                self.kernel = decorator(self.kernel)
        return self.kernel

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
    """
    def __init__(self, meta, num_warps=4, num_stages=2):
        self.meta = meta
        self.num_warps = num_warps
        self.num_stages = num_stages


def autotune(configs, key, reset_to_zero=None):
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
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    """
    def decorator(fn):
        def wrapper(kernel):
            return Autotuner(kernel, fn.arg_names, configs, key, reset_to_zero)

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
                    meta[v] = heur(*args, **meta)
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
        self.base  = base
        self.is_cuda = base.is_cuda
        self.device = base.device
    
    def data_ptr(self):
        return self.base.data_ptr()


def reinterpret(tensor, dtype):
    return TensorWrapper(tensor, dtype)
