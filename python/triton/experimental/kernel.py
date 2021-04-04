import inspect
import struct
import enum
import types
import torch
import ast
import triton._C.libtriton.triton as _triton
from abc import ABC, abstractmethod

########################
# Built-in Functions   #
########################


def load(ptr):
    pass


def arange(start, end):
    pass


def get_program_id(axis):
    pass


def store(ptr, arg):
    pass


class CodeGenerator(ast.NodeVisitor):
    def broadcast(self, lhs, rhs):
        lhs_ty = lhs.type
        rhs_ty = rhs.type
        # op(block, scalar)
        if lhs_ty.is_block() and not rhs_ty.is_block():
            rhs = self.builder.splat(rhs, lhs_ty.shape)
        # op(scalar, block)
        elif rhs_ty.is_block() and not lhs_ty.is_block():
            lhs = self.builder.splat(lhs, rhs_ty.shape)
        # op(block, block)
        elif lhs_ty.is_block() and rhs_ty.is_block() and lhs_ty.shape != rhs_ty.shape:
            raise NotImplementedError("Blocks must have the same shape")
        return lhs, rhs

    def __init__(self, module, prototype, symbols):
        self.module = module
        self.builder = module.builder
        self.prototype = prototype
        self.symbols = symbols

    def visit_Module(self, node):
        self.module.add_new_scope()
        ast.NodeVisitor.generic_visit(self, node)
        self.module.pop_scope()

    def visit_FunctionDef(self, node):
        module = self.module
        arg_names = ast.NodeVisitor.visit(self, node.args)
        # initialize function
        fn = module.get_or_insert_function(node.name, self.prototype)
        for i, arg_name in enumerate(arg_names):
            fn.args[i].name = arg_name
            module.set_value(arg_name, fn.args[i])
            module.scope.set_type(arg_name, fn.args[i].type)
        entry = _triton.ir.basic_block.create(module.get_context(), "entry", fn)
        module.add_new_scope()
        module.seal_block(entry)
        module.builder.set_insert_block(entry)
        # visit function body
        for stmt in node.body:
            ast.NodeVisitor.visit(self, stmt)
        # finalize function
        module.builder.ret_void()
        module.pop_scope()

    def visit_arguments(self, node):
        names = []
        for arg in node.args:
            names += [ast.NodeVisitor.visit(self, arg)]
        return names

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
        self.module.set_value(name, value)

    def visit_Name(self, node):
        return node.id

    def visit_Store(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Load(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_BinOp(self, node):
        lhs = ast.NodeVisitor.visit(self, node.left)
        rhs = ast.NodeVisitor.visit(self, node.right)
        lhs = self.module.get_value(lhs)
        rhs = self.module.get_value(rhs)
        lhs, rhs = self.broadcast(lhs, rhs)
        # -------------------
        # Handle ADD operator
        # -------------------
        if type(node.op) == ast.Add:
            if lhs.type.scalar.is_ptr():  # ptr + offset
                return self.builder.gep(lhs, [rhs])
            elif lhs.type.scalar.is_floating():  # float + float
                return self.builder.fadd(lhs, rhs)
            else:  # int + int
                return self.builder.add(lhs, rhs)
        raise NotImplementedError("Unsupported op: {}".format(expr.op))

    def visit_Call(self, node):
        fn = ast.NodeVisitor.visit(self, node.func)
        if isinstance(fn, str):
            fn = self.symbols[fn]
        name = fn.__name__

        args = [ast.NodeVisitor.visit(self, arg) for arg in node.args]
        assert not node.keywords, "keywords not supported"
        assert not any(arg is None for arg in args)
        if name == 'get_program_id':
            is_valid = isinstance(args[0], _triton.ir.constant_int)
            assert is_valid, "expected constant integer"
            return self.builder.get_program_id(args[0].value)
        if name == 'arange':
            is_valid_0 = isinstance(args[0], _triton.ir.constant_int)
            is_valid_1 = isinstance(args[1], _triton.ir.constant_int)
            assert is_valid_0 and is_valid_1, "expected constant integer"
            return self.builder.get_range(args[0].value, args[1].value)
        if name == 'load':
            return self.builder.load(*args)
        if name == 'store':
            return self.builder.store(*args)
        print(args)

    def visit_Num(self, node):
        val = node.n
        ty = type(val)
        if ty == int:
            return self.builder.get_int32(val)
        if ty == float:
            return self.builder.get_float(val)
        raise NotImplementedError("Unsupported constant type: {}".format(ty))

    def visit_Attribute(self, node):
        lhs = ast.NodeVisitor.visit(self, node.value)
        if isinstance(lhs, str):
            lhs = self.symbols[lhs]
        return getattr(lhs, node.attr)

    def visit_Expr(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def generic_visit(self, node):
        typename = type(node).__name__
        raise NotImplementedError("Unsupported node: {}".format(typename))


suffixes = {
    int: 'I', float: 'f', bool: 'B',\
    torch.float16: 'f16', torch.float32: 'f32', torch.float64: 'f64',
    torch.bool: 'i1', \
    torch.int8: 'i8', torch.int16: 'i16', torch.int32: 'i32', torch.int64: 'i64',
}

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


def as_ir_type(module, obj):
    ctx = module.get_context()
    if isinstance(obj, torch.Tensor):
        ty = type_map[suffixes[obj.dtype]](ctx)
        return _triton.ir.type.make_ptr(ty, 1)
    return type_map[suffixes[obj.__class__]](ctx)


class binary:
    def __init__(self, module, kernel, num_warps, shared_mem):
        self.module = module
        self.kernel = kernel
        self.shared_mem = shared_mem
        self.num_warps = num_warps

    def __call__(self, stream, args, grid_0, grid_1=1, grid_2=1):
        stream.enqueue(self.kernel, grid_0, grid_1, grid_2, self.num_warps * 32, 1, 1, args, self.shared_mem)


def kernel(fn):
    num_warps = 4

    kernel.cache[fn] = dict()

    def wrapper(*wargs):
        # device inference
        tensor_idxs = [i for i, arg in enumerate(wargs) if isinstance(arg, torch.Tensor)]
        if len(tensor_idxs) == 0:
            raise ValueError("No Tensor argument found.")
        device = wargs[tensor_idxs[0]].device
        # type inference
        types_key = [None] * len(wargs)
        for i, arg in enumerate(wargs):
            prefix = 'P' if i in tensor_idxs else ''
            suffix = suffixes[arg.dtype] if i in tensor_idxs else suffixes[arg.__class__]
            types_key[i] = prefix + suffix
        types_key = '_'.join(types_key)
        # retrieve from cache
        key = f'{device.type}_{device.index}_{types_key}'
        if key not in kernel.cache[fn]:
            # create IR module
            module = _triton.ir.module("")
            # Generate Triton IR
            arg_types = [as_ir_type(module, arg) for arg in wargs]
            ret_type = _triton.ir.type.get_void(module.get_context())
            prototype = _triton.ir.type.make_function(ret_type, arg_types)
            tree = ast.parse(inspect.getsource(fn))
            CodeGenerator(module, prototype, globals()).visit(tree)
            tt_device = _triton.driver.cu_device(device.index, False)
            # Compile to machine code
            mod, ker, shared_mem = _triton.codegen.add_passes_to_emit_bin(module, tt_device, num_warps)
            caller = binary(mod, ker, num_warps, shared_mem)
            kernel.cache[fn][key] = caller
        # create callable kernel from IR
        caller = kernel.cache[fn][key]
        # pack arguments
        fmt = ''.join(['P' if i in tensor_idxs else suffixes[arg.__class__] for i, arg in enumerate(wargs)])
        args = [arg.data_ptr() if i in tensor_idxs else arg for i, arg in enumerate(wargs)]
        params = struct.pack(fmt, *args)
        # run function
        cu_stream = torch.cuda.current_stream(device.index).cuda_stream
        stream = _triton.driver.cu_stream(cu_stream, False)
        grid = (1, 1, 1)
        caller(stream, params, *grid)

    return wrapper


kernel.cache = dict()


@kernel
def add(Xptr, Yptr, Zptr):
    pid = get_program_id(0)
    off = arange(0, 128)
    x = load(Xptr + off)
    y = load(Yptr + off)
    store(Zptr + off, x + y)


x = torch.rand(128, device='cuda')
y = torch.rand(128, device='cuda')
z = torch.empty_like(x)
add(x, y, z)
print(z)
print(x + y)
