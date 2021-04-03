import inspect
import struct
import types
import torch
import triton._C.libtriton.triton as _triton


def broadcast(builder, lhs, rhs):
    lhs_ty = lhs.type
    rhs_ty = rhs.type
    # op(block, scalar)
    if lhs_ty.is_block() and not rhs_ty.is_block():
        rhs = builder.splat(rhs, lhs_ty.shape)
    # op(scalar, block)
    elif rhs_ty.is_block() and not lhs_ty.is_block():
        lhs = builder.splat(lhs, rhs_ty.shape)
    # op(block, block)
    elif lhs_ty.is_block() and rhs_ty.is_block() and lhs_ty.shape != rhs_ty.shape:
        raise NotImplementedError("Blocks must have the same shape")
    return lhs, rhs


class value:
    def __init__(self, ctx, handle):
        self.ctx = ctx
        self.builder = ctx.module.builder
        self.handle = handle
        self.type = handle.type

    def __add__(self, other):
        # implicit broadcast
        lhs, rhs = broadcast(self.builder, self.handle, other.handle)
        if self.type.scalar.is_ptr():
            handle = self.builder.gep(lhs, [rhs])
        elif self.type.scalar.is_floating():
            handle = self.builder.fadd(lhs, rhs)
        else:
            handle = self.builder.add(lhs, rhs)
        return value(self.ctx, handle)

    def __sub__(self, other):
        if self.type.is_floating():
            handle = self.builder.fsub(self.handle, other.handle)
        else:
            handle = self.builder.sub(self.handle, other.handle)
        return value(self.ctx, handle)

    def __mul__(self, other):
        if self.type.is_floating():
            handle = self.builder.fmul(self.handle, other.handle)
        else:
            handle = self.builder.mul(self.handle, other.handle)
        return value(self.ctx, handle)

    def __div__(self, other):
        if self.type.is_floating():
            handle = self.builder.fdiv(self.handle, other.handle)
        else:
            handle = self.builder.sdiv(self.handle, other.handle)
        return value(self.ctx, handle)

    def __mod__(self, other):
        if self.type.is_floating():
            handle = self.builder.frem(self.handle, other.handle)
        else:
            handle = self.builder.srem(self.handle, other.handle)
        return value(self.ctx, handle)

    def __lshift__(self, other):
        if self.type.is_floating():
            handle = self.builder.fshl(self.handle, other.handle)
        else:
            handle = self.builder.shl(self.handle, other.handle)
        return value(self.ctx, handle)


class context:
    def __init__(self):
        self.__dict__['module'] = _triton.ir.module("")

    def __setattr__(self, name, value):
        self.module.set_value(name, value.handle)

    def __getattr__(self, name):
        ret = self.module.get_value(name)
        return value(self, ret)

    def arange(self, start, end):
        assert isinstance(start, int), "start must be int"
        assert isinstance(end, int), "end must be int"
        builder = self.module.builder
        handle = builder.get_range(start, end)
        return value(self, handle)

    def get_program_id(self, axis):
        builder = self.module.builder
        handle = builder.get_program_id(axis)
        return value(self, handle)


class For:
    def __enter__(self):
        pass

    def __exit__(self):
        pass


class If:
    def __enter__(self):
        pass

    def __exit__(self):
        pass


def load(ptr):
    handle = ptr.builder.load(ptr.handle)
    return value(ptr.ctx, handle)


def store(ptr, arg):
    handle = ptr.builder.store(ptr.handle, arg.handle)
    return value(ptr.ctx, handle)


def init_function(module, name, arg_names, prototype):
    fn = module.get_or_insert_function(name, prototype)
    for i, arg_name in enumerate(arg_names):
        fn.args[i].name = arg_name
        module.set_value(arg_name, fn.args[i])
        module.get_scope().types[arg_name] = fn.args[i].type
    entry = _triton.ir.basic_block.create(module.get_context(), "entry", fn)
    module.seal_block(entry)
    module.builder.set_insert_block(entry)
    return fn


def finalize_function(module):
    module.builder.ret_void()


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
            ctx = context()
            module = ctx.module
            # Generate Triton IR
            arg_names = inspect.getfullargspec(fn).args[1:]
            arg_types = [as_ir_type(module, arg) for arg in wargs]
            ret_type = _triton.ir.type.get_void(module.get_context())
            prototype = _triton.ir.type.make_function(ret_type, arg_types)
            module.add_new_scope()
            handle = init_function(module, fn.__name__, arg_names, prototype)
            # Call decorated function
            params = [value(ctx, h) for h in handle.args]
            fn(ctx, *params)
            finalize_function(module)
            module.pop_scope()
            # generate binary module from Triton-IR
            tt_device = _triton.driver.cu_device(device.index, False)
            # Compile to machine code
            print('emitting bin')
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
def add(ctx, Xptr, Yptr, Zptr):
    ctx.pid = ctx.get_program_id(0)
    ctx.off = ctx.arange(0, 128)
    ctx.x = load(Xptr + ctx.off)
    ctx.y = load(Yptr + ctx.off)
    store(Zptr + ctx.off, ctx.x + ctx.y)


x = torch.rand(128, device='cuda')
y = torch.rand(128, device='cuda')
z = torch.empty_like(x)
add(x, y, z)
print(z)

print(x + y)