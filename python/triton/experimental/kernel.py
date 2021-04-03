import inspect
import struct
import torch
import triton._C.libtriton.triton as _triton


class value:
    def __init__(self, ctx, handle):
        self.ctx = ctx
        self.builder = ctx.builder
        self.handle = handle
        self.type = handle.type

    def __add__(self, other):
        if self.type.is_ptr():
            handle = self.builder.gep(self.handle, other.handle)
        elif self.type.is_floating():
            handle = self.builder.fadd(self.handle, other.handle)
        else:
            handle = self.builder.add(self.handle, other.handle)
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


def load(ptr):
    handle = ptr.builder.load(ptr)
    return value(ptr.ctx, handle)


def store(ptr, arg):
    handle = ptr.builder.store(arg.handle, ptr)
    return value(ptr.ctx, handle)


def make_arange(ctx):
    def arange(start, end):
        assert isinstance(start, int), "start must be int"
        assert isinstance(end, int), "end must be int"
        lhs = ctx.buidler.get_int32(start)
        rhs = ctx.builder.get_int32(end)
        handle = ctx.builder.create_range(lhs, rhs)
        return value(ctx, handle)

    return arange


class context:
    def __init__(self):
        self.module = _triton.ir.module("")

    def __setattr__(self, name, value):
        self.module.set_value(name, value)

    def __getattr__(self, name):
        return self.module.get_value(name)


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


def init_function(module, name, arg_names, prototype):
    fn = module.get_or_insert_function(name, prototype)
    for i, arg_name in enumerate(arg_names):
        fn.args[i].name = arg_name
        module.set_value(arg_name, fn.args[i])
        module.get_scope().types[arg_name] = fn.args[i].type
    entry = _triton.ir.basic_block.create(module.ctx, "entry", fn)
    module.seal_block(entry)
    module.builder.set_block(entry)


def finalize_function(module):
    module.builder.ret_void()


suffixes = {
    int: 'I', float: 'f', bool: 'B',\
    torch.float16: 'f16', torch.float32: 'f32', torch.float64: 'f64',
    torch.bool: 'i1', \
    torch.int8: 'i8', torch.int16: 'i16', torch.int32: 'i32', torch.int64: 'i64',
}


def kernel(fn):
    kernel.cache[fn] = dict()

    def wrapper(*wargs):
        # device inference
        tensor_idxs = [i for i, arg in enumerate(wargs) if isinstance(arg, torch.Tensor)]
        if len(tensor_idxs) == 0:
            raise ValueError("No Tensor argument found. Please specify device.")
        device = wargs[tensor_idxs[0]].device
        # type inference
        types = [None] * len(wargs)
        for i, arg in enumerate(wargs):
            prefix = 'P' if i in tensor_idxs else ''
            suffix = suffixes[arg.dtype] if i in tensor_idxs else suffixes[arg.__class__]
            types[i] = prefix + suffix
        types = '_'.join(types)
        # retrieve from cache
        key = f'{device.type}_{device.index}_{types}'
        if key not in kernel.cache[fn]:
            # create IR module
            ctx = context()
            module = ctx.module
            # Generate Triton IR
            arg_names = inspect.getfullargspec(fn).args
            params = []
            init_function(module, fn.__name__, arg_names, prototype)
            fn(ctx, *params)
            finalize_function(module)
            exit()
            # Compile to machine code
            kernel.cache[fn][key] = _triton.rt.kernel(ctx.module, device)
        # create callable kernel from IR
        caller = kernel.cache[fn][key]
        # pack arguments
        fmt = ['P' if i in tensor_idxs else suffixes[arg.__class__] for i, arg in enumerate(args)]
        params = struct.pack(fmt, *wargs)
        # run function
        cu_stream = torch.cuda.current_stream(device.index).cuda_stream
        stream = _triton.driver.cu_stream(cu_stream, False)
        grid = (1, )
        caller(params, stream, grid)

    return wrapper


kernel.cache = dict()


@kernel
def add(ctx, X, Y):
    ctx.pid = get_program_id(0)
    ctx.off = pid * BLOCK + arange(0, BLOCK)
    ctx.val = load(X + ctx.off)
    store(Y + ctx.off, ctx.val)


x = torch.tensor(128, device='cuda')
y = torch.tensor(128, device='cuda')
add(x, y)