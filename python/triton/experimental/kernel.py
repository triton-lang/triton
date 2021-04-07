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


def builtin(fn):
    def wrapper(*args):
        ret = fn(*args)
        ret = ir_value(args[0].builder, ret)
        return ret

    return wrapper


class ir_value:
    def __init__(self, builder, handle):
        assert isinstance(handle, _triton.ir.value)
        self.builder = builder
        self.type = handle.type
        self._handle = handle

    @builtin
    def __add__(self, other):
        if self.type.scalar.is_ptr():  # ptr + offset
            return self.builder.gep(self._handle, [other._handle])
        elif self.type.scalar.is_floating():  # float + float
            return self.builder.fadd(self._handle, other._handle)
        elif self.type.scalar.is_int():  # int + int
            return self.builder.add(self._handle, other._handle)

    @builtin
    def __mul__(self, other):
        if self.type.scalar.is_floating():  # float * float
            return self.builder.fmul(self._handle, other._handle)
        elif self.type.scalar.is_int():  # int * int
            return self.builder.mul(self._handle, other._handle)

    @builtin
    def __gt__(self, other):
        if self.type.scalar.is_floating():  # float > float
            return self.builder.fcmpOGT(self._handle, other._handle)
        elif self.type.scalar.is_int():  # int > int
            return self.builder.icmpSGT(self._handle, other._handle)

    @builtin
    def __lt__(self, other):
        if self.type.scalar.is_floating():  # float < float
            return self.builder.fcmpOLT(self._handle, other._handle)
        elif self.type.scalar.is_int():  # int < int
            return self.builder.icmpSLT(self._handle, other._handle)


class float32:
    @staticmethod
    def make_ir(context):
        return _triton.ir.type.get_fp32(context)


class float16:
    @staticmethod
    def make_ir(context):
        return _triton.ir.type.get_fp16(context)


class Stub:
    pass


def load(ptr):
    pass


def dot(a, b):
    pass


def select(cond, true_val, false_val):
    pass


def cast(arg, dtype):
    pass


def arange(start, end):
    return Stub()


def program_id(axis):
    return Stub()


def store(ptr, arg):
    pass


def zeros(*shape):
    return Stub()


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
        elif lhs_ty.is_block() and rhs_ty.is_block():
            lhs_shape = lhs_ty.shape
            rhs_shape = rhs_ty.shape
            if len(lhs_shape) != len(rhs_shape):
                raise ValueError("Cannot broadcast blocks of different shapes")
            ret_shape = []
            for i, (l, r) in enumerate(zip(lhs_shape, rhs_shape)):
                if l == 1:
                    ret_shape.append(r)
                elif r == 1:
                    ret_shape.append(l)
                elif l == r:
                    ret_shape.append(l)
                else:
                    raise ValueError(f'Incompatible shape {l} and {r} at dimension {i}')
            if lhs_shape != ret_shape:
                lhs = self.builder.broadcast(lhs, ret_shape)
            if rhs_shape != ret_shape:
                rhs = self.builder.broadcast(rhs, ret_shape)
        return lhs, rhs

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
        if isinstance(ret, ir_value):
            return self._wrap_ir(self.module.get_value(name))
        elif isinstance(ret, int):
            return self._wrap_ir(self.builder.get_int32(ret))
        elif isinstance(ret, float):
            return self._wrap_ir(self.builder.get_float32(ret))
        else:
            return ret

    def visit_compound_statement(self, stmts, add_scope=False):
        if add_scope:
            self.module.add_new_scope()
        for stmt in stmts:
            ast.NodeVisitor.visit(self, stmt)
        if add_scope:
            self.module.pop_scope()

    def __init__(self, module, prototype, gscope, attributes, kwargs):
        self.module = module
        self.builder = module.builder
        self.prototype = prototype
        self.gscope = gscope
        self.lscope = dict()
        self.attributes = attributes
        self.kwargs = kwargs
        self.builtins = {'range': __builtins__.range}
        self._wrap_ir = lambda x: ir_value(self.builder, x)

    def visit_Module(self, node):
        self.module.add_new_scope()
        ast.NodeVisitor.generic_visit(self, node)
        self.module.pop_scope()

    def visit_FunctionDef(self, node):
        module = self.module
        arg_names, kwarg_names = ast.NodeVisitor.visit(self, node.args)
        # store keyword arguments in local scope
        self.lscope[kwarg_names] = self.kwargs
        # initialize function
        fn = module.get_or_insert_function(node.name, self.prototype)
        for i, arg_name in enumerate(arg_names):
            if i in self.attributes:
                is_ptr = fn.args[i].type.is_ptr()
                attr = 'aligned' if is_ptr else 'multiple_of'
                attr = getattr(_triton.ir.attribute_kind, attr)
                attr = _triton.ir.attribute(attr, self.attributes[i])
                fn.add_attr(i + 1, attr)
            fn.args[i].name = arg_name
            module.set_value(arg_name, fn.args[i])
            module.scope.set_type(arg_name, fn.args[i].type)
            self.lscope[arg_name] = self._wrap_ir(fn.args[i])
        entry = _triton.ir.basic_block.create(module.context, "entry", fn)
        module.seal_block(entry)
        module.builder.set_insert_block(entry)
        # visit function body
        self.visit_compound_statement(node.body, add_scope=True)
        # finalize function
        module.builder.ret_void()

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
        if isinstance(value, ir_value):
            self.module.set_value(name, value._handle)
            self.module.scope.set_type(name, value.type)
        self.lscope[name] = value

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

    def visit_BinOp(self, node):
        lhs = ast.NodeVisitor.visit(self, node.left)
        rhs = ast.NodeVisitor.visit(self, node.right)
        if isinstance(lhs, int):
            lhs = self._wrap_ir(self.builder.get_int32(lhs))
        if isinstance(rhs, int):
            rhs = self._wrap_ir(self.builder.get_int32(rhs))
        lhs, rhs = self.broadcast(lhs._handle, rhs._handle)
        lhs = self._wrap_ir(lhs)
        rhs = self._wrap_ir(rhs)
        if type(node.op) == ast.Add:
            return lhs + rhs
        if type(node.op) == ast.Mult:
            return lhs * rhs
        if type(node.op) == ast.Gt:
            return lhs > rhs
        if type(node.op) == ast.Lt:
            return lhs < rhs
        raise NotImplementedError(f"Unsupported op: {node.op}")

    def visit_UnaryOp(self, node):
        operand = ast.NodeVisitor.visit(self, node.operand)
        if isinstance(operand, int):
            operand = self.builder.get_int32(operand)
        # Handle non-constant
        _0f = self.builder.get_float32(0)
        _0i = self.builder.get_int32(0)
        if operand.type.is_block():
            _0f = self.builder.splat(_0f)
            _0i = self.builder.splat(_0i)
        # HANDLE MINUS OPERATOR
        if type(node.op) == ast.USub:
            if operand.type.is_floating():
                return self._wrap_ir(self.builder.fsub(_0f, operand._handle))
            elif operand.type.is_int():
                return self._wrap_ir(self.builder.sub(_0i, operand._handle))

        raise NotImplementedError(f"Unsupported op: {node.op}")

    def visit_Str(self, node):
        return ast.literal_eval(node)

    def visit_Subscript(self, node):
        lhs = ast.NodeVisitor.visit(self, node.value)
        slices = ast.NodeVisitor.visit(self, node.slice)
        if isinstance(lhs, dict):
            return dict.__getitem__(lhs, slices)
        assert isinstance(lhs, ir_value)
        assert node.ctx.__class__.__name__ == "Load"
        # subscripting ir-value
        shapes = []
        curr = 0
        for s in slices:
            if s == None:
                shapes += [1]
            elif s == (None, None, None):
                shapes += [lhs.type.shape[curr]]
                curr += 1
            else:
                raise NotImplementedError(f"Unsupported slice type: {s}")
        return self._wrap_ir(self.builder.reshape(lhs._handle, shapes))

    def visit_ExtSlice(self, node):
        return [ast.NodeVisitor.visit(self, dim) for dim in node.dims]

    def visit_For(self, node):
        iterator = ast.NodeVisitor.visit(self, node.iter.func)
        assert iterator == __builtins__.range
        # create nodes
        st_target = ast.Name(id=node.target.id, ctx=ast.Store())
        ld_target = ast.Name(id=node.target.id, ctx=ast.Load())
        init_node = ast.Assign(targets=[st_target], value=node.iter.args[0])
        pos_cond_node = ast.BinOp(ld_target, ast.Lt(), node.iter.args[1])
        neg_cond_node = ast.BinOp(ld_target, ast.Gt(), node.iter.args[1])
        pos_step_node = ast.BinOp(node.iter.args[2], ast.Gt(), ast.Num(0))
        cond_node = ast.Call()
        cond_node.func = ast.Name(id="select", ctx=ast.Load())
        cond_node.args = [pos_step_node, pos_cond_node, neg_cond_node]
        cond_node.keywords = []
        #cond_node = neg_cond_node
        step_node = ast.AugAssign(target=st_target, op=ast.Add(), value=node.iter.args[2])
        # code generation
        current_bb = self.builder.get_insert_block()
        loop_bb = _triton.ir.basic_block.create(self.module.context, "loop", current_bb.parent)
        next_bb = _triton.ir.basic_block.create(self.module.context, "postloop", current_bb.parent)

        def continue_fn():
            ast.NodeVisitor.visit(self, step_node)
            cond = ast.NodeVisitor.visit(self, cond_node)
            return self.builder.cond_br(cond._handle, loop_bb, next_bb)

        ast.NodeVisitor.visit(self, init_node)
        cond = ast.NodeVisitor.visit(self, cond_node)
        self.builder.cond_br(cond._handle, loop_bb, next_bb)
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

    def visit_Call(self, node):
        fn = ast.NodeVisitor.visit(self, node.func)
        name = fn.__name__

        args = [ast.NodeVisitor.visit(self, arg) for arg in node.args]
        assert not node.keywords, "keywords not supported"
        assert not any(arg is None for arg in args)
        if name == 'program_id':
            return self._wrap_ir(self.builder.get_program_id(int(args[0]._handle)))
        if name == 'arange':
            return self._wrap_ir(self.builder.get_range(int(args[0]._handle), int(args[1]._handle)))
        if name == 'load':
            return self._wrap_ir(self.builder.load(*[arg._handle for arg in args]))
        if name == 'store':
            return self._wrap_ir(self.builder.store(*[arg._handle for arg in args]))
        if name == 'zeros':
            _0 = self.builder.get_float32(0)
            shape = [int(x._handle) for x in args]
            return self._wrap_ir(self.builder.splat(_0, shape))
        if name == 'cast':
            # return type
            src_ty = args[0].type
            ret_ty = args[1].make_ir(self.module.context)
            if src_ty.is_block:
                ret_ty = _triton.ir.type.make_block(ret_ty, src_ty.shape)
            # FP Truncation
            if src_ty.scalar.is_floating() and ret_ty.scalar.is_floating() and\
               src_ty.scalar.fp_mantissa_width > ret_ty.scalar.fp_mantissa_width:
                return self._wrap_ir(self.builder.fp_trunc(args[0]._handle, ret_ty))
            raise NotImplementedError(f"cast from {src_ty} to {ret_ty}")
        if name == 'dot':
            M, K = args[0].type.shape
            K, N = args[1].type.shape
            assert args[0].type.scalar.is_floating()
            assert args[1].type.scalar.is_floating()
            _0 = self.builder.get_float32(0)
            _0 = self.builder.splat(_0, [M, N])
            return self._wrap_ir(self.builder.dot(args[0]._handle, args[1]._handle, _0))
        if name == 'select':
            return self._wrap_ir(self.builder.select(*[arg._handle for arg in args]))
        raise NotImplementedError(f"Unsupported function: {name}")

    def visit_Num(self, node):
        val = node.n
        ty = type(val)
        if ty == int:
            return self._wrap_ir(self.builder.get_int32(val))
        if ty == float:
            return self._wrap_ir(self.builder.get_float32(val))
        raise NotImplementedError("Unsupported constant type: {}".format(ty))

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
    ctx = module.context
    if isinstance(obj, torch.Tensor):
        ty = type_map[suffixes[obj.dtype]](ctx)
        return _triton.ir.type.make_ptr(ty, 1)
    return type_map[suffixes[obj.__class__]](ctx)


def cdiv(a, b):
    return (a + b - 1) // b


def pow2_divisor(N):
    if N % 16 == 0: return 16
    if N % 8 == 0: return 8
    if N % 4 == 0: return 4
    if N % 2 == 0: return 2
    return 1


class Binary:
    def __init__(self, module, kernel, num_warps, shared_mem):
        self.module = module
        self.kernel = kernel
        self.shared_mem = shared_mem
        self.num_warps = num_warps

    def __call__(self, stream, args, grid_0, grid_1=1, grid_2=1):
        stream.enqueue(self.kernel, grid_0, grid_1, grid_2, self.num_warps * 32, 1, 1, args, self.shared_mem)


class Kernel:
    def __init__(self, fn, grid):
        self.fn = fn
        self.grid = grid

    def __call__(self, *wargs, num_warps=4, **meta):
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
        # attribute key
        args = [arg.data_ptr() if i in tensor_idxs else arg for i, arg in enumerate(wargs)]
        attributes = {i: pow2_divisor(a) for i, a in enumerate(args) if isinstance(a, int)}
        attr_key = '_'.join(map(str, attributes.values()))
        # cache
        key = f'{device.type}_{device.index}_{types_key}_{attr_key}'
        cache = self.fn.cache
        if key not in cache:
            # create IR module
            module = _triton.ir.module("")
            module.context.builder = module.builder
            # Generate Triton IR
            arg_types = [as_ir_type(module, arg) for arg in wargs]
            ret_type = _triton.ir.type.get_void(module.context)
            prototype = _triton.ir.type.make_function(ret_type, arg_types)
            tree = ast.parse(inspect.getsource(self.fn.src))
            code_gen = CodeGenerator(module, prototype, gscope=globals(), attributes=attributes, kwargs=meta)
            code_gen.visit(tree)
            tt_device = _triton.driver.cu_device(device.index, False)
            # Compile to machine code
            mod, ker, shared_mem = _triton.codegen.add_passes_to_emit_bin(module, tt_device, num_warps)
            cache[key] = Binary(mod, ker, num_warps, shared_mem)
            pass
        # pack arguments
        fmt = ''.join(['P' if i in tensor_idxs else suffixes[arg.__class__] for i, arg in enumerate(wargs)])
        params = struct.pack(fmt, *args)
        # enqueue cached function into stream
        binary = cache[key]
        cu_stream = torch.cuda.current_stream(device.index).cuda_stream
        stream = _triton.driver.cu_stream(cu_stream, False)
        binary(stream, params, *self.grid(meta))


class Autotuner:
    def __init__(self, kernel, configs):
        if not configs:
            self.configs = [Config(dict(), num_warps=4)]
        else:
            self.configs = configs
        self.kernel = kernel

    def __call__(self, *args, **meta):
        for config in self.configs:
            # check for conflicts, i.e. meta-parameters both provided
            # as kwargs and by the autotuner
            conflicts = meta.keys() & config.meta.keys()
            if conflicts:
                raise ValueError(
                    f"Conflicting meta-parameters: {', '.join(conflicts)}."
                    " Make sure that you don't re-define auto-tuned symbols."
                )
            # augment meta-parameters with tunable ones
            meta.update(config.meta)
            self.kernel(*args, num_warps=config.num_warps, **meta)


class JITFunction:
    def __init__(self, src, configs):
        self.src = src
        self.configs = configs
        self.cache = dict()

    def compile(self, module, device, num_warps):
        pass

    def __getitem__(self, grid_fn):
        return Autotuner(Kernel(self, grid), self.configs)


class Config:
    def __init__(self, meta, num_warps=4):
        self.meta = meta
        self.num_warps = num_warps


def jit(configs=None):
    if configs is None:
        configs = []

    def decorator(fn):
        return JITFunction(fn, configs)

    return decorator


@jit(configs=[
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
])
def matmul(C, A, B, M, N, K, lda, ldb, ldc, **META):
    # extract meta-parameters
    BLOCK_M = META['BLOCK_M']
    BLOCK_N = META['BLOCK_N']
    BLOCK_K = META['BLOCK_K']
    # matrix multiplication
    pid_m = program_id(0)
    pid_n = program_id(1)
    rm = pid_m * BLOCK_M + arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + arange(0, BLOCK_N)
    rk = arange(0, BLOCK_K)
    A = A + rm[:, None] * lda + rk[None, :]
    B = B + rk[:, None] * ldb + rn[None, :]
    c = zeros(BLOCK_M, BLOCK_N)
    for k in range(K, 0, -BLOCK_K):
        c += dot(load(A), load(B))
        A += BLOCK_K
        B += BLOCK_K * ldb
    C = C + rm[:, None] * ldc + rn[None, :]
    store(C, cast(c, float16))


M, N, K = 512, 512, 512
A = torch.randn((M, K), dtype=torch.float16, device='cuda')
B = torch.randn((K, N), dtype=torch.float16, device='cuda')
C = torch.empty((M, N), dtype=torch.float16, device='cuda')
# Launch triton kernel
grid = lambda META: (cdiv(M, META['BLOCK_M']), cdiv(N, META['BLOCK_N']))
matmul[grid](C, A, B, M, N, K, A.stride(0), B.stride(0), C.stride(0))
# Compare results
assert torch.allclose(C, torch.matmul(A, B), atol=1e-3, rtol=1e-3)
