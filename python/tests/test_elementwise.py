import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel

torch_type = {
    "int32": torch.int32,
    "float32": torch.float32,
    "float64": torch.float64
}

torch_ops = {
    "log": "log",
    "cos": "cos",
    "sin": "sin",
    "sqrt": "sqrt",
    "abs": "abs",
    "exp": "exp",
    "sigmoid": "sigmoid",
}

libdevice = '/usr/local/cuda/nvvm/libdevice/libdevice.10.bc'

""" @pytest.mark.parametrize('expr, output_type, input0_type',
    [('log', 'float32', 'float32'),
     ('log', 'float64', 'float64'),
     ('cos', 'float32', 'float32'),
     ('cos', 'float64', 'float64'),
     ('sin', 'float32', 'float32'),
     ('sin', 'float64', 'float64'),
     ('sqrt', 'float32', 'float32'),
     ('sqrt', 'float64', 'float64'),
     ('abs', 'float32', 'float32'),
     #('abs', 'float64', 'float64'),
     ('exp', 'float32', 'float32')
     ('sigmoid', 'float32', 'float32'),
    ])
def test_single_input(expr, output_type, input0_type):
    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)
    
    shape = (128, )
    # limit the range of integers so that the sum does not overflow
    x = torch.abs(torch.randn(shape, dtype=torch_type[input0_type], device="cuda"))
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': "tl." + expr + "(x)"})
    
    # triton result
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    kernel[(1,)](x, y, BLOCK=shape[0], extern_libs={"libdevice": libdevice})
    # reference result
    y_ref = getattr(torch, torch_ops[expr])(x)
    # compare
    assert_close(y, y_ref)"""

'''



def atomic_cas(ptr: tl.tensor,
               cmp: tl.tensor,
               val: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    element_ty = ptr.type.scalar.element_ty
    if element_ty.primitive_bitwidth not in [16, 32, 64]:
        raise ValueError("atomic_cas only supports elements with width {16, 32, 64}")
    return tl.tensor(builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle), val.type)


def atom_red_typechecking_impl(ptr: tl.tensor,
                               val: tl.tensor,
                               mask: tl.tensor,
                               op: str,
                               builder: ir.builder) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    if not ptr.type.scalar.is_ptr():
        raise ValueError("Pointer argument of store instruction is " + ptr.type.__repr__())

    element_ty = ptr.type.scalar.element_ty
    if element_ty is tl.float16 and op != 'add':
        raise ValueError("atomic_" + op + " does not support fp16")
    if element_ty in [tl.int1, tl.int8, tl.int16, tl.bfloat16]:
        raise ValueError("atomic_" + op + " does not support " + element_ty)
    if ptr.type.is_block():
        if mask:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if val:
            val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
    val = cast(val, ptr.type.scalar.element_ty, builder)
    if not mask:
        mask_ir = builder.get_int1(True)
        mask_ty = tl.int1
        if ptr.type.is_block():
            mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
            mask_ty = tl.block_type(tl.int1, ptr.type.get_block_shapes())
        mask = tl.tensor(mask_ir, mask_ty)
    return ptr, val, mask


def atomic_max(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'max', builder)
    sca_ty = val.type.scalar
    # direct call to atomic_max for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MAX,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle),
                             val.type)
        else:
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle),
                             val.type)
    # for float
    # return atomic_smax(i_ptr, i_val) if val >= 0
    # return atomic_umin(i_ptr, i_val) if val < 0
    i_val = bitcast(val, tl.int32, builder)
    i_ptr = bitcast(ptr, tl.pointer_type(tl.int32, 1), builder)
    pos = greater_equal(val, tl.tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder)
    neg = less_than(val, tl.tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder)
    pos_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, i_ptr.handle, i_val.handle, and_(mask, pos, builder).handle), i_val.type)
    neg_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, i_ptr.handle, i_val.handle, and_(mask, neg, builder).handle), i_val.type)
    return where(pos, pos_ret, neg_ret, builder)


def atomic_min(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'min', builder)
    sca_ty = val.type.scalar
    # direct call to atomic_min for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MIN,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle),
                             val.type)
        else:
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle),
                             val.type)
    # for float
    # return atomic_smin(i_ptr, i_val) if val >= 0
    # return atomic_umax(i_ptr, i_val) if val < 0
    i_val = bitcast(val, tl.int32, builder)
    i_ptr = bitcast(ptr, tl.pointer_type(tl.int32, 1), builder)
    pos = greater_equal(val, tl.tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder)
    neg = less_than(val, tl.tensor(ir.constant_float.get(sca_ty.to_ir(builder), 0), sca_ty), builder)
    pos_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MIN,
                                                  i_ptr.handle,
                                                  i_val.handle,
                                                  and_(mask, pos, builder).handle),
                        i_val.type)
    neg_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX,
                                                  i_ptr.handle,
                                                  i_val.handle,
                                                  and_(mask, neg, builder).handle),
                        i_val.type)
    return where(pos, pos_ret, neg_ret, builder)


def atomic_add(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'add', builder)
    sca_ty = val.type.scalar
    op = ir.ATOMIC_OP.FADD if sca_ty.is_floating() else ir.ATOMIC_OP.ADD
    return tl.tensor(builder.create_atomic_rmw(op, ptr.handle, val.handle, mask.handle), val.type)


def atomic_and(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'and', builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.AND, ptr.handle, val.handle, mask.handle), val.type)


def atomic_or(ptr: tl.tensor,
              val: tl.tensor,
              mask: tl.tensor,
              builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'or', builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.OR, ptr.handle, val.handle, mask.handle), val.type)


def atomic_xor(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'xor', builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XOR, ptr.handle, val.handle, mask.handle), val.type)


def atomic_xchg(ptr: tl.tensor,
                val: tl.tensor,
                mask: tl.tensor,
                builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'xchg', builder)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XCHG, ptr.handle, val.handle, mask.handle), val.type)


@builtin
def where(condition, x, y, _builder=None):
    """
    Returns a tensor of elements from either :code:`x` or :code:`y`, depending on :code:`condition`.

    Note that :code:`x` and :code:`y` are always evaluated regardless of the value of :code:`condition`.

    If you want to avoid unintended memory operations, use the :code:`mask` arguments in `triton.load` and `triton.store` instead.

    The shape of :code:`x` and :code:`y` are both broadcast to the shape of :code:`condition`.
    :code:`x` and :code:`y` must have the data type.

    :param condition: When True (nonzero), yield x, otherwise yield y.
    :type condition: Block of triton.bool
    :param x: values selected at indices where condition is True.
    :param y: values selected at indices where condition is False.
    """
    condition = _to_tensor(condition, _builder)
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    return semantic.where(condition, x, y, _builder)


# -----------------------
# Math
# -----------------------

@builtin
def umulhi(x, y, _builder=None):
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    return semantic.umulhi(x, y, _builder)


@builtin
def fdiv(x, y, ieee_rounding=False, _builder=None):
    ieee_rounding = _constexpr_to_value(ieee_rounding)
    return semantic.fdiv(x, y, ieee_rounding, _builder)
@triton.jit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type input: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div


@triton.jit
def minimum(x, y):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return triton.language.where(x < y, x, y)


@triton.jit
def maximum(x, y):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param input: the first input tensor
    :type input: Block
    :param other: the second input tensor
    :type other: Block
    """
    return triton.language.where(x > y, x, y)

'''

def test_two_input(expr, output_type, input0_type, input1_type):
    @triton.jit
    def kernel(X0, X1, Y, BLOCK: tl.constexpr):
        x0 = tl.load(X0 + tl.arange(0, BLOCK))
        x1 = tl.load(X1 + tl.arange(0, BLOCK))
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)
    
    shape = (128, )
    # limit the range of integers so that the sum does not overflow
    x0 = torch.abs(torch.randn(shape, dtype=torch_type[input0_type], device="cuda"))
    x1 = torch.abs(torch.randn(shape, dtype=torch_type[input1_type], device="cuda"))
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': "tl." + expr + "(x0, x1)"})
    
    # triton result
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    kernel[(1,)](x, y, BLOCK=shape[0], extern_libs={"libdevice": libdevice})
    # reference result
    y_ref = getattr(torch, torch_ops[expr])(x)
    # compare
    assert_close(y, y_ref)"""
