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