import triton._C.libtriton.triton.ir as ir

ctx = ir.context()
ctx.load_triton()

# TODO
builder = ir.builder(ctx)

module = builder.create_module()


i1_ty = builder.get_int1_ty()
i8_ty = builder.get_int8_ty()
i16_ty = builder.get_int16_ty()
i32_ty = builder.get_int32_ty()
i64_ty = builder.get_int64_ty()

f16_ty = builder.get_half_ty()

f16_ptr_ty = builder.get_ptr_ty(f16_ty)

func_ty = builder.get_function_ty([f16_ptr_ty, f16_ptr_ty, f16_ptr_ty], [])
func = builder.create_function('foo', func_ty)

# ...
entry = func.add_entry_block()
builder.set_insertion_point_to_start(entry)
offsets = builder.create_make_range(0, 128)
pid = builder.create_get_program_id(0)
_128 = builder.get_int32(128)
offset = builder.create_add(pid, _128)
offset = builder.create_broadcast(offset, [128])
offsets = builder.create_add(offset, offsets)


a_ptrs = builder.create_broadcast(entry.arg(0), [128])
b_ptrs = builder.create_broadcast(entry.arg(1), [128])

a_ptrs = builder.create_gep(a_ptrs, offsets)
b_ptrs = builder.create_gep(b_ptrs, offsets)

a = builder.create_load(a_ptrs)
b = builder.create_load(b_ptrs)

c = builder.create_fadd(a, b)
# c.set_attr("ieee_rounding", builder.get_bool_attr(True))

c_ptrs = builder.create_broadcast(entry.arg(2), [128])
c_ptrs = builder.create_gep(c_ptrs, offsets)
builder.create_store(c_ptrs, c)

# func.dump()

module.push_back(func)
module.dump()
