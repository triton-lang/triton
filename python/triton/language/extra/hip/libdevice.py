# from triton.language import core

# @core.extern
# def mulhi(arg0, arg1, _builder=None):
#     return core.extern_elementwise(
#         "", "", [arg0, arg1], {
#             (core.dtype("int32"), core.dtype("int32")): ("__ockl_mul_hi_i32", core.dtype("int32")),
#             (core.dtype("uint32"), core.dtype("uint32")): ("__ockl_mul_hi_u32", core.dtype("uint32")),
#             (core.dtype("int64"), core.dtype("int64")): ("__ockl_mul_hi_i64", core.dtype("int64")),
#             (core.dtype("uint64"), core.dtype("uint64")): ("__ockl_mul_hi_u64", core.dtype("uint64")),
#         }, is_pure=True, _builder=_builder)
