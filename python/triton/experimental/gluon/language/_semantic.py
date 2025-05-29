from triton.language import semantic as tl_semantic
from . import _core as ttgl
from triton._C.libtriton.gluon_ir import GluonOpBuilder


def arange(start, end, layout, builder: GluonOpBuilder):
    shape = [end - start]
    ret_ty = ttgl.distributed_type(ttgl.int32, shape, layout)
    return tl_semantic.arange(start, end, builder, ret_ty=ret_ty)


def splat(value, shape, layout, builder: GluonOpBuilder):
    ret_ty = ttgl.distributed_type(value.dtype, shape, layout)
    handle = builder.create_splat(ret_ty.to_ir(builder), value.handle)
    return ttgl.tensor(handle, ret_ty)


def full(shape, value, dtype, layout, builder: GluonOpBuilder):
    scalar = tl_semantic.full([], value, dtype, builder)
    return splat(scalar, shape, layout, builder)


def convert_layout(value, layout, builder: GluonOpBuilder):
    ty = value.type
    assert isinstance(ty, ttgl.distributed_type)
    ret_ty = ttgl.distributed_type(ty.element_ty, ty.shape, layout)
    handle = builder.create_convert_layout(ret_ty.to_ir(builder), value.handle)
    return ttgl.tensor(handle, ret_ty)


def allocate_shared(element_ty, shape, layout, value, builder: GluonOpBuilder):
    ty = ttgl.shared_memory_descriptor_type(element_ty, shape, layout, shape)
    handle = builder.create_local_alloc(ty.to_ir(builder), value.handle)
    return ttgl.shared_memory_descriptor(handle, element_ty, shape, layout, shape)


def shared_load(mem_desc, layout, builder: GluonOpBuilder):
    ret_ty = ttgl.distributed_type(mem_desc.dtype, mem_desc.shape, layout)
    handle = builder.create_local_load(ret_ty.to_ir(builder), mem_desc.handle)
    return ttgl.tensor(handle, ret_ty)


def shared_store(mem_desc, value, builder: GluonOpBuilder):
    builder.create_local_store(mem_desc.handle, value.handle)
