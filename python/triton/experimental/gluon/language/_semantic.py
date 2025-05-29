from typing import Sequence
from triton.language import semantic as tl_semantic
from . import _core as ttgl
from triton._C.libtriton.gluon_ir import GluonOpBuilder
from triton.compiler.code_generator import flatten_values_to_ir, unflatten_ir_values


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


def warp_specialize(args, default_partition, worker_partitions, worker_num_warps: Sequence[int],
                    worker_num_regs: Sequence[int], builder: GluonOpBuilder, generator):
    num_partitions = len(worker_partitions)
    assert num_partitions == len(
        worker_num_warps), f"warp specialize got {num_partitions} partitions but {len(worker_num_warps)} warp counts"
    assert num_partitions == len(
        worker_num_regs), f"warp specialize got {num_partitions} partitions but {len(worker_num_regs)} register counts"

    insert_pt = builder.get_insertion_point()

    # Emit the default partition to get the result types.
    default_block = builder.new_block()
    builder.set_insertion_point_to_start(default_block)
    default_results = generator.call_JitFunction(default_partition, args, kwargs={})
    mlir_results = flatten_values_to_ir(default_results)
    builder.create_warp_yield(mlir_results)
    result_types = [r.get_type() for r in mlir_results]

    # Create the warp specialize op.
    builder.restore_insertion_point(insert_pt)
    mlir_args = flatten_values_to_ir(args)
    ws_op = builder.create_warp_specialize(result_types, mlir_args, worker_num_warps)
    ws_op.get_default_region().push_back(default_block)
    ws_op.set_requested_registers(worker_num_regs)

    # Emit the partition regions.
    builder.create_block_with_parent(ws_op.get_partition_op_holder(), [])
    partitions_op = builder.create_warp_specialize_partitions(num_partitions)
    arg_types = [arg.get_type() for arg in mlir_args]
    for i in range(num_partitions):
        block = builder.create_block_with_parent(partitions_op.get_region(i), arg_types)
        block_args = [block.get_argument(j) for j in range(len(mlir_args))]
        block_args = unflatten_ir_values(block_args, [arg.type for arg in args])
        generator.call_JitFunction(worker_partitions[i], block_args, kwargs={})
        builder.create_warp_return()

    builder.set_insertion_point_after(ws_op.get_operation())
    mlir_results = [ws_op.get_result(i) for i in range(len(result_types))]
    return tuple(unflatten_ir_values(mlir_results, [r.type for r in default_results]))
