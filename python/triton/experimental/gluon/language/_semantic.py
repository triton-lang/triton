from typing import Sequence, List, TypeVar, Tuple
from triton.language.semantic import TritonSemantic
from . import _core as ttgl
from ._layouts import SliceLayout
from triton._C.libtriton.gluon_ir import GluonOpBuilder
from triton.compiler.code_generator import flatten_values_to_ir, unflatten_ir_values

TensorTy = TypeVar("TensorTy")


class GluonSemantic(TritonSemantic[TensorTy]):
    tensor = ttgl.tensor
    lang = ttgl

    builder: GluonOpBuilder

    def __init__(self, builder: GluonOpBuilder):
        self.builder = builder

    def _broadcast_shapes(self, lhs_shape: List[int], rhs_shape: List[int]):
        if len(lhs_shape) != len(rhs_shape):
            raise ValueError(f"Cannot broadcast, rank mismatch: {lhs_shape}, {rhs_shape}")

        ret_shape = []
        for i, left in enumerate(lhs_shape):
            right = rhs_shape[i]
            if left == 1:
                ret_shape.append(right)
            elif (right == 1) or (right == left):
                ret_shape.append(left)
            else:
                raise ValueError("Cannot make_shape_compatible: incompatible dimensions "
                                 "at index " + str(i) + ": " + str(left) + " and " + str(right))
        return ret_shape

    def expand_dims(self, input: TensorTy, axis: int) -> TensorTy:
        dst_shape = [ttgl._unwrap_if_constexpr(x) for x in input.shape]
        dst_shape.insert(axis, 1)

        if axis < 0:
            axis += len(input.shape)

        assert isinstance(input.type, ttgl.distributed_type)
        layout = input.type.layout
        assert isinstance(layout, SliceLayout)
        assert layout.dim == axis

        ret_ty = ttgl.distributed_type(input.type.scalar, dst_shape, layout.parent)
        handle = self.builder.create_expand_dims(input.handle, axis, ret_ty.to_ir(self.builder))
        return self.tensor(handle, ret_ty)

    def broadcast_impl_shape(self, input: TensorTy, shape: Tuple[int]) -> TensorTy:
        assert isinstance(input.type, ttgl.distributed_type), str(input.type)
        src_shape = input.type.get_block_shapes()
        if len(src_shape) != len(shape):
            raise ValueError(f"Cannot broadcast, rank mismatch: {src_shape}, {shape}")
        if shape == src_shape:
            return input
        for i, item in enumerate(src_shape):
            if shape[i] != item and item != 1:
                raise ValueError(f"Cannot broadcast, the expanded size of the tensor ({shape[i]})"
                                 f" must match the existing size ({item}) at non-singleton dimension"
                                 f" {i}: {src_shape}, {shape}")
        ret_ty = ttgl.distributed_type(input.type.scalar, shape, input.type.layout)
        handle = self.builder.create_broadcast(input.handle, ret_ty.to_ir(self.builder))
        return self.tensor(handle, ret_ty)

    def broadcast_impl_value(self, lhs: TensorTy, rhs: TensorTy) -> TensorTy:
        lhs_ty = lhs.type
        rhs_ty = rhs.type

        if not lhs_ty.is_block() or not rhs_ty.is_block():
            return super().broadcast_impl_value(lhs, rhs)

        assert isinstance(lhs_ty, ttgl.distributed_type)
        assert isinstance(rhs_ty, ttgl.distributed_type)

        lhs_shape = lhs_ty.get_block_shapes()
        rhs_shape = rhs_ty.get_block_shapes()
        ret_shape = self._broadcast_shapes(lhs_shape, rhs_shape)
        if lhs_ty.layout != rhs_ty.layout:
            raise ValueError(f"Layout mismatch in broadcast: {lhs_ty.layout} vs {rhs_ty.layout}")

        lhs = self.broadcast_impl_shape(lhs, ret_shape)
        rhs = self.broadcast_impl_shape(rhs, ret_shape)
        return lhs, rhs

    def arange(self, start, end, layout):
        shape = [end - start]
        ret_ty = ttgl.distributed_type(ttgl.int32, shape, layout)
        return super().arange(start, end, ret_ty=ret_ty)

    def splat(self, value, shape, layout):
        ret_ty = ttgl.distributed_type(value.dtype, shape, layout)
        handle = self.builder.create_splat(ret_ty.to_ir(self.builder), value.handle)
        return ttgl.tensor(handle, ret_ty)

    def full(self, shape, value, dtype, layout):
        scalar = self.make_scalar(value, dtype)
        return self.splat(scalar, shape, layout)

    def convert_layout(self, value, layout):
        ty = value.type
        assert isinstance(ty, ttgl.distributed_type)
        ret_ty = ttgl.distributed_type(ty.element_ty, ty.shape, layout)
        handle = self.builder.create_convert_layout(ret_ty.to_ir(self.builder), value.handle)
        return ttgl.tensor(handle, ret_ty)

    def allocate_shared(self, element_ty, shape, layout, value):
        ty = ttgl.shared_memory_descriptor_type(element_ty, shape, layout, shape)
        if value is not None:
            handle = self.builder.create_local_alloc(ty.to_ir(self.builder), value.handle)
        else:
            handle = self.builder.create_local_alloc(ty.to_ir(self.builder))
        return ttgl.shared_memory_descriptor(handle, element_ty, shape, layout, shape)

    def shared_load(self, mem_desc, layout):
        ret_ty = ttgl.distributed_type(mem_desc.dtype, mem_desc.shape, layout)
        handle = self.builder.create_local_load(ret_ty.to_ir(self.builder), mem_desc.handle)
        return ttgl.tensor(handle, ret_ty)

    def shared_store(self, mem_desc, value):
        self.builder.create_local_store(mem_desc.handle, value.handle)

    def shared_dealloc(self, mem_desc):
        self.builder.create_local_dealloc(mem_desc.handle)

    def _memdesc_subview(self, mem_desc, offsets, shape, layout):
        ty = ttgl.shared_memory_descriptor_type(mem_desc.dtype, shape, layout, mem_desc.type.alloc_shape)
        builder = self.builder
        handle = builder.create_memdesc_subview(ty.to_ir(builder), mem_desc.handle, offsets)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def memdesc_split(self, mem_desc, offset, size, dim, layout):
        offsets = [self.builder.get_int32(0)] * mem_desc.rank
        offsets[dim] = self.builder.get_int32(offset)
        shape = list(mem_desc.shape)
        shape[dim] = size
        return self._memdesc_subview(mem_desc, offsets, shape, layout)

    def memdesc_slice(self, mem_desc, index, shape, layout):
        assert mem_desc.rank > len(
            shape), f"source rank ({mem_desc.rank}) must be greater than result rank ({len(shape)})"

        offsets = [self.builder.get_int32(0)] * mem_desc.rank
        offsets[0] = self._convert_elem_to_ir_value(index, require_i64=False)
        return self._memdesc_subview(mem_desc, offsets, shape, layout)

    def memdesc_trans(self, mem_desc, order, layout):
        assert len(order) == len(
            mem_desc.shape), f"source rank ({mem_desc.rank}) and order length ({len(order)}) must match"

        shape = [mem_desc.shape[i] for i in order]
        alloc_shape = mem_desc.type.alloc_shape
        new_alloc_shape = alloc_shape[:len(alloc_shape) - mem_desc.rank]
        new_alloc_shape += [alloc_shape[:mem_desc.rank][i] for i in order]

        ty = ttgl.shared_memory_descriptor_type(mem_desc.dtype, shape, layout, new_alloc_shape)
        handle = self.builder.create_memdesc_trans(ty.to_ir(self.builder), mem_desc.handle, order)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def memdesc_reshape(self, mem_desc, shape, layout):
        ty = ttgl.shared_memory_descriptor_type(mem_desc.dtype, shape, layout, mem_desc.type.alloc_shape)
        handle = self.builder.create_memdesc_reshape(ty.to_ir(self.builder), mem_desc.handle)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def memdesc_reinterpret(self, mem_desc, dtype, shape, layout):
        ty = ttgl.shared_memory_descriptor_type(dtype, shape, layout, shape)
        handle = self.builder.create_memdesc_reinterpret(ty.to_ir(self.builder), mem_desc.handle)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def warp_specialize(self, args, default_partition, worker_partitions, worker_num_warps: Sequence[int],
                        worker_num_regs: Sequence[int], generator):
        num_partitions = len(worker_partitions)
        assert num_partitions == len(
            worker_num_warps
        ), f"warp specialize got {num_partitions} partitions but {len(worker_num_warps)} warp counts"
        assert num_partitions == len(
            worker_num_regs
        ), f"warp specialize got {num_partitions} partitions but {len(worker_num_regs)} register counts"

        builder = self.builder
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
