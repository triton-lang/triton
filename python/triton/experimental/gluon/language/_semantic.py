from typing import Sequence, List, TypeVar, Tuple, Callable
import math
from triton.language.semantic import TritonSemantic
from . import _core as ttgl
from ._layouts import AutoLayout, DistributedLayout, SliceLayout
from triton._C.libtriton.gluon_ir import GluonOpBuilder
from triton.compiler.code_generator import flatten_values_to_ir, unflatten_ir_values

TensorTy = TypeVar("TensorTy")


def _check(cond: bool, msg_fn: Callable[[], str], category=ValueError):
    if not cond:
        raise category(msg_fn())


class GluonCallerContext:

    def __init__(self, num_warps: int):
        self.num_warps = num_warps

    def mangle(self):
        return f"_NW{self.num_warps}"

    def initialize_callee(self, fn, builder):
        fn.set_attr("ttg.num-warps", builder.get_int32_attr(self.num_warps))


class GluonSemantic(TritonSemantic[TensorTy]):
    tensor = ttgl.tensor
    lang = ttgl

    builder: GluonOpBuilder

    def __init__(self, builder: GluonOpBuilder):
        self.builder = builder

    def _wrap_handle_infer_layout(self, handle, scalar_ty, shape):
        if shape == []:
            ty = scalar_ty
        else:
            ty = ttgl.distributed_type(scalar_ty, shape, self.builder.get_gluon_layout_from_tensor(handle))
        return self.tensor(handle, ty)

    def _wrap_tensor_infer_layout(self, tensor):
        return self._wrap_handle_infer_layout(tensor.handle, tensor.type.scalar, tensor.shape)

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

        _check(isinstance(input.type, ttgl.distributed_type),
               lambda: f"expected expand_dims input to be a distributed_type but got: {input.type!r}")
        layout = input.type.layout
        _check(isinstance(layout, (SliceLayout, AutoLayout)),
               lambda: f"expected expand_dims input to have a SliceLayout, but got: {layout}")
        _check(
            isinstance(layout, AutoLayout) or layout.dim == axis,
            lambda: f"expected expand_dims input layout to be sliced in axis {axis} but got {layout.dim}")

        handle = self.builder.create_expand_dims(input.handle, axis)
        return self._wrap_handle_infer_layout(handle, input.type.scalar, dst_shape)

    def join(self, a: TensorTy, b: TensorTy) -> TensorTy:
        a, b = self.broadcast_impl_value(a, b)
        _check(a.shape != [], "Cannot join scalars in gluon")
        value = super().join(a, b)
        return self._wrap_tensor_infer_layout(value)

    def split(self, a: TensorTy) -> Tuple[TensorTy, TensorTy]:
        lhs, rhs = super().split(a)
        return self._wrap_tensor_infer_layout(lhs), self._wrap_tensor_infer_layout(rhs)

    def permute(self, input: TensorTy, dims: Tuple[int]) -> TensorTy:
        value = super().permute(input, dims)
        return self._wrap_tensor_infer_layout(value)

    def broadcast_impl_shape(self, input: TensorTy, shape: Tuple[int]) -> TensorTy:
        _check(isinstance(input.type, ttgl.distributed_type),
               lambda: f"expected expand_dims input to be a distributed_type but got: {input.type!r}")
        src_shape = input.type.get_block_shapes()
        _check(len(src_shape) == len(shape), lambda: f"Cannot broadcast, rank mismatch: {src_shape}, {shape}")
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

        _check(isinstance(lhs_ty, ttgl.distributed_type),
               lambda: f"expected broadcast left input to be a distributed_type but got: {lhs_ty!r}")
        _check(isinstance(rhs_ty, ttgl.distributed_type),
               lambda: f"expected broadcast right input to be a distributed_type but got: {rhs_ty!r}")

        lhs_shape = lhs_ty.get_block_shapes()
        rhs_shape = rhs_ty.get_block_shapes()
        ret_shape = self._broadcast_shapes(lhs_shape, rhs_shape)

        is_lhs_auto = isinstance(lhs_ty.layout, AutoLayout)
        is_rhs_auto = isinstance(rhs_ty.layout, AutoLayout)
        if is_lhs_auto and not is_rhs_auto:
            lhs = self.set_auto_layout(lhs, rhs_ty.layout)
        elif is_rhs_auto and not is_lhs_auto:
            rhs = self.set_auto_layout(rhs, lhs_ty.layout)
        elif lhs_ty.layout != rhs_ty.layout:
            raise ValueError(f"Layout mismatch in broadcast: {lhs_ty.layout} vs {rhs_ty.layout}")

        lhs = self.broadcast_impl_shape(lhs, ret_shape)
        rhs = self.broadcast_impl_shape(rhs, ret_shape)
        return lhs, rhs

    def arange(self, start, end, layout):
        shape = [end - start]
        if layout is None:
            layout = AutoLayout()
        ret_ty = ttgl.distributed_type(ttgl.int32, shape, layout)
        return super().arange(start, end, ret_ty=ret_ty)

    def reshape(self, input: TensorTy, dst_shape: List[int], can_reorder: bool):
        _check(not can_reorder, "can_reorder is not supported in gluon")
        value = super().reshape(input, dst_shape, can_reorder)
        return self._wrap_tensor_infer_layout(value)

    def splat(self, value, shape, layout):
        ret_ty = ttgl.distributed_type(value.dtype, shape, layout)
        handle = self.builder.create_splat(ret_ty.to_ir(self.builder), value.handle)
        return ttgl.tensor(handle, ret_ty)

    def full(self, shape, value, dtype, layout):
        scalar = self.make_scalar(value, dtype)
        if layout is None:
            layout = AutoLayout()
        return self.splat(scalar, shape, layout)

    def convert_layout(self, value, layout, assert_trivial=False):
        ty = value.type
        _check(isinstance(ty, ttgl.distributed_type),
               lambda: f"expected convert_layout input to be a distributed_type but got: {ty!r}")
        ret_ty = ttgl.distributed_type(ty.element_ty, ty.shape, layout)
        ret_ty_ir = ret_ty.to_ir(self.builder)
        if assert_trivial and not self.builder.is_convert_layout_trivial(ret_ty_ir, value.handle):
            raise TypeError(f"layout conversion from {ty.layout} to {layout} is not trivial")
        handle = self.builder.create_convert_layout(ret_ty_ir, value.handle)
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
        assert value.shape == mem_desc.shape, f"source shape {value.shape} and destination shape {mem_desc.shape} must match"
        assert value.dtype == mem_desc.dtype, f"source dtype {value.dtype} and destination dtype {mem_desc.dtype} must match"
        self.builder.create_local_store(mem_desc.handle, value.handle)

    def shared_dealloc(self, mem_desc):
        self.builder.create_local_dealloc(mem_desc.handle)

    def set_auto_layout(self, value, layout):
        src_ty = value.type
        assert isinstance(layout,
                          DistributedLayout), f"set_auto_layout must set to a distributed layout but got {layout}"
        assert isinstance(src_ty.layout,
                          AutoLayout), f"set_auto_layout input must have auto layout but got {value.type.layout}"
        handle = self.builder.create_set_auto_layout(layout._to_ir(self.builder), value.handle)
        res_ty = ttgl.distributed_type(src_ty.element_ty, src_ty.shape, layout)
        return self.tensor(handle, res_ty)

    def memdesc_slice(self, mem_desc, start, length, dim):
        offsets = [0] * mem_desc.rank
        offsets[dim] = start
        shape = list(mem_desc.shape)
        shape[dim] = length
        layout = mem_desc.layout
        ty = ttgl.shared_memory_descriptor_type(mem_desc.dtype, shape, layout, mem_desc.type.alloc_shape)
        builder = self.builder
        handle = builder.create_memdesc_subslice(ty.to_ir(builder), mem_desc.handle, offsets)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def memdesc_index(self, mem_desc, index):
        shape = mem_desc.shape[1:]
        index = self.to_tensor(index).handle
        layout = mem_desc.layout
        ty = ttgl.shared_memory_descriptor_type(mem_desc.dtype, shape, layout, mem_desc.type.alloc_shape)
        builder = self.builder
        handle = builder.create_memdesc_index(ty.to_ir(builder), mem_desc.handle, index)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def memdesc_trans(self, mem_desc, order):
        assert len(order) == len(
            mem_desc.shape), f"source rank ({mem_desc.rank}) and order length ({len(order)}) must match"

        shape = [mem_desc.shape[i] for i in order]
        alloc_shape = mem_desc.type.alloc_shape
        new_alloc_shape = alloc_shape[:len(alloc_shape) - mem_desc.rank]
        new_alloc_shape += [alloc_shape[len(alloc_shape) - mem_desc.rank:][i] for i in order]

        handle = self.builder.create_memdesc_trans(mem_desc.handle, order)
        layout = self.builder.get_gluon_layout_from_memdesc(handle)
        return ttgl.shared_memory_descriptor(handle, element_ty=mem_desc.dtype, shape=shape,
                                             alloc_shape=new_alloc_shape, layout=layout)

    def memdesc_reshape(self, mem_desc, shape):
        _check(
            math.prod(shape) == math.prod(mem_desc.shape),
            lambda: (f"memdesc_reshape total elements mismatch: "
                     f"{mem_desc.shape} -> {shape}"),
        )

        handle = self.builder.create_memdesc_reshape(mem_desc.handle, shape)
        layout = self.builder.get_gluon_layout_from_memdesc(handle)
        alloc_shape = mem_desc.type.alloc_shape
        prefix_len = len(alloc_shape) - mem_desc.rank
        new_alloc_shape = alloc_shape[:prefix_len] + list(shape)

        return ttgl.shared_memory_descriptor(
            handle,
            element_ty=mem_desc.dtype,
            shape=shape,
            alloc_shape=new_alloc_shape,
            layout=layout,
        )

    def memdesc_reinterpret(self, mem_desc, dtype, shape, layout):
        ty = ttgl.shared_memory_descriptor_type(dtype, shape, layout, shape)
        handle = self.builder.create_memdesc_reinterpret(ty.to_ir(self.builder), mem_desc.handle)
        return ttgl.shared_memory_descriptor(handle, **ty.__dict__)

    def wrap_tensor(self, x, scalar_ty, ret_shape, layout):
        if ret_shape:
            res_ty = ttgl.distributed_type(scalar_ty, ret_shape, layout)
        else:
            res_ty = scalar_ty
        return self.tensor(x, res_ty)

    @staticmethod
    def _check_same_layout(xs):
        for x in xs:
            _check(isinstance(x.type, ttgl.distributed_type), lambda: f"expected distributed_type but got: {x.type!r}")
        layouts = [x.type.layout for x in xs]
        l0 = layouts[0]
        _check(all(l == l0 for l in layouts[1:]),
               lambda: f"Expected inputs to have matching layouts, but got: {layouts}")

    def associative_scan(self, inputs: Sequence[TensorTy], axis: int, region_builder_fn,
                         reverse: bool) -> Tuple[TensorTy, ...]:
        shape = inputs[0].type.shape
        rank = len(shape)

        assert -rank <= axis < rank, f"scan axis {axis} must be < inputs rank ({rank})"

        if axis < 0:
            axis += rank

        for t in inputs:
            assert t.type.shape == shape, "all scan inputs must have the same shape"

        scan_op = self.builder.create_scan([t.handle for t in inputs], axis, reverse)
        region_builder_fn(scan_op)
        assert scan_op.verify()

        return tuple(
            self._wrap_handle_infer_layout(scan_op.get_result(i), inputs[i].type.scalar, shape)
            for i in range(len(inputs)))

    def reduction(self, inputs: Sequence[TensorTy], axis: int, region_builder_fn) -> Tuple[TensorTy, ...]:
        _check(axis is not None, lambda: "All-reduce is not yet implemented in gluon")
        # get result shape
        shape = inputs[0].type.shape
        rank = len(shape)
        _check(0 <= axis < rank, lambda: f"expected reduction axis to be in the range [0, {rank}) but got {axis}")
        self._check_same_layout(inputs)
        ret_shape = [s for i, s in enumerate(shape) if i != axis]
        assert all(t.type.shape == shape for t in inputs), "all reduction inputs must have the same shape"

        reduce_op = self.builder.create_reduce([t.handle for t in inputs], axis)
        region_builder_fn(reduce_op)
        assert reduce_op.verify()

        return tuple(
            self._wrap_handle_infer_layout(reduce_op.get_result(i), inputs[i].type.scalar, ret_shape)
            for i in range(len(inputs)))

    def histogram(self, input: TensorTy, num_bins: int, mask: TensorTy, layout) -> TensorTy:
        _check(len(input.shape) == 1, lambda: "histogram only supports 1D input")
        _check(input.dtype.is_int(), lambda: "histogram only supports integer input")
        _check(layout is not None, lambda: "histogram requires a destination layout")
        if mask is not None:
            mask, input = self.broadcast_impl_value(mask, input)
            _check(mask.type.scalar.is_bool(), lambda: "Mask must have boolean scalar type")
            mask = mask.handle
        layout_attr = layout._to_ir(self.builder)
        handle = self.builder.create_histogram(input.handle, num_bins, mask, layout_attr)
        return self.wrap_tensor(handle, ttgl.int32, [num_bins], layout)

    def warp_specialize(self, default_args, default_partition, worker_args, worker_partitions,
                        worker_num_warps: Sequence[int], worker_num_regs: Sequence[int], generator):
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
        default_results = generator.call_JitFunction(default_partition, default_args, kwargs={})
        mlir_results = []
        if default_results is not None:
            mlir_results = flatten_values_to_ir(default_results)
        builder.create_warp_yield(mlir_results)
        result_types = [r.get_type() for r in mlir_results]

        # Create the warp specialize op.
        builder.restore_insertion_point(insert_pt)
        mlir_args = flatten_values_to_ir(worker_args)
        ws_op = builder.create_warp_specialize(result_types, mlir_args, worker_num_warps)
        ws_op.get_default_region().push_back(default_block)
        ws_op.set_requested_registers(worker_num_regs)

        # Emit the partition regions.
        builder.create_block_with_parent(ws_op.get_partition_op_holder(), [])
        partitions_op = builder.create_warp_specialize_partitions(num_partitions)
        arg_types = [arg.get_type() for arg in mlir_args]
        for i in range(num_partitions):
            caller_context = GluonCallerContext(num_warps=worker_num_warps[i])
            block = builder.create_block_with_parent(partitions_op.get_region(i), arg_types)
            block_args = [block.get_argument(j) for j in range(len(mlir_args))]
            block_args = unflatten_ir_values(block_args, [arg.type for arg in worker_args])
            generator.call_JitFunction(worker_partitions[i], block_args, kwargs={}, caller_context=caller_context)
            builder.create_warp_return()

        builder.set_insertion_point_after(ws_op.get_operation())
        mlir_results = [ws_op.get_result(i) for i in range(len(result_types))]
        if default_results is None:
            return
        return tuple(unflatten_ir_values(mlir_results, [r.type for r in default_results]))
