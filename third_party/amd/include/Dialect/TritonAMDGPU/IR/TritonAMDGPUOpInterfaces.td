#ifndef TRITON_AMDGPU_OP_INTERFACES
#define TRITON_AMDGPU_OP_INTERFACES

include "mlir/IR/OpBase.td"

def TDMOpInterface : OpInterface<"TDMOpInterface"> {
  let description = [{
    Common interface for AMD TDM (Tensor DMA) data-movement operations.
  }];

  let cppNamespace = "::mlir::triton::amdgpu";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the tensor descriptor",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getDesc">
  ];
}

def BufferOpInterface : OpInterface<"BufferOpInterface"> {
  let description = [{
    This interface is implemented by buffer load/store operations.
    It provides methods to access common properties such base pointer, offset,
    mask and others.

    `ptr` is a scalar global pointer or, for loads and stores, a buffer
    descriptor (`amdg.make_buffer_desc`). A pointer gets a default descriptor
    with a 2^31 - 2 byte bound. `offsets` and `soffset` count elements of the
    accessed type. Element `i` accesses `base + soffset + offsets[i]`.
    Bounds are those of the descriptor (see `amdg.make_buffer_desc`).
  }];

  let cppNamespace = "::mlir::triton::amdgpu";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get operation base: a scalar pointer or a buffer descriptor.",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getPtr">,
    InterfaceMethod<
      /*desc=*/"Get mutable operation base ptr.",
      /*retType=*/"::mlir::OpOperand &",
      /*methodName=*/"getPtrMutable">,
    InterfaceMethod<
      /*desc=*/"Get operation offset tensor.",
      /*retType=*/"::mlir::TypedValue<::mlir::TensorType>",
      /*methodName=*/"getOffsets">,
    InterfaceMethod<
      /*desc=*/"Get mutable operation offset tensor.",
      /*retType=*/"::mlir::OpOperand &",
      /*methodName=*/"getOffsetsMutable">,
    InterfaceMethod<
      /*desc=*/"Get operation stride.",
      /*retType=*/"::mlir::TypedValue<::mlir::IntegerType>",
      /*methodName=*/"getStride">,
    InterfaceMethod<
      /*desc=*/"Get mutable operation stride.",
      /*retType=*/"::mlir::MutableOperandRange ",
      /*methodName=*/"getStrideMutable">,
    InterfaceMethod<
      /*desc=*/"Get the scalar offset (an i32 or an i32 tensor), in elements, or null.",
      /*retType=*/"::mlir::Value",
      /*methodName=*/"getSoffset",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return {};">
  ];
}

#endif // TRITON_AMDGPU_OP_INTERFACES
