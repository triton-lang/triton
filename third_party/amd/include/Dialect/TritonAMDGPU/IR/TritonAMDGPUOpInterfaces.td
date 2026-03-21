#ifndef TRITON_AMDGPU_OP_INTERFACES
#define TRITON_AMDGPU_OP_INTERFACES

include "mlir/IR/OpBase.td"

def BufferOpInterface : OpInterface<"BufferOpInterface"> {
  let description = [{
    This interface is implemented by buffer load/store operations.
    It provides methods to access common properties such base pointer, offset, mask and others.
  }];

  let cppNamespace = "::mlir::triton::amdgpu";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get operation base ptr.",
      /*retType=*/"::mlir::TypedValue<::mlir::triton::PointerType>",
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
      /*methodName=*/"getStrideMutable">
  ];
}

#endif // TRITON_AMDGPU_OP_INTERFACES
