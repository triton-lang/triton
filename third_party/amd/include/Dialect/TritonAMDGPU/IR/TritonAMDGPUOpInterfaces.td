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
    It provides methods to access common properties such as base pointer,
    voffset, mask and others.
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
      /*desc=*/"Get operation voffset tensor.",
      /*retType=*/"::mlir::TypedValue<::mlir::TensorType>",
      /*methodName=*/"getVoffset">,
    InterfaceMethod<
      /*desc=*/"Get mutable operation voffset tensor.",
      /*retType=*/"::mlir::OpOperand &",
      /*methodName=*/"getVoffsetMutable">,
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
