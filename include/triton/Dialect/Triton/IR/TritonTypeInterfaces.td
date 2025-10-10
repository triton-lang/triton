#ifndef TRITON_TYPE_INTERFACES
#define TRITON_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

// A minimal, IR-level interface that abstracts over tensor-like types and
// memory-descriptor-like types, exposing common layout and element properties.
// Implementations may be dynamically attached by target backends (e.g.,
// TritonGPU for RankedTensorType and MemDescType).
def TT_TensorOrMemDescLike : TypeInterface<"TensorOrMemDescLike"> {
  let cppNamespace = "::mlir::triton";
  let methods = [
    InterfaceMethod<"Returns the encoding attribute, if any",
      "mlir::Attribute", "getEncoding", (ins)>,
    InterfaceMethod<"Returns element type",
      "mlir::Type", "getElementType", (ins)>,
    InterfaceMethod<"Returns the logical shape",
      "llvm::ArrayRef<int64_t>", "getShape", (ins)>,
    InterfaceMethod<"Returns the allocation shape (may equal logical shape)",
      "llvm::ArrayRef<int64_t>", "getAllocShape", (ins)>,
    InterfaceMethod<"Returns the memory space attribute (may be null)",
      "mlir::Attribute", "getMemorySpace", (ins)>,
    InterfaceMethod<"Returns whether the underlying memory is mutable",
      "bool", "getMutableMemory", (ins)>
  ];
}

#endif // TRITON_TYPE_INTERFACES
