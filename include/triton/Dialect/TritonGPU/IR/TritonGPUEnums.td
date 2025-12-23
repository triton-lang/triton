#ifndef TRITONGPU_ENUMS
#define TRITONGPU_ENUMS

include "mlir/IR/EnumAttr.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"

// Bitmask enum describing which memory domains a barrier/fence orders.
def TTG_AddrSpace : I32BitEnumAttr<
    "AddrSpace", "",
    [
      I32BitEnumAttrCase<"None", 0b0000, "none">,
      I32BitEnumAttrCase<"Local", 0b0001, "local">,
      I32BitEnumAttrCase<"GlobalRead", 0b0010, "global_read">,
      I32BitEnumAttrCase<"GlobalWrite", 0b0100, "global_write">,
      I32BitEnumAttrCase<"TensorRead", 0b1000, "tensor_read">,
      I32BitEnumAttrCase<"TensorWrite", 0b10000, "tensor_write">,
      I32BitEnumAttrCase<"All", 0b11111, "all">
    ]> {
  let cppNamespace = "::mlir::triton::gpu";
}

#endif // TRITONGPU_ENUMS
