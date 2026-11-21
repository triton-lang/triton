#ifndef TRITONGPU_ENUMS
#define TRITONGPU_ENUMS

include "mlir/IR/EnumAttr.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"

def TTG_NVMmaPadding : I32EnumAttr<"NVMmaPadding", "NVIDIA shared memory packing", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"Fp4", 1, "fp4">,
    I32EnumAttrCase<"B6x16P32", 2, "b6x16_p32">,
    I32EnumAttrCase<"B6P2x16", 3, "b6p2x16">
  ]> {
  let cppNamespace = "::mlir::triton::gpu";
}

def TTG_SharedKind : I32EnumAttr<
    "SharedKind", "shared-memory access kind",
    [
      I32EnumAttrCase<"Generic", 0, "generic">,
      I32EnumAttrCase<"Async", 1, "async">,
      I32EnumAttrCase<"Barrier", 2, "barrier">
    ]> {
  let cppNamespace = "::mlir::triton::gpu";
}

def TTG_BarrierScope : I32EnumAttr<
    "BarrierScope", "barrier synchronization scope",
    [
      I32EnumAttrCase<"CTA", 0, "cta">,
      I32EnumAttrCase<"Warp", 1, "warp">
    ]> {
  let cppNamespace = "::mlir::triton::gpu";
}

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
