#ifndef PROTONGPU_ATTR_DEFS
#define PROTONGPU_ATTR_DEFS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "proton/dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"

def GranularityAttr : I32EnumAttr<
  "Granularity", "",
  [
    I32EnumAttrCase<"CTA", 0, "cta">,
    I32EnumAttrCase<"WARP", 1, "warp">,
    I32EnumAttrCase<"WARPGROUP_2", 2, "warpgroup_2">,
    I32EnumAttrCase<"WARPGROUP_4", 3, "warpgroup_4">,
    I32EnumAttrCase<"WARPGROUP_8", 4, "warpgroup_8">
  ]> {
  let cppNamespace = "::mlir::triton::proton::gpu";
}

def PTG_StackMemorySpace : AttrDef<ProtonGPU_Dialect, "StackMemorySpace"> {
  let cppNamespace = "::mlir::triton::proton::gpu";
  let mnemonic = "stack_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to stack memory.
  }];
}

def PTG_HeapMemorySpace : AttrDef<ProtonGPU_Dialect, "HeapMemorySpace"> {
  let cppNamespace = "::mlir::triton::proton::gpu";
  let mnemonic = "heap_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to heap memory.
  }];
}

#endif // PROTONGPU_ATTR_DEFS
