#ifndef PROTONGPU_ATTR_DEFS
#define PROTONGPU_ATTR_DEFS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "proton/dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"

def GranularityAttr : I32EnumAttr<
  "Granularity", "",
  [
    I32EnumAttrCase<"WARPGROUP", 0, "warpgroup">,
    I32EnumAttrCase<"WARP", 1, "warp">,
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

#endif // PROTONGPU_ATTR_DEFS
