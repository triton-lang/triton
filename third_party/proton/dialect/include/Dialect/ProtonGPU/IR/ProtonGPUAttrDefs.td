#ifndef PROTONGPU_ATTR_DEFS
#define PROTONGPU_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

def GranularityAttr : I32EnumAttr<
  "Granularity", "",
  [
    I32EnumAttrCase<"WARPGROUP", 0, "warpgroup">,
    I32EnumAttrCase<"WARP", 1, "warp">,
  ]> {
  let cppNamespace = "::mlir::triton::proton::gpu";
}

#endif // PROTONGPU_ATTR_DEFS
