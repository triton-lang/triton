#ifndef PROTON_ATTR_DEFS
#define PROTON_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

def MetricAttr : I32EnumAttr<
  "Metric", "",
  [
    I32EnumAttrCase<"CYCLE", 0, "cycle">,
  ]> {
  let cppNamespace = "::mlir::triton::proton";
}

#endif // PROTON_ATTR_DEFS
