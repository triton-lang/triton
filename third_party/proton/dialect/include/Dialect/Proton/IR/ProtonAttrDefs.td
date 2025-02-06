#ifndef PROTON_ATTR_DEFS
#define PROTON_ATTR_DEFS

include "mlir/IR/AttrTypeBase.td"
include "ProtonDialect.td"

class Proton_Attr<string name, list<Trait> traits = [],
                  string baseCppClass = "::mlir::Attribute">
  : AttrDef<Proton_Dialect, name, traits, baseCppClass> {
}

def MetricAttr : I32EnumAttr<
  "Metric", "",
  [
    I32EnumAttrCase<"CYCLE", 0, "cycle">,
  ]> {
  let cppNamespace = "::mlir::triton::proton";
}

#endif // PROTON_ATTR_DEFS
