#ifndef TRITON_ATTR_ENUMS
#define TRITON_ATTR_ENUMS

include "mlir/IR/EnumAttr.td"

def TT_CacheModifierAttr : I32EnumAttr<
    "CacheModifier", "",
    [
        I32EnumAttrCase<"NONE", 1, "none">,
        I32EnumAttrCase<"CA", 2, "ca">,
        I32EnumAttrCase<"CG", 3, "cg">,
        I32EnumAttrCase<"WB", 4, "wb">,
        I32EnumAttrCase<"CS", 5, "cs">,
        I32EnumAttrCase<"WT", 6, "wt">,
        I32EnumAttrCase<"CV", 7, "cv">,
    ]> {
  let cppNamespace = "::mlir::triton";
}

#endif // TRITON_ATTR_ENUMS
