#ifndef TRITONINSTRUMENT_ATTR_DEFS
#define TRITONINSTRUMENT_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

def TT_MemTypeAttr : I32EnumAttr<
    "MemType", "",
    [
        I32EnumAttrCase<"SHARED", 0, "shared">,
        I32EnumAttrCase<"TENSOR", 1, "tensor">,
    ]> {
    let cppNamespace = "::mlir::triton::instrument";
}

#endif // TRITONINSTRUMENT_ATTR_DEFS
