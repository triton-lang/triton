#ifndef TRITON_ATTR_ENUMS
#define TRITON_ATTR_ENUMS

include "mlir/IR/EnumAttr.td"

def TT_ReportValidityAttr : I32EnumAttr<
    "ReportValidity", "",
    [
        I32EnumAttrCase<"NONE", 1, "none">,
        I32EnumAttrCase<"PER_16B_FP32", 2, "per_16B_fp32">,
        I32EnumAttrCase<"PER_16B_FP16", 3, "per_16B_fp16">,
        I32EnumAttrCase<"PER_16B_FP8", 4, "per_16B_fp8">,
        I32EnumAttrCase<"PER_16B_FP4", 5, "per_16B_fp4">,
        I32EnumAttrCase<"PER_ELEM_1B", 6, "per_elem_1B">
    ]> {
    let cppNamespace = "::mlir::triton";
}

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
