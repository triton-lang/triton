#ifndef TRITONINSTRUMENT_ATTR_DEFS
#define TRITONINSTRUMENT_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

def TT_MemTypeAttr : I32EnumAttr<
    "MemType", "",
    [
        I32EnumAttrCase<"SHARED_MEM", 0, "shared_mem">,
        I32EnumAttrCase<"TENSOR_MEM", 1, "tensor_mem">,
    ]> {
    let cppNamespace = "::mlir::triton::instrument";
}

def TTI_TensorMapAccessKind : I32EnumAttr<"TensorMapAccessKind", "", [
    I32EnumAttrCase<"Read", 0, "read">,
    I32EnumAttrCase<"Publish", 1, "publish">,
    I32EnumAttrCase<"Acquire", 2, "acquire">,
    I32EnumAttrCase<"Use", 3, "use">
]> {
    let cppNamespace = "::mlir::triton::instrument";
}

#endif // TRITONINSTRUMENT_ATTR_DEFS
