#ifndef TRITON_TYPE_ENUMS
#define TRITON_TYPE_ENUMS

include "mlir/IR/EnumAttr.td"

// Address space of a Triton pointer type (`!tt.ptr`):
//   Global: generic global memory
//   Descriptor: pointer to tensor descriptor
//   Constant: read only global memory
def TT_PtrAddrSpaceAttr : I32EnumAttr<
    "PtrAddrSpace", "",
    [
        I32EnumAttrCase<"Global", 1, "global">,
        I32EnumAttrCase<"Descriptor", 0, "descriptor">,
        I32EnumAttrCase<"Constant", 4, "constant">,
    ]> {
    let cppNamespace = "::mlir::triton";
}

#endif // TRITON_TYPE_ENUMS
