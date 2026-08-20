#ifndef TRITON_TYPE_ENUMS
#define TRITON_TYPE_ENUMS

include "mlir/IR/EnumAttr.td"

// Address space of a Triton pointer type (`!tt.ptr`). Case values mirror the
// LLVM address spaces:
//   Global     (1): NVPTXAS ADDRESS_SPACE_GLOBAL   / AMDGPUAS GLOBAL_ADDRESS
//   Descriptor (0): NVPTXAS ADDRESS_SPACE_GENERIC  / AMDGPUAS FLAT_ADDRESS
//   Constant   (4): NVPTXAS ADDRESS_SPACE_GLOBAL   / AMDGPUAS CONSTANT_ADDRESS
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
