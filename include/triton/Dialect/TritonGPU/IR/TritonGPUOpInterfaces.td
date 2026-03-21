#ifndef TRITONGPU_OP_INTERFACES
#define TRITONGPU_OP_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def UpcastFpOpInterface : OpInterface<"UpcastFpOpInterface"> {
    let description = [{
        This interface is for operations that upcast floating-point numbers.
    }];

    let cppNamespace = "::mlir::triton::gpu";

    let methods = [
        InterfaceMethod<
            /*desc=*/"Infer destination encoding",
            /*retType=*/"mlir::Attribute",
            /*methodName=*/"inferDstEncoding",
            /*args=*/(ins "unsigned":$opIdx, "mlir::Attribute":$srcEnc)
        >,
        InterfaceMethod<
            /*desc=*/"Infer operand encoding from dst encoding",
            /*retType=*/"mlir::Attribute",
            /*methodName=*/"inferSrcEncoding",
            /*args=*/(ins "unsigned":$opIdx, "mlir::Attribute":$dstEnc)
        >
    ];
}

def VerifyMemDescLayoutsTrait : NativeOpTrait<"VerifyMemDescLayoutsTrait">;

// A trait equivalent to InferTypeOpAdaptor, but that checks for structural
// equivalence of the layouts of the result rather than just layout equality.
def InferMemDescTypeOpWithLayoutEquivalence : InferTypeOpAdaptorBase<[{
  static bool isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
    if (lhs.size() != rhs.size())
      return false;
    return llvm::all_of(llvm::zip(lhs, rhs), [](auto tup) {
      auto [lhs, rhs] = tup;
      return succeeded(OpTrait::impl::verifyEquivalentMemDescType(lhs, rhs));
    });
  }
}]>;

#endif // TRITONGPU_OP_INTERFACES
