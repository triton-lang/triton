#ifndef TRITON_INTERFACES
#define TRITON_INTERFACES

include "mlir/IR/OpBase.td"

def TensorSizeTrait : NativeOpTrait<"TensorSizeTrait">;
def VerifyTensorLayoutsTrait : NativeOpTrait<"VerifyTensorLayoutsTrait">;
def DotLike : NativeOpTrait<"DotLike">;
def SameOperandsEncoding : NativeOpTrait<"SameOperandsEncoding">;
def SameOperandsAndResultEncoding : NativeOpTrait<"SameOperandsAndResultEncoding">;
def SameLoadStoreOperandsShape : NativeOpTrait<"SameLoadStoreOperandsShape">;
def SameLoadStoreOperandsAndResultShape : NativeOpTrait<"SameLoadStoreOperandsAndResultShape">;
def SameLoadStoreOperandsEncoding : NativeOpTrait<"SameLoadStoreOperandsEncoding">;
def SameLoadStoreOperandsAndResultEncoding : NativeOpTrait<"SameLoadStoreOperandsAndResultEncoding">;

#endif // TRITON_INTERFACES
