#ifndef TRITON_INTERFACES
#define TRITON_INTERFACES

include "mlir/IR/OpBase.td"

def TensorSizeTrait : NativeOpTrait<"TensorSizeTrait">;
def VerifyTensorLayoutsTrait : NativeOpTrait<"VerifyTensorLayoutsTrait">;
def SameOperandsEncoding : NativeOpTrait<"SameOperandsEncoding">;
def SameOperandsAndResultEncoding : NativeOpTrait<"SameOperandsAndResultEncoding">;
def SameLoadStoreOperandsShape : NativeOpTrait<"SameLoadStoreOperandsShape">;
def SameLoadStoreOperandsAndResultShape : NativeOpTrait<"SameLoadStoreOperandsAndResultShape">;
def SameLoadStoreOperandsEncoding : NativeOpTrait<"SameLoadStoreOperandsEncoding">;
def SameLoadStoreOperandsAndResultEncoding : NativeOpTrait<"SameLoadStoreOperandsAndResultEncoding">;

// Like SameOperandsAndResult{Shape,Encoding}, except those require at least one
// result, whereas these allow 0 results.
def SameOperandsAndResultIfPresentShape : NativeOpTrait<"SameOperandsAndResultIfPresentShape">;
def SameOperandsAndResultIfPresentEncoding : NativeOpTrait<"SameOperandsAndResultIfPresentEncoding">;

#endif // TRITON_INTERFACES
