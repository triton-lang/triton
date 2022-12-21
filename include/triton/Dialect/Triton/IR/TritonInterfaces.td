#ifndef TRITON_INTERFACES
#define TRITON_INTERFACES

include "mlir/IR/OpBase.td"

def TensorSizeTrait : NativeOpTrait<"TensorSizeTrait">;
def SameOperandsAndResultEncoding : NativeOpTrait<"SameOperandsAndResultEncoding">;
def SameOperandsEncoding : NativeOpTrait<"SameOperandsEncoding">;


#endif // TRITON_INTERFACES