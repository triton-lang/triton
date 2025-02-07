#ifndef TRITON_NVIDIAGPU_OP_INTERFACES
#define TRITON_NVIDIAGPU_OP_INTERFACES

include "mlir/IR/OpBase.td"

def MMAv5OpInterface : OpInterface<"MMAv5OpInterface"> {
  let description = [{
  }];

  let cppNamespace = "::mlir::triton::nvidia_gpu";

  // We can add more methods as needed.
  let methods = [
    InterfaceMethod<"",
                    "::mlir::Value",
                    "useAccumulator">,
    InterfaceMethod<"",
                    "void",
                    "setUseAccumulator",
                    (ins "::mlir::Value":$flag)>,
    InterfaceMethod<"",
                    "void",
                    "setBarrier",
                    (ins "::mlir::Value":$barrier)>,
    InterfaceMethod<"",
                    "::mlir::Value",
                    "getAccumulator">,
    InterfaceMethod<"",
                    "void",
                    "setAccumulator",
                    (ins "::mlir::Value":$accum)>,
    InterfaceMethod<"",
                    "::mlir::Value",
                    "getPredicate">,
    InterfaceMethod<"",
                    "void",
                    "setPredicate",
                    (ins "::mlir::Value":$pred)>,
  ];
}
#endif // TRITON_NVIDIAGPU_OP_INTERFACES
