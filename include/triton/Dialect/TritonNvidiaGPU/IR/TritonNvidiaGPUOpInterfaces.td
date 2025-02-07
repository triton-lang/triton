#ifndef TRITON_NVIDIAGPU_OP_INTERFACES
#define TRITON_NVIDIAGPU_OP_INTERFACES

include "mlir/IR/OpBase.td"

def MMAv5OpInterface : OpInterface<"MMAv5OpInterface"> {
  let description = [{
  }];

  let cppNamespace = "::mlir::triton::nvidia_gpu";

  let methods = [
    InterfaceMethod<"",
                    "::mlir::Value",
                    "useAccumulator">,
    // InterfaceMethod<"",
    //                 "",
    //                 "">,
    // InterfaceMethod<"",
    //                 "",
    //                 "">,
    // InterfaceMethod<"",
    //                 "",
    //                 "">,
  ];
}
#endif // TRITON_NVIDIAGPU_OP_INTERFACES
