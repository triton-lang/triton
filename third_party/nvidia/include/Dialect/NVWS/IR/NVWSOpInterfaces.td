#ifndef NVWS_OP_INTERFACES
#define NVWS_OP_INTERFACES

include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"

def NVWS_DescriptorLoadOpInterface : OpInterface<"DescriptorLoadOpInterface", [TT_DescriptorOpInterface]> {
  let cppNamespace = "::mlir::triton::nvws";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the transaction counts",
      /*retType=*/"int",
      /*methodName=*/"getTxCount",
      /*args=*/(ins)>,
  ];
}

def NVWS_ArefStageInterface : OpInterface<"ArefStageInterface"> {
  let cppNamespace = "::mlir::triton::nvws";

  let description = [{
     This interface implements setStage/getStage for aref ops
  }];

  // We can add more methods as needed.
  let methods = [
    InterfaceMethod<"Return aref stage",
                    "::mlir::Value",
                    "getStage">,
    InterfaceMethod<"Set aref stage",
                    "void",
                    "setStage",
                    (ins "::mlir::Value":$stage)>,
  ];
}

#endif // NVWS_OP_INTERFACES
