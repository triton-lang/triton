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

#endif // NVWS_OP_INTERFACES
