#ifndef TRITON_NVIDIAGPU_OP_INTERFACES
#define TRITON_NVIDIAGPU_OP_INTERFACES

include "mlir/IR/OpBase.td"

def MMAv5OpInterface : OpInterface<"MMAv5OpInterface"> {
  let description = [{
     This interface is implemented by MMAv5 dot and dot scaled ops.
  }];

  let cppNamespace = "::mlir::triton::nvidia_gpu";

  // We can add more methods as needed.
  let methods = [
    InterfaceMethod<"Return the A operand.",
                    "::mlir::TypedValue<::mlir::triton::gpu::MemDescType>",
                    "getA">,
    InterfaceMethod<"Return the accumulator init flag.",
                    "::mlir::Value",
                    "useAccumulator">,
    InterfaceMethod<"Set the accumulator init flag.",
                    "void",
                    "setUseAccumulator",
                    (ins "::mlir::Value":$flag)>,
    InterfaceMethod<"Associate a new completion barrier to this MMAv5 op.",
                    "void",
                    "addCompletionBarrier",
                    (ins "::mlir::Value":$barrier, "::mlir::Value":$pred)>,
    InterfaceMethod<"Get mutable completion barriers of this MMAv5 op.",
                    "::mlir::MutableOperandRange",
                    "getBarriersMutable">,
    InterfaceMethod<"Get mutable completion barriers predicates of this MMAv5 op.",
                    "::mlir::MutableOperandRange",
                    "getBarrierPredsMutable">,
    InterfaceMethod<"Return the accumulator.",
                    "::mlir::TypedValue<::mlir::triton::gpu::MemDescType>",
                    "getAccumulator">,
    InterfaceMethod<"Set the accumulator.",
                    "void",
                    "setAccumulator",
                    (ins "::mlir::Value":$accum)>,
    InterfaceMethod<"Set value isAsync",
                    "void",
                    "setIsAsync",
                    (ins "bool":$isAsync)>,
    InterfaceMethod<"Return the predicate of this op.",
                    "::mlir::Value",
                    "getPredicate">,
    InterfaceMethod<"Set the predicate of this op.",
                    "void",
                    "setPredicate",
                    (ins "::mlir::Value":$pred)>,
    InterfaceMethod<"Get the memory dependencies of the accumulator.",
                    "::mlir::Value",
                    "getAccDep">,
    InterfaceMethod<"Get the mutable memory dependencies of the accumulator.",
                    "::mlir::MutableOperandRange",
                    "getAccDepMutable">,
    InterfaceMethod<"Get the produced write dependency of the accumulator.",
                    "::mlir::Value",
                    "getToken">,
  ];
}

def ArefEnterOpInterface : OpInterface<"ArefEnterOpInterface"> {
  let description = [{
    This interface is implemented by ArefPutEnterOp and ArefGetEnterOp.
  }];

  let cppNamespace = "::mlir::triton::nvidia_gpu";

  let methods = [
    InterfaceMethod<"Return the Aref operand.",
                    "::mlir::TypedValue<::mlir::triton::nvidia_gpu::ArefType>",
                    "getAref">,
    InterfaceMethod<"Return the index of this ArefEnterOp.",
                    "::mlir::TypedValue<::mlir::IntegerType>",
                    "getIndex">,
    InterfaceMethod<"Set the index of this ArefEnterOp.",
                    "void",
                    "setIndex",
                    (ins "::mlir::Value":$index)>,
    InterfaceMethod<"Return the buffers of this ArefEnterOp.",
                    "::mlir::Operation::result_range",
                    "getBuffers">,
    InterfaceMethod<"Return the tokens of this ArefEnterOp.",
                    "::mlir::Operation::result_range",
                    "getTokens">,
  ];
}

def ArefExitOpInterface : OpInterface<"ArefExitOpInterface"> {
  let description = [{
    This interface is implemented by ArefPutExitOp and ArefGetExitOp.
  }];

  let cppNamespace = "::mlir::triton::nvidia_gpu";

  let methods = [
    InterfaceMethod<"Return the Aref operand.",
                    "::mlir::TypedValue<::mlir::triton::nvidia_gpu::ArefType>",
                    "getAref">,
    InterfaceMethod<"Return the index of this ArefExitOp.",
                    "::mlir::TypedValue<::mlir::IntegerType>",
                    "getIndex">,
    InterfaceMethod<"Return producer or consumer op attributes of this ArefExitOp.",
                    "::mlir::ArrayAttr",
                    "getAssociatedOpAttrs">,
    InterfaceMethod<"Set the index of this ArefExitOp.",
                    "void",
                    "setIndex",
                    (ins "::mlir::Value":$index)>,
    InterfaceMethod<"Set producer or consumer op attributes of this ArefExitOp.",
                    "void",
                    "setAssociatedOpAttrs",
                    (ins "::mlir::ArrayAttr":$attributes)>,
  ];
}
#endif // TRITON_NVIDIAGPU_OP_INTERFACES
