#ifndef TRITONGPU_OPS
#define TRITONGPU_OPS

include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // NoSideEffect
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType

def TT_BoolTensor : TensorOf<[I1]>;

class TTG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonGPU_Dialect, mnemonic, traits>;

def TTG_ConvertLayoutOp : TTG_Op<"convert_layout",
                                 [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "convert layout";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);
}

def TTG_AsyncWaitOp : TTG_Op<"async_wait"> {
  let summary = "async wait";

  let arguments = (ins I32Attr:$num);
}

// def TTG_CopyAsyncOp : TTG_Op<"copy_async"> {}

// Port Arith_CmpIOp & Arith_CmpFOp to TritonGPU.
def TTG_CmpIOp : TTG_Op<"cmpi"> {
  let summary = "integer comparison operation";

  let description = [{}];

  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       TT_IntegerTensor:$lhs,
                       TT_IntegerTensor:$rhs);

  let results = (outs TT_BoolTensor:$result);
}

def TTG_CmpFOp : TTG_Op<"cmpf"> {
  let summary = "floating-point comparison operation";

  let description = [{}];

  let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                       TT_FloatTensor:$lhs,
                       TT_FloatTensor:$rhs);

  let results = (outs TT_BoolTensor:$result);
}

#endif
