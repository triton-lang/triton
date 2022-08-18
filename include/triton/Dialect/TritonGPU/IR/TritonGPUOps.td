#ifndef TRITONGPU_OPS
#define TRITONGPU_OPS

include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // NoSideEffect
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType

def TT_BoolTensor : TensorOf<[I1]>;

def TT_BoolLike : AnyTypeOf<[I1, TT_BoolTensor]>;
def TT_IntegerLike : AnyTypeOf<[TT_Int, TT_IntegerTensor]>;
def TT_FloatLike : AnyTypeOf<[TT_Float, TT_FloatTensor]>;

class TTG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonGPU_Dialect, mnemonic, traits>;

def TTG_ConvertLayoutOp : TTG_Op<"convert_layout",
                                 [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "convert layout";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

def TTG_AsyncWaitOp : TTG_Op<"async_wait"> {
  let summary = "async wait";

  let arguments = (ins I32Attr:$num);

  let assemblyFormat = "attr-dict";
}

def TTG_CopyAsyncOp : TTG_Op<"copy_async",
                             [MemoryEffects<[MemRead, MemWrite]>,
                              TypesMatchWith<"infer mask type from ptr type",
                                              "ptr", "mask",
                                              "getI1SameShape($_self)">,
                              TypesMatchWith<"infer other type from ptr type",
                                             "ptr", "other",
                                             "getPointeeType($_self)">]> {
  let summary = "copy async";

  let arguments = (ins TT_PtrTensor:$ptr, BoolLike:$mask, TT_Type:$other,
                       TT_CacheModifierAttr:$cache, TT_EvictionPolicyAttr:$evict,
                       BoolAttr:$isVolatile,
                       BoolAttr:$isOtherUnspecified);

  let results = (outs TT_Type:$result);

  let assemblyFormat = "$ptr`,` $mask`,` $other attr-dict `:` type($ptr) `->` type($result)";

  // result needs to be of shared layout
  let verifier = [{ return ::verify(*this); }];
}

// Port Arith_CmpIOp & Arith_CmpFOp to TritonGPU.
// This is needed because Arith's Cmp ops don't
// handle encodings
// https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Arithmetic/IR/ArithmeticOps.td#L111
def TTG_CmpIOp : TTG_Op<"cmpi"> {
  let summary = "integer comparison operation";

  let description = [{}];

  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       TT_IntegerLike:$lhs,
                       TT_IntegerLike:$rhs);

  let results = (outs TT_BoolLike:$result);
}

def TTG_CmpFOp : TTG_Op<"cmpf"> {
  let summary = "floating-point comparison operation";

  let description = [{}];

  let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                       TT_FloatLike:$lhs,
                       TT_FloatLike:$rhs);

  let results = (outs TT_BoolLike:$result);
}

#endif
