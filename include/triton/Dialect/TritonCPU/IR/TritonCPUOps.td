#ifndef TRITONCPU_OPS
#define TRITONCPU_OPS

include "triton/Dialect/TritonCPU/IR/TritonCPUDialect.td"
include "triton/Dialect/TritonCPU/IR/TritonCPUTypes.td"
include "triton/Dialect/TritonCPU/IR/TritonCPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class TTC_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonCPU_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def TTC_ExtractMemRefOp : TTC_Op<"extract_memref", [NoMemoryEffect]> {
  let summary = "Extract base memref from a block pointer";

  let description = [{
    Extract base memref from a block pointer. It covers whole base tensor memory,
    not only the block referenced. Base pointer, shape, and strides are used
    in the resulting memref. Offsets and block shape are ignored.

  }];

  let arguments = (ins TT_TensorPtr:$src);

  let results = (outs AnyRankedOrUnrankedMemRef:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTC_ExtractIndicesOp : TTC_Op<"extract_indices", [NoMemoryEffect]> {
  let summary = "Extract indices from a block pointer.";

  let description = [{
    Extract indices that can be used to access the block using its base memref.
    Indices are supposed to be used for vector loads/stores with the base
    memref extracted from the same block pointer.
  }];

  let arguments = (ins TT_TensorPtr:$src);

  let results = (outs Variadic<Index>:$result);

  let builders = [
    OpBuilder<(ins "Value":$src)>
  ];

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTC_PtrToMemRefOp : TTC_Op<"ptr_to_memref", [NoMemoryEffect]> {
  let summary = "Build a memref for a pointer.";

  let description = [{
    Build memref with static shape, offset, strides, and specified base pointer.
  }];

  let arguments = (ins TT_Ptr:$src);

  let results = (outs AnyStaticShapeMemRef:$result);

  let hasCanonicalizer = 0;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif
