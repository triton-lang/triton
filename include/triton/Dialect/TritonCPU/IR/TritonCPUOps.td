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

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

def TTC_PrintOp : TTC_Op<"print", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Print at most a single scalar or vector (converted from tensor) on each line";

  let description = [{
    For converting tensor types to vector types.
    It only takes a single scalar or vector (tensor) element.
  }];

  let arguments = (ins StrAttr:$prefix, BoolAttr:$hex,
                   Variadic<AnyTypeOf<[TT_Float, TT_Int, TT_Ptr, TTC_Vector]>>:$val);

  let assemblyFormat = [{
    $prefix attr-dict (`:` $val^ `:` type($val))?
  }];

  let hasVerifier = 1;
}

def TT_AssertOp : TTC_Op<"assert", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "For correctness checking";
  let description = [{
    Takes a condition tensor, a message string, a file string, a function string, and a line number.
    If the condition is false, the message is printed, and the program is aborted.
  }];
  let arguments = (ins I1:$condition, StrAttr:$message, StrAttr:$file, StrAttr:$func, I32Attr:$line);
  let assemblyFormat = "$condition `,` $message `,` $file `,` $func `,` $line attr-dict `:` type($condition)";
}


#endif
