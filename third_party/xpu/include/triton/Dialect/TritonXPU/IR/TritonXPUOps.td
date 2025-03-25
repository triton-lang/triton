//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITONXPU_OPS
#define TRITONXPU_OPS

include "mlir/IR/OpBase.td" // Trait
include "mlir/Dialect/Arith/IR/ArithBase.td" // Arith_CmpFPredicateAttr
include "triton/Dialect/TritonXPU/IR/TritonXPUDialect.td" // For TritonXPU_Dialect
include "triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.td" // For Attr
include "triton/Dialect/TritonXPU/IR/TritonXPUTypes.td" // For types
include "triton/Dialect/TritonXPU/IR/TritonXPUInterfaces.td" // For types


//===----------------------------------------------------------------------===//
// TRITONXPU op definitions
//===----------------------------------------------------------------------===//


class TTX_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonXPU_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// Memory Ops
//===----------------------------------------------------------------------===//

def TTX_AllocaOp : TTX_Op<"alloca"> {
    let summary = "alloca";
    let arguments = (ins SI64Attr:$size);
    let results = (outs TT_PtrLike:$result);
}

def TTX_GM2LMOp : TTX_Op<"gm2lm", [AttrSizedOperandSegments]> {
    let summary = "gm2lm";
    let arguments = (ins TT_PtrLike:$ptr,
                         Optional<TT_IntLike>:$len,
                         Optional<TT_PtrLike>:$bufPtr,
                         SI32Attr:$offsetState,
                         SI32Attr:$fixedStride,
                         SI64Attr:$rowLen,
                         SI64Attr:$rowStride,
                         SI32Attr:$lrie,
                         SI32Attr:$tensorColSize,
                         BoolAttr:$SVOpt,
                         BoolAttr:$async,
                         BoolAttr:$atomicSim);
    let results = (outs TT_PtrLike:$result);
}

def TTX_LM2GMOp : TTX_Op<"lm2gm", [AttrSizedOperandSegments]> {
    let summary = "lm2gm";
    let arguments = (ins TT_PtrLike:$ptr,
                         TTX_Type:$value,
                         Optional<TT_IntLike>:$len,
                         Optional<TT_PtrLike>:$bufPtr,
                         SI32Attr:$tensorColSize,
                         SI32Attr:$offsetState,
                         SI64Attr:$rowLen,
                         SI64Attr:$rowStride,
                         BoolAttr:$atomicSim);
}

def TTX_LoadOp : TTX_Op<"load", [AttrSizedOperandSegments,
                                 MemoryEffects<[MemRead]>]> {
    let summary = "load";
    let arguments = (ins TT_PtrLike:$ptr,
                     Optional<TT_BoolLike>:$mask,
                     Optional<TT_Type>:$other,
                     Optional<TT_Int>:$index,
                     SI32Attr:$stride,
                     SI32Attr:$tensorColSize,
                     BoolAttr:$isDiscrete,
                     BoolAttr:$SVOpt,
                     BoolAttr:$bf16Tofp32Unordered);
    let results = (outs TT_Type:$result);
}

def TTX_StoreOp : TTX_Op<"store", [AttrSizedOperandSegments,
                                   MemoryEffects<[MemWrite]>]> {
    let summary = "store";
    let arguments = (ins TT_PtrLike:$ptr,
                     TT_Type:$value,
                     Optional<TT_BoolLike>:$mask,
                     Optional<TT_Int>:$index,
                     SI32Attr:$tensorColSize,
                     BoolAttr:$bf16Tofp32Unordered);
}

def TTX_SM2GMOp : TTX_Op<"sm2gm", [AttrSizedOperandSegments]> {
    let summary = "sm2gm";
    let arguments = (ins TT_PtrLike:$ptr,
                         Optional<TTX_Type>:$value,
                         Optional<TT_IntLike>:$len,
                         Optional<TT_PtrLike>:$bufPtr,
                         SI32Attr:$offsetState);
}

def TTX_StoreSMOp : TTX_Op<"storeSM", [MemoryEffects<[MemWrite]>, AttrSizedOperandSegments]> {
    let summary = "store to SM";
    let arguments = (ins TT_PtrLike:$ptr, TT_Type:$value, Optional<TT_BoolLike>:$mask, Optional<TT_Int>:$index);
}


//===----------------------------------------------------------------------===//
// Context Ops
//===----------------------------------------------------------------------===//

def TTX_GetCoreIdOp : TTX_Op<"get_core_id"> {
    let summary = "get_core_id";
    let results = (outs TTX_Type:$result);
}

def TTX_GetThreadIdOp : TTX_Op<"get_thread_id"> {
    let summary = "get_thread_id";
    let description = [{
        threadType(0): tid = core_id() * cluster_num() + cluster_id()
        threadType(1): tid = core_num() * cluster_id() + core_id()
    }];
    let arguments = (ins SI32Attr:$threadType);
    let results = (outs TTX_Type:$result);
}

def TTX_GetClusterIdOp : TTX_Op<"get_cluster_id", [NoMemoryEffect]> {
  let summary = "Get the ID of the current Cluster";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTX_GetNumClusterOp : TTX_Op<"get_num_cluster", [NoMemoryEffect]> {
  let summary = "Get the number of launched Cluster";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// MakeRange-Liks Ops
//===----------------------------------------------------------------------===//
// TODO[dyq]: combine TTX_MakeRangeOp && TTX_InterleaveOp && TTX_OutRangeOp

// borrowed from triton/include/triton/Dialect/Triton/IR/TritonOps.td
def TTX_MakeRangeOp : TTX_Op<"make_range", [AttrSizedOperandSegments,
                                            Pure]> {
    let summary = "make range";

    let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

    // WARNING: MLIR generates getStart()/getEnd() functions which return
    // uint32_t, even though these arguments are to be interpreted as *signed*
    // int32 values.  If this matters, use get{Start,End}Attr().getInt(), which
    // return int64_t.
    let arguments = (ins I32Attr:$start,
                         I32Attr:$end,
                         I32Attr:$realSize,
                         Optional<TT_Int>:$loopIndex,
                         Optional<TT_Int>:$unrollIndex);

    let results = (outs TT_IntTensor:$result);

    let assemblyFormat = "$loopIndex $unrollIndex attr-dict `:` type($loopIndex) type($unrollIndex) `->` type($result)";

    // let hasFolder = 1;
    let hasVerifier = 1;
}

def TTX_InterleaveOp : TTX_Op<"interleave", [AttrSizedOperandSegments,
                                            Pure]> {
    let summary = "interleave";

    let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

    // WARNING: MLIR generates getStart()/getEnd() functions which return
    // uint32_t, even though these arguments are to be interpreted as *signed*
    // int32 values.  If this matters, use get{Start,End}Attr().getInt(), which
    // return int64_t.
    let arguments = (ins I32Attr:$start,
                         I32Attr:$end,
                         Optional<TT_Int>:$loopIndex,
                         Optional<TT_Int>:$unrollIndex);

    let results = (outs TT_IntTensor:$result);

    let assemblyFormat = "$loopIndex $unrollIndex attr-dict `:` type($loopIndex) type($unrollIndex) `->` type($result)";

    // let hasFolder = 1;
    let hasVerifier = 1;
}

def TTX_OutRangeOp : TTX_Op<"out_range"> {
    let summary = "out_range";
    let description = [{
        Returns idx.
    }];
    let arguments = (ins I32Attr:$groupsize, I32Attr:$rowspercore, TT_Int:$index);
    let results = (outs TT_IntTensor:$result);
}

//===----------------------------------------------------------------------===//
// Vectorization Ops
//===----------------------------------------------------------------------===//

class TTX_VUnaryFOp<string mnemonic, list<Trait> traits = []> :
    TTX_Op<mnemonic, traits # [SameOperandsAndResultType]>,
    Arguments<(ins TTX_VectorLike:$value)>,
    Results<(outs TTX_VectorLike:$result)> {
  let assemblyFormat = "$value `,` attr-dict `:` type($result)";
}

class TTX_BinaryOp<string mnemonic, list<Trait> traits = []> :
    TTX_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class TTX_VVIntBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTX_BinaryOp<mnemonic, traits>,
    Arguments<(ins TTX_VectorLike:$lhs, TTX_VectorLike:$rhs)>,
    Results<(outs TTX_VectorLike:$result)>;

class TTX_VVFloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTX_BinaryOp<mnemonic, traits>,
    Arguments<(ins TTX_VectorLike:$lhs, TTX_VectorLike:$rhs)>,
    Results<(outs TTX_VectorLike:$result)>;

class TTX_SVFloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTX_Op<mnemonic, traits>,
    Arguments<(ins TTX_Type:$lhs, TTX_Type:$rhs, I32Attr:$elemState)>,
    Results<(outs TTX_VectorLike:$result)> {
    let summary = mnemonic;
}

def TTX_VExpFOp : TTX_VUnaryFOp<"vexpf">;
def TTX_VLogFOp : TTX_VUnaryFOp<"vlogf">;
def TTX_VSinFOp : TTX_VUnaryFOp<"vsinf">;
def TTX_VCosFOp : TTX_VUnaryFOp<"vcosf">;
def TTX_VSqrtFOp : TTX_VUnaryFOp<"vsqrtf">;
def TTX_VAbsFOp : TTX_VUnaryFOp<"vabsf">;
def TTX_VSIToFPOp : TTX_Op<"vsitofp"> {
    let summary = "vsitofp";
    let arguments = (ins TTX_VectorLike:$value);
    let results = (outs TTX_VectorLike:$result);
}

def TTX_VvaddFOp : TTX_VVFloatBinaryOp<"vvaddf">;
def TTX_VvsubFOp : TTX_VVFloatBinaryOp<"vvsubf">;
def TTX_VvmulFOp : TTX_VVFloatBinaryOp<"vvmulf">;
def TTX_VvdivFOp : TTX_VVFloatBinaryOp<"vvdivf">;
def TTX_VvmaxFOp : TTX_VVFloatBinaryOp<"vvmaxf">;
def TTX_VvminFOp : TTX_VVFloatBinaryOp<"vvminf">;
def TTX_VvmaxNumFOp : TTX_VVFloatBinaryOp<"vvmaxnumf">;
def TTX_VvminNumFOp : TTX_VVFloatBinaryOp<"vvminnumf">;

def TTX_SvaddFOp : TTX_SVFloatBinaryOp<"svaddf">;
def TTX_SvmulFOp : TTX_SVFloatBinaryOp<"svmulf">;
def TTX_SvsubFOp : TTX_SVFloatBinaryOp<"svsubf">;
def TTX_SvmaxFOp : TTX_SVFloatBinaryOp<"svmaxf">;

def TTX_VvorIOp : TTX_VVIntBinaryOp<"vvori">;
def TTX_VvxorIOp : TTX_VVIntBinaryOp<"vvxori">;
def TTX_VvandIOp : TTX_VVIntBinaryOp<"vvandi">;
def TTX_VvaddIOp : TTX_VVIntBinaryOp<"vvaddi">;
def TTX_VvsubIOp : TTX_VVIntBinaryOp<"vvsubi">;
def TTX_VvmulIOp : TTX_VVIntBinaryOp<"vvmuli">;

def TTX_VMacFOp : TTX_Op<"vmacf", [SameOperandsAndResultType]> {
    let summary = "vmacf";
    let arguments = (ins TTX_VectorLike:$value, TTX_VectorLike:$mulData, TTX_VectorLike:$addData);
    let results = (outs TTX_VectorLike:$result);
    let assemblyFormat = "$value `,` $mulData `,` $addData attr-dict `:` type($result)";
}

def TTX_VConstOp : TTX_Op<"vconst"> {
    let summary = "vconst";
    let arguments = (ins AnyAttr:$value);
    let results = (outs TTX_VectorLike:$result);
}

def TTX_VSplatOp : TTX_Op<"vsplat"> {
    let summary = "vsplat";
    let arguments = (ins TTX_Type:$src);
    let results = (outs TTX_VectorLike:$result);
}

def TTX_VSelectOp : TTX_Op<"vselect"> {
    let summary = "vselect";
    let arguments = (ins TTX_Type:$condition,
                       TTX_Type:$true_value,
                       TTX_Type:$false_value);
    let results = (outs TTX_VectorLike:$result);
}

def TTX_VExtFOp : TTX_Op<"vextf"> {
    let summary = "vextf";
    let arguments = (ins TTX_VectorLike:$value);
    let results = (outs TTX_VectorLike:$result);
}

def TTX_VTruncFOp : TTX_Op<"vtruncf"> {
    let summary = "vtruncf";
    let arguments = (ins TTX_VectorLike:$value);
    let results = (outs TTX_VectorLike:$result);
}

def TTX_VCmpFOp : TTX_Op<"vcmpf"> {
    let summary = "vcmpf";
    let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                        TTX_VectorLike:$lhs,
                        TTX_VectorLike:$rhs);
    let results = (outs TTX_VectorLike:$result);
}

//===----------------------------------------------------------------------===//
// Other Ops
//===----------------------------------------------------------------------===//

def TTX_MinOp : TTX_Op<"min"> {
    let summary = "min";
    let arguments = (ins TTX_Type:$lhs, TTX_Type:$rhs);
    let results = (outs TTX_Type:$res);
}

def TTX_ExtractOp : TTX_Op<"extract"> {
    let summary = "extract the element from the tensor";
    let description = [{
        Returns an 1D int32.
        Extract the element from the tensor according to index.
    }];
    let arguments = (ins I32Attr:$index, TTX_Type:$tensor);
    let results = (outs TTX_Type:$result);
}

def TTX_ExtractSliceOp : TTX_Op<"extract_slice"> {
    let summary = "extract slice a part of elements from the tensor";
    let description = [{
        Returns an 1D int32.
        Truncate a part of elements from the tensor.
    }];
    let arguments = (ins TTX_Type:$tensor);
    let results = (outs TTX_Type:$result);
}

// borrowed from triton/include/triton/Dialect/Triton/IR/TritonOps.td
def TTX_ConvertLayoutOp : TTX_Op<"convert_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType,
                                  Pure]> {
  let summary = "convert layout";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

// borrowed from triton/include/triton/Dialect/Triton/IR/TritonOps.td
def TTX_ReduceOp: TTX_Op<"reduce",
                       [Pure,
                        SameOperandsEncoding,
                        SingleBlock,
                        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Reduction using generic combination algorithm";
    let arguments = (ins Variadic<TTX_Type>:$srcs, I32Attr:$axis, TT_Int:$loopIndex);
    let results = (outs Variadic<TT_Type>:$result);
    let regions = (region SizedRegion<1>:$combineOp);
    let builders = [
        OpBuilder<(ins "ValueRange":$srcs, "int":$axis, "Value":$loopIndex)>,
    ];
    let hasVerifier = 1;
    let hasRegionVerifier = 1;
    let extraClassDeclaration = [{
      llvm::SmallVector<RankedTensorType> getInputTypes();
      llvm::SmallVector<Type> getElementTypes();
      unsigned getNumOperands();
    }];
}

def TTX_ReduceReturnOp: TTX_Op<"reduce.return",
                             [HasParent<"ReduceOp">, Pure, Terminator, ReturnLike]> {
    let summary = "terminator for reduce operator";
    let arguments = (ins Variadic<AnyType>:$result);
    let assemblyFormat = "$result attr-dict `:` type($result)";
}

// borrowed from triton/include/triton/Dialect/Triton/IR/TritonOps.td
def TTX_BroadcastOp : TTX_Op<"broadcast", [Pure]> {
    let summary = "broadcast a tensor";

    let description = [{
      For a given tensor, broadcast changes one or more dimensions with size 1
      to a new size, e.g. tensor<1x32x1xf32> -> tensor<2x32x4xf32>.  You cannot
      change the size of a non-1 dimension.
    }];

    let arguments = (ins TTX_TensorVector:$src);

    let results = (outs TTX_TensorVector:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

#endif // TRITONXPU_OPS
