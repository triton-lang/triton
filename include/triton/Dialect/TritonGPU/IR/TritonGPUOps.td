#ifndef TRITONGPU_OPS
#define TRITONGPU_OPS

include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class TTG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonGPU_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def TTG_ConvertLayoutOp : TTG_Op<"convert_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType,
                                  Pure]> {
  let summary = "convert layout";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTG_AsyncWaitOp : TTG_Op<"async_wait"> {
  let summary = "async wait";

  let arguments = (ins Variadic<TTG_AsyncToken>:$asyncToken, I32Attr:$num);

  let results = (outs TTG_AsyncToken:$retToken);

  let assemblyFormat = "$asyncToken attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

def TTG_AsyncCommitGroupOp : TTG_Op<"async_commit_group"> {
  let summary = "async commit group";

  let results = (outs TTG_AsyncToken:$asyncToken);
  let arguments = (ins Variadic<TTG_AsyncToken>:$inputTokens);

  let assemblyFormat = [{
    $inputTokens attr-dict
  }];

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

def TTG_AsyncCopyGlobalToLocalOp : TTG_Op<"async_copy_global_to_local", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TypesMatchWith<"infer mask type from src type",
                 "src", "mask", "getI1SameShape($_self)",
                 "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
  TypesMatchWith<"infer other type from src type",
                 "src", "other", "getPointeeType($_self)",
                 "($_op.getOperands().size() <= 4) || std::equal_to<>()">
]> {
  let summary = "copy data from global memory to local memory asynchronously";

  let hasVerifier = 1;
  let description = [{
    This operation copies data from global memory to local memory asynchronously.
    This is analogue to tt.load except the data are copied to local memory pointed
    by by the memory descriptor instead of a distributed tensor. The rest of the
    operands are the same as tt.load.
  }];

  let arguments = (
    ins TT_PtrTensor:$src,
    TTG_MemDescType:$result,
    Optional<I1Tensor>:$mask,
    Optional<TT_Type>:$other,
    DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile
  );

  let builders = [
      OpBuilder<(ins "Value":$src, "Value":$result,
                     "triton::CacheModifier":$cache,
                     "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
  ];

  let results = (outs TTG_AsyncToken:$token);

  let extraClassDeclaration = [{
    static DenseSet<unsigned> getEligibleLoadByteWidth(int computeCapability) {
      DenseSet<unsigned> validLoadBytes;
      if (computeCapability >= 80) {
        validLoadBytes = {4, 8, 16};
      }
      return validLoadBytes;
    }
  }];

  // Specify cacheModifier and evictionPolicy explicitly, instead of leaving
  // them in attr-dict, because this way their values get printed as strings,
  // rather than as opaque integers.
  //
  // Note there are no commas between other, cacheModifier, and evictionPolicy,
  // due to limitations in MLIR's asm parser.
  let assemblyFormat = [{
    $src `,` $result (`mask` $mask^)? (`other` $other^)?
    oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
    attr-dict `:` type($src) `->` type($result)
  }];
}


// Allocate shared memory
def TTG_LocalAllocOp : TTG_Op<"local_alloc", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "allocate tensor";
  let description = [{
    This operation allocates buffer in shared memory and return a descriptor
    containing the address and a view of the buffer.

    Explicitly deallocating a buffer is optional; see local_dealloc.
  }];
  let arguments = (
    ins
    Optional<TT_Tensor>:$src,
    OptionalAttr<I32Attr>:$alignment
  );

  let builders = [
    OpBuilder<(ins "Type":$result),
              [{ build($_builder, $_state, result, Value(), IntegerAttr()); }]>,
    OpBuilder<(ins "Type":$result, "Value":$src),
              [{ build($_builder, $_state, result, src, IntegerAttr()); }]>,
    OpBuilder<(ins "Type":$result, "Value":$src, "int32_t":$alignment),
              [{ build($_builder, $_state, result, src, $_builder.getI32IntegerAttr(alignment)); }]>
  ];

  let extraClassDeclaration = [{
    bool isSharedMemoryAlloc() {
      return getType().getMemorySpace() &&
             isa<SharedMemorySpaceAttr>(getType().getMemorySpace());
    }
    int32_t getAlignmentOrDefault();
  }];
  let assemblyFormat = [{$src attr-dict `:` functional-type(operands, results)}];

  let results = (outs TTG_MemDescType:$result);
  let hasFolder = 1;
  let hasVerifier = 1;
}

// Deallocate shared memory
def TTG_LocalDeallocOp : TTG_Op<"local_dealloc", [MemoryEffects<[MemFree<SharedMemory>]>]> {
  let summary = "dealloc buffer";

  let description = [{
    This operation deallocates a buffer explicitly. Using the buffer after this
    operation is undefined.

    This operation is optional.  If you don't explicitly dealloc a buffer, the
    compiler assumes it's deallocated at the first point that post-dominates all
    uses of the alloc.

    Because we assume a memdesc is dead at the first point that post-dominates
    its uses, ops that wait for an async operation on a memdesc to complete
    (such as ttng.warp_group_dot_wait) should also take the memdesc as an
    operand.
  }];

  let arguments = (ins TTG_MemDescType:$src);

  // Use qualified() otherwise "!ttg.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src attr-dict `:` qualified(type($src))}];
}

def TTG_MemDescSubviewOp : TTG_Op<"memdesc_subview", [Pure]> {
  let summary = "take a subview of the descriptor.";

  let description = [{
    This operation returns a new descriptor representing a subview of the buffer.
    It doesn't affect the underlying memory. The subview can be rank-reduced.

    For example, suppose that
     - the input shape is 2x4x16xf16,
     - the output shape is 4x4xf16, and
     - offsets = [1, 0, 4].

    Then in Python syntax, the subview covers input[1][0:4][4:8].
  }];
  let arguments = (
    ins TTG_MemDescType:$src, Variadic<I32>:$offsets);

  // Use qualified() otherwise "!ttg.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src `[` $offsets `]` attr-dict `:` qualified(type($src)) `->` qualified(type($result))}];

  let results = (outs TTG_MemDescType:$result);

  let hasVerifier = 1;
}

def TTG_MemDescTransOp : TTG_Op<"memdesc_trans", [Pure,
                                                  TransposeOpInterface,
                                                  InferTypeOpWithLayoutEquivalence,
                                                  SameOperandsAndResultElementType]> {
  let summary = "transpose the descriptor";

  let description = [{
    This operation returns a new descriptor
    representing a transposed view of the buffer.
  }];

  let arguments = (ins TTG_MemDescType:$src, Variadic<I32>:$order);

  let arguments = (
    ins TTG_MemDescType:$src,
    DenseI32ArrayAttr:$order
  );

  let results = (outs TTG_MemDescType:$result);

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  let hasFolder = 1;
}

def TTG_LocalLoadOp : TTG_Op<"local_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load a buffer from local memory into a distributed tensor";

  let description = [{
    Load a tensor from the local memory descriptor into a distributed tensor.
  }];
  let arguments = (ins TTG_MemDescType:$src, Optional<TTG_AsyncToken> :$token);

  let builders = [
      OpBuilder<(ins "Type":$retType, "Value":$src),
      [{
      build($_builder, $_state, retType, src, /*token=*/static_cast<mlir::Value>(nullptr));
      }]>];

  // Use qualified() otherwise "!ttg.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src (`token` $token^)? attr-dict `:` qualified(type($src)) `->` type($result)}];

  let results = (outs TT_Tensor:$result);
}

def TTG_LocalStoreOp : TTG_Op<"local_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store a distributed tensor into a buffer in local memory";

  let description = [{
    Store a distributed tensor into a buffer in local memory.
  }];
  let arguments = (ins TT_Tensor:$src, TTG_MemDescType:$dst);

  let hasVerifier = 1;
  // Use qualified() otherwise "!ttg.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `->` qualified(type($dst))
  }];
}

def TTG_UpcastMXFPOp : TTG_Op<"upcast_mxfp", [Pure]> {
  let summary = "Convert an mxfp tensor to bf16/fp16";

  let hasVerifier = 1;

  let description = [{
    Compute the bf16 encoded in the given mxfp number as per
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  }];
  let arguments = (
    ins
    TT_Tensor:$src,
    TT_Tensor:$scale,
    TT_ScaleDotElemTypeAttr:$fp_type,
    BoolAttr:$fastMath
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src `,` $scale  `fp_type` `=` $fp_type attr-dict `:` type($src) `,` type($scale) `->` type($result)
  }];

  let extraClassDeclaration = [{
    static RankedTensorType deduceOutputType(
        TypedValue<RankedTensorType> inputTensor, ScaleDotElemType inputElemType, Type outputElemType);
  }];
}

// Allocate global memory
def TTG_GlobalScratchAllocOp : TTG_Op<"global_scratch_alloc", [MemoryEffects<[MemAlloc<GlobalMemory>]>]> {
  let summary = "allocate a global memory buffer";
  let description = [{
    This operation allocates a buffer in global memory that is private to the current program.
  }];
  let arguments = (
    ins
    I32Attr:$nbytes,
    I32Attr:$alignment
  );
  let results = (outs TT_Ptr:$result);

  let builders = [
    OpBuilder<(ins "Type":$result, "int32_t":$nbytes, "int32_t":$alignment),
              [{ build($_builder, $_state, result,
                       $_builder.getI32IntegerAttr(nbytes), $_builder.getI32IntegerAttr(alignment)); }]>
  ];

  let assemblyFormat = [{attr-dict `:` qualified(type($result))}];
}

#endif
