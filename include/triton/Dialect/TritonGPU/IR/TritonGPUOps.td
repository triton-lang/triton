#ifndef TRITONGPU_OPS
#define TRITONGPU_OPS

include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def ResultsAreSharedEncoding: NativeOpTrait<"ResultsAreSharedEncoding">;

class TTG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonGPU_Dialect, mnemonic, traits>;

def TTG_ConvertLayoutOp : TTG_Op<"convert_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType,
                                  Pure]> {
  let summary = "convert layout";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let hasCanonicalizeMethod = 1;

  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

def TTG_AsyncWaitOp : TTG_Op<"async_wait"> {
  let summary = "async wait";

  let arguments = (ins I32Attr:$num);

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

def TTG_AsyncCommitGroupOp : TTG_Op<"async_commit_group"> {
  let summary = "async commit group";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}


// Port Arith_CmpIOp & Arith_CmpFOp & Std_SelectOp to TritonGPU.
// This is needed because these ops don't
// handle encodings
// e.g., https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td#L111
def TTG_CmpIOp : TTG_Op<"cmpi", [Pure, Elementwise,
                                 SameOperandsAndResultShape, 
                                 SameOperandsAndResultEncoding]> {
  let summary = "integer comparison operation";

  let description = [{}];

  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       TT_IntLike:$lhs,
                       TT_IntLike:$rhs);

  let results = (outs TT_BoolLike:$result);
}

def TTG_CmpFOp : TTG_Op<"cmpf", [Pure, Elementwise,
                                 SameOperandsAndResultShape, 
                                 SameOperandsAndResultEncoding]> {
  let summary = "floating-point comparison operation";

  let description = [{}];

  let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                       TT_FloatLike:$lhs,
                       TT_FloatLike:$rhs);

  let results = (outs TT_BoolLike:$result);
}

// TODO: migrate to arith::SelectOp on LLVM16
def TTG_SelectOp : TTG_Op<"select", [Pure, Elementwise,
                                     SameOperandsAndResultShape,
                                     SameOperandsAndResultEncoding]> {
  let summary = "select operation";

  let description = [{}];

  let arguments = (ins TT_BoolLike:$condition,
                       TT_Tensor:$true_value,
                       TT_Tensor:$false_value);

  let results = (outs TT_Tensor:$result);
}



def TTG_ExtractSliceOp : TTG_Op<"extract_slice",
                                [AttrSizedOperandSegments,
                                 ResultsAreSharedEncoding,
                                 Pure,
                                 OffsetSizeAndStrideOpInterface
                                 ]> {
  let summary = "extract slice operation";
  let description = [{
    same as tensor.extract_slice, but with int32 index. The motivations for re-implementing it are:
    We reimplement ExtractSliceOp with int32 index, because:
    - we want to enforce int32 indexing on GPUs since Triton tensors fit in SRAM
    - we still want to use indexWidth = 64 when lowering to LLVM because our loops can have
      64-bit induction variables and scf.for uses indexType for bounds/ivs
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<I32>:$offsets,
    Variadic<I32>:$sizes,
    Variadic<I32>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let builders = [
    // Build an ExtractSliceOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = [{
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Returns the type of the base tensor operand.
    RankedTensorType getSourceType() {
      return getSource().getType().cast<RankedTensorType>();
    }

    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }
  }];

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];
}

//

def TTG_InsertSliceAsyncOp : TTG_Op<"insert_slice_async",
                                    [AttrSizedOperandSegments,
                                     ResultsAreSharedEncoding,
                                     MemoryEffects<[MemRead]>,
                                     TypesMatchWith<"infer mask type from src type",
                                                    "src", "mask", "getI1SameShape($_self)",
                                                    "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
                                     TypesMatchWith<"infer other type from src type",
                                                    "src", "other", "getPointeeType($_self)",
                                                    "($_op.getOperands().size() <= 4) || std::equal_to<>()">]> {
  let summary = "insert slice async";

  let description = [{
      This operation inserts a tensor `$src` into another tensor `$dst` as specified by the operation’s
      `$index` argument and `$axis` attribute.

      It returns a copy of `$dst` with the proper slice updated asynchronously with the value of `$src`.
      This operation is non-blocking, and `$results` will have the updated value after the corresponding async_wait.

      When converting from `tt.load` to `triton_gpu.insert_slice_async`, the `$evict`, `$cache`, and `$isVolatile` fields
      might be ignored on certain hardware. For example, on NVIDIA GPUs, the cache policy is determined by the backend,
      and `$evict` and `$isVolatile` are ignored because they apply to L1 cache only.

      The insert_slice_async operation supports the following arguments:

      * src: the tensor that is inserted.
      * dst: the tensor into which the `$src` tensor is inserted.
      * index: the index of the `$src` tensor at the given `$axis` from which the `$dst` tensor is inserted into
      * mask: optional tensor-rank number of boolean masks which specify which
              elements of the `$src` tensor are inserted into the `$dst` tensor.
      * other: optional tensor-rank number of other tensors which specify what
              values are inserted into the `$dst` tensor if the corresponding
              element of the `$mask` tensor is false.

      In the future, we may decompose this operation into a sequence of:

      * `async` operation to specify a sequence of asynchronous operations
      * `load` operation to load a tensor from global memory
      * `insert_slice` operations to insert the `$src` tensor into the `$dst` tensor

      Example:

      ```
      %1 = triton_gpu.alloc_tensor : tensor<2x32xf32>
      %2 = triton_gpu.insert_slice_async %0, %1, %index { axis = 0 } : tensor<32x!tt.ptr<f32>, #AL> -> tensor<2x32xf32, #A>
      triiton_gpu.async_wait { num = 0 : i32 }
      ```
  }];

  let arguments = (ins TT_PtrTensor:$src, TT_Tensor:$dst, I32:$index,
                       Optional<I1Tensor>:$mask, Optional<TT_Type>:$other,
                       TT_CacheModifierAttr:$cache, TT_EvictionPolicyAttr:$evict,
                       BoolAttr:$isVolatile, I32Attr:$axis);

  let builders = [
      OpBuilder<(ins "Value":$src, "Value":$dst, "Value":$index,
                     "triton::CacheModifier":$cache,
                     "triton::EvictionPolicy":$evict, "bool":$isVolatile, "int":$axis)>,
      OpBuilder<(ins "Value":$src, "Value":$dst, "Value":$index, "Value":$mask,
                     "triton::CacheModifier":$cache,
                     "triton::EvictionPolicy":$evict, "bool":$isVolatile, "int":$axis)>,
      OpBuilder<(ins "Value":$src, "Value":$dst, "Value":$index,
                     "Value":$mask, "Value":$other,
                     "triton::CacheModifier":$cache,
                     "triton::EvictionPolicy":$evict, "bool":$isVolatile, "int":$axis)>,
  ];

  let results = (outs TT_Tensor:$result);

  //let assemblyFormat = [{
  //  $src `,` $dst ``
  //  $index, $mask, $other
  //  attr-dict `:` type($src) `->` type($dst)
  //}];

  let extraClassDeclaration = [{
    static DenseSet<unsigned> getEligibleLoadByteWidth(int computeCapability) {
      DenseSet<unsigned> validLoadBytes;
      if (computeCapability >= 80) {
        validLoadBytes = {4, 8, 16};
      }
      return validLoadBytes;
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def TTG_AllocTensorOp : TTG_Op<"alloc_tensor", [MemoryEffects<[MemAlloc]>,  // Allocate shared memory
                                                ResultsAreSharedEncoding]> {
  let summary = "allocate tensor";

  let description = [{
    This operation defines a tensor of a particular shape.
    The contents of the tensor are supposed to be in shared memory.

    Note: This op can be repalced to a `bufferization.alloc_tensor` in LLVM 16.
  }];

  let assemblyFormat = [{attr-dict `:` type($result)}];

  let results = (outs TT_Tensor:$result);
}

#endif
