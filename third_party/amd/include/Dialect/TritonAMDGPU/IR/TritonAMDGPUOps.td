/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#ifndef TRITON_AMDGPU_OPS
#define TRITON_AMDGPU_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "TritonAMDGPUDialect.td"
include "TritonAMDGPUAttrDefs.td"


class TT_AMDGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonAMDGPU_Dialect, mnemonic, !listconcat(traits, [])>;

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

//===----------------------------------------------------------------------===//
// ExtractSliceOp
//===----------------------------------------------------------------------===//

def ExtractSliceOp : TT_AMDGPU_Op<"extract_slice", [Pure]> {
  let summary = "extract slice operation";
  let description = [{
    The "extract_slice" operation enables extracting a slice of a tensor in
    registers.

    The "extract_slice" operation supports the following arguments:

    * source: the base tensor on which to create a view tensor
    * offsets: offsets into the base tensor at which to create the view

    In distributed layouts, tensors are divided into CTA tiles.
    A CTA tile represents the smallest contiguous portion of a tensor that is
    distributed across all threads and warps within a workgroup.
    The ExtractSlice operation extracts a portion of the tensor that is a
    multiple of CTA tiles.

    The source and destination must have matching linear layouts at the CTA
    tile level. This ensures that the extract_slice is a no-op, meaning no data
    rearrangement between threads is required to extract the destination tensor
    with the given shape and layout.

      +-------+-------+
      |  W0   |  W1   |
      |       |       |
      |   +   |   +   |
      |  W2   |  W3   |  <-- Single CTA tile (distributed across warps W0-W3)
      |       |       |
      |   +   |   +   |
      |       |       |
      +-------+-------+
      |          Source Tensor                    Extracted Slice
      |             .                           +--------------+
      |             .                           |  W0  |  W1   |
      |             .                           |      |       |
      |                                         |  +   |   +   |
      |                                         |  W2  |  W3   |
      |                                         |      |       |
      |                                         |  +   |   +   |
      |                                         |      |       |
      |                                         +-------+------+
      |                                         |  W0  |   W1  |
      |                                         |      |       |
      |                                         |  +   |   +   |
      |                                         |  W2     W3   |
      |                                         |      |       |
      |                                         |  +   |   +   |
      |                                         |      |       |
      |                                         +--------------+


    This op is designed to work on logical tensors directly, avoiding the need
    for complex layout reinterpretation or reshaping. For example, the tt.split
    operation only supports splitting along the innermost dimension,
    and requires that the resulting innermost dimension provide 2 elements per thread,
    distributed across registers. In contrast, extract_slice op imposes no constraints
    on the extraction dimension or the size of dimensions.

    Example 1:

    ```mlir
    #blocked = #ttg.blocked<{sizePerThread = [1, 8],
        threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
    #blocked1 = #ttg.blocked<{sizePerThread = [1, 8],
        threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
    %1 = ttg.convert_layout %0 : tensor<128x128xf16, #blocked>
        -> tensor<128x128xf16, #blocked1>
    // create a slice of base tensor %1 with static offsets
    %2 = amdgpu.extract_slice %0 [0, 0] :
      tensor<128x128xf16, #blocked1> to tensor<128x32xf16, #blocked1>
    ```

    Example 1 shows how "extract_slice" operation may be used. In this example a
    new slice of 128x32 is created. "extract_slice" works on tensors
    where the desired slice has the same layout on a CTA tile as the source tensor.
    "%0" cannot be sliced directly as the resulting slice does not satisfy this condition.
    Therefore it needs to be converted to a layout suitable for slicing.
    "#blocked1" layout is appropriate for this as it keeps the
    sizePerThread the same thus keeping coalescing properties the same.
    In order to utilize all threads in a warp, "threadsPerWarp" is set to
    [16,4] for this new layout. This layout conversion carried out before
    using "extract_slice" ensures slicing still uses all threads efficiently. The
    size of the slice is determined by the result type.
    }];

  let arguments = (ins
    AnyRankedTensor:$source,
    DenseI64ArrayAttr:$static_offsets
  );
  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSource().getType().getRank();
      return {rank, rank, rank};
    }
  }];

  let assemblyFormat = [{
    $source $static_offsets attr-dict `:` type($source) `to` type($result)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def ConcatOp : TT_AMDGPU_Op<"concat", [Pure]> {
  let summary = "concat operation";
  let description = [{
    The "concat" operation combines a list of source n-dimensional tensors into a single larger destination tensor.

    All source tensors must have the same shape, element type, and encoding.
    The concatenation dimension is inferred from the source and destination shapes provided by the user.
    For example, two tensors of shape 64x128 can produce a destination shape of 128x128,
    indicating concatenation along dimension 0; or 64x256, indicating concatenation along dimension 1.

    Generally, source tensors passed as op arguments can be arranged into the resulting shape in multiple ways.
    For example, given four tensors of shape 64x64:
      concat s0<64x64>, s1<64x64>, s2<64x64>, s3<64x64> -> <128x128>

    They can be laid out in different configurations within the result tensor:
      1) s0 s1     2) s0 s2
         s2 s3        s1 s3

    From a logical tensor perspective, the source tensors are treated as elements of a tensor of tensors.
    In other words, the 1-D array of input tensors is conceptually reshaped into an n-D grid.
    The semantics of this op assume a row-major order (or its n-D generalization),
    meaning the fastest-varying dimension is filled first, and the slowest-varying dimension is filled last.
    In the example above, this corresponds to layout 1).

    The source and destination tensors must have identical linear layouts at the CTA tile level.
    That is, all base vectors for input dimensions must match, except for the register input dimension.
    The register basis must align on the subset that defines the logical tensor shape of a single CTA tile.

    This ensures that the concatenation is a no-op, meaning no data rearrangement among threads is required
    to assemble the destination tensor with the given shape and layout.
    However, the order of CTA tiles within the layout does not need to match between source and destination layouts.
    It is the responsibility of the op's lowering logic to handle this correctly.

    This op is designed to work on logical tensors directly, avoiding the need for complex layout reinterpretation or reshaping.
    For example, the `tt.join` operation only supports concatenation along the innermost dimension,
    and requires that the resulting innermost dimension provide 2 elements per thread, distributed across registers.
    In contrast, this `concat` op imposes no constraints on the concatenation dimension or the size of dimensions.

    * sources: a list of the input tensors.

    Example 1:

    ```mlir
    #blocked = #ttg.blocked<{sizePerThread = [1, 8],
        threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
    %0 = amdgpu.concat %arg0, %arg1: tensor<32x64xf32, #blocked>,tensor<32x64xf32, #blocked>,
      -> tensor<64x64xf32, #blocked>
    ```

    Example 2:
    ```mlir
    #src_layout = #ttg.linear<{register=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 64], [64, 0]], lane=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp=[[0, 32], [32, 0]], block=[]}>
    #dst_layout = #ttg.linear<{register=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 64], [0, 128], [64, 0], [128, 0]], lane=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp=[[0, 32], [32, 0]], block=[]}>
    %0 = amdgpu.concat %arg0, %arg1, %arg2, %arg3 : tensor<128x128xf16, #src_layout>, tensor<128x128xf16, #src_layout>, tensor<128x128xf16, #src_layout>,
                                                    tensor<128x128xf16, #src_layout> -> tensor<256x256xf16, #dst_layout>
    ```

    }];

  let arguments = (ins Variadic<TT_Tensor>:$sources);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $sources attr-dict `:` type($sources) `->` type($result)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// InstructionSchedHint
//===----------------------------------------------------------------------===//

def InstructionSchedHint : TT_AMDGPU_Op<"instruction_sched_hint", []> {
  let summary = "A placeholder op for instruction scheduling hints within a basic block";
  let description = [{
    A placeholder op for instruction scheduling hints applied to instructions within
    a basic block where the placeholder op is located. This op is primarily intended
    to be used to adjust instruction scheduling inside the resulting main loop
    of a `tt.dot` operation. It's easier to identify dot ops at a high level and, thus,
    to mark intended scheduling regions. The hint ops are eventually lowered
    into LLVM AMDGPU instruction scheduling primitives, which are meant to control
    how different kinds of instructions (valu/mfma, global/shared memory, etc.) should
    interleave for better instruction level parallelism.
  }];

  let arguments = (ins TritonAMDGPU_SchedHintVariantAttr:$variant);

  let assemblyFormat = [{ attr-dict }];
}

//===----------------------------------------------------------------------===//
// CondBarrierOp
//===----------------------------------------------------------------------===//

def CondBarrierOp : TT_AMDGPU_Op<"cond_barrier"> {
  let summary = "Conditionally set barriers to synchronize partial threads in a block";

  let description = [{
      condBarrierOp sets barrier instruction only when the given argument is true.
      This provides a way to synchronize partial threads in a block, deliberately
      diverges the execution sequences. However, user should guarantee all threads
      converge at the end by calling condBarrierOp(true) with the remaining threads.
      Conceptually, this is similar to having an execution barrier inside an if statement.
      This op allows us to avoid blocking the whole block when suitable to help scheduling.
      NB. This doesn't set any memory fence.
  }];

  let arguments = (ins I1:$pred);

  let assemblyFormat = "$pred attr-dict";
}

//===----------------------------------------------------------------------===//
// BufferLoadOp
//===----------------------------------------------------------------------===//

def BufferLoadOp : TT_AMDGPU_Op<"buffer_load", [
  SameLoadStoreOperandsAndResultEncoding,
  AttrSizedOperandSegments,
  TypesMatchWith<"result element type matches the pointed type of ptr", "result", "ptr", "getPointerTypeToElement($_self)">,
  TypesMatchWith<"result and offsets have the same shape", "result", "offsets", "getI32SameShape($_self)">,
  TypesMatchWith<"result and mask have the same shape", "result", "mask", "getI1SameShape($_self)",
                 "(cast<BufferLoadOp>($_op).getMask() == nullptr) || std::equal_to<>()">,
  TypesMatchWith<"result and other have the same type", "result", "other", "$_self",
                 "(cast<BufferLoadOp>($_op).getOther() == nullptr) || std::equal_to<>()">,
]>{
    let summary = "Load from a scalar base pointer and a tensor offset";
    let description = [{
      AMD Buffer load operation. Buffer store is similar to
      a normal store but it accesses global memory via a scalar base pointer
      and a tensor of offsets instead of a tensor of pointers. The other fields
      are similar to a normal load, i.e., the `mask` is a boolean vector that
      determines if a given element should be read from memory, and `other` is the
      element that should be returned on lane `i` when `mask[i] == 0`.
      Stride is the distance between the beginning of contiguous memory chunks.
      When performing a load of a block, the `stride` is the address difference between
      the first elements of each row in bytes. Compiler tries to obtain the `stride`
      when it converts to the buffer ops because it is important for optimizing
      the cache memory access.
    }];
    let arguments = (ins
      Arg<TT_Ptr, "Global memory scalar base pointer to load from", [MemRead<GlobalMemory>]>:$ptr,
      I32Tensor:$offsets,
      Optional<I32>:$stride,
      DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
      Optional<TT_BoolTensor>:$mask,
      Optional<TT_Tensor>:$other
    );
    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
      $ptr `[` $offsets `]` (`,` $mask^)? (`,` $other^)?
      oilist(`cacheModifier` `=` $cache)
      (`stride` `=` $stride^)?
      attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// BufferLoadToLocalOp
//===----------------------------------------------------------------------===//

def BufferLoadToLocalOp : TT_AMDGPU_Op<"buffer_load_to_local", [
  AttrSizedOperandSegments,
  TypesMatchWith<"dest element type matches pointee type of ptr", "dest", "ptr", "getPointerTypeToElement($_self)">,
  TypesMatchWith<"infer mask shape from offsets",
                 "offsets", "mask", "getI1SameShape($_self)",
                 "(cast<BufferLoadToLocalOp>($_op).getMask() == nullptr) || std::equal_to<>()">,
  TypesMatchWith<"other matches shape and layout of offsets and the element type matches the pointee type of ptr",
                 "offsets", "other", "cast<TensorType>($_self).clone(getPointeeType($ptr.getType()))",
                 "(cast<BufferLoadToLocalOp>($_op).getOther() == nullptr) || std::equal_to<>()">,
]>{
    let summary = "Load from a scalar base pointer and a tensor offset to shared memory";
    let description = [{
      AMD Buffer load operation. Similar to amdgpu.buffer_load op but directly wirtes to shared memory instead of into registers. }];
    let arguments = (ins
      Arg<TTG_MemDescType, "Shared memory slice to write to", [MemWrite<SharedMemory>]>:$dest,
      Arg<TT_Ptr, "Global memory scalar base pointer to load from", [MemRead<GlobalMemory>]>:$ptr,
      I32Tensor:$offsets,
      Optional<TT_BoolTensor>:$mask,
      Optional<TT_Tensor>:$other,
      Optional<I32>:$stride,
      DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache
    );
    let results = (outs TTG_AsyncToken:$token);

    let assemblyFormat = [{
      $ptr `[` $offsets `]` (`mask` `=` $mask^)? (`other` `=` $other^)? (`stride` `=` $stride^)?
      oilist(`cacheModifier` `=` $cache) `into` $dest
      attr-dict `:` type($ptr) `[` type($offsets) `]` type($other) `->` type($dest)
    }];
}

//===----------------------------------------------------------------------===//
// BufferAtomicRMWOp
//===----------------------------------------------------------------------===//

def BufferAtomicRMWOp : TT_AMDGPU_Op<"buffer_atomic_rmw", [
  AttrSizedOperandSegments,
  SameLoadStoreOperandsAndResultEncoding,
  TypesMatchWith<"result element type matches the value type", "result", "value", "$_self">,
  TypesMatchWith<"result element type matches the pointed type of ptr", "result", "ptr", "getPointerTypeToElement($_self)">,
  TypesMatchWith<"result and offsets have the same shape", "result", "offsets", "getI32SameShape($_self)">,
  TypesMatchWith<"result and mask have the same shape", "result", "mask", "getI1SameShape($_self)",
                 "(cast<BufferAtomicRMWOp>($_op).getMask() == nullptr) || std::equal_to<>()">,
  TypesMatchWith<"value element type matches the pointed type of ptr", "value", "ptr", "getPointerTypeToElement($_self)">,
  TypesMatchWith<"value and offsets have the same shape", "value", "offsets", "getI32SameShape($_self)">,
  TypesMatchWith<"value and mask have the same shape", "value", "mask", "getI1SameShape($_self)",
                 "(cast<BufferAtomicRMWOp>($_op).getMask() == nullptr) || std::equal_to<>()">,
]>{
    let summary = "Atomic RMW op which reads, modifies, and writes to a scalar base pointer and a tensor offset";
    let description = [{
        AMD Buffer atomic RMW operation. Buffer atomics are similar to normal atomics, but access global memory via a
        scalar base pointer and a tensor of offsets instead of a tensor of pointers.
        Similar to other buffer ops, the `mask` is a boolean vector that determines if a given element should be processed with
        the atomic RMW op. Elements with `mask[i] == 0` are dropped (i.e., the atomic is not executed).
        Similar to TT_AtomicRMWOp: Buffer atomic RMW ops load data at $ptr, do $rmw_op with $val, and store result to $ptr with
        the specified memory semantics and scope. Atomic RMW ops return the pre-op value if used, otherwise the value is implicitly dropped.
        Stride is the distance between the beginning of contiguous memory chunks. When performing a RMW, the `stride` is
        the address difference between the first elements of each row in bytes. Compiler tries to obtain the `stride`
        when it converts to the buffer ops because it is important for optimizing the cache memory access.
    }];
    let arguments = (ins
      TT_AtomicRMWAttr:$atomic_rmw_op,
      Arg<TT_Ptr, "Global memory pointer", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$ptr,
      I32Tensor:$offsets,
      TT_Tensor:$value,
      Optional<I32>:$stride,
      TT_MemSemanticAttr:$sem,
      TT_MemSyncScopeAttr:$scope,
      Optional<TT_BoolTensor>:$mask
    );
    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
        $atomic_rmw_op `,` $sem `,` $scope `,` $value `,` $ptr `[` $offsets `]` (`,` $mask^)?
        (`stride` `=` $stride^)?
        attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// BufferAtomicCASOp
//===----------------------------------------------------------------------===//
def BufferAtomicCASOp : TT_AMDGPU_Op<"buffer_atomic_cas", [
  SameLoadStoreOperandsAndResultEncoding,
  TypesMatchWith<"result element type matches the val type", "result", "val", "$_self">,
  TypesMatchWith<"result element type matches the cmp type", "result", "cmp", "$_self">,
  TypesMatchWith<"result element type matches the pointed type of ptr", "result", "ptr", "getPointerTypeToElement($_self)">,
  TypesMatchWith<"result and offsets have the same shape", "result", "offsets", "getI32SameShape($_self)">,
  TypesMatchWith<"val and offsets have the same shape", "val", "offsets", "getI32SameShape($_self)">,
  TypesMatchWith<"val and cmp have the same shape", "val", "cmp", "$_self">,
]>{
    let summary = "Atomic CAS op which does compare-exchange to a scalar base pointer and a tensor offset";
    let description = [{
        AMD Buffer Atomic CAS operation. Buffer atomics are similar to normal atomics, but access global memory via a
        scalar base pointer and a tensor of offsets instead of a tensor of pointers.
        Similar to TT_AtomicCASOp: Buffer atomic CAS op loads data at $ptr, and stores $val to $ptr atomically if value at $ptr equals $cmp, with
        the specified memory semantics and scope. Atomic CAS ops return the pre-op value if used, otherwise the value is implicitly dropped.
        Stride is the distance between the beginning of contiguous memory chunks. When performing a CAS, the `stride` is
        the address difference between the first elements of each row in bytes. Compiler tries to obtain the `stride`
        when it converts to the buffer ops because it is important for optimizing the cache memory access.
    }];
    let arguments = (ins
      Arg<TT_Ptr, "Global memory pointer", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$ptr,
      I32Tensor:$offsets,
      TT_Tensor:$cmp,
      TT_Tensor:$val,
      Optional<I32>:$stride,
      TT_MemSemanticAttr:$sem,
      TT_MemSyncScopeAttr:$scope
    );
    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
        $sem `,` $scope `,` $cmp `,` $val `,` $ptr `[` $offsets `]`
        (`stride` `=` $stride^)?
        attr-dict `:` type($result)
    }];
}

//===----------------------------------------------------------------------===//
// BufferStoreOp
//===----------------------------------------------------------------------===//

def BufferStoreOp : TT_AMDGPU_Op<"buffer_store", [
  AttrSizedOperandSegments,
  SameLoadStoreOperandsEncoding,
  TypesMatchWith<"value element type matches the pointed type of ptr", "value", "ptr", "getPointerTypeToElement($_self)">,
  TypesMatchWith<"value and offsets have the same shape", "value", "offsets", "getI32SameShape($_self)">,
  TypesMatchWith<"value and mask have the same shape", "value", "mask", "getI1SameShape($_self)",
                 "(cast<BufferStoreOp>($_op).getMask() == nullptr) || std::equal_to<>()">,
]>{
    let summary = "Store into scalar base pointer and a tensor offset";
    let description = [{
      AMD Buffer store operation. Buffer store is similar to
      normal store but it accesses global memory via a scalar base pointer
      and a tensor of offsets instead of a tensor of pointers. The other fields
      are similar to a normal store , i.e., the `mask` is a boolean vector that
      determines if a given element should be written to memory, and `value` is the
      tensor of elements that should be written on lane `i` when `mask[i] == 1`.
      Stride is the distance between the beginning of contiguous memory chunks.
      When performing a block store, the `stride` is the address difference between
      the first elements of each row in bytes. Compiler tries to obtain the `stride`
      when it converts to the buffer ops because it is important for optimizing
      the cache memory access.
    }];
    let arguments = (ins
      TT_Tensor:$value,
      Arg<TT_Ptr, "Global memory scalar base pointer to write to", [MemWrite<GlobalMemory>]>:$ptr,
      I32Tensor:$offsets,
      Optional<I32>:$stride,
      DefaultValuedAttr<TT_CacheModifierAttr, "mlir::triton::CacheModifier::NONE">:$cache,
      Optional<TT_BoolTensor>:$mask
    );

    let assemblyFormat = [{
      $value `,` $ptr `[` $offsets `]` (`,` $mask^)?
      oilist(`cacheModifier` `=` $cache)
      (`stride` `=` $stride^)?
      attr-dict `:` type($value)
    }];
}

//===----------------------------------------------------------------------===//
// UpcastMXFPOp
//===----------------------------------------------------------------------===//

def TTG_UpcastMXFPOp : TT_AMDGPU_Op<"upcast_mxfp", [Pure]> {
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

//===----------------------------------------------------------------------===//
// MaskedLoadOp
//===----------------------------------------------------------------------===//
def MaskedLoadOp : TT_AMDGPU_Op<"masked_load", []> {
  let summary = "Masked load operation";
  let description = [{
    Load operation with masking support. If the mask is true, loads from the given pointer. Works with LLVM types as a utility op for making LLVM conversion easier.
  }];
  let arguments = (ins
    LLVM_AnyPointer:$ptr,
    I1:$mask,
    LLVM_Type:$falseVal,
    DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<BoolAttr, "false">:$forceNoAlias
  );

  let results = (outs LLVM_Type:$result);

  let assemblyFormat = [{
    $ptr `,` $mask `,` $falseVal
    oilist(`cacheModifier` `=` $cache)
    (`forceNoAlias` $forceNoAlias^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// MaskedStoreOp
//===----------------------------------------------------------------------===//
def MaskedStoreOp : TT_AMDGPU_Op<"masked_store", []> {
  let summary = "Masked Store operation";
  let description = [{
    Store operation with masking support. If the mask is true, Store from the given pointer. Works with LLVM types as a utility op for making LLVM conversion easier.
  }];
  let arguments = (ins
    LLVM_AnyPointer:$ptr,
    LLVM_Type:$value,
    I1:$mask,
    DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<BoolAttr, "false">:$forceNoAlias
  );

  let assemblyFormat = [{
    $ptr `,` $value `,` $mask
    oilist(`cacheModifier` `=` $cache)
    (`forceNoAlias` $forceNoAlias^)?
    attr-dict `:` type(operands)
  }];
}

//===----------------------------------------------------------------------===//
// InThreadTransposeOp
//===----------------------------------------------------------------------===//

def InThreadTransposeOp : TT_AMDGPU_Op<"in_thread_transpose", [Pure]> {
  let summary = "Perform transpose of register values belonging to each threads";

  let hasVerifier = 1;

  let description = [{
    This operation performs a layout transpose over values in registers per thread.
    Specifically, given the input layout's blocked layout, it transposes the two last dimensions(rank-1 and rank-2)
    along the register dimension of the underlying linear layout.

    Conversion example:
    * input layout: blocked layout with sizePerThread=[2, 2], order=[0, 1]. It's linear layout register bases = [[1, 0], [2, 0], [0, 1], [0, 2]]
    * output layout: same thread and warp bases as in input, register bases = [[0, 1], [0, 2], [1, 0], [2, 0]]

    This operation enables efficient coalesced loading from HBM with following vectorized writing to shared memory
    in cases when HBM and shared memory order differ and target AMD hardware does not natively support this transposition.
    This is a specific variant of ttg.convert_layout and will be converted to ttg.convert_layout when lowering to llvm.
    We do not want this conversion to be optimized out, because we need to explicitly materialize instructions
    to transpose within each thread after loading from HBM and before writing to shared memory.
  }];

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

  let extraClassDeclaration = [{
    static mlir::triton::LinearLayout deduceOutputLayout(mlir::ArrayRef<int64_t> shape,
                                 mlir::triton::gpu::BlockedEncodingAttr srcEncoding);
  }];
}

//===----------------------------------------------------------------------===//
// LocalLoadPackedTransposedOp
//===----------------------------------------------------------------------===//

def LocalLoadPackedTransposedOp : TT_AMDGPU_Op<"local_load_packed_tranposed", [LocalLoadTrait]> {
    let summary = "Load a transposed packed tensor from shared memory into a distributed tensor";
    let description = [{
      Requires a M/N packed and M/N contiguous tensor in shared memory and will yield a K packed K contiguous tensor in registers.
      The packing change will change the shape of the tensor by doubling the M/N dimension and halving the K dimension.
      For example if A is 16x64 in shared memory, the result of this operation will be 32x32.
    }];
  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemRead<SharedMemory>]>:$src,
    Optional<TTG_AsyncToken>:$token
  );
  let results = (outs TT_Tensor:$result);

  let builders = [
      OpBuilder<(ins "Type":$retType, "Value":$src),
      [{
      build($_builder, $_state, retType, src, /*token=*/static_cast<mlir::Value>(nullptr));
      }]>];

  // Use qualified() otherwise "!ttg.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src (`token` $token^)? attr-dict `:` qualified(type($src)) `->` type($result)}];
  let hasVerifier = 1;
}

#endif
