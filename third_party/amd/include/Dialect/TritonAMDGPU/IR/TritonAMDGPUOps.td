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
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "TritonAMDGPUDialect.td"
include "TritonAMDGPUAttrDefs.td"


class TT_AMDGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonAMDGPU_Dialect, mnemonic, !listconcat(traits, [])> {
}

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

//===----------------------------------------------------------------------===//
// ExtractSliceOp
//===----------------------------------------------------------------------===//

def ExtractSliceOp
    : TT_AMDGPU_Op<"extract_slice", [Pure]> {
  let summary = "extract slice operation";
  let description = [{
    The "extract_slice" operation enables extracting a slice of a tensor in
    registers.

    The "extract_slice" operation supports the following arguments:

    * source: the base tensor on which to create a view tensor
    * offsets: offsets into the base tensor at which to create the view

    Example 1:

    ```mlir
    #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8],
        threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
    #blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8],
        threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
    %1 = triton_gpu.convert_layout %0 : tensor<128x128xf16, #blocked>
        -> tensor<128x128xf16, #blocked1>
    // create a slice of base tensor %1 with static offsets
    %2 = amdgpu.extract_slice %0 [0, 0] :
      tensor<128x128xf16, #blocked1> to tensor<128x32xf16, #blocked1>
    ```

    Example 1 shows how "extract_slice" operation may be used. In this example a
    new slice of 128x32 is created. "extract_slice" works on tensors with layout
    where the desired slice has the same layout as the source tensor.
    "%0" cannot be sliced directly as the resulting slice cannot have the same
    layout as "%0". Therefore it needs to be converted to a layout suitable
    for slicing. "#blocked1" layout is appropriate for this as it keeps the
    sizePerThread the same thus keeping coalescing properties the same.
    In order to utilize all threads in a warp, "threadsPerWarp" is set to
    [16,4] for this new layout. This layout conversion carried out before
    using "extract_slice" ensures slicing still uses all threads efficiently. The
    size of the slice is determined by the result type.
    }];

  let arguments = (ins AnyRankedTensor:$source,
      DenseI64ArrayAttr:$static_offsets);
  let results = (outs AnyRankedTensor:$result);

  let builders = [
      // Build a ExtractSliceOp with static offsets and the same result type
      OpBuilder<(ins "RankedTensorType":$resultType,
          "Value":$source,
          "ArrayRef<int64_t>": $static_offsets)>,
  ];

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
}

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

  let arguments = (ins
    TritonAMDGPU_InstCounter:$numDsReadsA,
    TritonAMDGPU_InstCounter:$numDsReadsB,
    TritonAMDGPU_InstCounter:$numDsWritesA,
    TritonAMDGPU_InstCounter:$numDsWritesB,
    TritonAMDGPU_InstCounter:$numGlobalLoadsA,
    TritonAMDGPU_InstCounter:$numGlobalLoadsB,
    BoolAttr:$isBufferLoadsAEnabled,
    BoolAttr:$isBufferLoadsBEnabled,
    TritonAMDGPU_InstCounter:$numMMAs
  );

  let builders = [
    OpBuilder<(ins), [{
      auto ctx = $_state.getContext();
      auto noneType = NoneType::get(ctx);
      auto emptyAttr = amdgpu::InstCounterAttr::get(ctx, 0, noneType);
      build($_builder, $_state, emptyAttr, emptyAttr, emptyAttr, emptyAttr,
            emptyAttr, emptyAttr, false, false, emptyAttr);
    }]>
  ];

  let assemblyFormat = [{ attr-dict }];
}

def condBarrierOp :
  TT_AMDGPU_Op<"cond_barrier">,
  Arguments<(ins I1:$pred)> {
  let summary = "set barrier conditionally";

  let description = [{
      condBarrierOp sets barrier instruction only when the given argument is true.
      This provides a way to synchronize partial threads in a block, deliberately
      diverges the execution sequences. However, user should guarantee all threads
      converge at the end by calling condBarrierOp(true) with the remaining threads.
      NB. This doesn't set memory fence.
  }];

  let assemblyFormat = "attr-dict $pred";
}

//
// AMD Buffer operations.
//
def BufferLoadOp : TT_AMDGPU_Op<"buffer_load", [
  SameLoadStoreOperandsAndResultEncoding,
  AttrSizedOperandSegments,
  MemoryEffects<[MemRead<GlobalMemory>]>,
  TypesMatchWith<"result element type matches the pointed type of ptr", "result", "ptr", "getPointerTypeToElement($_self)">,
  TypesMatchWith<"result and offsets have the same shape", "result", "offsets", "getI32SameShape($_self)">,
  TypesMatchWith<"result and mask have the same shape", "result", "mask", "getI1SameShape($_self)",
                 "($_op.getOperands().size() <= 2) || std::equal_to<>()">,
  TypesMatchWith<"result and other have the same type", "result", "other", "$_self",
                 "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
]>{
    let summary = "Load from a scalar base pointer and a tensor offset";
    let description = [{
      AMD Buffer load operation. Buffer store is similar to
      a normal store but it accesses global memory via a scalar base pointer
      and a tensor of offsets instead of a tensor of pointers. The other fields
      are similar to a normal load, i.e., the `mask` is a boolean vector that
      determines if a given element should be read from memory, and `other` is the
      element that should be returned on lane `i` when `mask[i] == 0`.
    }];
    let arguments = (
      ins
      TT_Ptr:$ptr,
      I32Tensor:$offsets,
      Optional<TT_BoolTensor>:$mask,
      Optional<TT_Tensor>:$other
    );
    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
      $ptr `[` $offsets `]` (`,` $mask^)? (`,` $other^)?
      attr-dict `:` type($result)
    }];
}

def BufferStoreOp : TT_AMDGPU_Op<"buffer_store", [
  SameLoadStoreOperandsEncoding,
  MemoryEffects<[MemWrite<GlobalMemory>]>,
  TypesMatchWith<"value element type matches the pointed type of ptr", "value", "ptr", "getPointerTypeToElement($_self)">,
  TypesMatchWith<"value and offsets have the same shape", "value", "offsets", "getI32SameShape($_self)">,
  TypesMatchWith<"value and mask have the same shape", "value", "mask", "getI1SameShape($_self)",
                 "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
]>{
    let summary = "Store into scalar base pointer and a tensor offset";
    let description = [{
      AMD Buffer store operation. Buffer store is similar to
      normal store but it accesses global memory via a scalar base pointer
      and a tensor of offsets instead of a tensor of pointers. The other fields
      are similar to a normal store , i.e., the `mask` is a boolean vector that
      determines if a given element should be written to memory, and `value` is the
      tensor of elements that should be written on lane `i` when `mask[i] == 1`.
    }];
    let arguments = (
      ins
      TT_Tensor:$value,
      TT_Ptr:$ptr,
      I32Tensor:$offsets,
      Optional<TT_BoolTensor>:$mask
    );

    let assemblyFormat = [{
      $value `,` $ptr `[` $offsets `]` (`,` $mask^)?
      attr-dict `:` type($value)
    }];
}

#endif
