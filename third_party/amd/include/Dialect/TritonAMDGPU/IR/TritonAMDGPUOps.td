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
      DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
      Optional<TT_BoolTensor>:$mask,
      Optional<TT_Tensor>:$other
    );
    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
      $ptr `[` $offsets `]` (`,` $mask^)? (`,` $other^)?
      oilist(`cacheModifier` `=` $cache)
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
      DefaultValuedAttr<TT_CacheModifierAttr, "mlir::triton::CacheModifier::NONE">:$cache,
      Optional<TT_BoolTensor>:$mask
    );

    let assemblyFormat = [{
      $value `,` $ptr `[` $offsets `]` (`,` $mask^)?
      oilist(`cacheModifier` `=` $cache)
      attr-dict `:` type($value)
    }];
}

#endif
