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
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ViewLikeInterface.td" // OffsetSizeAndStrideOpInterface
include "TritonAMDGPUDialect.td"
include "TritonAMDGPUAttrDefs.td"


class TT_AMDGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonAMDGPU_Dialect, mnemonic, !listconcat(traits, [])> {
}

//===----------------------------------------------------------------------===//
// ViewSliceOp
//===----------------------------------------------------------------------===//

def TritonAMDGPU_ViewSliceOp
    : TritonAMDGPU_Op<"view_slice", [AttrSizedOperandSegments,
                                     OffsetSizeAndStrideOpInterface, Pure]> {
  let summary = "view slice operation";
  let description = [{
    The "view_slice" operation enables "viewing" a slice of a tensor in
    registers without data exchange.

    The "view_slice" operation supports the following arguments:

    * source: the base tensor on which to create a "view" tensor
    * offsets: offsets into the base tensor at which to create the "view"
    * size: size of the result "view" tensor
    * strides: the number of strides for each dimension

    Currently only 2D tensors are supported.

    Example 1:

    ```mlir
    %1 = triton_gpu.convert_layout %0 : tensor<128x128x!tt.ptr<f16>, #blocked>
        -> tensor<128x128x!tt.ptr<f16>, #blocked2>
    // create a slice of base tensor %1 with
    // static offsets and sizes for each dimension
    %2 = amdgpu.view_slice %0[0, 0] [128, 8] [1, 1] :
      tensor<128x128x!tt.ptr<f16>, #blocked2> to
      tensor<128x8x!tt.ptr<f16>, #blocked2>
    ```
    }];

  let arguments = (ins AnyRankedTensor:$source, Variadic<I32>:$offsets,
      Variadic<I32>:$sizes, Variadic<I32>:$strides,
      DenseI64ArrayAttr:$static_offsets, DenseI64ArrayAttr:$static_sizes,
      DenseI64ArrayAttr:$static_strides);
  let results = (outs AnyRankedTensor:$result);

  let builders = [
      // Build a ViewSliceOp with mixed static and dynamic entries and the same
      // result type
      OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
          "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
          "ArrayRef<OpFoldResult>":$strides,
          CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = [{
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSource().getType().getRank();
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
    TritonAMDGPU_InstCounter:$numMMAs
  );

  let builders = [
    OpBuilder<(ins), [{
      auto ctx = $_state.getContext();
      auto type = IntegerType::get(ctx, 32);
      auto emptyAttr = amdgpu::InstCounterAttr::get(ctx, 0, type);
      build($_builder, $_state, emptyAttr, emptyAttr, emptyAttr, emptyAttr,
            emptyAttr, emptyAttr, emptyAttr);
    }]>
  ];

  let assemblyFormat = [{ attr-dict }];
}

#endif
