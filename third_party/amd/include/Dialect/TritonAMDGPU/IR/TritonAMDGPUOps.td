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

class TritonAMDGPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<TritonAMDGPU_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ViewSliceOp
//===----------------------------------------------------------------------===//

def TritonAMDGPU_ViewSliceOp
    : TritonAMDGPU_Op<"view_slice", [AttrSizedOperandSegments,
                                     OffsetSizeAndStrideOpInterface, Pure]> {
  let summary = "view slice operation";
  let description = [{
    The "view_slice" operation enables viewing a slice of a tensor in
    registers without data exchange.

    The "view_slice" operation supports the following arguments:

    * source: the base tensor on which to create a view tensor
    * offsets: offsets into the base tensor at which to create the view
    * size: size of the result "view" tensor
    * strides: the number of strides for each dimension

    Example 1:

    ```mlir
    #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8],
        threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
    #blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8],
        threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
    %1 = triton_gpu.convert_layout %0 : tensor<128x128xf16, #blocked>
        -> tensor<128x128xf16, #blocked1>
    // create a slice of base tensor %1 with
    // static offsets and sizes for each dimension
    %2 = amdgpu.view_slice %0[0, 0] [128, 32] [1, 1] :
      tensor<128x128xf16, #blocked1> to tensor<128x32xf16, #blocked1>
    ```

    Example 1 shows how "view_slice" operation may be used. In this example a
    new view of 128x32 is created. "view_slice" works on tensors with layout
    where the desired slice has the same layout as the source tensor.
    "%0" cannot be sliced directly as the resulting slice cannot have the same
    layout as "%0". Therefore it needs to be converted to a layout suitable
    for slicing. "#blocked1" layout is appropriate for this as it keeps the
    sizePerThread the same thus keeping coalescing properties the same.
    In order to utilize all threads in a warp, "threadsPerWarp" is set to
    [16,4] for this new layout. This layout conversion carried out before
    using "view_slice" ensures slicing still uses all threads efficiently.
    }];

  let arguments = (ins AnyRankedTensor:$source,
      Variadic<I32>:$offsets,
      Variadic<I32>:$sizes,
      Variadic<I32>:$strides,
      DenseI64ArrayAttr:$static_offsets,
      DenseI64ArrayAttr:$static_sizes,
      DenseI64ArrayAttr:$static_strides);
  let results = (outs AnyRankedTensor:$result);

  let builders = [
      // Build a ViewSliceOp with mixed static and dynamic entries and the same
      // result type
      OpBuilder<(ins "RankedTensorType":$resultType,
          "Value":$source,
          "ArrayRef<OpFoldResult>":$offsets,
          "ArrayRef<OpFoldResult>":$sizes,
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

#endif
