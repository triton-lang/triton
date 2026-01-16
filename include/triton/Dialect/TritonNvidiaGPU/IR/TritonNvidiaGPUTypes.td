// Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRITONNVIDIAGPU_TYPES
#define TRITONNVIDIAGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td"
include "triton/Dialect/Triton/IR/TritonTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// TritonNvidiaGPU Type Definitions
//===----------------------------------------------------------------------===//

class TTNG_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TritonNvidiaGPU_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

//===----------------------------------------------------------------------===//
// TensorDescIm2ColType
//===----------------------------------------------------------------------===//

def TTNG_TensorDescIm2ColType : TTNG_TypeDef<"TensorDescIm2Col", "tensordesc_im2col",
                                              [TT_TensorDescInterface]> {
  let summary = "Im2col tensor descriptor type for NVIDIA TMA operations";

  let description = [{
    Tensor descriptor type for im2col (image-to-column) tensor memory access.
    This is used for convolution-friendly access patterns with TMA on NVIDIA GPUs.

    Im2col mode transforms a multi-dimensional tensor into a 2D matrix format
    suitable for matrix multiplication, which is commonly used in convolution
    operations.

    Parameters:
    - blockType: The shape and element type of the data block being accessed

    This type implements TensorDescInterface, sharing common operations with
    the tiled TensorDescType in the base Triton dialect.

    See NVIDIA PTX documentation for im2col tensor mode:
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-mode
  }];

  let parameters = (ins
    "RankedTensorType":$blockType
  );

  let assemblyFormat = [{
    `<` $blockType `>`
  }];

  let builders = [
    // Builder with signedness for integer types
    TypeBuilder<(ins
      "RankedTensorType":$blockType,
      "bool":$isSigned
    ), [{
      if (auto intTy = llvm::dyn_cast<IntegerType>(blockType.getElementType())) {
        auto sem = isSigned ? IntegerType::Signed : IntegerType::Unsigned;
        auto elemTy = IntegerType::get($_ctxt, intTy.getWidth(), sem);
        blockType = blockType.clone(elemTy);
      }
      return Base::get($_ctxt, blockType);
    }]>
  ];

  let genVerifyDecl = 1;
}

#endif // TRITONNVIDIAGPU_TYPES
