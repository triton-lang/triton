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

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

// clang-format off
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::amdgpu;

void mlir::triton::amdgpu::TritonAMDGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/TritonAMDGPU/IR/Ops.cpp.inc"
      >();
}

#include "Dialect/TritonAMDGPU/IR/TritonAMDGPUEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/TritonAMDGPU/IR/Ops.cpp.inc"

namespace mlir::triton::amdgpu {

LogicalResult ExtractSliceOp::verify() {
  auto srcTy = getSource().getType();
  auto srcLayout = srcTy.getEncoding();
  auto srcElementType = getElementTypeOrSelf(srcTy);
  auto resultTy = getResult().getType();
  auto resultLayout = resultTy.getEncoding();
  auto resultElementType = getElementTypeOrSelf(resultTy);

  if (srcElementType != resultElementType) {
    return emitError("result element type must match source element type");
  }
  if (srcLayout != resultLayout) {
    return emitError("result layout must match source layout");
  }
  if (srcTy.getRank() != resultTy.getRank()) {
    return emitError("result rank must be equal to source rank");
  }
  if (srcTy.getRank() != 2) {
    return emitError("currently only 2D tensors are supported");
  }

  auto srcShape = srcTy.getShape();
  auto shapePerCTATile = mlir::triton::gpu::getShapePerCTATile(srcLayout);
  shapePerCTATile[0] =
      std::min(static_cast<unsigned>(srcShape[0]), shapePerCTATile[0]);
  shapePerCTATile[1] =
      std::min(static_cast<unsigned>(srcShape[1]), shapePerCTATile[1]);

  // ExtractSlice only supports slicing where offsets and sizes are multiples of
  // shapePerCTATile. This condition ensures that slice has the same layout as
  // the original tensor.

  auto offsets = getStaticOffsets();
  if (offsets.size() != 2) {
    return emitError("invalid offset shape ") << offsets;
  }

  SmallVector<int64_t, 2> sizes;
  for (auto i = 0; i < 2; ++i) {
    auto resultDimSize = resultTy.getDimSize(i);
    auto srcDimSize = srcTy.getDimSize(i);
    if (resultDimSize == 0) {
      return emitError("result tensor dimension size zero at dimension ") << i;
    }
    if (srcDimSize == 0) {
      return emitError("source tensor dimension size zero at dimension ") << i;
    }
    if (resultDimSize > srcDimSize) {
      return emitError(
                 "result shape cannot be larger than input shape at dimension ")
             << i;
    }
    if (offsets[i] + resultDimSize > srcDimSize) {
      return emitError("invalid offset ")
             << offsets[i] << " at dimension " << i;
    }
    sizes.push_back(resultDimSize);
  }

  if (sizes[0] % shapePerCTATile[0] != 0 ||
      sizes[1] % shapePerCTATile[1] != 0) {
    return emitError() << "sizes [" << sizes
                       << "] must be a multiple of shapePerCTATile ["
                       << shapePerCTATile << "]";
  }

  if (offsets[0] % shapePerCTATile[0] != 0 ||
      offsets[1] % shapePerCTATile[1] != 0) {
    return emitError() << "offset [" << offsets
                       << "] must be a multiple of shapePerCTATile ["
                       << shapePerCTATile << "]";
  }

  return success();
}
} // namespace mlir::triton::amdgpu
