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
#include "mlir/IR/OperationSupport.h"
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

#define GET_ATTRDEF_CLASSES
#include "Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/TritonAMDGPU/IR/Ops.cpp.inc"

namespace mlir::triton::amdgpu {

LogicalResult ViewSliceOp::verify() {
  auto srcTy = getSource().getType();
  auto srcLayout = srcTy.getEncoding();
  auto srcElementType = dyn_cast<RankedTensorType>(srcTy).getElementType();
  auto resultTy = getResult().getType();
  auto resultLayout = resultTy.getEncoding();
  auto resultElementType =
      dyn_cast<RankedTensorType>(resultTy).getElementType();

  if (srcElementType != resultElementType) {
    return emitError("result type must match source type");
  }

  if (srcLayout != resultLayout)
    return emitError("result layout must match source layout");

  auto srcShape = srcTy.getShape();
  auto shapePerCTA = mlir::triton::gpu::getShapePerCTATile(srcLayout, srcShape);
  shapePerCTA[0] = std::min(srcShape[0], (long)shapePerCTA[0]);
  shapePerCTA[1] = std::min(srcShape[1], (long)shapePerCTA[1]);

  auto offsets = getStaticOffsets();
  auto sizes = getStaticSizes();

  // ViewSlice only supports slicing where offsets and sizes are multiples of
  // shapePerCTA. This condition ensures that slice has the same layout as the
  // original tensor.

  if (offsets[0] % shapePerCTA[0] != 0 || offsets[1] % shapePerCTA[1] != 0) {
    return emitError("incorrect static offset");
  }

  if (sizes[0] % shapePerCTA[0] != 0 || sizes[1] % shapePerCTA[1] != 0) {
    return emitError("incorrect static size");
  }

  if (!hasUnitStride()) {
    return emitError("unsupported stride");
  }

  return success();
}
} // namespace mlir::triton::amdgpu