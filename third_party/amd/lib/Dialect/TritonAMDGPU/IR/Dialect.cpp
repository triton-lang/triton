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

struct CanonicalizeExtractSliceOp
    : public mlir::OpRewritePattern<amdgpu::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(amdgpu::ExtractSliceOp op,
                  PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    auto resultType = cast<RankedTensorType>(result.getType());
    auto source = op.getSource();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto offsets = op.getStaticOffsets();

    if (resultType == sourceType) {
      result.replaceAllUsesWith(source);
      return success();
    }
    return failure();
  }
};

void ExtractSliceOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<CanonicalizeExtractSliceOp>(context);
}

LogicalResult UpcastMXFPOp::verify() {
  auto fpType = getFpType();

  auto xTy = getSrc().getType();
  auto scaleTy = getScale().getType();
  Builder b(getContext());
  if (xTy.getElementType() != b.getBF16Type() &&
      xTy.getElementType() != b.getF16Type() &&
      xTy.getElementType() != b.getI8Type()) {
    return emitOpError(
        "element type of the first operand must be bf16/fp16 or i8");
  }

  if (scaleTy.getElementType() != b.getI8Type()) {
    return emitOpError("element type of the second operand must be uint8");
  }

  auto xShape = xTy.getShape();
  auto scaleShape = scaleTy.getShape();

  if (xShape.size() != scaleShape.size() || xShape.size() < 2) {
    return emitOpError(
        "operands must have the same number of dimensions, at least 2");
  }

  if (!(fpType == ScaleDotElemType::E2M1 || fpType == ScaleDotElemType::E4M3 ||
        fpType == ScaleDotElemType::E5M2)) {
    return emitOpError("NYI: fpType must be E2M1, E4M3, or E5M2");
  }

  auto layoutX = xTy.getEncoding();
  auto layoutScale = scaleTy.getEncoding();
  if (bool(layoutX) != bool(layoutScale)) {
    return emitOpError(
        "Expected either both or neither operands to have an encoding");
  }
  // Nothing to check if no encoding. This is used to infer the return type in
  // AccelerateMatmul.cpp
  if (!layoutX) {
    return success();
  }

  auto dotEncoding = dyn_cast<gpu::DotOperandEncodingAttr>(layoutX);
  if (!dotEncoding) {
    return emitOpError("Expected a DotOperandEncodingAttr for values");
  }
  if (!isa<gpu::BlockedEncodingAttr, gpu::LinearEncodingAttr>(layoutScale)) {
    return emitOpError(
        "Expected a BlockOperandEncoding or LinearOperandEncoding "
        "for scales");
  }

  // Change to support fp8 types
  const auto elemsPacked = fpType == ScaleDotElemType::E2M1 ? 2 : 1;
  // Figure out the K dimension for the input A/B. For A/B scale, the K
  // dimension is always the last dimension.
  const int opIdx = dotEncoding.getOpIdx();
  const bool hasBatch = xShape.size() == 3;
  const int kIdx = (opIdx == 0 ? 1 : 0) + hasBatch;

  if (xShape[kIdx] != (32 / elemsPacked) * scaleShape.back()) {
    return emitOpError("K dimension of first operand must be 16 times "
                       "larger than last/K dimension of the second operand");
  }

  // Check other dimensions match too. For input A/B, we need to figure out the
  // index for the M/N dimension. For scale, it's always {(batch), M/N, K}.
  const int mnIdx = (opIdx == 0 ? 0 : 1) + hasBatch;
  if (hasBatch && xShape[0] != scaleShape[0])
    return emitOpError("batch dimension must match between operands");
  if (xShape[mnIdx] != scaleShape[hasBatch]) {
    return emitOpError("M/N dimension must match between operands");
  }

  return success();
}

RankedTensorType
UpcastMXFPOp::deduceOutputType(TypedValue<RankedTensorType> inputTensor,
                               ScaleDotElemType inputElemType,
                               Type outputElemType) {
  MLIRContext *ctx = inputTensor.getContext();
  auto xTy = inputTensor.getType();
  if (inputElemType != ScaleDotElemType::E2M1)
    return xTy;

  auto xShape = xTy.getShape();
  auto newShape = llvm::to_vector(xShape);
  auto encoding = xTy.getEncoding();
  if (!encoding) {
    newShape.back() *= 2;
    return RankedTensorType::get(xShape, outputElemType);
  }

  auto oldEncoding = cast<DotOperandEncodingAttr>(encoding);
  auto newVEncoding = DotOperandEncodingAttr::get(ctx, oldEncoding.getOpIdx(),
                                                  oldEncoding.getParent(),
                                                  oldEncoding.getKWidth() * 2);
  // Figure out the K dimension for the input A/B, given that the return
  // type is upcasted A/B type so we need to update the proper dim size.
  const int opIdx = oldEncoding.getOpIdx();
  const bool hasBatch = xShape.size() == 3;
  const int kIdx = (opIdx == 0 ? 1 : 0) + hasBatch;
  newShape[kIdx] *= 2;
  return RankedTensorType::get(newShape, outputElemType, newVEncoding);
}

LogicalResult ConcatOp::verify() {
  auto sources = getSources();
  auto coords = getCoords();

  auto expectedNumSources = product(coords);
  if (sources.size() != expectedNumSources) {
    return emitError() << "dims spec [" << coords
                       << "] does not match the number of provided sources ["
                       << sources.size() << "]";
  }

  auto srcType = dyn_cast<RankedTensorType>(sources.front().getType());
  if (!srcType)
    return emitError() << "expected source type is `RankedTensorType`";

  for (auto source : sources) {
    auto currType = dyn_cast<RankedTensorType>(source.getType());
    if (srcType != currType)
      return emitError() << "sources are expected to have the same type";
  }

  auto result = getResult();
  auto dstType = dyn_cast<RankedTensorType>(result.getType());
  if (dstType.getElementType() != srcType.getElementType())
    return emitError() << "sources and the destination are expected to have "
                          "the same element type";

  auto dstShape = dstType.getShape();
  auto srcShape = srcType.getShape();
  if (dstShape.size() != srcShape.size())
    return emitError()
           << "sources and the destination must have the same shape size";

  if (dstShape.size() != coords.size())
    return emitError() << "shape size of the destination and concat. coords "
                          "must be the same";

  for (auto [idx, coordValue] : llvm::enumerate(coords)) {
    auto scaledSrcDim = srcShape[idx] * coordValue;
    if (dstShape[idx] != scaledSrcDim) {
      return emitError() << "mismatch along dim [" << idx
                         << "]. Expected size `" << dstShape[idx] << "`; give `"
                         << scaledSrcDim << "` after concatenation";
    }
  }

  return success();
}

struct CanonicalizeConcatOpFromExtractSlice
    : public mlir::OpRewritePattern<amdgpu::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(amdgpu::ExtractSliceOp op,
                  PatternRewriter &rewriter) const override {
    auto concatOp = op.getSource().getDefiningOp<amdgpu::ConcatOp>();
    if (!concatOp)
      return failure();

    auto offset = op.getStaticOffsets();
    auto coords = concatOp.getCoords();
    if (coords.size() != offset.size())
      return failure();

    auto sliceResult = op.getResult();
    auto sliceResultType = sliceResult.getType();
    auto sliceResultShape = sliceResultType.getShape();

    auto concatItem = concatOp.getSources().front();
    auto concatItemType = dyn_cast<RankedTensorType>(concatItem.getType());
    if (!concatItemType)
      return failure();

    if (sliceResultType != concatItemType)
      return failure();

    auto concatItemShape = concatItemType.getShape();
    SmallVector<int64_t> dimScales(concatItemShape.size(), 1);
    int64_t concatItemIndex = 0;
    std::exclusive_scan(coords.begin(), coords.end(), dimScales.begin(), 1,
                        std::multiplies<>());
    for (auto [idx, itemDimSize] : llvm::enumerate(concatItemShape)) {
      if ((offset[idx] % itemDimSize) != 0)
        return failure();
      const auto sliceCoords = offset[idx] / itemDimSize;
      concatItemIndex += sliceCoords * dimScales[idx];
    }
    assert(concatItemIndex < concatOp->getNumOperands() &&
           "concat index must be in bounds");
    Value concreteConcatItem = concatOp->getOperand(concatItemIndex);
    rewriter.replaceOp(op, concreteConcatItem);

    return success();
  }
};

struct CanonicalizeConcatOp : public mlir::OpRewritePattern<amdgpu::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(amdgpu::ConcatOp op,
                  PatternRewriter &rewriter) const override {

    auto result = op.getResult();
    auto sources = op.getSources();
    auto offsets = op.getCoords();
    if (sources.size() == 1) {
      assert(product(offsets) == 1);
      auto source = sources.front();
      result.replaceAllUsesWith(source);
      return success();
    }

    return failure();
  }
};

void ConcatOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *context) {
  patterns.add<CanonicalizeConcatOpFromExtractSlice>(context);
  patterns.add<CanonicalizeConcatOp>(context);
}
} // namespace mlir::triton::amdgpu
