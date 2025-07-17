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
#include "third_party/amd/include/Utils/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/TypeSwitch.h"

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

// Check that the source and destination tensor layouts match on a CTA tile.
// This means that lane and warp bases of linear layout must match, and the
// register basis must be the same up to a number of registers contained within
// a CTA tile.
bool hasMatchingCTATileLayoutForSliceConcat(
    RankedTensorType srcTy, RankedTensorType dstTy,
    std::function<void(const Twine &)> emitError) {
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();
  auto srcLL = triton::gpu::toLinearLayout(srcTy);
  auto dstLL = triton::gpu::toLinearLayout(dstTy);

  MLIRContext *ctx = srcTy.getContext();
  auto kReg = StringAttr::get(ctx, "register");
  srcLL = srcLL.removeZeroBasesAlongDim(kReg);
  dstLL = dstLL.removeZeroBasesAlongDim(kReg);

  auto getBases = [&](StringRef name) {
    auto key = StringAttr::get(ctx, name);
    return std::pair{srcLL.getBases().lookup(key),
                     dstLL.getBases().lookup(key)};
  };

  auto [regSrc, regDst] = getBases("register");
  auto [laneSrc, laneDst] = getBases("lane");
  auto [warpSrc, warpDst] = getBases("warp");

  auto shapeCTASrc = mlir::triton::gpu::getShapePerCTATile(srcTy);
  auto shapeCTADst = mlir::triton::gpu::getShapePerCTATile(dstTy);
  if (shapeCTASrc != shapeCTADst) {
    emitError(
        "CTA tile shapes must match between source and destination tensors.");
    return false;
  }

  // Compute number of basis vectors that desribe registers from one CTA tile.
  unsigned numCTAs = 1;
  for (size_t d = 0, rank = srcShape.size(); d < rank; ++d) {
    assert(srcShape[d] % shapeCTASrc[d] == 0 &&
           "Source shape must be multiple of CTA tile shape");
    numCTAs *= srcShape[d] / shapeCTASrc[d];
  }

  assert(llvm::isPowerOf2_32(numCTAs) &&
         "expect number of CTAs to be power of 2");

  unsigned totalElemsPerThreadNoBroadcastLog = regSrc.size();
  unsigned elemsPerThreadPerCTALog =
      totalElemsPerThreadNoBroadcastLog - llvm::Log2_32(numCTAs);
  unsigned regCompareLen = elemsPerThreadPerCTALog;

  auto compareBasis = [&](auto &srcBasis, auto &dstBasis, StringRef message,
                          int limit = -1) {
    int n = (limit < 0 ? srcBasis.size()
                       : std::min<unsigned>(srcBasis.size(), limit));
    if (dstBasis.size() < n) {
      emitError(message);
      return false;
    }
    for (size_t i = 0; i < n; ++i) {
      if (srcBasis[i] != dstBasis[i]) {
        emitError(message);
        return false;
      }
    }
    return true;
  };

  if (!compareBasis(regSrc, regDst,
                    "Register basis must match on a CTA tile between source "
                    "and destination.",
                    regCompareLen))
    return false;

  if (laneSrc != laneDst || warpSrc != warpDst) {
    emitError("Lane and warp dim basis must match between source and "
              "destination layout.");
    return false;
  }
  return true;
}

LogicalResult ExtractSliceOp::verify() {
  // Basic type/rank checks.
  auto srcTypeVal = getSource().getType();
  auto dstTypeVal = getResult().getType();
  auto srcTy = mlir::cast<RankedTensorType>(srcTypeVal);
  auto dstTy = mlir::cast<RankedTensorType>(dstTypeVal);

  auto srcElm = getElementTypeOrSelf(srcTy);
  auto resElm = getElementTypeOrSelf(dstTy);
  if (srcElm != resElm)
    return emitError("result element type must match source element type");
  if (srcTy.getRank() != dstTy.getRank())
    return emitError("result rank must be equal to source rank");

  // Per-dimension shape/offset checks
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();
  auto offsets = getStaticOffsets();
  auto shapePerCTATile = mlir::triton::gpu::getShapePerCTATile(srcTy);
  size_t rank = srcShape.size();

  auto failDim = [&](StringRef msg, int i) -> LogicalResult {
    return emitError(msg) << " at dimension " << i;
  };

  for (size_t i = 0; i < rank; ++i) {
    if (dstShape[i] > srcShape[i])
      return failDim("result shape cannot exceed source shape", i);
    if (offsets[i] + dstShape[i] > srcShape[i])
      return failDim("invalid offset", i);
    if (dstShape[i] % shapePerCTATile[i] != 0)
      return emitError("result shape must be multiple of shapePerCTATile");
    if (offsets[i] % shapePerCTATile[i] != 0)
      return emitError("offset must be multiple of shapePerCTATile");
  }

  // Verify that source and destination layout match on a CTA tile.
  if (!hasMatchingCTATileLayoutForSliceConcat(
          srcTy, dstTy, [&](const Twine &msg) { emitError() << msg; }))
    return failure();

  return success();
}

// This pattern optimizes the combination of extract_slice and concat
// operations. When extract_slice is used to extract a portion that exactly
// matches one of the original tensors concatenated by a concat operation, we
// can eliminate extract_slice op and use the original tensor directly.
struct CononicalizeExtractSliceAndConcat
    : public mlir::OpRewritePattern<amdgpu::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(amdgpu::ExtractSliceOp op,
                  PatternRewriter &rewriter) const override {
    // Try to match preceding Concat op
    auto concatOp = op.getSource().getDefiningOp<amdgpu::ConcatOp>();
    if (!concatOp)
      return failure();

    auto offset = op.getStaticOffsets();
    auto sliceResult = op.getResult();
    auto sliceResultType = sliceResult.getType();
    RankedTensorType dstType =
        cast<RankedTensorType>(concatOp.getResult().getType());
    auto dstShape = dstType.getShape();

    auto concatItem = concatOp.getSources().front();
    auto concatItemType = dyn_cast<RankedTensorType>(concatItem.getType());
    if (!concatItemType)
      return failure();

    if (sliceResultType != concatItemType)
      return failure();

    // Calculate which concat operand contains our slice
    auto srcShape = concatItemType.getShape();
    auto rank = srcShape.size();
    std::vector<unsigned> defaultOrder(rank);
    std::iota(defaultOrder.rbegin(), defaultOrder.rend(), 0);

    // Convert multidimensional offset to concat operand index
    auto multiDimSrcIdx = LLVM::AMD::multiDimElementwise<int64_t, int64_t>(
        offset, srcShape, std::divides<unsigned>());
    auto srcToDstShape = LLVM::AMD::multiDimElementwise<int64_t, int64_t>(
        dstShape, srcShape, std::divides<unsigned>());
    auto linearSrcIdx =
        mlir::LLVM::linearize(multiDimSrcIdx, srcToDstShape, defaultOrder);

    // Replace extract_slice with the concat operand
    assert(linearSrcIdx < concatOp->getNumOperands() &&
           "concat index must be in bounds");
    Value concreteConcatItem = concatOp->getOperand(linearSrcIdx);
    rewriter.replaceOp(op, concreteConcatItem);

    return success();
  }
};

void ExtractSliceOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<CononicalizeExtractSliceAndConcat>(context);
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

LogicalResult InThreadTransposeOp::verify() {
  auto srcTy = getSrc().getType();
  auto dstTy = getResult().getType();
  if (srcTy.getElementType() != dstTy.getElementType()) {
    return emitOpError("Expect input and output tensor to have same dtype");
  }

  auto shape = srcTy.getShape();
  if (shape != dstTy.getShape()) {
    return emitOpError("Expect equal input and output shapes");
  }

  if (shape.size() != 2) {
    return emitOpError("Expect 2d tensor");
  }

  auto srcEncoding = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
  if (!srcEncoding) {
    return emitOpError("Expect input tensor in Blocked encoding");
  }

  auto expectedLinearLayout = deduceOutputLayout(shape, srcEncoding);
  auto dstLinearLayout = triton::gpu::toLinearLayout(dstTy);
  if (dstLinearLayout != expectedLinearLayout) {
    return emitOpError("Expect output layout to be transposed per thread: " +
                       expectedLinearLayout.toString());
  }
  return success();
}

LinearLayout
InThreadTransposeOp::deduceOutputLayout(ArrayRef<int64_t> shape,
                                        gpu::BlockedEncodingAttr srcEncoding) {
  auto srcLL = srcEncoding.toLinearLayout(shape);
  SmallVector<unsigned> newRegOrder(srcEncoding.getOrder());
  int rank = shape.size();
  assert(rank == 2 && "InThreadTransposeOp do not support non 2d tensors yet");
  std::swap(newRegOrder[rank - 2], newRegOrder[rank - 1]);

  // Make in-register transposed tile
  auto ctx = srcEncoding.getContext();
  auto regDimName = StringAttr::get(ctx, "register");
  auto inThreadTransposedTile = identityStandardND(
      regDimName, srcEncoding.getSizePerThread(), newRegOrder);
  // make sure basis in same order as in srcLayout
  SmallVector<StringAttr> outDimNames(srcLL.getOutDimNames());
  inThreadTransposedTile = inThreadTransposedTile.transposeOuts(outDimNames);

  // Copy original bases, and replace register tile with transposed one
  LinearLayout::BasesT bases = srcLL.getBases();
  auto &regBase = *bases.find(regDimName);
  int regsTransposed = inThreadTransposedTile.getInDimSizeLog2(regDimName);
  for (int i = 0; i < regsTransposed; ++i)
    regBase.second[i] = inThreadTransposedTile.getBasis(regDimName, i);

  LinearLayout transposedLL(bases, SmallVector<StringAttr>(outDimNames));
  return transposedLL;
}

LogicalResult ConcatOp::verify() {
  auto sources = getSources();
  auto result = getResult();

  auto srcType = cast<RankedTensorType>(sources.front().getType());
  auto dstType = cast<RankedTensorType>(result.getType());

  auto srcShape = srcType.getShape();
  auto dstShape = dstType.getShape();
  unsigned rank = srcShape.size();

  // 1) Shape related checks.
  if (rank != dstShape.size())
    return emitError()
           << "Source and destination tensors must have the same rank.";

  unsigned numTiles = 1;
  for (int i = 0; i < rank; ++i) {
    if (dstShape[i] % srcShape[i] != 0)
      return emitError() << "Source and destination tensor shapes don't match.";
    numTiles *= dstShape[i] / srcShape[i];
  }

  if (numTiles != sources.size())
    return emitError() << "Number of source tiles (" << sources.size()
                       << ") doesn't match required count (" << numTiles
                       << ").";

  // 2) Check that all sources have same type and element type match.
  for (auto src : sources) {
    auto curr = dyn_cast<RankedTensorType>(src.getType());
    if (curr != srcType)
      return emitError() << "All sources must have identical tensor types.";
  }

  if (dstType.getElementType() != srcType.getElementType())
    return emitError()
           << "Element types of sources and destination must match.";

  // 3) Check that all source and destination layouts match on a CTA tile.
  if (!hasMatchingCTATileLayoutForSliceConcat(
          srcType, dstType, [&](const Twine &msg) { emitError() << msg; }))
    return failure();

  return success();
}

LogicalResult LocalLoadPackedTransposedOp::verify() {
  auto srcTy = getSrc().getType();
  auto dstTy = getType();
  auto srcShape = srcTy.getShape();

  auto dotEnc = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  if (!dotEnc)
    return emitOpError("only works with DotOperandEncodingAttr dst encoding");

  auto sharedEnc =
      dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding());
  if (!sharedEnc)
    return emitOpError(
        "only works with SwizzledSharedEncodingAttr src encoding");

  auto order = sharedEnc.getOrder();
  bool isA = dotEnc.getOpIdx() == 0;

  // operand A: [0, 1] / [1, 2, 0]
  // operand B: [1, 0] / [2, 1, 0]
  bool hasBatchDim = srcShape.size() == 3;

  if (isA) {
    bool matchingOrderA =
        order.equals({0, 1}) || (hasBatchDim && order.equals({1, 2, 0}));
    if (!matchingOrderA)
      return emitOpError("Order of dimensions don't match expected");

    SmallVector<int64_t> srcShapeBasedOnDstA(dstTy.getShape());
    srcShapeBasedOnDstA[hasBatchDim ? 1 : 0] /= 2;
    srcShapeBasedOnDstA[hasBatchDim ? 2 : 1] *= 2;

    bool aDimMatch = srcShape.equals(ArrayRef(srcShapeBasedOnDstA));
    if (!aDimMatch)
      return emitOpError(
          "Input and output dimensions don't match after packing changes");
  } else {
    bool matchingOrderB =
        order.equals({1, 0}) || (hasBatchDim && order.equals({2, 1, 0}));
    if (!matchingOrderB)
      return emitOpError("Order of dimensions don't match expected");

    SmallVector<int64_t> srcShapeBasedOnDstB(dstTy.getShape());
    srcShapeBasedOnDstB[hasBatchDim ? 1 : 0] *= 2;
    srcShapeBasedOnDstB[hasBatchDim ? 2 : 1] /= 2;

    bool bDimMatch = srcShape.equals(ArrayRef(srcShapeBasedOnDstB));
    if (!bDimMatch)
      return emitOpError(
          "Input and output dimensions don't match after packing changes");
  }

  return success();
}

// This pattern removes a concatOp if it has a single input operand.
// This scenario can potentially happen as a result of ops refinement.
mlir::LogicalResult foldConcatOpFromSingleSource(amdgpu::ConcatOp op,
                                                 PatternRewriter &rewriter) {
  auto sources = op.getSources();
  if (sources.size() == 1) {
    auto source = sources.front();
    auto result = op.getResult();
    result.replaceAllUsesWith(source);
    return success();
  }
  return failure();
}

void ConcatOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *context) {
  patterns.add(foldConcatOpFromSingleSource);
}
} // namespace mlir::triton::amdgpu
