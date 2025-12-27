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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "third_party/amd/include/Utils/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include <limits>

// clang-format off
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.cpp.inc"
// clang-format on

#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"

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

  addInterfaces<TritonInlinerInterface>();
}

#include "Dialect/TritonAMDGPU/IR/TritonAMDGPUEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/TritonAMDGPU/IR/TritonAMDGPUAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/TritonAMDGPU/IR/Ops.cpp.inc"
#include "Dialect/TritonAMDGPU/IR/TritonAMDGPUOpInterfaces.cpp.inc"

namespace mlir::triton::amdgpu {

std::string getStringFromCoords(mlir::triton::AMD::ElemLocationKey coords) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "[";
  llvm::interleaveComma(coords, os,
                        [&](const auto &coord) { os << coord.second; });
  os << "]";
  return os.str();
}

// Helper function to verify TDM block dimensions
static LogicalResult verifyTDMBlockSize(Operation *op,
                                        ArrayRef<int64_t> blockShape) {
  constexpr int64_t maxBlockSize = std::numeric_limits<uint16_t>::max();
  for (size_t i = 0; i < blockShape.size(); ++i) {
    if (blockShape[i] > maxBlockSize) {
      return op->emitOpError("TDM block dimension ")
             << i << " (" << blockShape[i] << ") exceeds maximum size of "
             << maxBlockSize;
    }
  }
  return success();
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
  size_t rank = srcShape.size();

  auto failDim = [&](StringRef msg, int i) -> LogicalResult {
    return emitError(msg) << " at dimension " << i;
  };

  for (size_t i = 0; i < rank; ++i) {
    if (dstShape[i] > srcShape[i])
      return failDim("result shape cannot exceed source shape", i);
    if (offsets[i] + dstShape[i] > srcShape[i])
      return failDim("invalid offset", i);
  }

  auto linearLayoutSrc = triton::gpu::toLinearLayout(srcTy);
  auto linearLayoutDst = triton::gpu::toLinearLayout(dstTy);
  auto ctx = srcTy.getContext();

  auto getBases = [&](StringRef name) {
    auto key = StringAttr::get(ctx, name);
    return std::pair{linearLayoutSrc.getBases().lookup(key),
                     linearLayoutDst.getBases().lookup(key)};
  };

  StringAttr kReg = StringAttr::get(ctx, "register");
  auto dstRegBases = linearLayoutDst.getBases().lookup(kReg);

  int dstRegCount = 1 << dstRegBases.size();
  SmallVector<Value> resultVals;

  // Algorithm:
  // 1. for every dst register
  // 2.   get dst element coordinates relative to tile start
  // 3.   add coordinates of tile start relative to parent tensor
  // 4.   check if exists source register which holds dst value

  // 1. for every dst register
  for (int regId = 0; regId < dstRegCount; ++regId) {
    // 2.   get dst element coordinates relative to tile start
    auto elemCoords = mlir::triton::AMD::getElemCoordinatesFromRegisters(
        linearLayoutDst, regId, ctx);
    // 3.   add coordinates of tile start relative to parent tensor

    for (int i = 0; i < rank; ++i)
      elemCoords[i].second += offsets[i];

    // 4.   check if exists source register which holds dst value
    std::optional<int> srcReg = mlir::triton::AMD::getRegFromCoordinates(
        linearLayoutSrc, elemCoords, ctx);

    if (!srcReg.has_value()) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "No source register holds the element for destination index "
         << getStringFromCoords(elemCoords);
      return emitError(os.str());
    }
  }

  auto [laneSrc, laneDst] = getBases("lane");
  auto [warpSrc, warpDst] = getBases("warp");
  if (laneSrc != laneDst || warpSrc != warpDst) {
    return emitError("Lane and warp dim basis must match between source and "
                     "destination layout.");
  }

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
      xTy.getElementType() != b.getI8Type() &&
      xTy.getElementType() != b.getType<Float8E4M3FNType>() &&
      xTy.getElementType() != b.getType<Float8E5M2Type>()) {
    return emitOpError("element type of the first operand must be bf16/fp16, "
                       "OCP fp8/bf8 or i8");
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
  if (!(inputElemType == ScaleDotElemType::E2M1 ||
        inputElemType == ScaleDotElemType::E4M3 ||
        inputElemType == ScaleDotElemType::E5M2))
    return xTy;

  auto factor = inputElemType == ScaleDotElemType::E2M1 ? 2 : 1;
  auto xShape = xTy.getShape();
  auto newShape = llvm::to_vector(xShape);
  auto encoding = xTy.getEncoding();
  if (!encoding) {
    newShape.back() *= factor;
    return RankedTensorType::get(xShape, outputElemType);
  }

  auto oldEncoding = cast<DotOperandEncodingAttr>(encoding);
  auto newVEncoding = DotOperandEncodingAttr::get(
      ctx, oldEncoding.getOpIdx(), oldEncoding.getParent(),
      oldEncoding.getKWidth() * factor);
  // Figure out the K dimension for the input A/B, given that the return
  // type is upcasted A/B type so we need to update the proper dim size.
  const int opIdx = oldEncoding.getOpIdx();
  const bool hasBatch = xShape.size() == 3;
  const int kIdx = (opIdx == 0 ? 1 : 0) + hasBatch;
  newShape[kIdx] *= factor;
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
    return emitOpError(
        "Expect output layout to be transposed per thread: " +
        expectedLinearLayout.toString() +
        "\nGot following dst layout: " + dstLinearLayout.toString());
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
  SmallVector<unsigned> sizePerThread{srcEncoding.getSizePerThread()};
  // Trim sizePerThread to tensor shape,
  // to ensure deduced layout does not refer to elements outside of tensor
  for (int i = 0; i < rank; ++i) {
    sizePerThread[i] =
        std::min(sizePerThread[i], static_cast<unsigned>(shape[i]));
  }
  auto ctx = srcEncoding.getContext();
  auto regDimName = StringAttr::get(ctx, "register");
  auto inThreadTransposedTile =
      identityStandardND(regDimName, sizePerThread, newRegOrder);
  // make sure basis in same order as in srcLayout
  SmallVector<StringAttr> outDimNames(srcLL.getOutDimNames());
  inThreadTransposedTile = inThreadTransposedTile.transposeOuts(outDimNames);

  // Copy original bases, and replace register tile with transposed one
  LinearLayout::BasesT bases = srcLL.getBases();
  auto &regBase = *bases.find(regDimName);
  int regBasesTransposed = inThreadTransposedTile.getInDimSizeLog2(regDimName);
  for (int baseIdx = 0; baseIdx < regBasesTransposed; ++baseIdx)
    regBase.second[baseIdx] =
        inThreadTransposedTile.getBasis(regDimName, baseIdx);
  int regBasesInTile = llvm::Log2_32(product(srcEncoding.getSizePerThread()));
  for (int baseIdx = regBasesTransposed; baseIdx < regBasesInTile; ++baseIdx)
    llvm::for_each(regBase.second[baseIdx], [](int32_t &val) { val = 0; });

  LinearLayout transposedLL(bases, SmallVector<StringAttr>(outDimNames));
  return transposedLL;
}

LogicalResult ScaledUpcastFp4Op::verify() {
  RankedTensorType inputTy = getInput().getType();
  RankedTensorType outputTy = getOutput().getType();
  RankedTensorType scaleTy = getScale().getType();
  auto axis = getAxis();

  if (outputTy.getShape() != scaleTy.getShape())
    return emitError() << "scale and output should have the same shape";

  // Reuse Fp4ToFpOp's verifier to check types of input and output
  auto rank = inputTy.getRank();

  if (rank != outputTy.getRank())
    return emitError() << "source rank " << rank << " != result rank "
                       << outputTy.getRank();

  auto srcShape = inputTy.getShape();
  auto resShape = outputTy.getShape();

  if (!(0 <= axis && axis < rank))
    return emitError() << "axis " << axis << " out of range for rank " << rank;

  for (int i = 0; i < rank; ++i) {
    if (i == axis) {
      if (resShape[i] != srcShape[i] * 2)
        return emitError() << "axis " << axis
                           << " dimension must be 2x source dimension (src="
                           << srcShape[i] << ", dst=" << resShape[i] << ")";
    } else {
      if (resShape[i] != srcShape[i])
        return emitError() << "dimension " << i
                           << " mismatch (src=" << srcShape[i]
                           << ", dst=" << resShape[i] << ", axis=" << axis
                           << ")";
    }
  }
  return success();
}

Attribute ScaledUpcastFp4Op::inferDstEncoding(unsigned opIdx,
                                              Attribute srcEnc) {
  // The layout of scale is the same as that of the result
  if (opIdx == 1)
    return srcEnc;
  Attribute dstEnc;
  auto shape = getInput().getType().getShape();

  auto iface =
      srcEnc.getDialect()
          .getRegisteredInterface<triton::DialectInferLayoutInterface>();
  // Given the fp4 operand is packed, we can reuse the infer utility of
  // Fp4ToFpOp
  auto result =
      iface->inferFp4ToFpOpEncoding(shape, getAxis(), srcEnc, dstEnc,
                                    /*fwdInference*/ true, std::nullopt);
  assert(succeeded(result));
  return dstEnc;
}

Attribute ScaledUpcastFp4Op::inferSrcEncoding(unsigned opIdx,
                                              Attribute dstEnc) {
  // The layout of scale is the same as that of the result
  if (opIdx == 1)
    return dstEnc;
  Attribute srcEnc;
  auto shape = getInput().getType().getShape();

  auto iface =
      dstEnc.getDialect()
          .getRegisteredInterface<triton::DialectInferLayoutInterface>();
  // Given the fp4 operand is packed, we can reuse the infer utility of
  // Fp4ToFpOp
  if (succeeded(iface->inferFp4ToFpOpEncoding(shape, getAxis(), dstEnc, srcEnc,
                                              /*fwdInference*/ false,
                                              std::nullopt))) {
    return srcEnc;
  }
  return {};
}

Attribute ScaledUpcastFp8Op::inferDstEncoding(unsigned opIdx,
                                              Attribute srcEnc) {
  return srcEnc;
}

Attribute ScaledUpcastFp8Op::inferSrcEncoding(unsigned opIdx,
                                              Attribute dstEnc) {
  return dstEnc;
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

  auto linearLayoutSrc = triton::gpu::toLinearLayout(srcType);
  auto linearLayoutDst = triton::gpu::toLinearLayout(dstType);
  auto ctx = srcType.getContext();

  auto getBases = [&](StringRef name) {
    auto key = StringAttr::get(ctx, name);
    return std::pair{linearLayoutSrc.getBases().lookup(key),
                     linearLayoutDst.getBases().lookup(key)};
  };

  auto srcToDstShape = LLVM::AMD::multiDimElementwise<int64_t, int64_t>(
      dstShape, srcShape, std::divides<unsigned>());
  std::vector<unsigned> defaultOrder(rank);
  std::iota(defaultOrder.rbegin(), defaultOrder.rend(), 0);

  StringAttr kReg = StringAttr::get(ctx, "register");
  auto dstRegBases = linearLayoutDst.getBases().lookup(kReg);
  int dstRegCount = 1 << dstRegBases.size();

  // Algorithm:
  // 1. for all elements in dst tensor
  // 2.   get dst value location in tensor
  // 3.   find, which input tile holds the dst value
  // 4.   subtract dst coordinates and start coordinates of the tile
  // 5.   check if exist source register which holds dst value

  // 1. for all elements in dst tensor
  for (int regId = 0; regId < dstRegCount; ++regId) {
    // 2.   get dst value location in tensor
    auto elemCoords = mlir::triton::AMD::getElemCoordinatesFromRegisters(
        linearLayoutDst, regId, ctx);
    auto elemCoordsArray = llvm::to_vector(llvm::make_second_range(elemCoords));

    // 3.   find, which input tile holds the dst value
    auto multiDimOperandIdx = LLVM::AMD::multiDimElementwise<int32_t, int64_t>(
        elemCoordsArray, srcShape, std::divides<unsigned>());
    auto linearOperandIdx =
        mlir::LLVM::linearize(multiDimOperandIdx, srcToDstShape, defaultOrder);

    // 4.   subtract dst coordinates and start coordinates of the tile

    for (int dim = 0; dim < rank; ++dim)
      elemCoords[dim].second -= multiDimOperandIdx[dim] * srcShape[dim];

    std::optional<int> srcReg = mlir::triton::AMD::getRegFromCoordinates(
        linearLayoutSrc, elemCoords, ctx);
    // 5.   check if exist source register which holds dst value

    if (!srcReg.has_value()) {
      auto coordsStr = getStringFromCoords(elemCoords);
      std::string msg =
          "No source register holds the element for destination index " +
          coordsStr;
      return emitError(msg);
    }
  }

  auto [laneSrc, laneDst] = getBases("lane");
  auto [warpSrc, warpDst] = getBases("warp");
  if (laneSrc != laneDst || warpSrc != warpDst) {
    return emitError("Lane and warp dim basis must match between source and "
                     "destination layout.");
  }

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

LogicalResult AsyncTDMCopyGlobalToLocalOp::verify() {
  auto tensorDescTy = getDesc().getType();
  auto smemTy = getResult().getType();

  // Check that every dimension of the block shape is <= 2^16
  auto blockShape = tensorDescTy.getBlockType().getShape();
  auto verifyResult = verifyTDMBlockSize(getOperation(), blockShape);
  if (failed(verifyResult))
    return verifyResult;

  auto swizzledEnc =
      llvm::dyn_cast<gpu::SwizzledSharedEncodingAttr>(smemTy.getEncoding());
  if (swizzledEnc && swizzledEnc.getMaxPhase() != 1)
    return emitOpError("TDM does not support swizzling");

  auto paddedEnc =
      llvm::dyn_cast<gpu::PaddedSharedEncodingAttr>(smemTy.getEncoding());
  if (!paddedEnc && !swizzledEnc)
    return emitOpError("Invalid shared memory layout for TDM");

  Type elementType = smemTy.getElementType();
  auto elementBitWidth = elementType.getIntOrFloatBitWidth();
  if (paddedEnc) {
    unsigned dwordSize = 32;
    for (auto [interval, padding] :
         llvm::zip(paddedEnc.getIntervals(), paddedEnc.getPaddings())) {
      auto intervalInDwords = interval * elementBitWidth / dwordSize;
      if (intervalInDwords < 2)
        return emitOpError("TDM padding interval must be at least 2 dwords");

      auto paddingInDwords = padding * elementBitWidth / dwordSize;
      if (paddingInDwords < 1)
        return emitOpError("TDM padding amount must be at least 1 dword");
    }
  }

  return success();
}

// -- AsyncCopyLocalToGlobalOp --
LogicalResult AsyncCopyLocalToGlobalOp::verify() {
  // Verify the source is local memory (shared memory)
  auto srcTy = getSrc().getType();
  if (!isa<gpu::SharedMemorySpaceAttr>(srcTy.getMemorySpace()))
    return emitOpError("source must be in shared memory");

  return success();
}

LogicalResult AsyncTDMCopyLocalToGlobalOp::verify() {
  auto tensorDescTy = getDesc().getType();
  auto smemTy = getSrc().getType();

  // Check that every dimension of the block shape is <= 2^16
  auto blockShape = tensorDescTy.getBlockType().getShape();
  auto verifyResult = verifyTDMBlockSize(getOperation(), blockShape);
  if (failed(verifyResult))
    return verifyResult;

  auto swizzledEnc =
      llvm::dyn_cast<gpu::SwizzledSharedEncodingAttr>(smemTy.getEncoding());
  if (swizzledEnc && swizzledEnc.getMaxPhase() != 1)
    return emitOpError("TDM does not support swizzling");

  auto paddedEnc =
      llvm::dyn_cast<gpu::PaddedSharedEncodingAttr>(smemTy.getEncoding());
  if (paddedEnc)
    return emitOpError("TDM store does not support padding");

  if (!paddedEnc && !swizzledEnc)
    return emitOpError("Invalid shared memory layout for TDM");

  return success();
}

// -- InitBarrierOp --
LogicalResult InitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  return success();
}

// -- WaitBarrierOp --
LogicalResult WaitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- ArriveBarrierOp --
LogicalResult ArriveBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  return success();
}

// -- AsyncCopyMbarrierArriveOp --
LogicalResult AsyncCopyMbarrierArriveOp::verify() {
  if (failed(verifyBarrierType(*this, getBarrier().getType())))
    return failure();
  return success();
}

} // namespace mlir::triton::amdgpu
