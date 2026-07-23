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
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include <limits>
#include <optional>

// clang-format off
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.cpp.inc"
// clang-format on

#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TDMUtility.h"

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

namespace {

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
LogicalResult verifyTDMBlockSize(Operation *op, ArrayRef<int64_t> blockShape) {
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

// Verify the descriptor and allocation carry a consistent TDM shared layout
LogicalResult verifyTDMLayoutConsistency(Operation *op,
                                         triton::TensorDescType descTy,
                                         gpu::MemDescType smemTy) {
  Attribute descLayout = descTy.getSharedLayout();
  if (!descLayout)
    return success();
  Attribute allocLayout = smemTy.getEncoding();

  bool compatible = descLayout == allocLayout;
  // Padded layouts bake in the tile shape, so compare the physical padding
  // only.
  auto descPad = llvm::dyn_cast<gpu::PaddedSharedEncodingAttr>(descLayout);
  auto allocPad = llvm::dyn_cast<gpu::PaddedSharedEncodingAttr>(allocLayout);
  if (descPad && allocPad)
    compatible = descPad.getIntervals() == allocPad.getIntervals() &&
                 descPad.getPaddings() == allocPad.getPaddings();

  if (!compatible)
    return op->emitOpError("shared layout of the tensor descriptor (")
           << descLayout
           << ") is inconsistent with the shared memory allocation layout ("
           << allocLayout
           << "); TDM uses a single shared layout so they must match";
  return success();
}

// Verify the TDM layout constraints common to all TDM ops
LogicalResult verifyTDMCommonLayout(Operation *op,
                                    triton::TensorDescType descTy,
                                    gpu::MemDescType smemTy) {
  if (failed(verifyTDMBlockSize(op, descTy.getShape())))
    return failure();

  auto swizzledEnc =
      llvm::dyn_cast<gpu::SwizzledSharedEncodingAttr>(smemTy.getEncoding());
  if (swizzledEnc && swizzledEnc.getMaxPhase() != 1)
    return op->emitOpError("TDM does not support swizzling");

  return verifyTDMLayoutConsistency(op, descTy, smemTy);
}

// The v_cvt_scale_pk8 lowering upcasts 8 register-consecutive fp4 values with a
// single scale, check that 8 elements in the scale layout are all broadcasts.
bool pk8GroupSharesSingleScale(const LinearLayout &scaleLL,
                               StringAttr kRegister) {
  if (scaleLL.getInDimSizeLog2(kRegister) < 3)
    return false;
  auto outDims = llvm::to_vector(scaleLL.getOutDimNames());
  return scaleLL.resizeInDim(kRegister, /*newSize=*/8)
      .sublayoutIsZero({kRegister}, outDims);
}

// Validates the output/scale shape relationship and returns the number of
// output elements each scale covers along the scaled axis (>= 1), or nullopt if
// the shapes are incompatible. A block of 1 means the scales are expanded to
// the output shape; anything larger means compact scales.
std::optional<int64_t> scaledUpcastFp4ScaleBlock(ScaledUpcastFp4Op op) {
  auto outputShape = op.getOutput().getType().getShape();
  auto scaleShape = op.getScale().getType().getShape();
  int64_t axis = op.getAxis();
  if (axis < 0 || axis >= static_cast<int64_t>(outputShape.size()))
    return std::nullopt;
  if (outputShape.size() != scaleShape.size())
    return std::nullopt;
  int64_t outputAxis = outputShape[axis];
  int64_t scaleAxis = scaleShape[axis];
  if (scaleAxis <= 0 || outputAxis % scaleAxis != 0)
    return std::nullopt;

  return outputAxis / scaleAxis;
}

std::optional<LinearLayout>
computeCompactScaleLayoutForOp(ScaledUpcastFp4Op op, Attribute outputEnc) {
  auto scaleBlock = scaledUpcastFp4ScaleBlock(op);
  if (!scaleBlock || *scaleBlock == 1)
    return std::nullopt;
  if (!outputEnc || !isa<gpu::LayoutEncodingTrait>(outputEnc))
    return std::nullopt;
  auto outputLL =
      gpu::toLinearLayout(op.getOutput().getType().getShape(),
                          cast<gpu::LayoutEncodingTrait>(outputEnc));
  return ScaledUpcastFp4Op::computeScaleLayout(outputLL, op.getAxis(),
                                               *scaleBlock);
}

std::optional<Attribute>
inferScaledUpcastFp4ScaleEncoding(ScaledUpcastFp4Op op, Attribute outputEnc) {
  auto scaleBlock = scaledUpcastFp4ScaleBlock(op);
  if (!scaleBlock)
    return std::nullopt;
  // Expanded scales have the same axis extent as the output and reuse
  // outputEnc.
  if (*scaleBlock == 1)
    return outputEnc;
  // Compact scales: drop the broadcast register bases so the inferred layout
  // keeps a single register per distinct scale value.
  auto compact = computeCompactScaleLayoutForOp(op, outputEnc);
  if (!compact)
    return std::nullopt;

  auto kRegister = StringAttr::get(outputEnc.getContext(), "register");
  return gpu::LinearEncodingAttr::get(
      outputEnc.getContext(), compact->removeZeroBasesAlongDim(kRegister));
}

LogicalResult verifyScaledUpcastFp4ScaleLayout(ScaledUpcastFp4Op op) {
  RankedTensorType scaleTy = op.getScale().getType();
  auto scaleEnc = scaleTy.getEncoding();
  auto outputEnc = op.getOutput().getType().getEncoding();
  if (!scaleEnc != !outputEnc)
    return op.emitError()
           << "scale and output must both have an encoding, or neither";
  if (!scaleEnc)
    return success();

  // Reduce a scale encoding to one register per distinct scale, so the final
  // comparison matches up to redundant register broadcasting (a thread may keep
  // one register per distinct scale rather than replicating it across every
  // output register it covers).
  auto kRegister = StringAttr::get(op.getContext(), "register");
  auto stripped = [&](Attribute enc) {
    return mlir::triton::gpu::toLinearLayout(
               scaleTy.getShape(), cast<gpu::LayoutEncodingTrait>(enc))
        .removeZeroBasesAlongDim(kRegister);
  };

  std::optional<LinearLayout> expectedLL;
  if (auto compact = computeCompactScaleLayoutForOp(op, outputEnc)) {
    // Compact scales: the 8 register-consecutive fp4 of a v_cvt_scale_pk8 group
    // must share a single scale; otherwise the lowering has no single held
    // scale to apply to the group (e.g. a group that spans two scale blocks
    // along the scaled axis, or one that crosses rows of a non-scaled dim).
    if (!pk8GroupSharesSingleScale(*compact, kRegister))
      return op.emitError()
             << "the 8 elements of a v_cvt_scale_pk8 group would not share a "
                "single scale; the scaled axis must place 8 register-"
                "consecutive elements within one scale block";
    expectedLL = compact->removeZeroBasesAlongDim(kRegister);
  } else if (auto expectedEnc =
                 inferScaledUpcastFp4ScaleEncoding(op, outputEnc)) {
    // Expanded scales reuse the output encoding.
    expectedLL = stripped(*expectedEnc);
  } else {
    return op.emitError() << "could not infer expected scale encoding";
  }

  if (stripped(scaleEnc) != *expectedLL)
    return op.emitError()
           << "scale encoding is not compatible with the inferred scale layout";
  return success();
}

} // namespace

// Derive the layout of a scale tensor from the upcast output layout. A compact
// scale holds a single value per `elementsPerScale` consecutive output elements
// along `axis`, so the scale that an output element at coordinate `k` needs is
// `scale[..., k / elementsPerScale, ...]`, i.e. the output coordinate with its
// low `log2(elementsPerScale)` bits dropped (right-shifted). Output positions
// that fall inside the same scale tile map to the same scale and so collapse to
// broadcast (zero) bases along the scaled axis. `elementsPerScale` must be a
// power of two.
std::optional<LinearLayout>
ScaledUpcastFp4Op::computeScaleLayout(const LinearLayout &outputLayout,
                                      int64_t axis, int64_t elementsPerScale) {
  // For expanded scales the scale layout is the same as the output layout.
  if (elementsPerScale == 1)
    return outputLayout;
  if (elementsPerScale <= 0 || !llvm::isPowerOf2_64(elementsPerScale))
    return std::nullopt;

  auto ctx = outputLayout.getOutDimNames().begin()->getContext();
  auto axisDim = StringAttr::get(ctx, llvm::formatv("dim{0}", axis).str());

  if (!outputLayout.hasOutDim(axisDim) ||
      outputLayout.getOutDimSize(axisDim) % elementsPerScale != 0)
    return std::nullopt;

  // Build a `divisor` map from output coordinates to scale coordinates that
  // floor-divides (right shift) the scaled axis by `elementsPerScale` and
  // leaves every other dimension unchanged (identity).
  LinearLayout scaleDivisor = LinearLayout::empty();
  for (StringAttr outDim : outputLayout.getOutDimNames()) {
    int32_t size = outputLayout.getOutDimSize(outDim);
    if (outDim != axisDim) {
      scaleDivisor *= LinearLayout::identity1D(size, outDim, outDim);
      continue;
    }

    // floor-divide the scaled axis by elementsPerScale: drop the low
    // log2(elementsPerScale) bits, shift the remaining bits down.
    scaleDivisor *=
        LinearLayout::zeros1D(elementsPerScale, outDim, outDim) *
        LinearLayout::identity1D(size / elementsPerScale, outDim, outDim);
  }

  // Compose the divisor with the output layout to get the scale layout. For
  // each input it yields the scale coordinate that the output element at that
  // input needs.
  return outputLayout.compose(scaleDivisor);
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
struct CanonicalizeExtractSliceAndConcat
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
  patterns.add<CanonicalizeExtractSliceAndConcat>(context);
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
  auto outputShape = outputTy.getShape();
  auto scaleShape = scaleTy.getShape();
  if (outputShape.size() != scaleShape.size())
    return emitError() << "scale and output must have the same rank";

  int64_t axis = getAxis();
  int64_t rank = outputTy.getRank();
  if (axis < 0 || axis >= rank)
    return emitError() << "axis out of range: " << getAxis() << " for rank "
                       << rank;

  for (int64_t dim = 0; dim < rank; ++dim) {
    if (dim == axis)
      continue;
    if (outputShape[dim] != scaleShape[dim])
      return emitError()
             << "scale and output must match on non-axis dimensions";
  }
  if (scaleShape[axis] <= 0 || outputShape[axis] % scaleShape[axis] != 0)
    return emitError() << "expected output.shape[axis] to be divisible by "
                          "scale.shape[axis], but got output["
                       << axis << "]=" << outputShape[axis] << ", scale["
                       << axis << "]=" << scaleShape[axis];

  if (failed(verifyScaledUpcastFp4ScaleLayout(*this)))
    return failure();

  return mlir::triton::gpu::Fp4ToFpOp::verifyFp4ToFp(*this, inputTy, outputTy,
                                                     getAxis());
}

Attribute ScaledUpcastFp4Op::inferDstEncoding(unsigned opIdx,
                                              Attribute srcEnc) {
  // The layout of scale is either identical to the output or a quotient along
  // the scaled axis (compact scales).
  if (opIdx == 1) {
    auto scaleEnc = inferScaledUpcastFp4ScaleEncoding(*this, srcEnc);
    return scaleEnc.value_or(Attribute());
  }
  Attribute dstEnc;
  auto shape = getOutput().getType().getShape();

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
  // The layout of scale is either identical to the output or a quotient along
  // the scaled axis (compact scales).
  if (opIdx == 1) {
    auto scaleEnc = inferScaledUpcastFp4ScaleEncoding(*this, dstEnc);
    return scaleEnc.value_or(Attribute());
  }
  Attribute srcEnc;
  auto shape = getOutput().getType().getShape();

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

LogicalResult BufferLoadToLocalOp::verify() {
  auto mod = getOperation()->getParentOfType<ModuleOp>();
  if (!mod)
    return success();

  TargetFeatures features = TargetFeatures::fromModuleOp(mod);
  if (features.getArch().empty() || features.supportsBufferLoadToLocal())
    return success();
  return emitError() << "BufferLoadToLocal unsupported on target architecture";
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
static mlir::LogicalResult
foldConcatOpFromSingleSource(amdgpu::ConcatOp op, PatternRewriter &rewriter) {
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

namespace {
// Axis-aligned warp hint rule (see triton-lang/triton#10056).
// Legal iff the active warps form a regular axis-aligned bit pattern: after
// anchoring at i0 (the lowest active warp), the varying warp-id bits span
// exactly log2(K) positions, so the set is selectable by one mask test.  The
// granular diagnostics in validateWarpUsedHint below mirror these checks.
bool isAxisAlignedWarpHint(uint32_t hint, int64_t numWarps) {
  if (!llvm::isPowerOf2_64(numWarps) || numWarps >= 32 || hint == 0)
    return false;

  // Bits above num_warps - 1 must be zero (no warp at those positions).
  uint32_t numWarpsMask = (uint32_t{1} << numWarps) - 1;
  if ((hint & ~numWarpsMask) != 0)
    return false;

  unsigned K = llvm::popcount(hint);
  if (!llvm::isPowerOf2_32(K))
    return false;

  unsigned i0 = llvm::countr_zero(hint);
  uint32_t support = 0;
  for (uint32_t mask = hint; mask != 0; mask &= mask - 1) {
    unsigned w = llvm::countr_zero(mask);
    support |= static_cast<uint32_t>(w ^ i0);
  }
  return static_cast<unsigned>(llvm::popcount(support)) == llvm::Log2_32(K);
}

// Validate `warp_used_hint` against the axis-aligned hint rule (see
// TritonAMDGPUOps.td).  Encoding-specific rules (e.g.
// PartitionedSharedEncoding) live in verify() since they need the result type.
LogicalResult validateWarpUsedHint(Operation *op, uint32_t hint,
                                   int64_t numWarps) {
  if (!llvm::isPowerOf2_64(numWarps))
    return op->emitOpError("num_warps must be a power of two when using "
                           "warp_used_hint, got ")
           << numWarps;

  if (numWarps >= 32)
    return op->emitOpError("num_warps must be less than 32 when using "
                           "warp_used_hint, got ")
           << numWarps;

  if (hint == 0)
    return op->emitOpError("warp_used_hint must have at least one bit set");

  // Bits above num_warps - 1 must be zero (no warp at those positions).
  uint32_t numWarpsMask = (uint32_t{1} << numWarps) - 1;
  if ((hint & ~numWarpsMask) != 0)
    return op->emitOpError("warp_used_hint = ")
           << llvm::formatv("{0:x}", hint)
           << " sets bits beyond num_warps = " << numWarps;

  unsigned K = llvm::popcount(hint);
  if (!llvm::isPowerOf2_32(K))
    return op->emitOpError("popcount(warp_used_hint) = ")
           << K << " must be a power of two (got hint "
           << llvm::formatv("{0:x}", hint) << ")";

  // Axis-aligned check delegated to isAxisAlignedWarpHint above.  All the
  // granular conditions have passed, so a false result here means specifically
  // that the active set is not axis-aligned.
  if (!isAxisAlignedWarpHint(hint, numWarps)) {
    unsigned logK = llvm::Log2_32(K);
    return op->emitOpError("warp_used_hint = ")
           << llvm::formatv("{0:x}", hint) << " is not axis-aligned: K = " << K
           << " active warps must span exactly log2(K) = " << logK
           << " warpId bit positions";
  }

  return success();
}

LogicalResult verifyTDMSharedMemoryEncoding(Operation *op,
                                            gpu::MemDescType smemTy) {
  auto enc = smemTy.getEncoding();
  auto paddedEnc = llvm::dyn_cast<gpu::PaddedSharedEncodingAttr>(enc);
  auto swizzledEnc = llvm::dyn_cast<gpu::SwizzledSharedEncodingAttr>(enc);

  // Check for PartitionedSharedEncodingAttr and validate its inner layout.
  auto partitionedEnc = llvm::dyn_cast<gpu::PartitionedSharedEncodingAttr>(enc);
  if (partitionedEnc) {
    auto partitionLayout = partitionedEnc.getPartitionLayout();
    auto innerSwizzled =
        llvm::dyn_cast<gpu::SwizzledSharedEncodingAttr>(partitionLayout);
    if (innerSwizzled && innerSwizzled.getMaxPhase() != 1)
      return op->emitOpError(
          "TDM does not support swizzling in partitioned layout");

    auto innerPadded =
        llvm::dyn_cast<gpu::PaddedSharedEncodingAttr>(partitionLayout);
    if (!innerPadded && !innerSwizzled)
      return op->emitOpError(
          "Invalid inner layout for partitioned shared memory in TDM");
  }

  if (!paddedEnc && !swizzledEnc && !partitionedEnc)
    return op->emitOpError("Invalid shared memory layout for TDM");

  Type elementType = smemTy.getElementType();
  auto elementBitWidth = elementType.getIntOrFloatBitWidth();
  if (paddedEnc) {
    unsigned dwordSize = 32;
    for (auto [interval, padding] :
         llvm::zip(paddedEnc.getIntervals(), paddedEnc.getPaddings())) {
      auto intervalInDwords = interval * elementBitWidth / dwordSize;
      if (intervalInDwords < 2)
        return op->emitOpError(
            "TDM padding interval must be at least 2 dwords");

      auto paddingInDwords = padding * elementBitWidth / dwordSize;
      if (paddingInDwords < 1)
        return op->emitOpError("TDM padding amount must be at least 1 dword");
    }
  }

  return success();
}

LogicalResult verifyPartitionedHintFitsSingleInstruction(
    Operation *op, gpu::MemDescType smemTy, uint32_t hint,
    std::optional<size_t> memberIdx = std::nullopt) {
  auto partitionedEnc =
      llvm::dyn_cast<gpu::PartitionedSharedEncodingAttr>(smemTy.getEncoding());
  if (!partitionedEnc)
    return success();

  unsigned numLogicalPieces = partitionedEnc.getNumLogicalPieces();
  assert(numLogicalPieces > 0 &&
         "PartitionedSharedEncoding must have numLogicalPieces >= 1");
  unsigned K = llvm::popcount(hint);
  if (K % numLogicalPieces == 0)
    return success();

  InFlightDiagnostic diag =
      op->emitOpError("warp_used_hint with a partitioned shared encoding must "
                      "select K active warps such that numLogicalPieces "
                      "divides K so the copy fits in a single TDM instruction");
  if (memberIdx)
    diag << " (member " << *memberIdx << " got K = " << K
         << ", numLogicalPieces = " << numLogicalPieces << ")";
  else
    diag << " (got K = " << K << ", numLogicalPieces = " << numLogicalPieces
         << ", partitionDim = " << partitionedEnc.getPartitionDim() << ")";
  return failure();
}
} // namespace

LogicalResult AsyncTDMCopyGlobalToLocalOp::verify() {
  auto tensorDescTy = getDesc().getType();
  auto smemTy = getResult().getType();

  if (failed(verifyTDMCommonLayout(getOperation(), tensorDescTy, smemTy)))
    return failure();

  if (failed(verifyTDMSharedMemoryEncoding(getOperation(), smemTy)))
    return failure();

  if (auto warpUsedHintAttr = getWarpUsedHintAttr()) {
    int numWarps = gpu::lookupNumWarps(*this);
    uint32_t hint = static_cast<uint32_t>(warpUsedHintAttr.getInt());
    if (failed(validateWarpUsedHint(getOperation(), hint, numWarps)))
      return failure();

    if (failed(verifyPartitionedHintFitsSingleInstruction(getOperation(),
                                                          smemTy, hint)))
      return failure();
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

LogicalResult AsyncTDMFusedCopyGlobalToLocalOp::verify() {
  size_t numMembers = getDescs().size();
  if (numMembers < 2 || numMembers > 4)
    return emitOpError("requires 2 to 4 members");

  if (getDests().size() != numMembers)
    return emitOpError(
        "requires the same number of descriptors and destinations");
  if (getWarpUsedHints().size() != numMembers)
    return emitOpError("requires one warp_used_hint per member");

  auto firstDescTy = cast<triton::TensorDescType>(getDescs().front().getType());
  unsigned rank = firstDescTy.getShape().size();
  uint32_t hintUnion = 0;
  int numWarps = gpu::lookupNumWarps(*this);
  for (auto [idx, member] :
       llvm::enumerate(llvm::zip(getDescs(), getDests(), getWarpUsedHints()))) {
    auto [desc, dest, hint] = member;
    auto tensorDescTy = cast<triton::TensorDescType>(desc.getType());
    auto smemTy = cast<gpu::MemDescType>(dest.getType());
    if (failed(verifyTDMCommonLayout(getOperation(), tensorDescTy, smemTy)))
      return failure();
    if (failed(verifyTDMSharedMemoryEncoding(getOperation(), smemTy)))
      return failure();

    if (tensorDescTy.getShape().size() != rank)
      return emitOpError(
          "requires all member descriptors to have the same rank");

    uint32_t hintValue = static_cast<uint32_t>(hint);
    if (failed(validateWarpUsedHint(getOperation(), hintValue, numWarps)))
      return failure();

    if (hintUnion & hintValue)
      return emitOpError("requires pairwise-disjoint warp_used_hint values");
    hintUnion |= hintValue;

    if (failed(verifyPartitionedHintFitsSingleInstruction(
            getOperation(), smemTy, hintValue, idx)))
      return failure();
  }

  return success();
}

LogicalResult AsyncTDMCopyLocalToGlobalOp::verify() {
  auto tensorDescTy = getDesc().getType();
  auto smemTy = getSrc().getType();

  if (failed(verifyTDMCommonLayout(getOperation(), tensorDescTy, smemTy)))
    return failure();

  auto enc = smemTy.getEncoding();
  auto paddedEnc = llvm::dyn_cast<gpu::PaddedSharedEncodingAttr>(enc);
  if (!paddedEnc && !llvm::isa<gpu::SwizzledSharedEncodingAttr>(enc))
    return emitOpError("Invalid shared memory layout for TDM");

  auto blockShape = tensorDescTy.getShape();
  if (paddedEnc) {
    // Check if we can apply the padding workaround, see the lowering to LLVM
    // for more details.
    auto intervals = paddedEnc.getIntervals();
    if (intervals.size() != 1)
      return emitOpError("TDM store only supports single interval paddings.");

    auto shapePerCTA = triton::gpu::getShapePerCTA(paddedEnc, blockShape);
    if (intervals[0] != shapePerCTA.back())
      return emitOpError("TDM store padding is only supported when padding "
                         "interval equals the innermost block dimension (got "
                         "padInterval=")
             << intervals[0] << ", innermost dimension=" << blockShape.back()
             << ")";
  }

  return success();
}

LogicalResult AsyncTDMScatterOp::verify() {
  auto tensorDescTy = getDesc().getType();
  auto smemTy = getSrc().getType();

  // TDM scatter mode only supports 2D tensors
  auto blockShape = tensorDescTy.getShape();
  if (blockShape.size() != 2)
    return emitOpError("TDM scatter only supports 2D tensors, got ")
           << blockShape.size() << "D";

  if (failed(verifyTDMCommonLayout(getOperation(), tensorDescTy, smemTy)))
    return failure();

  auto enc = smemTy.getEncoding();
  if (!llvm::isa<gpu::PaddedSharedEncodingAttr>(enc) &&
      !llvm::isa<gpu::SwizzledSharedEncodingAttr>(enc))
    return emitOpError("Invalid shared memory layout for TDM");

  if (smemTy.getElementType().getIntOrFloatBitWidth() < 8)
    return emitOpError("TDM scatter requires element types of at least 8 bits");

  auto dstRowIndicesType = cast<RankedTensorType>(getDstRowIndices().getType());
  if (dstRowIndicesType.getRank() != 1)
    return emitOpError("dst_row_indices must be a 1D tensor");

  // Element type (i16 or i32) is already verified by ODS constraint
  // TensorOf<[I16, I32]>

  int64_t numIndices = dstRowIndicesType.getShape()[0];
  if (!llvm::isPowerOf2_64(numIndices))
    return emitOpError("dst_row_indices size must be a power of 2, got ")
           << numIndices;

  if (auto paddedEnc = llvm::dyn_cast<gpu::PaddedSharedEncodingAttr>(enc)) {
    // Check if we can apply the padding workaround, see the lowering to LLVM
    // for more details.
    auto intervals = paddedEnc.getIntervals();
    if (intervals.size() != 1)
      return emitOpError("TDM scatter only supports single interval paddings.");

    if (intervals[0] != blockShape.back())
      return emitOpError("TDM scatter padding is only supported when padding "
                         "interval equals the innermost block dimension (got "
                         "padInterval=")
             << intervals[0] << ", innermost dimension=" << blockShape.back()
             << ")";
  }

  return success();
}

LogicalResult AsyncTDMGatherOp::verify() {
  auto tensorDescTy = getDesc().getType();
  auto smemTy = getDst().getType();

  // TDM gather mode only supports 2D tensors
  auto blockShape = tensorDescTy.getShape();
  if (blockShape.size() != 2)
    return emitOpError("TDM gather only supports 2D tensors, got ")
           << blockShape.size() << "D";

  if (failed(verifyTDMCommonLayout(getOperation(), tensorDescTy, smemTy)))
    return failure();

  auto enc = smemTy.getEncoding();
  if (!llvm::isa<gpu::PaddedSharedEncodingAttr>(enc) &&
      !llvm::isa<gpu::SwizzledSharedEncodingAttr>(enc))
    return emitOpError("Invalid shared memory layout for TDM");

  if (smemTy.getElementType().getIntOrFloatBitWidth() < 8)
    return emitOpError("TDM gather requires element types of at least 8 bits");

  auto srcRowIndicesType = cast<RankedTensorType>(getSrcRowIndices().getType());
  if (srcRowIndicesType.getRank() != 1)
    return emitOpError("src_row_indices must be a 1D tensor");

  // Element type (i16 or i32) is already verified by ODS constraint
  // TensorOf<[I16, I32]>

  int64_t numIndices = srcRowIndicesType.getShape()[0];
  if (!llvm::isPowerOf2_64(numIndices))
    return emitOpError("src_row_indices size must be a power of 2, got ")
           << numIndices;

  auto paddedEnc = llvm::dyn_cast<gpu::PaddedSharedEncodingAttr>(enc);
  if (paddedEnc) {
    if (!(paddedEnc.getIntervals().size() == 1 &&
          paddedEnc.getPaddings().size() == 1))
      return emitOpError(
          "TDM gather does not support multiple interval-padding pairs");

    if (blockShape.back() % paddedEnc.getIntervals()[0] != 0)
      return emitOpError(
                 "TDM gather padding interval must divide the innermost "
                 "block dimension (got padInterval=")
             << paddedEnc.getIntervals()[0]
             << ", innermost dimension=" << blockShape.back() << ")";
  }

  auto shapePerCTA = triton::gpu::getShapePerCTA(smemTy);
  auto sharedOrder = triton::gpu::getOrder(
      cast<triton::gpu::SharedEncodingTrait>(smemTy.getEncoding()),
      shapePerCTA);
  if (sharedOrder[0] != (sharedOrder.size() - 1))
    return emitOpError("TDM gather only supports row-major shared order");

  // TDM gather reads the descriptor from SGPRs — all lanes in a warp see
  // the same descriptor. The index layout must broadcast the same values
  // to all lanes (all lane bits must be free).
  if (srcRowIndicesType.getEncoding()) {
    auto indexLL = triton::gpu::toLinearLayout(srcRowIndicesType);
    auto kLane = mlir::StringAttr::get(getContext(), "lane");
    auto kBlock = mlir::StringAttr::get(getContext(), "block");
    auto freeVarMasks = indexLL.getFreeVariableMasks();
    unsigned laneFreeMask = freeVarMasks.lookup(kLane);
    unsigned numLanes = indexLL.getInDimSize(kLane);
    if (laneFreeMask != (numLanes - 1))
      return emitOpError(
          "index layout distributes values across lanes, which is "
          "incompatible with the warp-level TDM instruction. Change layout "
          "to broadcast the same indices to all lanes in a warp.");

    // Because indices only describe rows the CGA layout of the indices and the
    // destination must only match on the row dimension.
    // How the tensor is distributed across the columns is not relevant for the
    // indicies and is only encoded in the CGA layout of the destination.
    auto sharedLL = paddedEnc ? paddedEnc.getLinearComponent()
                              : triton::gpu::toLinearLayout(smemTy);
    auto kDim0 = mlir::StringAttr::get(getContext(), "dim0");
    auto indexBlockIt = indexLL.getBases().find(kBlock);
    auto sharedBlockIt = sharedLL.getBases().find(kBlock);

    bool indexHasBlockBasis = indexBlockIt != indexLL.getBases().end() &&
                              !indexBlockIt->second.empty();
    bool sharedHasBlockBasis = sharedBlockIt != sharedLL.getBases().end() &&
                               !sharedBlockIt->second.empty();

    if (indexHasBlockBasis != sharedHasBlockBasis) {
      return emitOpError("TDM gather index and destination layout must both "
                         "have a block basis or neither have a block basis");
    } else if (indexHasBlockBasis && sharedHasBlockBasis) {
      auto indexRowCGA = indexLL.sublayout({kBlock}, {kDim0});
      auto sharedRowCGA = sharedLL.sublayout({kBlock}, {kDim0});
      if (!indexRowCGA.equalIgnoringOutDimSizes(sharedRowCGA))
        return emitOpError("TDM gather index and shared encoding must have "
                           "the same block basis for the row dimension");
    }
  }

  return success();
}

// -- UpdateTensorDescriptorOp --
LogicalResult UpdateTensorDescriptorOp::verify() {
  auto descTy = getDesc().getType();
  size_t rank = descTy.getShape().size();

  if (!getAddOffsets().empty() && getAddOffsets().size() != rank)
    return emitOpError("expected ")
           << rank << " add_offsets to match descriptor rank, got "
           << getAddOffsets().size();

  if (!getSetBounds().empty() && getSetBounds().size() != rank)
    return emitOpError("expected ")
           << rank << " set_bounds to match descriptor rank, got "
           << getSetBounds().size();

  // At least one mutation parameter must be provided -- a no-op update is
  // either a user mistake or should be folded by canonicalizer.
  if (getAddOffsets().empty() && getSetBounds().empty() && !getPred())
    return emitOpError("must provide at least one of add_offsets, set_bounds, "
                       "or pred");

  if (getClampBounds()) {
    if (getAddOffsets().empty())
      return emitOpError("clamp_bounds requires add_offsets");
    if (!getSetBounds().empty())
      return emitOpError("clamp_bounds and set_bounds are mutually exclusive");
  }

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

TypedValue<gpu::MemDescType> InitBarrierOp::getBarrier() { return getAlloc(); }

// -- WaitBarrierOp --
LogicalResult WaitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

TypedValue<gpu::MemDescType> WaitBarrierOp::getBarrier() { return getAlloc(); }

// -- ArriveBarrierOp --
LogicalResult ArriveBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  return success();
}

TypedValue<gpu::MemDescType> ArriveBarrierOp::getBarrier() {
  return getAlloc();
}

// -- AsyncCopyMbarrierArriveOp --
LogicalResult AsyncCopyMbarrierArriveOp::verify() {
  if (failed(verifyBarrierType(*this, getBarrier().getType())))
    return failure();
  return success();
}

// -- TDMPrefetchOp --
// This op optionally returns the prefetch offsets (testing-only). When
// `returnOffsets` is absent, it produces no results. When present, it yields an
// int64 tensor of the prefetch addresses relative to the tensor base. The
// tensor shape is:
//   [num_programs, block_shape[:-1], block_shape[-1] / elements_per_prefetch]
// i.e., the last dimension is scaled by how many elements fit in one 256-byte
// prefetch. Values are the byte offsets added to the base pointer for each
// prefetch instruction.
LogicalResult TDMPrefetchOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  TDMPrefetchOp::Adaptor ad(operands, attributes, properties, regions);

  // If returnOffsets is not set the op will not return any results
  if (!ad.getReturnOffsets().has_value()) {
    return success();
  }

  auto descType = cast<triton::TensorDescType>(ad.getDesc().getType());
  auto blockShape = descType.getShape();
  auto elementType = descType.getElementType();

  // Lookup the module to get the number of threads per warp, number of warps
  // and number of CTAs
  ModuleOp mod;
  for (auto operand : operands) {
    if (auto op = operand.getDefiningOp()) {
      mod = op->getParentOfType<ModuleOp>();
      break;
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (parentOp) {
        mod = parentOp->getParentOfType<ModuleOp>();
        break;
      }
    }
  }
  assert(mod);

  auto threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  auto numWarps = triton::gpu::lookupNumWarps(mod);
  auto numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);

  // Prefetches 256 bytes into L2
  const int bytesPerPrefetch = 256;
  int elemPerPrefetch =
      (bytesPerPrefetch * 8) / elementType.getIntOrFloatBitWidth();

  // Scale the block shape by the number of elements per prefetch
  SmallVector<int64_t> scaledBlockShape(blockShape.begin(), blockShape.end());
  scaledBlockShape.back() =
      ceil<int64_t>(scaledBlockShape.back(), elemPerPrefetch);

  // Use the default blocked encoding to unroll the TDM tile
  auto enc = triton::gpu::getDefaultBlockedEncoding(
      context, scaledBlockShape, numWarps, threadsPerWarp, numCTAs);
  IntegerType i64Type = IntegerType::get(context, 64);
  auto tensorTy = RankedTensorType::get(scaledBlockShape, i64Type, enc);

  inferredReturnTypes.push_back(tensorTy);

  return success();
}

// -- ClusterBarrierSignalOp --
LogicalResult ClusterBarrierArriveOp::verify() {
  int numCTAs = triton::gpu::lookupNumCTAs(getOperation());
  if (numCTAs <= 1)
    return emitOpError("requires ttg.num-ctas > 1");
  return success();
}

// -- ClusterBarrierWaitOp --
LogicalResult ClusterBarrierWaitOp::verify() {
  int numCTAs = triton::gpu::lookupNumCTAs(getOperation());
  if (numCTAs <= 1)
    return emitOpError("requires ttg.num-ctas > 1");
  return success();
}

// -- PredicatedOpInterface implementations --

Value BufferLoadOp::getPredicateOperand() { return getMask(); }
void BufferLoadOp::setPredicateOperand(Value pred) {
  getMaskMutable().assign(pred);
}
Type BufferLoadOp::getPredicateOperandTypeLike() {
  return getOffsets().getType();
}

Value BufferLoadToLocalOp::getPredicateOperand() { return getMask(); }
void BufferLoadToLocalOp::setPredicateOperand(Value pred) {
  getMaskMutable().assign(pred);
}
Type BufferLoadToLocalOp::getPredicateOperandTypeLike() {
  return getOffsets().getType();
}

Value BufferAtomicRMWOp::getPredicateOperand() { return getMask(); }
void BufferAtomicRMWOp::setPredicateOperand(Value pred) {
  getMaskMutable().assign(pred);
}
Type BufferAtomicRMWOp::getPredicateOperandTypeLike() {
  return getOffsets().getType();
}

Value BufferStoreOp::getPredicateOperand() { return getMask(); }
void BufferStoreOp::setPredicateOperand(Value pred) {
  getMaskMutable().assign(pred);
}
Type BufferStoreOp::getPredicateOperandTypeLike() {
  return getOffsets().getType();
}

Value AsyncCopyLocalToGlobalOp::getPredicateOperand() { return getMask(); }
void AsyncCopyLocalToGlobalOp::setPredicateOperand(Value pred) {
  getMaskMutable().assign(pred);
}
Type AsyncCopyLocalToGlobalOp::getPredicateOperandTypeLike() {
  return getDst().getType();
}

Value TDMPrefetchOp::getPredicateOperand() { return getPred(); }
void TDMPrefetchOp::setPredicateOperand(Value pred) {
  getPredMutable().assign(pred);
}
Type TDMPrefetchOp::getPredicateOperandTypeLike() {
  return IntegerType::get(getContext(), 1);
}

} // namespace mlir::triton::amdgpu
