#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace {
using tt::DotOp;
using ttg::BlockedEncodingAttr;
using ttg::ConvertLayoutOp;
using ttg::DotOperandEncodingAttr;
using ttg::MfmaEncodingAttr;
using ttg::SliceEncodingAttr;

SmallVector<unsigned, 2>
warpsPerTileMFMA(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps) {
  // TODO: needs to be updated with appropriate shapePerWarp etc.
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  mlir::ForwardSliceOptions fwdOpt;
  fwdOpt.filter = filter;
  mlir::BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = filter;
  auto slices = mlir::getSlice(dotOp, bwdOpt, fwdOpt);
  for (Operation *op : slices)
    if (isa<tt::DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<int64_t, 2> tensorShape = {shape[0], shape[1]};
  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {32, 32};
  bool changed = false;

  do {
    changed = false;
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (tensorShape[0] / (shapePerWarp[0] * 2) / ret[0] >=
        tensorShape[1] / shapePerWarp[1] / ret[1]) {
      if (ret[0] < tensorShape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);

  if (ret[1] * shapePerWarp[1] > tensorShape[1]) {
    return {ret[1], ret[0]};
  }

  return ret;
}

class BlockedToMFMA : public mlir::RewritePattern {
  int mfmaVersion;
  int enforcedNonKDim;

public:
  BlockedToMFMA(mlir::MLIRContext *context, int mfmaVersion, int nonKDim)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context),
        mfmaVersion(mfmaVersion), enforcedNonKDim(nonKDim) {}

  bool isChainDot(tt::DotOp &dotOp) const {
    auto filter = [&dotOp](Operation *op) {
      return op->getParentRegion() == dotOp->getParentRegion();
    };
    mlir::ForwardSliceOptions fwdOpt;
    fwdOpt.filter = filter;
    mlir::BackwardSliceOptions bwdOpt;
    bwdOpt.omitBlockArguments = true;
    bwdOpt.filter = filter;
    auto slices = mlir::getSlice(dotOp, bwdOpt, fwdOpt);
    for (Operation *op : slices) {
      if (isa<tt::DotOp>(op) && (op != dotOp))
        return true;
    }
    return false;
  }

  /// @brief Choose MFMA instruction parameters
  /// @param dot target dot operation
  /// @return pair {nonKDim, kDim} sizes of one MFMA instruction arguments
  std::pair<int64_t, int64_t> chooseMfmaDimensions(tt::DotOp dot) const {
    // number of matrix elements along k dim per one MFMA intruction
    int64_t kDim = -1;
    auto opType = dot.getA().getType().cast<RankedTensorType>();
    auto elemType = opType.getElementType();

    auto resType = dot.getD().getType().cast<RankedTensorType>();
    auto resShape = resType.getShape();

    int64_t nonKDim = -1;
    if (enforcedNonKDim != 0) {
      nonKDim = enforcedNonKDim;
    } else {
      nonKDim = -1;
      int minSize = std::min(resShape[0], resShape[1]);
      if (minSize >= 32)
        nonKDim = 32;
      if (minSize >= 16 && minSize < 32)
        nonKDim = 16;
      if (minSize < 16)
        nonKDim = 4;
      assert(nonKDim != -1);
    }
    switch (nonKDim) {
    case 32:
      if (elemType.isF32())
        kDim = 2;
      if (elemType.isF16())
        kDim = 8;
      if (elemType.isBF16()) {
        if (mfmaVersion == 1)
          kDim = 4;
        if (mfmaVersion >= 2)
          kDim = 8;
      }
      if (elemType.isFloat8E4M3FNUZ() || elemType.isFloat8E5M2FNUZ()) {
        assert(mfmaVersion == 3);
        kDim = 16;
      }
      if (elemType.isInteger(8)) {
        if (mfmaVersion == 3) {
          kDim = 16;
        }
        else {
          kDim = 8;
        }
      }
      break;
    case 16:
      if (elemType.isF32())
        kDim = 4;
      if (elemType.isF16())
        kDim = 16;
      if (elemType.isBF16()) {
        if (mfmaVersion == 1)
          kDim = 8;
        if (mfmaVersion >= 2)
          kDim = 16;
      }
      if (elemType.isFloat8E4M3FNUZ() || elemType.isFloat8E5M2FNUZ()) {
        assert(mfmaVersion == 3);
        kDim = 32;
      }
      if (elemType.isInteger(8)) {
        if (mfmaVersion == 3) {
          kDim = 32;
        }
        else {
          kDim = 16;
        }
      }
      break;
    case 4:
      if (elemType.isF32())
        kDim = 16;
      if (elemType.isF16())
        kDim = 64;
      if (elemType.isBF16()) {
        if (mfmaVersion == 1)
          kDim = 32;
        if (mfmaVersion >= 2)
          kDim = 64;
      }
      if (elemType.isInteger(8)) {
        kDim = 64;
      }
      break;
    default:
      llvm::report_fatal_error("unsupported nonKDim size in MFMA dot");
    }
    assert(kDim != -1);
    assert(nonKDim != -1);
    assert(resShape[0] % nonKDim == 0 && resShape[1] % nonKDim == 0);
    assert(opType.getShape()[1] % kDim == 0);
    return {nonKDim, kDim};
  }

  /**
   * @brief Convert layout and cast element type of a given tensor
   *
   * If old element type is different from new element type, this function
   * creates two new operations:
   * 1. %converted_value = layout_convert %value, newEncoding
   * 2. %casted_value = cast(fext, ftrunc, etc.) %value, newElemType
   *
   * If old element type is same as new element type, this function creates only
   * one operation: %converted_value = layout_convert %value, newEncoding
   *
   * @param rewriter
   * @param value original tensor value, which we need to convert and cast
   * @param newEncoding new encoding for the tenosr
   * @param newElemType new element type for the tensor
   * @return converted and optionaly casted tensor value
   */
  Value convertAndCastTensor(mlir::PatternRewriter &rewriter, Value value,
                             ::mlir::Attribute newEncoding,
                             Type newElemType) const {
    assert(newElemType.isIntOrFloat());

    auto loc = value.getLoc();
    auto oldType = value.getType().cast<RankedTensorType>();
    auto oldElemType = oldType.getElementType();

    assert(oldElemType.isIntOrFloat());
    assert(oldElemType.isIntOrIndex() == newElemType.isIntOrIndex());

    auto convertedType =
        RankedTensorType::get(oldType.getShape(), oldElemType, newEncoding);

    Value convertedTensor =
        rewriter.create<ttg::ConvertLayoutOp>(loc, convertedType, value);

    if (newElemType == oldElemType)
      return convertedTensor;

    Type castedType = convertedType.cloneWith(std::nullopt, newElemType);

    Value castedTensor;

    if (newElemType.isIntOrIndex()) {
      unsigned oldWidth = oldElemType.getIntOrFloatBitWidth();
      unsigned newWidth = newElemType.getIntOrFloatBitWidth();
      if (oldWidth == newWidth)
        castedTensor = rewriter.create<mlir::arith::BitcastOp>(
            loc, convertedType, convertedTensor);
      else if (oldWidth > newWidth)
        castedTensor = rewriter.create<mlir::arith::TruncIOp>(loc, castedType,
                                                              convertedTensor);
      else if (oldElemType.isSignedInteger())
        castedTensor = rewriter.create<mlir::arith::ExtSIOp>(loc, castedType,
                                                             convertedTensor);
      else
        castedTensor = rewriter.create<mlir::arith::ExtUIOp>(loc, castedType,
                                                             convertedTensor);
    } else {
      if (oldElemType.isF16() && newElemType.isF32())
        castedTensor = rewriter.create<mlir::arith::ExtFOp>(loc, castedType,
                                                            convertedTensor);
      else if (oldElemType.isF32() && newElemType.isF16())
        castedTensor = rewriter.create<mlir::arith::TruncFOp>(loc, castedType,
                                                              convertedTensor);
      else
        castedTensor =
            rewriter.create<tt::FpToFpOp>(loc, castedType, convertedTensor);
    }
    return castedTensor;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<tt::DotOp>(op);

    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        !oldRetType.getEncoding().isa<ttg::BlockedEncodingAttr>())
      return failure();

    if (!supportMFMA(dotOp))
      return failure();

    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());

    // get MFMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();
    auto ctx = oldAType.getContext();

    ttg::MfmaEncodingAttr mfmaEnc;

    auto [nonKDim, kDim] = chooseMfmaDimensions(dotOp);

    auto warpsPerTile = warpsPerTileMFMA(dotOp, retShape, numWarps);

    bool isTransposed = isChainDot(dotOp);
    mfmaEnc = ttg::MfmaEncodingAttr::get(oldRetType.getContext(), nonKDim,
                                         warpsPerTile, isTransposed, CTALayout);

    Type mfmaAccType;
    if (oldRetType.getElementType().isIntOrIndex())
      mfmaAccType = rewriter.getIntegerType(32);
    else
      mfmaAccType = rewriter.getF32Type();

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = convertAndCastTensor(rewriter, oldAcc, mfmaEnc, mfmaAccType);

    // kWidth is a number of consecutive elements per one instruction per one
    // thread
    auto kWidth = kDim;
    // in mfma 32x32 case argument matrix groups elements in 2 groups
    // in mfma 16x16 case argument matrix groups elements in 4 groups
    // in mfma 4x4 case arguemnt matrix groups in 16 groups
    switch (nonKDim) {
    case 32:
      kWidth /= 2;
      break;
    case 16:
      kWidth /= 4;
      break;
    case 4:
      kWidth /= 16;
      break;
    default:
      llvm::report_fatal_error("unsupported kDim in mfma dot");
    }
    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth));
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth));
    a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<tt::DotOp>(dotOp.getLoc(), newAcc.getType(),
                                             a, b, newAcc, dotOp.getAllowTF32(),
                                             dotOp.getMaxNumImpreciseAcc());

    Value dotOutput =
        convertAndCastTensor(rewriter, newDot, oldRetType.getEncoding(),
                             oldRetType.getElementType());

    rewriter.replaceOp(op, dotOutput);

    return success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUAccelerateMatmulPass
    : public TritonAMDGPUAccelerateMatmulBase<
          TritonAMDGPUAccelerateMatmulPass> {
public:
  TritonAMDGPUAccelerateMatmulPass() = default;
  TritonAMDGPUAccelerateMatmulPass(int matrixCoreVersion,
                                   int matrixInstructionSize) {
    this->matrixCoreVersion = matrixCoreVersion;
    this->matrixInstructionSize = matrixInstructionSize;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    if (matrixCoreVersion == 1 || matrixCoreVersion == 2 ||
        matrixCoreVersion == 3)
      patterns.add<::BlockedToMFMA>(context, matrixCoreVersion,
                                    matrixInstructionSize);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUAccelerateMatmulPass(int matrixCoreVersion,
                                             int matrixInstructionSize) {
  return std::make_unique<TritonAMDGPUAccelerateMatmulPass>(
      matrixCoreVersion, matrixInstructionSize);
}
