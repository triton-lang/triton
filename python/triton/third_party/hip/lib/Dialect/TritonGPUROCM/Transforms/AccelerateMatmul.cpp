#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/AnalysisROCM/Utility.h"
#include "triton/Dialect/TritonGPUROCM/IR/Dialect.h"
#include "triton/Dialect/TritonGPUROCM/Transforms/Passes.h"
#include "triton/Dialect/TritonGPUROCM/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu_rocm;
namespace {
using tt::DotOp;
using ttg::BlockedEncodingAttr;
using ttg::ConvertLayoutOp;
using ttg::DotOperandEncodingAttr;
using ttg::MmaEncodingAttr;
using ttg::SliceEncodingAttr;

// higher mma version is prefered, will fallback to lower version if not
// supported
static int getMMAVersionSafe(int computeCapability, tt::DotOp op) {
  int baseVersion = 0;
  if (computeCapability < 75) {
    baseVersion = 1;
  } else if (computeCapability < 90) {
    baseVersion = 2;
  } else if (computeCapability < 100) {
    baseVersion = 3;
  } else {
    assert(false && "computeCapability not supported");
  }

  for (; baseVersion >= 1; baseVersion--) {
    if (supportMMA(op, baseVersion)) {
      return baseVersion;
    }
  }

  return 0;
}

SmallVector<unsigned, 2>
warpsPerTileV2(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, {filter});
  for (Operation *op : slices)
    if (isa<tt::DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, 8};
  // TODO (@daadaada): double-check.
  // original logic in
  // https://github.com/openai/triton/blob/master/lib/codegen/analysis/layout.cc#L252
  // seems buggy for shape = [32, 16] ?
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

#if 1
SmallVector<unsigned, 2> warpsPerTileMI200(tt::DotOp dotOp,
                                           const ArrayRef<int64_t> shape,
                                           int numWarps) {
  // TODO: needs to be updated with appropriate shapePerWarp etc.
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, filter);
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
    if (tensorShape[0] / (shapePerWarp[0] *2 )  / ret[0] >=
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

SmallVector<unsigned, 2>
warpsPerTileV3(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
               const SmallVector<unsigned, 3> &instrShape) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getResult(), &slices);
  if (llvm::find_if(slices, [](Operation *op) { return isa<tt::DotOp>(op); }) !=
      slices.end())
    return {(unsigned)numWarps, 1};

  // For MMAv3, the smallest indivisible unit of warp shape is (4, 1).
  SmallVector<unsigned, 2> ret = {4, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, instrShape[1]};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] > shapePerWarp[0] * ret[0]) {
      ret[0] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

class BlockedToMFMA : public mlir::RewritePattern {
  int mfmaVersion;
public:
  BlockedToMFMA(mlir::MLIRContext *context, int mfmaVersion)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context), mfmaVersion(mfmaVersion) {}

  bool isChainDot(tt::DotOp &dotOp) const {
    auto filter = [&dotOp](Operation *op) {
      return op->getParentRegion() == dotOp->getParentRegion();
    };
    auto slices = mlir::getSlice(dotOp, filter);
    for (Operation *op : slices) {
      if (isa<tt::DotOp>(op) && (op != dotOp))
        return true;
    }
    return false;
  }

  /// @brief Choose MFMA instruction parameters
  /// @param dot target dot operation
  /// @param mfmaVersion
  /// @param nonKDim
  /// @return pair {nonKDim, kDim} sizes of one MFMA instruction arguments
  std::pair<int64_t, int64_t> chooseMfmaDimensions(tt::DotOp dot,
                                                   int mfmaVersion,
                                                   int64_t nonKDim) const {
    // number of matrix elements along k dim per one MFMA intruction
    int64_t kDim = -1;
    auto opType = dot.getA().getType().cast<RankedTensorType>();
    auto elemType = opType.getElementType();
    if (nonKDim == 32) {
      if (elemType.isF32())
        kDim = 2;
      if (elemType.isF16())
        kDim = 8;
      if (elemType.isBF16()) {
        if (mfmaVersion == 1)
          kDim = 4;
        if (mfmaVersion == 2)
          kDim = 8;
      }
      if (elemType.isInteger(8))
        kDim = 8;
    } else {
      if (elemType.isF32())
        kDim = 4;
      if (elemType.isF16())
        kDim = 16;
      if (elemType.isBF16()) {
        if (mfmaVersion == 1)
          kDim = 8;
        if (mfmaVersion == 2)
          kDim = 16;
      }
      if (elemType.isInteger(8))
        kDim = 16;
    }
    assert(kDim != -1);
    return {nonKDim, kDim};
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<tt::DotOp>(op);

    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        !oldRetType.getEncoding().isa<ttg::BlockedEncodingAttr>())
      return failure();

    // TODO replace with nonKDim with some heuristic in chooseMfmaDimensions
    // function
    int64_t externalNonKDim = 32;

    const char *mfmaType = std::getenv("MFMA_TYPE");
    if (mfmaType) {
      externalNonKDim = std::stol(mfmaType);
      assert(externalNonKDim == 32 || externalNonKDim == 16);
    }

    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());

    // get MFMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUROCMDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();
    auto ctx = oldAType.getContext();

    ttg::MfmaEncodingAttr mfmaEnc;

    auto [nonKDim, kDim] =
        chooseMfmaDimensions(dotOp, mfmaVersion, externalNonKDim);

    auto warpsPerTile = warpsPerTileMI200(dotOp, retShape, numWarps);

    bool isTransposed = isChainDot(dotOp);
    mfmaEnc = ttg::MfmaEncodingAttr::get(oldRetType.getContext(), nonKDim,
                                         warpsPerTile, isTransposed, CTALayout);

    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), mfmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);
    auto oldAOrder = oldAType.getEncoding()
                         .cast<ttg::DotOperandEncodingAttr>()
                         .getParent()
                         .cast<ttg::BlockedEncodingAttr>()
                         .getOrder();
    auto oldBOrder = oldBType.getEncoding()
                         .cast<ttg::DotOperandEncodingAttr>()
                         .getParent()
                         .cast<ttg::BlockedEncodingAttr>()
                         .getOrder();

    // kWidth is a number of consecutive elements per one instruction per one thread
    auto kWidth = kDim;
    // in mfma 32x32 case argument matrix groups elements in 2 groups
    // in mfma 16x16 case argument matrix groups elements in 4 groups
    if (nonKDim == 32) {
      kWidth /= 2;
    } else {
      assert(nonKDim == 16);
      kWidth /= 4;
    }
    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth));
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth));
    a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<tt::DotOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, dotOp.getAllowTF32());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(
        op, oldRetType, newDot.getResult());
    return success();
  }
};
#endif

class BlockedToMMA : public mlir::RewritePattern {
  int computeCapability;
  mutable int mmaV1Counter{}; // used to generate ID for MMAv1 encoding
  mutable llvm::SmallVector<llvm::SetVector<Operation *>> dotOpSetVector;
  mutable llvm::SmallVector<unsigned> mmaV3InstrNs;

  static bool bwdFilter(Operation *op) {
    return op->getNumOperands() == 1 &&
           (isa<tt::FpToFpOp, tt::BitcastOp, ttg::ConvertLayoutOp>(op) ||
            op->getDialect()->getTypeID() ==
                mlir::TypeID::get<arith::ArithDialect>());
  }

  // finds the first different value bitwidth in the chain of
  // shape-preserving unary ops  that x depends on
  static int computeOrigBitWidth(Value x) {
    int finalBitWidth = getElementTypeOrSelf(x).getIntOrFloatBitWidth();
    int origBitWidth = finalBitWidth;
    SetVector<Operation *> slice;
    mlir::getBackwardSlice(x, &slice, bwdFilter);
    Operation *firstOp = slice.empty() ? nullptr : *slice.begin();
    if (firstOp)
      if (Value arg = firstOp->getOperand(0))
        if (RankedTensorType argTy = arg.getType().dyn_cast<RankedTensorType>())
          origBitWidth = argTy.getElementType().getIntOrFloatBitWidth();
    return origBitWidth;
  }

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context),
        computeCapability(computeCapability) {}

  static SmallVector<unsigned, 3>
  getWarpsPerTile(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int version,
                  int numWarps, const SmallVector<unsigned, 3> &instrShape) {
    switch (version) {
    case 2:
      return warpsPerTileV2(dotOp, shape, numWarps);
    case 3:
      return warpsPerTileV3(dotOp, shape, numWarps, instrShape);
    default:
      assert(false && "not supported version");
      return {0, 0};
    }
  }

  unsigned getMmaV3InstrN(tt::DotOp dotOp, unsigned currN) const {
    auto type = dotOp.getResult().getType().cast<RankedTensorType>();
    if (type.getEncoding().isa<MmaEncodingAttr>())
      return currN;
    for (size_t i = 0; i < dotOpSetVector.size(); ++i) {
      if (dotOpSetVector[i].count(dotOp.getOperation()) > 0)
        return mmaV3InstrNs[i];
    }

    SetVector<Operation *> slices;
    mlir::getForwardSlice(dotOp.getResult(), &slices);
    mlir::getBackwardSlice(dotOp.getOperation(), &slices);
    unsigned N = currN;
    llvm::SetVector<Operation *> dotOpSet;
    for (Operation *iter : slices) {
      if (auto nextDotOp = dyn_cast<tt::DotOp>(iter)) {
        auto type = nextDotOp.getResult().getType().cast<RankedTensorType>();
        auto AType = nextDotOp.getOperand(0).getType().cast<RankedTensorType>();
        auto shapePerCTA = ttg::getShapePerCTA(type);
        auto instrShape = mmaVersionToInstrShape(3, shapePerCTA, AType);
        dotOpSet.insert(iter);
        if (instrShape[1] < N)
          N = instrShape[1];
      }
    }
    mmaV3InstrNs.push_back(N);
    dotOpSetVector.push_back(dotOpSet);
    return N;
  }

  static Value getMMAv3Operand(Value v, mlir::PatternRewriter &rewriter,
                               int opIdx) {
    auto cvtOp = dyn_cast_or_null<ttg::ConvertLayoutOp>(v.getDefiningOp());
    auto arg = cvtOp.getSrc();
    auto argType = arg.getType().cast<RankedTensorType>();
    auto eltType = argType.getElementType();
    assert(argType.getEncoding() && "unexpected tensor type");
    auto newOrder = ttg::getOrder(argType.getEncoding());

    // MMAv3 with transpose only supports f16 and bf16 data type
    // fallback to MMAv3 without transpose for other data types
    if (!eltType.isF16() && !eltType.isBF16()) {
      if (opIdx == 1) {
        newOrder = {0, 1};
      } else {
        newOrder = {1, 0};
      }
    }

    auto CTALayout = ttg::getCTALayout(argType.getEncoding());
    auto newLayout = ttg::SharedEncodingAttr::get(
        argType.getContext(), argType.getShape(), newOrder, CTALayout,
        argType.getElementType());
    auto newType = RankedTensorType::get(argType.getShape(),
                                         argType.getElementType(), newLayout);

    return rewriter.create<ttg::ConvertLayoutOp>(arg.getLoc(), newType, arg);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    auto dotOp = cast<tt::DotOp>(op);
    auto ctx = op->getContext();
    // TODO: Check data-types and SM compatibility
    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        oldRetType.getEncoding().isa<ttg::MmaEncodingAttr>())
      return failure();

    auto AType = dotOp.getOperand(0).getType().cast<RankedTensorType>();
    auto BType = dotOp.getOperand(1).getType().cast<RankedTensorType>();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = ttg::getShapePerCTA(oldRetType);
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUROCMDialect::getNumWarps(mod);
    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());

    int versionMajor = getMMAVersionSafe(computeCapability, dotOp);
    if (!versionMajor)
      return failure();

    auto instrShape =
        mmaVersionToInstrShape(versionMajor, retShapePerCTA, AType);
    if (versionMajor == 3)
      instrShape[1] = getMmaV3InstrN(dotOp, instrShape[1]);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();

    ttg::MmaEncodingAttr mmaEnc;
    if (versionMajor == 1) {
      SetVector<Operation *> aBwdSlices, bBwdSlices;
      auto isCvt = [](Operation *op) { return isa<ConvertLayoutOp>(op); };
      getBackwardSlice(a, &aBwdSlices, {isCvt});
      getBackwardSlice(b, &bBwdSlices, {isCvt});
      // get the source of the first conversion found in slices
      auto getCvtArgOrder = [](Operation *op) {
        return cast<ConvertLayoutOp>(op)
            .getOperand()
            .getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .cast<BlockedEncodingAttr>()
            .getOrder();
      };
      bool isARow = true;
      bool isBRow = true;
      Operation *aOp = a.getDefiningOp();
      Operation *bOp = b.getDefiningOp();
      if (!aBwdSlices.empty())
        aOp = aBwdSlices[0];
      if (!bBwdSlices.empty())
        bOp = bBwdSlices[0];
      if (aOp)
        isARow = getCvtArgOrder(aOp)[0] == 1;
      if (bOp)
        isBRow = getCvtArgOrder(bOp)[0] == 1;

      mmaEnc = ttg::MmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor, numWarps, CTALayout,
          instrShape, oldAType.getShape(), oldBType.getShape(), retShapePerCTA,
          isARow, isBRow, mmaV1Counter++);
    } else if (versionMajor == 2 || versionMajor == 3) {
      int versionMinor = computeCapability == 75 ? 1 : 0;
      auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                          numWarps, instrShape);
      mmaEnc = ttg::MmaEncodingAttr::get(oldRetType.getContext(), versionMajor,
                                         versionMinor, warpsPerTile, CTALayout,
                                         instrShape);
    }
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);
    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(oldAcc.getLoc(),
                                                        newRetType, oldAcc);

    if (versionMajor == 3) {
      a = getMMAv3Operand(a, rewriter, 0);
      b = getMMAv3Operand(b, rewriter, 1);
    } else {

      // convert operands
      int minBitwidth =
          std::min(computeOrigBitWidth(a), computeOrigBitWidth(b));
      Type minType = IntegerType::get(ctx, minBitwidth);
      // convert A operand
      auto newAEncoding = ttg::DotOperandEncodingAttr::get(
          oldAType.getContext(), 0, newRetType.getEncoding(),
          minBitwidth > 0 ? minType : oldAType.getElementType());
      auto newAType = RankedTensorType::get(
          oldAType.getShape(), oldAType.getElementType(), newAEncoding);
      a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
      // convert B operand
      auto newBEncoding = ttg::DotOperandEncodingAttr::get(
          oldBType.getContext(), 1, newRetType.getEncoding(),
          minBitwidth > 0 ? minType : oldBType.getElementType());
      auto newBType = RankedTensorType::get(
          oldBType.getShape(), oldBType.getElementType(), newBEncoding);
      b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    }
    // convert dot instruction
    auto newDot = rewriter.create<tt::DotOp>(dotOp.getLoc(), newRetType, a, b,
                                             newAcc, dotOp.getAllowTF32());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, oldRetType,
                                                      newDot.getResult());
    return success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPUROCM/Transforms/Passes.h.inc"

class TritonGPUAccelerateMatmulPass
    : public TritonGPUAccelerateMatmulBase<TritonGPUAccelerateMatmulPass> {
public:
  TritonGPUAccelerateMatmulPass() = default;
  TritonGPUAccelerateMatmulPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
#if 1
    if (computeCapability == 1 || computeCapability == 2) {
      int mfmaVersion = computeCapability;
      patterns.add<::BlockedToMFMA>(context, mfmaVersion);
    }
#else
    patterns.add<::BlockedToMMA>(context, computeCapability);
#endif
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonGPUROCMAccelerateMatmulPass(int computeCapability) {
  return std::make_unique<TritonGPUAccelerateMatmulPass>(computeCapability);
}
