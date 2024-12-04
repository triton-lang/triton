#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace gpu {

namespace {

// Get the highest version supported for the hardware and the dot.
static int getMMAVersionSafe(int computeCapability, DotOp op) {
  // List supported mma version in order of preference.
  SmallVector<int> versionsSupported;
  if (computeCapability < 75) {
    versionsSupported = {1};
  } else if (computeCapability < 90) {
    versionsSupported = {2};
  } else if (computeCapability < 100) {
    versionsSupported = {3, 2};
  } else {
    assert(false && "computeCapability not supported");
  }
  for (int baseVersion : versionsSupported) {
    if (supportMMA(op, baseVersion))
      return baseVersion;
    if (baseVersion == 3)
      op.emitRemark() << "Warning: can't use MMA V3 for the dot op";
  }
  return 0;
}

SmallVector<unsigned> warpsPerTileV2(DotOp dotOp, const ArrayRef<int64_t> shape,
                                     int numWarps) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};

  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion() &&
           !isa<TransOp>(op);
  };
  auto slices = multiRootGetSlice(dotOp, {filter}, {filter});
  bool hasChainedDot = false;
  for (Operation *op : slices) {
    if (isa<DotOp>(op) && (op != dotOp)) {
      auto chainedDot = cast<DotOp>(op);
      auto resTy = chainedDot.getResult().getType();
      if (resTy.getRank() != rank) {
        continue;
      }
      if (auto mmaEncoding =
              dyn_cast<NvidiaMmaEncodingAttr>(resTy.getEncoding())) {
        return getWarpsPerCTA(mmaEncoding);
      }
      hasChainedDot = true;
    }
  }
  if (hasChainedDot) {
    if (shape[0] >= shape[1]) {
      return {(unsigned)numWarps, 1};
    } else {
      return {1, (unsigned)numWarps};
    }
  }

  assert(rank == 2);
  SmallVector<int64_t> shapePerWarp = {16, 8};
  SmallVector<int64_t> warps = {1, 1};
  // Compute repM and repN
  SmallVector<int64_t> reps = {ceil(shape[0], shapePerWarp[0]),
                               ceil(shape[1], shapePerWarp[1])};
  // The formula for the number of registers given the reps is
  // repM * 4 * repK + repN * 2 * repK + regsC
  // where regsC = repM * repN * 4, which does not depend on the warp shape
  //
  // As such, to minimize the register pressure, we need to balance
  // repM and repN. We then untie towards M, as the lhs tile has 4 elements,
  // and the rhs tile has just 2.
  while (product(warps) < numWarps) {
    if (reps[0] >= reps[1]) {
      warps[0] *= 2;
      // Too many warps for this mma (repM == repN == 1).
      // We allocate the remainin warps to the left (arbitrary choice)
      if (reps[0] != 1) {
        reps[0] /= 2;
      }
    } else {
      warps[1] *= 2;
      reps[1] /= 2;
    }
  }
  return {(unsigned)warps[0], (unsigned)warps[1]};
}

SmallVector<unsigned, 2>
warpsPerTileV3(DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
               const SmallVector<unsigned, 3> &instrShape) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getResult(), &slices);
  // Contains a chained dot. We prefer to assign warps to one axis
  // to facilitate use cases like flash attention, allowing reductions within
  // the same warp.
  if (llvm::find_if(slices, [](Operation *op) {
        return op->hasTrait<OpTrait::DotLike>();
      }) != slices.end())
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

// Returns a shared memory allocation that can be used by a dotMMA op for the
// given value.
static Value getSharedMemoryMMAOperand(Value v, mlir::PatternRewriter &rewriter,
                                       int opIdx, bool allowTranspose) {
  OpBuilder::InsertionGuard g(rewriter);
  Value arg = v;
  if (auto cvtOp = v.getDefiningOp<ConvertLayoutOp>())
    arg = cvtOp.getSrc();
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto newOrder = getOrder(argType.getEncoding());

  // If the MMA op doesn't support transpose pick the layout expected by the MMA
  // op.
  if (!allowTranspose) {
    if (opIdx == 1) {
      newOrder = {0, 1};
    } else {
      newOrder = {1, 0};
    }
  }

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CTALayout = getCTALayout(argType.getEncoding());
  auto newLayout =
      SharedEncodingAttr::get(argType.getContext(), argType.getShape(),
                              newOrder, CTALayout, argType.getElementType());
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return rewriter.create<LocalAllocOp>(arg.getLoc(), newType, arg);
}

SmallVector<unsigned, 3>
getWarpsPerTile(DotOp dotOp, const ArrayRef<int64_t> shape, int version,
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

class BlockedToMMA : public mlir::OpRewritePattern<DotOp> {
  int computeCapability;
  mutable llvm::DenseMap<Operation *, unsigned> dotOpInstNs;

  static bool bwdFilter(Operation *op) {
    return op->getNumOperands() == 1 &&
           (isa<FpToFpOp, BitcastOp, ConvertLayoutOp>(op) ||
            isPureUnaryInlineAsm(op) ||
            op->getDialect()->getTypeID() ==
                mlir::TypeID::get<arith::ArithDialect>());
  }

  // Finds the first different bitwidth in the chain of shape-preserving
  // unary ops that x depends on.
  // There are two primary scenarios:
  // (1) Upcasting: A sequence such as loading an fp16, followed by arithmetic
  // operations, then bitcasting to fp32, and finally computing in fp32.
  // (2) Downcasting: This might involve loading an fp32, performing arithmetic
  // operations, bitcasting to fp16, and finally computing in fp16.
  // In the upcasting scenario, element reordering converts the original
  // elements distribution to the order of higher precision primitives. As a
  // result, kwidth can be the bitwidth of the lower precision primitive.
  // Conversely, in the downcasting scenario, no reordering is performed,
  // making it directly use the lower precision primitive.
  static int computeOrigBitWidth(Value x) {
    int finalBitWidth = getElementTypeOrSelf(x).getIntOrFloatBitWidth();
    int origBitWidth = finalBitWidth;
    SetVector<Operation *> slice;
    mlir::BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = bwdFilter;
    getBackwardSlice(x, &slice, opt);
    for (auto op : slice) {
      if (Value arg = op->getOperand(0))
        if (auto argTy = dyn_cast<RankedTensorType>(arg.getType())) {
          auto argBitWidth = argTy.getElementType().getIntOrFloatBitWidth();
          if (argBitWidth != origBitWidth) {
            origBitWidth = std::min<int>(origBitWidth, argBitWidth);
            break;
          }
        }
    }
    return origBitWidth;
  }

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability)
      : OpRewritePattern<DotOp>(context), computeCapability(computeCapability) {
  }

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    if (computeCapability < 80) {
      dotOp.emitRemark()
          << "Dot op using MMA for compute capability " << computeCapability
          << " has been deprecated. It falls back to the FMA path.";
      return failure();
    }
    // TODO: Check data-types and SM compatibility
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    auto mod = dotOp->getParentOfType<mlir::ModuleOp>();
    int numWarps = TritonGPUDialect::getNumWarps(mod);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());

    int versionMajor = getMMAVersionSafe(computeCapability, dotOp);
    if (!(versionMajor >= 1 && versionMajor <= 3))
      return failure();

    auto instrShape = mmaVersionToInstrShape(
        versionMajor, retShapePerCTA, dotOp.getA().getType().getElementType(),
        numWarps);
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = dotOp.getA().getType();
    auto oldBType = dotOp.getB().getType();

    assert(versionMajor == 2 || versionMajor == 3);
    int versionMinor = computeCapability == 75 ? 1 : 0;
    auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                        numWarps, instrShape);
    auto mmaEnc = NvidiaMmaEncodingAttr::get(
        oldRetType.getContext(), versionMajor, versionMinor, warpsPerTile,
        CTALayout, instrShape);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);
    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc =
        rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

    Operation *newDot = nullptr;
    if (versionMajor == 3) {
      auto eltType = dotOp.getA().getType().getElementType();
      // In MMAV3 tranpose is only supported for f16 and bf16.
      bool allowTranspose = eltType.isF16() || eltType.isBF16();
      a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose);
      b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose);
      newDot = rewriter.create<triton::nvidia_gpu::WarpGroupDotOp>(
          dotOp.getLoc(), newRetType, a, b, newAcc, nullptr,
          dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc(), false);
    } else {
      // convert operands
      int minBitwidth =
          std::min(computeOrigBitWidth(a), computeOrigBitWidth(b));
      Type minType = rewriter.getIntegerType(minBitwidth);
      // convert A operand
      auto newAEncoding = DotOperandEncodingAttr::get(
          oldAType.getContext(), 0, newRetType.getEncoding(),
          minBitwidth > 0 ? minType : oldAType.getElementType());
      auto newAType = RankedTensorType::get(
          oldAType.getShape(), oldAType.getElementType(), newAEncoding);
      a = rewriter.create<ConvertLayoutOp>(a.getLoc(), newAType, a);
      // convert B operand
      auto newBEncoding = DotOperandEncodingAttr::get(
          oldBType.getContext(), 1, newRetType.getEncoding(),
          minBitwidth > 0 ? minType : oldBType.getElementType());
      auto newBType = RankedTensorType::get(
          oldBType.getShape(), oldBType.getElementType(), newBEncoding);
      b = rewriter.create<ConvertLayoutOp>(b.getLoc(), newBType, b);
      newDot = rewriter.create<DotOp>(dotOp.getLoc(), newRetType, a, b, newAcc,
                                      dotOp.getInputPrecision(),
                                      dotOp.getMaxNumImpreciseAcc());
    }
    // convert dot instruction
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType,
                                                 newDot->getResult(0));
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  return builder.create<FpToFpOp>(loc, tensorPromotedType, operand);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod, int computeCapability) {
  mod.walk([=](DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    NvidiaMmaEncodingAttr mmaLayout =
        dyn_cast<NvidiaMmaEncodingAttr>(D.getType().getEncoding());
    if (mmaLayout) {
      bool isNativeFP8 = AElType.isFloat8E5M2() || AElType.isFloat8E4M3FN();
      // promote operands for sm < 89 since fp8 mma is not natively supported
      // promote operands for sm >= 90 when mma is not v3
      if (!isNativeFP8 ||
          (isNativeFP8 && (computeCapability == 89 || mmaLayout.isHopper())))
        return;
      promoteType = builder.getF16Type();
    } else {
      // FMA case.
      Type AElType = dotOp.getA().getType().getElementType();
      Type DElType = D.getType().getElementType();
      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

class DecomposeScaledBlocked
    : public mlir::OpRewritePattern<triton::DotScaledOp> {
  int computeCapability;

public:
  DecomposeScaledBlocked(mlir::MLIRContext *context, int computeCapability)
      : mlir::OpRewritePattern<triton::DotScaledOp>(context),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotScaledOp scaledDotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability >= 100 || computeCapability < 80)
      return rewriter.notifyMatchFailure(
          scaledDotOp, "DotScaledOp just supported on Ampere and Hopper");

    auto oldRetType = scaledDotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    auto ctx = scaledDotOp.getContext();

    // Check that rhs scale is null
    assert(scaledDotOp.getRhsScale() == nullptr && "rhs scale NYI");

    // operands
    auto a = scaledDotOp.getLhs();
    auto b = scaledDotOp.getRhs();
    auto scale = scaledDotOp.getLhsScale();
    auto aType = scaledDotOp.getLhsType();
    auto bType = scaledDotOp.getRhsType();

    auto rank = oldRetType.getShape().size();
    if (rank != 2)
      return rewriter.notifyMatchFailure(scaledDotOp, "NYI: rank==3");

    assert((aType == ScaleDotElemType::E4M3 ||
            aType == ScaleDotElemType::E5M2 ||
            aType == ScaleDotElemType::E2M1) &&
           "NYI: lhs supports fp4 or fp8");
    assert(bType == ScaleDotElemType::E4M3 || bType == ScaleDotElemType::E5M2 ||
           bType == ScaleDotElemType::BF16 && "NYI: rhs supports fp8 and bf16");
    bool isFp4 = aType == ScaleDotElemType::E2M1;

    auto mmaEnc = getMMAEncoding(rewriter, scaledDotOp);
    auto versionMajor = mmaEnc.getVersionMajor();
    assert(versionMajor == 2 ||
           versionMajor == 3 && "NYI: MMAV2 and MMAV3 only");

    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = scaledDotOp.getC();
    auto newAcc =
        rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

    // TODO: This should be kWidth = 2 once MMAv2 supports kWidth=1 for 1 byte
    // types
    auto aKWidth = mmaEnc.isHopper() ? 2 : 8;
    auto bKWidth = mmaEnc.isHopper() ? 2 : 8;
    if (isFp4) {
      // Load 2x4-bit elements per thread
      aKWidth /= 2;
    }
    // [Note: A trick to avoid warp shuffles in the lowering]
    // Once we fully support LLs in the IR, we can craft an LL so that
    // broadcasting happens effectively in the convertLayoutOp lowering. For
    // this, we would just need to create an LL with
    // `bases[warps] = {(0, 0), (0, 0), ...}`

    auto newAEncoding = DotOperandEncodingAttr::get(ctx, 0, mmaEnc, aKWidth);

    // MMAv3 uses the first dimension for the M dimension, while MMAv2 uses the
    // penultimate (ugh)
    auto instrShapeM =
        mmaEnc.getInstrShape()[versionMajor == 3
                                   ? 0
                                   : mmaEnc.getInstrShape().size() - 2];
    auto warpSize = getWarpSize(newAEncoding);
    assert(instrShapeM <= warpSize);
    // Necessary choice to leave all the scales of the tile in that given warp
    auto threadsPerWarp =
        SmallVector<unsigned>{instrShapeM, warpSize / instrShapeM};

    // This has to align with the order in UpcastMXFPOp
    auto order = getMatrixOrder(rank, /*rowMajor=*/true);
    Attribute newScaleEncoding = triton::gpu::BlockedEncodingAttr::get(
        ctx, {1, 1}, threadsPerWarp, newAEncoding.getWarpsPerCTA(), order,
        mmaEnc.getCTALayout());

    // Lezcano: In the future we could just use the LLs unconditionally
    // Not doing it now as they are not as performant as Blocked encoding at
    // times E.g., we bail on them in the backwardMaterialization pass
    auto dotBroadcastsWarpLevel = mmaEnc.getWarpsPerCTA()[1] != 1;
    if (dotBroadcastsWarpLevel) {
      auto kRegister = StringAttr::get(ctx, "register");
      auto regs = identityStandardND(kRegister, {1, 1}, order);
      auto lanes =
          identityStandardND(StringAttr::get(ctx, "lane"), {16, 2}, order);

      // Extract warp layout from dotAEncoding
      // In the future we'll have some nice division utils, but until then...
      auto dotLL = *newAEncoding.toLinearLayout(a.getType().getShape());
      LinearLayout::BasesT scaleBases = dotLL.getBases();
      auto kWarp = StringAttr::get(ctx, "warp");
      auto &warpBases = scaleBases[kWarp];
      // The tile shape was [16, 2 * 4 * kWidth] with broadcasting in K
      // We divide the M dimension by 16
      auto div = 16;
      for (auto &warpBase : warpBases) {
        if (warpBase[rank - 2] != 0) {
          assert(warpBase[rank - 2] % div == 0);
          warpBase[rank - 2] /= div;
        }
      }

      LinearLayout::BasesT warpBlockBases;
      auto standardOutDims = llvm::to_vector(dotLL.getOutDimNames());
      warpBlockBases[kWarp] = warpBases;
      auto kBlock = StringAttr::get(ctx, "block");
      assert(scaleBases[kBlock].empty() && "NYI: CGAs");
      warpBlockBases[kBlock] = {};
      auto warpBlock = LinearLayout(std::move(warpBlockBases), standardOutDims);

      auto newLL =
          (regs * lanes) *
          warpBlock.transposeOuts(llvm::to_vector(lanes.getOutDimNames()));
      auto shape = scale.getType().getShape();

      // Broadcast to the correct shape Equivalent to
      // newLL = ensureLayoutNotSmallerThan(newLL.transposeOuts(getRepOrder),
      // shape);
      for (auto d : newAEncoding.getRepOrder()) {
        auto outDim = standardOutDims[d];
        auto dimSize = newLL.getOutDimSize(outDim);
        newLL *=
            LinearLayout::identity1D(shape[d] / dimSize, kRegister, outDim);
      }
      newLL = newLL.transposeOuts(standardOutDims);
      newScaleEncoding = LinearEncodingAttr::get(ctx, std::move(newLL));
    }

    a = createArg(rewriter, a, 0, aType, newAEncoding, scale, newScaleEncoding);

    Operation *newDot = nullptr;
    if (versionMajor == 2) {
      // Upcast B operand
      assert(bType != ScaleDotElemType::E2M1 && "NYI: rhs scale for fp4");
      auto newBEncoding = DotOperandEncodingAttr::get(ctx, 1, mmaEnc, bKWidth);
      b = createArg(rewriter, b, 1, bType, newBEncoding,
                    /*scale=*/std::nullopt, /*scaleEncoding=*/std::nullopt);
      newDot = rewriter.create<DotOp>(scaledDotOp.getLoc(), newRetType, a, b,
                                      newAcc);
    } else {
      assert(versionMajor == 3);
      // At the time of this writing, this is always true
      auto allowTranspose = b.getType().getElementType().isBF16();
      auto bShmem = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose);
      newDot = rewriter.create<triton::nvidia_gpu::WarpGroupDotOp>(
          scaledDotOp.getLoc(), newRetType, a, bShmem, newAcc, nullptr);
    }

    // convert dot instruction
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(scaledDotOp, oldRetType,
                                                 newDot->getResult(0));
    return success();
  }

private:
  TypedValue<RankedTensorType>
  createArg(mlir::PatternRewriter &rewriter, TypedValue<RankedTensorType> v,
            int idx, ScaleDotElemType type, std::optional<Attribute> vEncoding,
            std::optional<TypedValue<RankedTensorType>> opt_scale,
            std::optional<Attribute> scaleEncoding) const {
    auto ctx = rewriter.getContext();
    // Create a new tensor with a given encoding or remove the encoding
    auto maybeWithEncoding =
        [](RankedTensorType ty,
           std::optional<Attribute> enc) -> RankedTensorType {
      if (enc.has_value()) {
        return RankedTensorType::get(ty.getShape(), ty.getElementType(), *enc);
      } else {
        return RankedTensorType::get(ty.getShape(), ty.getElementType());
      }
    };

    auto newVType = maybeWithEncoding(v.getType(), vEncoding);
    TypedValue<RankedTensorType> ret =
        rewriter.create<ConvertLayoutOp>(v.getLoc(), newVType, v);

    // convert to bf16
    if (type != ScaleDotElemType::E2M1 && type != ScaleDotElemType::BF16) {
      assert(type == ScaleDotElemType::E5M2 || type == ScaleDotElemType::E4M3);
      auto vTypeBf16 = RankedTensorType::get(
          newVType.getShape(), rewriter.getBF16Type(), newVType.getEncoding());
      ret = cast<TypedValue<RankedTensorType>>(
          rewriter.create<FpToFpOp>(v.getLoc(), vTypeBf16, ret).getResult());
    }
    if (opt_scale.has_value()) {
      auto scale = *opt_scale;
      assert(idx == 0 && "NYI: rhs scale");
      auto newScaleDotElemType =
          maybeWithEncoding(scale.getType(), scaleEncoding);
      scale = rewriter.create<ConvertLayoutOp>(scale.getLoc(),
                                               newScaleDotElemType, scale);
      ret = rewriter.create<triton::gpu::UpcastMXFPOp>(v.getLoc(), ret, scale,
                                                       type);
    }
    return ret;
  }

  NvidiaMmaEncodingAttr getMMAEncoding(mlir::PatternRewriter &rewriter,
                                       DotScaledOp scaledDotOp) const {
    auto ctx = rewriter.getContext();
    auto a = scaledDotOp.getLhs();
    auto b = scaledDotOp.getRhs();
    auto scale = scaledDotOp.getLhsScale();
    auto aType = scaledDotOp.getLhsType();
    auto bType = scaledDotOp.getRhsType();

    // create a DotOp to be passed in to getMMAVersionSafe
    // We don't pass encodings as we just want to get the type and shape
    // to create a DotOp to be passed in to getMMAVersionSafe. We use the
    // rewriter to avoid duplicating createArg, but these ops are not going to
    // end up in the graph
    RankedTensorType aTType =
        createArg(rewriter, a, 0, aType, /*vEncoding=*/std::nullopt, scale,
                  /*scaleEncoding=*/std::nullopt)
            .getType();
    auto aTypeNoEnc =
        RankedTensorType::get(aTType.getShape(), aTType.getElementType());
    a = rewriter.create<ConvertLayoutOp>(scaledDotOp.getLoc(), aTypeNoEnc, a);

    RankedTensorType bTType =
        createArg(rewriter, b, 1, bType, /*vEncoding=*/std::nullopt,
                  /*scale=*/std::nullopt, /*scaleEncoding=*/std::nullopt)
            .getType();
    auto bTypeNoEnc =
        RankedTensorType::get(bTType.getShape(), bTType.getElementType());
    b = rewriter.create<ConvertLayoutOp>(scaledDotOp.getLoc(), bTypeNoEnc, b);
    auto dotOp = rewriter.create<DotOp>(
        scaledDotOp.getLoc(), scaledDotOp.getType(), a, b, scaledDotOp.getC());

    int versionMajor = 2;
    // We just support bf16 for MMAv3 on the rhs
    if (bType == ScaleDotElemType::BF16) {
      versionMajor = getMMAVersionSafe(computeCapability, dotOp);
    }
    int versionMinor = computeCapability == 75 ? 1 : 0;

    RankedTensorType oldRetType = dotOp.getType();
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    auto mod = dotOp->getParentOfType<mlir::ModuleOp>();
    int numWarps = TritonGPUDialect::getNumWarps(mod);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());

    auto instrShape = mmaVersionToInstrShape(
        versionMajor, retShapePerCTA, dotOp.getA().getType().getElementType(),
        numWarps);

    auto warpsPerCTA = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                       numWarps, instrShape);
    return NvidiaMmaEncodingAttr::get(ctx, versionMajor, versionMinor,
                                      warpsPerCTA, CTALayout, instrShape);
  }
};

static void updateValueType(Value v, Attribute encoding,
                            ArrayRef<int64_t> shape) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  auto newType =
      RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  v.setType(newType);
}

static TransOp updateUsers(Value result, const SetVector<Operation *> &slice) {
  TransOp transOp;
  if (llvm::any_of(result.getUsers(),
                   [&](Operation *user) { return slice.count(user) == 0; })) {
    OpBuilder builder(result.getContext());
    builder.setInsertionPointAfterValue(result);
    transOp =
        builder.create<TransOp>(result.getLoc(), result, ArrayRef({1, 0}));
    result.replaceUsesWithIf(transOp.getResult(), [&](OpOperand &operand) {
      return operand.getOwner() != transOp.getOperation() &&
             slice.count(operand.getOwner()) == 0;
    });
  }
  return transOp;
}

// Sync the transpose in the IR, this is done to avoid generating convert layout
// when we have a transpose right after a dot as mma layout cannot be propagated
// through transpose op. Once we have layouts that can represent transposed MMA
// we can remove this transformation.
static void sinkTransposeOp(TransOp input) {
  SmallVector<TransOp> queue = {input};
  while (!queue.empty()) {
    TransOp transOp = queue.back();
    Value currentValue = transOp.getResult();
    queue.pop_back();
    mlir::ForwardSliceOptions options;
    options.filter = [](Operation *op) {
      if (op->hasTrait<OpTrait::Elementwise>() && op->getNumOperands() == 1)
        return true;
      if (isa<scf::YieldOp>(op))
        return isa<scf::ForOp>(op->getParentOp());
      if (isa<ConvertLayoutOp>(op))
        return true;
      return false;
    };
    SetVector<Operation *> slice;
    mlir::getForwardSlice(currentValue, &slice, options);
    for (Operation *op : slice) {
      if (op->hasTrait<OpTrait::Elementwise>()) {
        // Update users of transpose op.
        if (op->getOperand(0) == transOp.getResult())
          op->setOperand(0, transOp.getOperand());
        // Update the type of the result.
        for (Value result : op->getResults()) {
          auto srcType = cast<RankedTensorType>(op->getOperand(0).getType());
          updateValueType(result, srcType.getEncoding(), srcType.getShape());
          updateUsers(result, slice);
        }
        continue;
      }
      if (auto cvtOp = dyn_cast<ConvertLayoutOp>(op)) {
        // Update users of transpose op.
        if (op->getOperand(0) == transOp.getResult())
          op->setOperand(0, transOp.getOperand());
        auto resultEncoding = cvtOp.getType().getEncoding();
        auto newDstEncoding = inferSrcEncoding(transOp, resultEncoding);
        auto srcType = cast<RankedTensorType>(cvtOp.getOperand().getType());
        updateValueType(cvtOp.getResult(), *newDstEncoding, srcType.getShape());
        updateUsers(cvtOp.getResult(), slice);
        continue;
      }
      assert(isa<scf::YieldOp>(op));
      auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
      assert(forOp);
      for (OpOperand &operand : op->getOpOperands()) {
        Operation *def = operand.get().getDefiningOp();
        if (def && (slice.count(def)) || def == transOp.getOperation()) {
          if (def == transOp.getOperation())
            operand.set(transOp.getOperand());
          Type newType = operand.get().getType();
          forOp.getResult(operand.getOperandNumber()).setType(newType);
          TransOp retTrans =
              updateUsers(forOp.getResult(operand.getOperandNumber()), slice);
          // Recursively try to propagate the new transpose inserted.
          if (retTrans)
            queue.push_back(retTrans);
          forOp.getRegionIterArg(operand.getOperandNumber()).setType(newType);
          TransOp argTrans = updateUsers(
              forOp.getRegionIterArg(operand.getOperandNumber()), slice);
          if (argTrans)
            queue.push_back(argTrans);
          OpBuilder builder(forOp);
          OpOperand &init = forOp.getInitsMutable()[operand.getOperandNumber()];
          Value initTranspose = builder.create<TransOp>(
              forOp.getLoc(), init.get(), ArrayRef({1, 0}));
          init.set(initTranspose);
        }
      }
    }
  }
}

// Transpose scaled_dot ops that have a scale on lhs.
static Operation *transposeDotOp(DotScaledOp dotOp) {
  OpBuilder builder(dotOp);
  Value lhs = dotOp.getLhs();
  std::array<int, 2> transOrder = {1, 0};
  Value lhsTransposed = builder.create<TransOp>(lhs.getLoc(), lhs, transOrder);
  Value rhs = dotOp.getRhs();
  Value rhsTransposed = builder.create<TransOp>(rhs.getLoc(), rhs, transOrder);
  Value c = dotOp.getC();
  Value cTransposed = builder.create<TransOp>(c.getLoc(), c, transOrder);
  Value result = builder.create<DotScaledOp>(
      dotOp.getLoc(), cTransposed.getType(), rhsTransposed, lhsTransposed,
      cTransposed, dotOp.getRhsScale(), dotOp.getLhsScale(), dotOp.getRhsType(),
      dotOp.getLhsType());
  Operation *transposedResult =
      builder.create<TransOp>(result.getLoc(), result, transOrder);
  dotOp.replaceAllUsesWith(transposedResult);
  dotOp.erase();
  return transposedResult;
}

static void transposeDots(ModuleOp m) {
  // TODO: extend to regular dot when it is profitable. For instance when we may
  // want to use rhs from register for mmav3.
  SmallVector<DotScaledOp> toTranspose;
  m.walk([&](DotScaledOp dotOp) -> void {
    if (dotOp.getLhsScale() == nullptr && dotOp.getRhsScale() != nullptr)
      toTranspose.push_back(dotOp);
  });
  SmallVector<Operation *> transposes;
  for (DotScaledOp dotOp : toTranspose) {
    Operation *transpose = transposeDotOp(dotOp);
    transposes.push_back(transpose);
  }

  for (Operation *transpose : transposes) {
    sinkTransposeOp(cast<TransOp>(transpose));
  }
}

#define GEN_PASS_DEF_TRITONGPUACCELERATEMATMUL
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUAccelerateMatmulPass
    : public impl::TritonGPUAccelerateMatmulBase<
          TritonGPUAccelerateMatmulPass> {
public:
  using impl::TritonGPUAccelerateMatmulBase<
      TritonGPUAccelerateMatmulPass>::TritonGPUAccelerateMatmulBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    auto computeCapability = getNVIDIAComputeCapability(m);
    transposeDots(m);

    mlir::RewritePatternSet patterns(context);
    patterns.add<BlockedToMMA, DecomposeScaledBlocked>(context,
                                                       computeCapability);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    // Now that we have picked the mma type, decompose dot that are not natively
    // supported.
    decomposeMixedModeDotOp(m, computeCapability);
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
