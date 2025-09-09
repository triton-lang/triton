#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

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
  } else if (computeCapability < 110) {
    versionsSupported = {5, 2};
  } else if (computeCapability < 130) {
    versionsSupported = {2};
  } else {
    assert(false && "computeCapability not supported");
  }
  for (int baseVersion : versionsSupported) {
    if (supportMMA(op, baseVersion))
      return baseVersion;
    if (baseVersion == 3) {
      auto remark = op.emitRemark()
                    << "MMA version 3 acceleration not applied due to "
                       "unsupported shapes or data types.";
      remark.attachNote() << "Target compute capability (" << computeCapability
                          << ") supports MMA v3.";
    }

    if (baseVersion == 5) {
      auto remark = op.emitRemark()
                    << "MMA version 5 acceleration not applied due to "
                       "unsupported shapes or data types.";
      remark.attachNote() << "Target compute capability (" << computeCapability
                          << ") supports MMA v5.";
    }
  }
  return 0;
}

SmallVector<unsigned> warpsPerTileV2(DotOpInterface dotOp,
                                     const ArrayRef<int64_t> shape,
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
    if (isa<DotOpInterface>(op) && (op != dotOp)) {
      auto resTy = cast<RankedTensorType>(op->getResult(0).getType());
      if (resTy.getRank() != rank) {
        continue;
      }
      if (auto mmaEncoding =
              dyn_cast<NvidiaMmaEncodingAttr>(resTy.getEncoding())) {
        return to_vector(mmaEncoding.getWarpsPerCTA());
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
      // We allocate the remaining warps to the left (arbitrary choice)
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
warpsPerTileV3(DotOpInterface dotOp, const ArrayRef<int64_t> shape,
               int numWarps, const SmallVector<unsigned, 3> &instrShape) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getD(), &slices);
  // Contains a chained dot. We prefer to assign warps to one axis
  // to facilitate use cases like flash attention, allowing reductions within
  // the same warp.
  if (llvm::find_if(slices, [](Operation *op) {
        return isa<mlir::triton::DotOpInterface>(op);
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
static Value
getSharedMemoryMMAOperand(Value v, mlir::PatternRewriter &rewriter, int opIdx,
                          bool allowTranspose, bool isMMAv5Fp4Padded = false,
                          bool forceTranspose = false,
                          Operation *op = nullptr /*only for diagnostic*/) {
  OpBuilder::InsertionGuard g(rewriter);
  Value arg = v;
  if (auto cvtOp = v.getDefiningOp<ConvertLayoutOp>())
    arg = cvtOp.getSrc();
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto order = getOrderForMemory(argType);

  // If the MMA op doesn't support transpose pick the layout expected by the MMA
  // op.
  llvm::SmallVector<unsigned> newOrder = order;
  if (!allowTranspose) {
    if (opIdx == 1) {
      newOrder = {0, 1};
    } else {
      newOrder = {1, 0};
    }
    if (forceTranspose)
      std::swap(newOrder[0], newOrder[1]);
  }

  if (newOrder != order && op) {
    op->emitWarning("Warning: Forcing a different order [")
        << newOrder[0] << ", " << newOrder[1]
        << "] on SMEM than the register order for the operand " << opIdx
        << ". Registers will be transposed before SMEM store and the pipelined "
           "load for this operand will be disabled, so poor performance is "
           "expected. Recommendation: consider transposing the operand in "
           "global "
           "memory to remove the need to transpose the tensor in registers.";
  }

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CTALayout = getCTALayout(argType.getEncoding());
  auto newLayout = NVMMASharedEncodingAttr::get(
      argType.getContext(), argType.getShape(), newOrder, CTALayout,
      argType.getElementType(), isMMAv5Fp4Padded);
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return rewriter.create<LocalAllocOp>(arg.getLoc(), newType, arg);
}

static LocalAllocOp
getSharedMemoryScale(Value arg, mlir::PatternRewriter &rewriter, Location loc) {
  OpBuilder::InsertionGuard g(rewriter);
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto newOrder = getOrderForMemory(argType);

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CTALayout = getCTALayout(argType.getEncoding());
  // No swizzling for scale for now
  auto newLayout = NVMMASharedEncodingAttr::get(
      argType.getContext(), /*swizzlingByteWidth=*/0,
      /*transposed=*/false,
      /*elementBitWidth=*/argType.getElementType().getIntOrFloatBitWidth(),
      /*fp4Padded=*/false, CTALayout);
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return rewriter.create<LocalAllocOp>(loc, newType, arg);
}

SmallVector<unsigned, 3>
getWarpsPerTile(DotOpInterface dotOp, const ArrayRef<int64_t> shape,
                int version, int numWarps,
                const SmallVector<unsigned, 3> &instrShape) {
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

static bool bwdFilter(Operation *op) {
  return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
         isView(op) ||
         isa<Fp4ToFpOp, LoadOp, DescriptorLoadOp, BroadcastOp, ConvertLayoutOp>(
             op);
}

// Finds the bitwidth with which the value x is loaded
static int computeOrigBitWidth(Value x) {
  SetVector<Operation *> slice;
  mlir::BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.filter = bwdFilter;
  (void)getBackwardSlice(x, &slice, opt);

  // TODO: This heuristic may be a bit too coarse and may need improving
  // If the chain contains a fp4 to fp16/bf16 conversion, then the original
  // bitwidth is 4.
  if (llvm::any_of(slice, [](Operation *op) { return isa<Fp4ToFpOp>(op); }))
    return 4;

  int origBitWidth = getElementTypeOrSelf(x).getIntOrFloatBitWidth();
  for (auto op : slice) {
    if (isa<LoadOp, DescriptorLoadOp>(op)) {
      if (auto tensorTy =
              dyn_cast<RankedTensorType>(op->getResultTypes().front())) {
        origBitWidth =
            std::min<int>(origBitWidth, tensorTy.getElementTypeBitWidth());
      }
    }
  }

  // If JoinOp occurred at least once, in backward layout propagation,
  // the kWidth will be split in half as we pass through the JoinOp.
  // Hence we divide origBitWidth by 2 here to compensate for that and
  // improve our load width.
  // This won't be optimal if there is a tree of multiple JoinOps, which
  // would require counting the max number of JoinOp's along any path.
  //
  // In the future we might want to do something like trying a large kWidth,
  // run layout backpropagation and see what's the contiguity that you
  // get at the loads that feed into it.
  if (llvm::any_of(slice, [](Operation *op) { return isa<JoinOp>(op); }))
    origBitWidth /= 2;

  return origBitWidth;
}

namespace {

// Common MMA encoding creation
struct MMAEncodingResult {
  NvidiaMmaEncodingAttr mmaEnc;
  RankedTensorType newRetType;
  Value newAcc;
  int versionMajor;
  int versionMinor;
};

// Unified implementation for DotOpInterface
static MMAEncodingResult createMMAEncodingForDot(DotOpInterface dotOp,
                                                 PatternRewriter &rewriter,
                                                 int computeCapability,
                                                 int versionMajor) {
  auto oldRetType = cast<RankedTensorType>(dotOp.getD().getType());
  auto oldAType = cast<RankedTensorType>(dotOp.getA().getType());

  int numWarps = lookupNumWarps(dotOp);

  int versionMinor = computeCapability == 75 ? 1 : 0;
  // Only MMAv2 and MMAv3 rely on computing instrShape/warpsPerTile here.
  if (!(versionMajor == 2 || versionMajor == 3)) {
    return {nullptr, RankedTensorType(), Value(), versionMajor, versionMinor};
  }

  auto CTALayout = getCTALayout(oldRetType.getEncoding());
  auto retShapePerCTA = getShapePerCTA(oldRetType);
  auto instrShape = mmaVersionToInstrShape(versionMajor, retShapePerCTA,
                                           oldAType.getElementType(), numWarps);
  auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                      numWarps, instrShape);

  auto mmaEnc = NvidiaMmaEncodingAttr::get(oldRetType.getContext(),
                                           versionMajor, versionMinor,
                                           warpsPerTile, CTALayout, instrShape);
  auto newRetType = oldRetType.cloneWithEncoding(mmaEnc);

  auto oldAcc = dotOp->getOperand(2);
  auto newAcc =
      rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

  return {mmaEnc, newRetType, newAcc, versionMajor, versionMinor};
}

// Common operand conversion
static Value convertDotOperandForMMA(Value v, int opIdx, int bitwidth,
                                     RankedTensorType newRetType,
                                     PatternRewriter &rewriter) {
  auto minType = bitwidth > 0 ? rewriter.getIntegerType(bitwidth) : v.getType();
  auto vType = cast<RankedTensorType>(v.getType());
  auto newVEncoding = DotOperandEncodingAttr::get(
      v.getContext(), opIdx, newRetType.getEncoding(), minType);
  auto newVType = vType.cloneWithEncoding(newVEncoding);
  return rewriter.create<ConvertLayoutOp>(v.getLoc(), newVType, v);
}

} // namespace

class BlockedToMMA : public mlir::OpRewritePattern<DotOp> {
  int computeCapability;
  mutable llvm::DenseMap<Operation *, unsigned> dotOpInstNs;

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability, int benefit)
      : OpRewritePattern<DotOp>(context, benefit),
        computeCapability(computeCapability) {}

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
    auto retType = dotOp.getType();
    if (!retType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(retType.getEncoding()))
      return failure();

    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());
    auto oldRetType = cast<RankedTensorType>(dotOp.getType());

    // Enable F64 MMA only on SM80/SM90 with high performance F64 tensorcore.
    // Otherwise, fallback to F64 FMA for better performance.
    if ((oldAType.getElementType().isF64() ||
         oldBType.getElementType().isF64() ||
         oldRetType.getElementType().isF64()) &&
        !(computeCapability == 80 || computeCapability == 90)) {
      return failure();
    }

    auto mmaVersion = getMMAVersionSafe(computeCapability, dotOp);
    auto mmaResult =
        createMMAEncodingForDot(dotOp, rewriter, computeCapability, mmaVersion);
    if (!(mmaResult.versionMajor >= 1 && mmaResult.versionMajor <= 3))
      return failure();

    Operation *newDot = nullptr;
    bool aFromLoad = comesFromLoadOrBlockArg(a);
    bool bFromLoad = comesFromLoadOrBlockArg(b);

    if (mmaResult.versionMajor == 3) {
      auto eltType = cast<RankedTensorType>(a.getType()).getElementType();
      bool allowTranspose = eltType.isF16() || eltType.isBF16();
      if (!aFromLoad) {
        int bitwidth = getElementTypeOrSelf(a).getIntOrFloatBitWidth();
        a = convertDotOperandForMMA(a, 0, bitwidth, mmaResult.newRetType,
                                    rewriter);
      } else {
        a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose,
                                      /*isMMAv5Fp4Padded=*/false,
                                      /*forceTranspose=*/false, dotOp);
      }
      b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose,
                                    /*isMMAv5Fp4Padded=*/false,
                                    /*forceTranspose=*/false, dotOp);

      newDot = rewriter.create<triton::nvidia_gpu::WarpGroupDotOp>(
          dotOp.getLoc(), mmaResult.newRetType, a, b, mmaResult.newAcc, nullptr,
          dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc(), false);
    } else {
      int minBitwidth =
          std::min(computeOrigBitWidth(a), computeOrigBitWidth(b));
      a = convertDotOperandForMMA(a, 0, minBitwidth, mmaResult.newRetType,
                                  rewriter);
      b = convertDotOperandForMMA(b, 1, minBitwidth, mmaResult.newRetType,
                                  rewriter);
      newDot = rewriter.create<DotOp>(
          dotOp.getLoc(), mmaResult.newRetType, a, b, mmaResult.newAcc,
          dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
    }

    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, dotOp.getType(),
                                                 newDot->getResult(0));
    return success();
  }
};

// Pick the layout to match MXFP scales layout in register so that it can be
// copied directly using tmem st.
static Attribute getTmemScales(RankedTensorType type, unsigned numWarps) {
  return triton::gpu::LinearEncodingAttr::get(
      type.getContext(), getScaleTMEMStoreLinearLayout(type, numWarps));
}

static bool canUseTwoCTAs(triton::DotOp dotOp) {
  RankedTensorType retType = dotOp.getType();
  auto retShapePerCTA = getShapePerCTA(retType);
  // TODO: we could support 2 CTAs matmul with numCTAs > 2.
  SmallVector<unsigned> splitNum = getCTASplitNum(retType.getEncoding());
  if (splitNum.size() != 2 || splitNum[0] != 2 || splitNum[1] != 1)
    return false;
  int m = retShapePerCTA[0];
  int n = retShapePerCTA[1];
  // minimum size supported by 2CTAs mmav5.
  if (m < 64 || n < 32)
    return false;
  Value b = dotOp.getB();
  // Skip convert layouts.
  while (auto cvtOp = b.getDefiningOp<ConvertLayoutOp>())
    b = cvtOp.getSrc();
  return llvm::isa_and_nonnull<triton::LoadOp, triton::DescriptorLoadOp,
                               triton::DescriptorGatherOp>(b.getDefiningOp());
}

static DistributedEncodingTrait
replaceCTALayout(DistributedEncodingTrait layout,
                 const triton::gpu::CTALayoutAttr &newCTALayout) {
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(layout)) {
    return BlockedEncodingAttr::get(
        layout.getContext(), blockedLayout.getSizePerThread(),
        blockedLayout.getThreadsPerWarp(), blockedLayout.getWarpsPerCTA(),
        blockedLayout.getOrder(), newCTALayout);
  } else if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    return SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        replaceCTALayout(sliceLayout.getParent(), newCTALayout));
  } else {
    llvm::report_fatal_error("not implemented");
    return layout;
  }
}

static Value splitBOperand(Value b, mlir::PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  MLIRContext *ctx = b.getContext();
  while (auto cvtOp = b.getDefiningOp<ConvertLayoutOp>())
    b = cvtOp.getSrc();
  auto loadOp = b.getDefiningOp();
  assert((isa<triton::LoadOp, triton::DescriptorLoadOp,
              triton::DescriptorGatherOp>(loadOp)) &&
         "expected LoadOp");
  RankedTensorType bType = cast<RankedTensorType>(b.getType());
  auto currentLayout = cast<DistributedEncodingTrait>(bType.getEncoding());
  auto newCTALayout =
      CTALayoutAttr::get(ctx, {1, 2}, {1, 2}, getCTAOrder(currentLayout));
  Attribute newLayout = replaceCTALayout(currentLayout, newCTALayout);
  rewriter.setInsertionPoint(loadOp);
  for (OpOperand &operand : loadOp->getOpOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!tensorType)
      continue;
    Value newOperand = rewriter.create<ConvertLayoutOp>(
        operand.get().getLoc(), tensorType.cloneWithEncoding(newLayout),
        operand.get());
    loadOp->setOperand(operand.getOperandNumber(), newOperand);
  }
  loadOp->getResult(0).setType(bType.cloneWithEncoding(newLayout));
  Value newB = loadOp->getResult(0);
  rewriter.setInsertionPointAfter(loadOp);
  auto cvt = rewriter.create<ConvertLayoutOp>(b.getLoc(), bType, newB);
  rewriter.replaceAllUsesExcept(newB, cvt.getResult(), cvt);
  return newB;
}

class BlockedToMMAv5 : public mlir::OpRewritePattern<DotOp> {
  int computeCapability;

public:
  BlockedToMMAv5(mlir::MLIRContext *context, int computeCapability, int benefit)
      : OpRewritePattern<DotOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = lookupNumWarps(dotOp);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());

    int versionMajor = getMMAVersionSafe(computeCapability, dotOp);
    if (versionMajor != 5)
      return failure();
    Location loc = dotOp.getLoc();
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    if (std::min(computeOrigBitWidth(a), computeOrigBitWidth(b)) >= 32 &&
        dotOp.getInputPrecision() != InputPrecision::TF32)
      return failure();
    auto oldAType = dotOp.getA().getType();
    auto oldBType = dotOp.getB().getType();
    bool useTwoCTAs = canUseTwoCTAs(dotOp);
    if (useTwoCTAs) {
      b = splitBOperand(b, rewriter);
    }
    // TF32 transpose is only supported with 128 swizzle mode with 32B
    // atomicity. As we currently don't support this layout we disallow
    // transpose for TF32 inputs.
    bool allowTranspose = !dotOp.getA().getType().getElementType().isF32();
    a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose);
    b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose);
    MLIRContext *context = dotOp->getContext();
    auto instrShape = mmaVersionToInstrShape(
        versionMajor, retShapePerCTA, oldAType.getElementType(), numWarps);
    ArrayRef<unsigned> CTASplitNum = CTALayout.getCTASplitNum();
    Attribute accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
        context, instrShape[0], instrShape[1], /*unpacked=*/true,
        CTASplitNum[0], CTASplitNum[1]);
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    Type accMemDescType = triton::gpu::MemDescType::get(
        oldRetType.getShape(), oldRetType.getElementType(), accEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/true);
    Attribute newDistributedEncoding = nvidia_gpu::getTmemCompatibleLayout(
        instrShape[0], instrShape[1], oldRetType, numWarps);
    auto newAccType = oldRetType.cloneWithEncoding(newDistributedEncoding);
    Value cvtAcc =
        rewriter.create<ConvertLayoutOp>(loc, newAccType, dotOp.getOperand(2));
    auto tokType = rewriter.getType<AsyncTokenType>();
    auto acc = rewriter.create<triton::nvidia_gpu::TMEMAllocOp>(
        loc, accMemDescType, tokType, cvtAcc);
    auto vTrue = rewriter.create<arith::ConstantIntOp>(dotOp.getLoc(), 1, 1);
    auto mma = rewriter.create<triton::nvidia_gpu::TCGen5MMAOp>(
        loc, tokType, a, b, acc, acc.getToken(), /*useD=*/vTrue,
        /*pred=*/vTrue);
    mma.setTwoCtas(useTwoCTAs);

    auto ld = rewriter.create<triton::nvidia_gpu::TMEMLoadOp>(
        loc, newAccType, tokType, acc, /*dep=*/mma.getToken());
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType, ld);
    return success();
  }
};

Value addSmemStageToScaleLoad(Value scale, mlir::PatternRewriter &rewriter) {
  /*
    Rewrite load(scale) -> local_load(local_alloc(load(scale))).
    This function does not add anything to the final IR when num_stages > 1,
    but it makes it easy to apply TMEM copy rewriting later.

    Since scales are stored in TMEM for MMAv5 scaled dot, loading of scales do
    not needs to be put into SMEM. But in practice, the software pipeliner puts
    loading of scales into multi-buffered SMEM. At that point, the SMEM
    allocation created here is eliminated.
   */
  OpBuilder::InsertionGuard g(rewriter);
  auto op = scale.getDefiningOp();
  Operation *loadConsumer = nullptr;

  if (!op)
    return scale;

  while (!isa<LoadOp, DescriptorLoadOp>(op)) {
    if (auto reshape = dyn_cast<ReshapeOp>(op)) {
      op = reshape.getSrc().getDefiningOp();
      loadConsumer = reshape;
    } else if (auto trans = dyn_cast<TransOp>(op)) {
      op = trans.getSrc().getDefiningOp();
      loadConsumer = trans;
    } else if (auto cvt = dyn_cast<ConvertLayoutOp>(op)) {
      op = cvt.getSrc().getDefiningOp();
      loadConsumer = cvt;
    } else {
      // Unrecognized pattern, bail out. In practice, this implies that MMA
      // pipelining will not apply to the scaled dot op, since scales will not
      // be in passed through SMEM to tc_gen5_mma_scaled.
      return scale;
    }
  }

  auto scaleAfterLoad = op->getResult(0);
  auto scaleSmemAlloc =
      getSharedMemoryScale(scaleAfterLoad, rewriter, op->getLoc());

  rewriter.setInsertionPointAfterValue(scaleSmemAlloc);
  auto localLoad = rewriter.create<LocalLoadOp>(
      op->getLoc(), scaleAfterLoad.getType(), scaleSmemAlloc);

  rewriter.replaceAllUsesExcept(scaleAfterLoad, localLoad.getResult(),
                                scaleSmemAlloc);

  if (loadConsumer) {
    return scale;
  } else {
    return localLoad;
  }
}

class ScaledBlockedToMMA : public mlir::OpRewritePattern<triton::DotScaledOp> {
  int computeCapability;

public:
  ScaledBlockedToMMA(mlir::MLIRContext *context, int computeCapability,
                     int benefit)
      : mlir::OpRewritePattern<triton::DotScaledOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotScaledOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability != 120)
      return failure();

    auto numCTAs = lookupNumCTAs(rewriter);
    if (numCTAs != 1) {
      return failure();
    }

    // TODO: support mxfp4 variants.
    if (!((dotOp.getAElemType() == ScaleDotElemType::E5M2 ||
           dotOp.getAElemType() == ScaleDotElemType::E4M3) &&
          (dotOp.getBElemType() == ScaleDotElemType::E5M2 ||
           dotOp.getBElemType() == ScaleDotElemType::E4M3))) {
      return rewriter.notifyMatchFailure(dotOp, "only E5M2/E4M3 is supported");
    }

    // Skip if any scale is missing. This pattern requires both scales.
    if (!dotOp.getAScale() || !dotOp.getBScale())
      return failure();

    auto aScaleType = dotOp.getAScale().getType();
    auto bScaleType = dotOp.getBScale().getType();

    if (mlir::isa<LinearEncodingAttr>(aScaleType.getEncoding()) ||
        mlir::isa<LinearEncodingAttr>(bScaleType.getEncoding())) {
      return failure();
    }

    // Common MMA encoding creation
    auto mmaResult =
        createMMAEncodingForDot(dotOp, rewriter, computeCapability, 2);

    // Operand processing
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    Operation *newDot = nullptr;

    // ScaledBlockedToMMA logic
    int bitwidthA = oldAType.getElementType().getIntOrFloatBitWidth();
    int bitwidthB = oldBType.getElementType().getIntOrFloatBitWidth();
    int minBitwidth = std::min(bitwidthA, bitwidthB);

    Value newA = convertDotOperandForMMA(a, 0, minBitwidth,
                                         mmaResult.newRetType, rewriter);
    Value newB = convertDotOperandForMMA(b, 1, minBitwidth,
                                         mmaResult.newRetType, rewriter);

    // Compute tiles per warp for each operand
    auto computeTilePerWarp = [&](Value operand, int operandIdx) -> unsigned {
      auto operandTy = cast<RankedTensorType>(operand.getType());
      auto dotEncoding = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
          operandTy.getEncoding());
      if (!dotEncoding)
        return 1;

      const int bitWidth = operandTy.getElementType().getIntOrFloatBitWidth();
      const int kWidth = dotEncoding.getKWidth();
      auto rep = mmaResult.mmaEnc.getRepForOperand(
          triton::gpu::getShapePerCTA(operandTy), bitWidth, kWidth,
          dotEncoding.getOpIdx());

      // repA = [repM, repK], repB = [repK, repN]
      // For operand A (idx 0): return rep[1] (repK)
      // For operand B (idx 1): return rep[2] (repN)
      if (operandIdx == 0) {
        return rep.size() >= 2 ? rep[1] : 1;
      } else {
        return rep.size() >= 3 ? rep[2] : 1;
      }
    };
    SmallVector<unsigned, 2> tilesPerWarp{computeTilePerWarp(newA, 0),
                                          computeTilePerWarp(newB, 1)};
    // Convert scales to Linear layout
    auto convertScale = [&](Value scale, int opIdx) -> Value {
      if (!scale)
        return Value();
      auto ty = cast<RankedTensorType>(scale.getType());
      SmallVector<int64_t> shape = llvm::to_vector(ty.getShape());
      MLIRContext *ctx = ty.getContext();
      const auto mmaWarps = mmaResult.mmaEnc.getWarpsPerCTA(); // [wM, wN]
      const auto instr = mmaResult.mmaEnc.getInstrShape(); // [instrM, instrN]
      const unsigned instrM = instr[0], instrN = instr[1];

      auto blocked = cast<triton::gpu::BlockedEncodingAttr>(ty.getEncoding());
      auto ll = triton::gpu::getSM120DotScaledScaleLayout(
          ctx, opIdx, shape, tilesPerWarp,
          /*warpsPerCTA=*/mmaWarps, instrM, instrN, blocked.getCTALayout());
      auto newEnc = triton::gpu::LinearEncodingAttr::get(ctx, ll);
      auto newTy = RankedTensorType::get(shape, ty.getElementType(), newEnc);
      return rewriter.create<ConvertLayoutOp>(scale.getLoc(), newTy, scale);
    };
    Value aScale = convertScale(dotOp.getAScale(), /*opIdx=*/0);
    Value bScale = convertScale(dotOp.getBScale(), /*opIdx=*/1);

    newDot = rewriter.create<triton::DotScaledOp>(
        dotOp.getLoc(), mmaResult.newRetType, newA, newB, mmaResult.newAcc,
        aScale, bScale, dotOp.getAElemType(), dotOp.getBElemType(),
        dotOp.getFastMath(), dotOp.getLhsKPack(), dotOp.getRhsKPack());
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, dotOp.getType(),
                                                 newDot->getResult(0));
    return success();
  }
};

class ScaledBlockedToMMAv5
    : public mlir::OpRewritePattern<triton::DotScaledOp> {
  int computeCapability;

public:
  ScaledBlockedToMMAv5(mlir::MLIRContext *context, int computeCapability,
                       int benefit)
      : mlir::OpRewritePattern<triton::DotScaledOp>(context, benefit),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotScaledOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    if (dotOp.getAScale() == nullptr || dotOp.getBScale() == nullptr) {
      return failure();
    }

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = lookupNumWarps(dotOp);
    auto CTALayout = getCTALayout(oldRetType.getEncoding());
    if ((computeCapability) / 10 != 10)
      return failure();
    if (numWarps != 4 && numWarps != 8)
      return failure();
    if (retShapePerCTA[0] < 128 || retShapePerCTA[1] < 8)
      return failure();
    Location loc = dotOp.getLoc();
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType();
    auto oldBType = b.getType();

    bool IsAMixedPrecFp4 = false;
    bool IsBMixedPrecFp4 = false;
    bool isAFP4 = dotOp.getAElemType() == ScaleDotElemType::E2M1;
    bool isBFP4 = dotOp.getBElemType() == ScaleDotElemType::E2M1;

    if (dotOp.getAElemType() != dotOp.getBElemType()) {
      if (isAFP4)
        IsAMixedPrecFp4 = true;
      else if (isBFP4)
        IsBMixedPrecFp4 = true;
    }
    // If we use txgen05.mma.kind.mxf864 we need to padd the fp4 operands:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-smem
    bool isMMAv5Fp4PaddedLhs = IsAMixedPrecFp4 || !dotOp.getLhsKPack();
    bool isMMAv5Fp4PaddedRhs = IsBMixedPrecFp4 || !dotOp.getRhsKPack();
    // For mixed-precision fp4 operands, set allowTranspose = false, to force
    // the packed axis, K, to be contiguous in SMEM
    a = getSharedMemoryMMAOperand(a, rewriter, 0,
                                  /*allowTranspose=*/!isAFP4,
                                  /*isMMAv5Fp4Padded=*/isMMAv5Fp4PaddedLhs,
                                  /*forceTranspose=*/!dotOp.getLhsKPack(),
                                  dotOp);
    b = getSharedMemoryMMAOperand(b, rewriter, 1,
                                  /*allowTranspose=*/!isBFP4,
                                  /*isMMAv5Fp4Padded=*/isMMAv5Fp4PaddedRhs,
                                  /*forceTranspose=*/!dotOp.getRhsKPack(),
                                  dotOp);

    MLIRContext *context = dotOp->getContext();
    unsigned m = 128;
    unsigned n = retShapePerCTA[1] >= 256 ? 256 : retShapePerCTA[1];

    ArrayRef<unsigned> CTASplitNum = CTALayout.getCTASplitNum();
    Attribute accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
        context, m, n, /*unpacked=*/true, CTASplitNum[0], CTASplitNum[1]);
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    Type accMemDescType = triton::gpu::MemDescType::get(
        oldRetType.getShape(), oldRetType.getElementType(), accEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/true);
    Attribute newDistributedEncoding =
        nvidia_gpu::getTmemCompatibleLayout(m, n, oldRetType, numWarps);
    auto newAccType = oldRetType.cloneWithEncoding(newDistributedEncoding);
    Value cvtAcc =
        rewriter.create<ConvertLayoutOp>(loc, newAccType, dotOp.getOperand(2));
    auto tokType = rewriter.getType<AsyncTokenType>();
    auto acc = rewriter.create<triton::nvidia_gpu::TMEMAllocOp>(
        loc, accMemDescType, tokType, cvtAcc);

    RankedTensorType oldScaleAType = dotOp.getAScale().getType();
    RankedTensorType oldScaleBType = dotOp.getBScale().getType();

    Attribute scaleEncoding =
        triton::nvidia_gpu::TensorMemoryScalesEncodingAttr::get(
            context, CTASplitNum[0], CTASplitNum[1]);
    Type scaleAType = triton::gpu::MemDescType::get(
        oldScaleAType.getShape(), oldScaleAType.getElementType(), scaleEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/false);
    Type scaleBType = triton::gpu::MemDescType::get(
        oldScaleBType.getShape(), oldScaleBType.getElementType(), scaleEncoding,
        tensorMemorySpace,
        /*mutableMemory=*/false);
    Attribute scaleALayout = getTmemScales(oldScaleAType, numWarps);
    Attribute scaleBLayout = getTmemScales(oldScaleBType, numWarps);
    RankedTensorType newScaleAType =
        oldScaleAType.cloneWithEncoding(scaleALayout);
    RankedTensorType newScaleBType =
        oldScaleBType.cloneWithEncoding(scaleBLayout);

    auto lhsScale = addSmemStageToScaleLoad(dotOp.getAScale(), rewriter);
    auto rhsScale = addSmemStageToScaleLoad(dotOp.getBScale(), rewriter);

    Value newScaleA =
        rewriter.create<ConvertLayoutOp>(loc, newScaleAType, lhsScale);
    Value newScaleB =
        rewriter.create<ConvertLayoutOp>(loc, newScaleBType, rhsScale);

    // We don't need to track memory dependencies for the scale operands since
    // they are not pipelined.
    auto scaleA = rewriter.create<triton::nvidia_gpu::TMEMAllocOp>(
        loc, scaleAType, /*token=*/Type(), newScaleA);
    auto scaleB = rewriter.create<triton::nvidia_gpu::TMEMAllocOp>(
        loc, scaleBType, /*token=*/Type(), newScaleB);

    auto vTrue = rewriter.create<arith::ConstantIntOp>(dotOp.getLoc(), 1, 1);
    auto mmaOp = rewriter.create<triton::nvidia_gpu::TCGen5MMAScaledOp>(
        loc, tokType, a, b, acc.getResult(), acc.getToken(), scaleA.getResult(),
        scaleB.getResult(), dotOp.getAElemType(), dotOp.getBElemType(),
        /*useD=*/vTrue, /*pred=*/vTrue);

    auto ld = rewriter.create<triton::nvidia_gpu::TMEMLoadOp>(
        loc, newAccType, tokType, acc, mmaOp.getToken());
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType, ld);
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  Type operandElType =
      cast<RankedTensorType>(operand.getType()).getElementType();
  if (type::isFloat8(operandElType)) {
    return builder.create<FpToFpOp>(loc, tensorPromotedType, operand);
  }
  return builder.create<arith::ExtFOp>(loc, tensorPromotedType, operand);
}

static bool mmav2SupportsFp8Operands(int computeCapability) {
  // promote operands for sm < 89 since fp8 mma is not natively supported
  // although PTX instructions for mma v2 w/ fp8 operands exist for sm90 and
  // sm100, they are emulated as fp16 upcasts + fp16 HMMA in SASS. sm120 has
  // hardware support for fp8 operands w/ mmav2.
  return computeCapability == 89 || computeCapability == 120;
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
      bool isNativeFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType>(AElType);
      // promote to f16 unless there's hardware support for fp8 operands
      if (!isNativeFP8 ||
          (isNativeFP8 && (mmav2SupportsFp8Operands(computeCapability) ||
                           mmaLayout.isHopper())))
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

// Transpose scaled_dot ops that have a scale on lhs.
static void transposeDotOp(DotScaledOp dotOp) {
  OpBuilder builder(dotOp);
  Value lhs = dotOp.getA();
  std::array<int, 2> transOrder = {1, 0};
  Value lhsTransposed = builder.create<TransOp>(lhs.getLoc(), lhs, transOrder);
  Value rhs = dotOp.getB();
  Value rhsTransposed = builder.create<TransOp>(rhs.getLoc(), rhs, transOrder);
  Value c = dotOp.getC();
  Value cTransposed = builder.create<TransOp>(c.getLoc(), c, transOrder);
  Value result = builder.create<DotScaledOp>(
      dotOp.getLoc(), cTransposed.getType(), rhsTransposed, lhsTransposed,
      cTransposed, dotOp.getBScale(), dotOp.getAScale(), dotOp.getBElemType(),
      dotOp.getAElemType(), dotOp.getFastMath());
  Operation *transposedResult =
      builder.create<TransOp>(result.getLoc(), result, transOrder);
  dotOp.replaceAllUsesWith(transposedResult);
  dotOp.erase();
}

static void transposeDots(ModuleOp m) {
  // TODO: extend to regular dot when it is profitable. For instance when we may
  // want to use rhs from register for mmav3.
  SmallVector<DotScaledOp> toTranspose;
  m.walk([&](DotScaledOp dotOp) -> void {
    if (dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr)
      toTranspose.push_back(dotOp);
  });
  for (DotScaledOp dotOp : toTranspose) {
    transposeDotOp(dotOp);
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
    // We could do this generically if we manage to improve the heuristics
    // reverted in these two PRs https://github.com/triton-lang/triton/pull/5834
    // https://github.com/triton-lang/triton/pull/5837
    transposeDots(m);

    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;
    constexpr int benefitMMAv5 = 10;
    constexpr int benefitSM120 = 10;

    patterns.add<BlockedToMMA>(context, computeCapability, benefitDefault);
    patterns.add<ScaledBlockedToMMA>(context, computeCapability, benefitSM120);
    populateDecomposeScaledBlockedPatterns(patterns, benefitDefault);
    patterns.add<BlockedToMMAv5, ScaledBlockedToMMAv5>(
        context, computeCapability, benefitMMAv5);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
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
