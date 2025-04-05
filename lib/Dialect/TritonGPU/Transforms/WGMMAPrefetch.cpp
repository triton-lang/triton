//===----------------------------------------------------------------------===//
//
// This file implements the WGMMA (Warp Group Matrix Multiply-Accumulate)
// prefetch optimization pass. The pass optimizes matrix multiplication
// operations by prefetching operands into registers, specifically targeting
// NVIDIA's WGMMA instructions.
//
// Key optimizations:
// 1. Prefetches operands for warp group dot operations within loops
// 2. Focuses on Register-Shared Memory GEMM (RSGEMM) patterns
// 3. Handles elementwise operations interleaved with WGMMA
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "tritongpu-wgmma-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
using namespace mlir;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUWGMMAPREFETCH
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

/// Checks if the given dot operation is a supported Register-Shared Memory
/// GEMM. Currently supports operations where:
/// 1. LHS (A) is in registers
/// 2. RHS (B) is in shared memory
///
/// @param dotOp The dot operation to check
/// @return success if the operation is supported, failure otherwise
LogicalResult isSupportedRSGEMM(ttng::WarpGroupDotOp dotOp) {
  Value operandA = dotOp.getA();
  Value operandB = dotOp.getB();

  // operandB is in shared memoryassert
  auto encA = dyn_cast<TensorOrMemDesc>(operandA.getType()).getEncoding();
  auto encB = dyn_cast<TensorOrMemDesc>(operandB.getType()).getEncoding();

  if (isa<DotOperandEncodingAttr>(encA) && isa<NVMMASharedEncodingAttr>(encB)) {
    return success();
  }

  return failure();
}

/// Main class implementing the WGMMA prefetch optimization
class WGMMAPrefetcher {
public:
  WGMMAPrefetcher() = delete;
  explicit WGMMAPrefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  /// Initialize the prefetcher by analyzing the loop and collecting dot
  /// operations
  LogicalResult initialize();

  /// Create a new optimized for loop with prefetching
  scf::ForOp createNewForOp();

private:
  /// The original for loop being optimized
  scf::ForOp forOp;

  /// The yield operation of the original for loop
  scf::YieldOp yieldOp;

  /// Width of the prefetch window in elements
  unsigned prefetchWidth;

  /// Collection of dot operations to be optimizedSetVector
  DenseSet<ttng::WarpGroupDotOp> dots;

  /// Maps for tracking various aspects of dot operations
  DenseMap<Value, Value> dot2aSrcMemDesc;
  DenseMap<Value, Value> dot2bSrcMemDesc;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2aValsLocalLoad;
  DenseMap<Value, SmallVector<Value>> dot2aValsElementWise;
  DenseMap<Value, ttng::WarpGroupDotWaitOp> dot2Wait;

  Value generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                         bool loadToReg, Attribute dotEncoding,
                         OpBuilder &builder, int64_t kWidth,
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt);

  /// Clone elementwise operations for prefetched values
  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           Value source, OpBuilder &builder);
};

void WGMMAPrefetcher::cloneElementwiseOps(Value &ret,
                                          const SmallVector<Value> &vals,
                                          Value source, OpBuilder &builder) {
  IRMapping mapping;
  mapping.map(source, ret);

  for (int i = 0; i < vals.size(); i++) {
    Value v = vals[i];
    Operation *op = v.getDefiningOp();
    assert(op->getNumResults() == 1 &&
           "Defining operation must have exactly one result");
    Value curr = builder.clone(*op, mapping)->getResult(0);
    if (isa<RankedTensorType>(curr.getType())) {
      auto retType = RankedTensorType::get(
          cast<RankedTensorType>(ret.getType()).getShape(),
          cast<RankedTensorType>(curr.getType()).getElementType(),
          cast<RankedTensorType>(curr.getDefiningOp()->getOperand(0).getType())
              .getEncoding());
      curr.setType(retType);
    }
    mapping.map(v, curr);
  }
  ret = mapping.lookup(vals.back());
}

Value WGMMAPrefetcher::generatePrefetch(Value v, unsigned opIdx,
                                        bool isPrologue, bool loadToReg,
                                        Attribute dotEncoding,
                                        OpBuilder &builder, int64_t kWidth,
                                        std::optional<int64_t> offsetK,
                                        std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<triton::gpu::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  auto rank = shape.size();
  SmallVector<int64_t> offset(rank, 0);
  Type elementType = type.getElementType();

  int64_t kIdx = opIdx == 0 ? rank - 1 : rank - 2;

  offset[kIdx] = isPrologue ? 0 : prefetchWidth;
  shape[kIdx] = isPrologue ? prefetchWidth : (shape[kIdx] - prefetchWidth);

  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;

  SmallVector<Value> offsetsVal;
  for (int64_t off : offset)
    offsetsVal.push_back(
        builder.create<arith::ConstantIntOp>(v.getLoc(), off, 32));
  Value newSmem = builder.create<triton::gpu::MemDescSubviewOp>(
      v.getLoc(),
      triton::gpu::MemDescType::get(
          shape, elementType, type.getEncoding(), type.getMemorySpace(),
          type.getMutableMemory(), type.getAllocShape()),
      v, offsetsVal);

  if (loadToReg) {
    auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
        builder.getContext(), opIdx, dotEncoding, kWidth);
    Value prefetchSlice = builder.create<triton::gpu::LocalLoadOp>(
        v.getLoc(), RankedTensorType::get(shape, elementType, dotOperandEnc),
        newSmem);
    return prefetchSlice;
  }

  return newSmem;
}

LogicalResult WGMMAPrefetcher::initialize() {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<nvidia_gpu::WarpGroupDotOp> dotsInFor;

  // Step 1: check the condition if the forloop can be prefetched
  for (Operation &op : *loop) {
    if (auto dotOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      dotsInFor.push_back(dotOp);
    } else if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // Only allow WarpGroupDotOp in the loop for now
      return failure();
    }
  }

  // Early exit conditions
  if (dotsInFor.empty() || dotsInFor.size() > 1) {
    return failure();
  }

  // Verify all dots are supported RSGEMM operations
  for (ttng::WarpGroupDotOp dotOp : dotsInFor) {
    if (isSupportedRSGEMM(dotOp).failed())
      return failure();
  }

  auto getPrefetchSrc = [](Value v) -> SmallVector<Value> {
    // walk back to conversion
    Operation *op = v.getDefiningOp();
    bool foundConvertFromShared = false;
    SmallVector<Value> rets;
    if (isa<LocalAllocOp>(op)) {
      return rets;
    }

    rets.push_back(op->getResult(0));
    if (dyn_cast<MemDescSubviewOp>(op))
      return rets;
    LDBG("Prefetch src: " << *op);
    while (op) {
      if (op->getNumOperands() != 1)
        break;
      if (!op->getResult(0).hasOneUse())
        break;
      rets.push_back(op->getOperand(0));
      if (auto cvt = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        // NYI for other encodings, for example if we have transpose
        // in the chain
        if (isa<DotOperandEncodingAttr>(cvt.getType().getEncoding()))
          foundConvertFromShared = true;
        break;
      }
      op = op->getOperand(0).getDefiningOp();
      if (op)
        LDBG("op: " << *op);
    }
    std::reverse(rets.begin(), rets.end());

    if (foundConvertFromShared)
      return rets;
    return {};
  };

  for (ttng::WarpGroupDotOp dotOp : dotsInFor) {
    // If getMaxNumImpreciseAcc > 0, WGMMA.cpp will have
    // extra treatment for dotOp (e.g., add accumulator).
    // Therefore, we disable the optimization here
    // when getMaxNumImpreciseAcc > 0;
    if (dotOp.getMaxNumImpreciseAcc() > 0) {
      return failure();
    }

    auto aType = dotOp.getA().getType();
    auto bType = dotOp.getB().getType();

    auto aEnc = mlir::cast<DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc = mlir::cast<NVMMASharedEncodingAttr>(bType.getEncoding());

    if (!aEnc || !bEnc) {
      return failure();
    }
    auto aElementBitWidth = aType.getElementTypeBitWidth();
    auto bElementBitWidth = bType.getElementTypeBitWidth();
    assert((aElementBitWidth == bElementBitWidth) &&
           "BitWidth of a and b for dot does not match");

    // Get Prefetchwidth based on the instruction shape in K dim
    auto dType = dotOp.getType();
    auto mmaEnncoding = cast<NvidiaMmaEncodingAttr>(dType.getEncoding());
    auto instrShape = mmaEnncoding.getInstrShape();
    prefetchWidth = instrShape[2];

    auto kSize = aType.getShape().back();
    if (kSize < prefetchWidth)
      return failure();

    auto aVals = getPrefetchSrc(dotOp.getA());
    auto bVals = getPrefetchSrc(dotOp.getB());

    if (aVals.size() && bVals.size()) {
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
      if (!dyn_cast<MemDescSubviewOp>(aSmem.getDefiningOp()) ||
          !dyn_cast<MemDescSubviewOp>(bSmem.getDefiningOp())) {
        return failure();
      }
      auto dotOpResult = dotOp.getResult();
      if (!dotOpResult.hasOneUse())
        return failure();

      auto dotOpUser = *(dotOpResult.getUsers().begin());
      auto dotWait = dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(dotOpUser);
      if (!dotWait)
        return failure();

      dots.insert(dotOp);
      dot2aVals[dotOp] = aVals;
      for (auto op : aVals) {
        if (isa<MemDescSubviewOp, LocalLoadOp>(op.getDefiningOp())) {
          dot2aValsLocalLoad[dotOp].push_back(op);
        } else {
          auto curOp = op.getDefiningOp();
          if (curOp->hasTrait<mlir::OpTrait::Elementwise>() &&
              isMemoryEffectFree(curOp))
            dot2aValsElementWise[dotOp].push_back(op);
          else
            return failure();
        }
      }

      dot2aSrcMemDesc[dotOp] = aSmem;
      dot2bSrcMemDesc[dotOp] = bSmem;
      dot2Wait[dotOp] = dotWait;
    }
  }

  return llvm::success();
}

scf::ForOp WGMMAPrefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);

  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  auto setInsertionPointBeforeYield = [](OpBuilder &builder,
                                         scf::ForOp newForOp) {
    if (newForOp.getBody()->mightHaveTerminator()) {
      builder.setInsertionPoint(newForOp.getBody()->getTerminator());
    } else {
      builder.setInsertionPointToEnd(newForOp.getBody());
    }
  };
  // Keep track of each user of dotwait
  DenseSet<Operation *> dotWaitUsers;

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (op.getNumRegions() > 0) {
      setInsertionPointBeforeYield(builder, newForOp);
    }
    // Stop sinking dotwait when find the user of dotwait
    if (!dotWaitUsers.empty() && dotWaitUsers.contains(&op)) {
      dotWaitUsers.clear();
      setInsertionPointBeforeYield(builder, newForOp);
    }
    for (auto operand : op.getOperands()) {
      if (auto def = operand.getDefiningOp()) {
        auto dot = dyn_cast<nvidia_gpu::WarpGroupDotOp>(def);
        if (dot && dots.contains(dot)) {
          setInsertionPointBeforeYield(builder, newForOp);
        }
      }
    }
    Operation *newOp = builder.clone(op, mapping);
    auto dot = dyn_cast<nvidia_gpu::WarpGroupDotOp>(&op);
    if (dot && dots.contains(dot)) {

      Attribute dotEncoding = dot.getType().getEncoding();

      int64_t kShape = dot.getA().getType().getShape().back();
      auto aEnc =
          dyn_cast<DotOperandEncodingAttr>(dot.getA().getType().getEncoding());
      int64_t kWidth = aEnc.getKWidth();
      int64_t subTileCnt = ceil<int64_t>(kShape, prefetchWidth);

      SmallVector<Value> PrefetchedA;

      // Prefetching
      for (int i = 0; i < subTileCnt; i++) {
        Value aRem = generatePrefetch(mapping.lookup(dot2aSrcMemDesc[dot]), 0,
                                      false, true, dotEncoding, builder, kWidth,
                                      prefetchWidth * i, prefetchWidth);
        PrefetchedA.push_back(aRem);
      }

      // Interleave elementwise with WGMMA
      nvidia_gpu::WarpGroupDotOp prevDot;
      Value UseC = nullptr;
      if (dot.getUseC()) {
        UseC = mapping.lookup(dot.getUseC());
      }
      Value OpC = mapping.lookup(dot.getC());

      for (int i = 0; i < subTileCnt; i++) {
        cloneElementwiseOps(PrefetchedA[i], dot2aValsElementWise[dot],
                            dot2aValsLocalLoad[dot].back(), builder);
        Value bSubtile = generatePrefetch(
            mapping.lookup(dot2bSrcMemDesc[dot]), 1, false, false, dotEncoding,
            builder, kWidth, prefetchWidth * i, prefetchWidth);
        if (i > 0) {
          OpC = prevDot.getResult();
          // for subtiles with index > 0, useC is set to 1 for
          // using the result of prevDot
          UseC =
              builder.create<mlir::arith::ConstantIntOp>(newOp->getLoc(), 1, 1);
        }
        auto newDotOp = builder.create<nvidia_gpu::WarpGroupDotOp>(
            newOp->getLoc(), dot.getType(), PrefetchedA[i], bSubtile, OpC, UseC,
            dot.getInputPrecision(), dot.getMaxNumImpreciseAcc(),
            dot.getIsAsync());
        prevDot = newDotOp;
      }
      newOp = (Operation *)prevDot;
    }
    auto dotWait = dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(newOp);
    if (dotWait) {
      dotWaitUsers.clear();
      for (auto dotWaitUser : op.getUsers()) {
        dotWaitUsers.insert(dotWaitUser);
      }
      builder.setInsertionPoint(dotWait);
    }

    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));

  builder.setInsertionPointToEnd(newForOp.getBody());
  if (!yieldValues.empty())
    builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);

  return newForOp;
}

} // namespace

/// Pass to perform WGMMA prefetch optimization
struct WGMMAPrefetchPass
    : public impl::TritonGPUWGMMAPrefetchBase<WGMMAPrefetchPass> {
  void runOnOperation() override {
    // The detailed explanation of this pass can be found in
    // https://github.com/triton-lang/triton/pull/6196

    // Step 1: Canonicalize convert ops for easier pattern matching
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }

    // Step 2: Walk through all for loops and apply prefetch optimization
    getOperation()->walk([&](scf::ForOp forOp) {
      WGMMAPrefetcher prefetcher(forOp);

      // Skip if initialization fails
      if (prefetcher.initialize().failed())
        return;

      // Create new optimized loop
      scf::ForOp newForOp = prefetcher.createNewForOp();

      // Replace old loop with new one
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
