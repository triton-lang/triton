#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/AddressRanges.h"

namespace {

using namespace mlir;

namespace ttng = triton::nvidia_gpu;
namespace ttg = triton::gpu;
namespace tt = triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

// If we don't know the effects of the op, we add all possible effects.
static void addAllValuelessEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
}

static bool
collectEffects(Operation *op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Collect effect instances the operation. Note that the implementation of
  // getEffects erases all effect instances that have the type other than the
  // template parameter so we collect them first in a local buffer and then
  // copy.
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects(localEffects);
    llvm::append_range(effects, localEffects);
    return true;
  }
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block)
          if (!collectEffects(&innerOp, effects))
            return false;
      }
    }
    return true;
  }

  // We need to be conservative here in case the op doesn't have the interface
  // and assume it can have any possible effect.
  addAllValuelessEffects(effects);
  return false;
}

struct AccessRange {
  SmallVector<std::optional<llvm::AddressRange>> ranges;
  unsigned rankOffset = 0;
};

// Simple local alias analysis that looks for a single underlying allocation and
// an access subrange.
static std::pair<Value, AccessRange> findBufferAccess(Value a) {
  // Handle block arguments.
  if (auto arg = dyn_cast<BlockArgument>(a)) {
    Operation *parentOp = arg.getOwner()->getParentOp();

    // Look through `ttg.warp_specialize` explicit captures.
    if (auto wsOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(parentOp)) {
      return findBufferAccess(
          wsOp.getParentOp().getExplicitCaptures()[arg.getArgNumber()]);
    }

    // Unknown block argument.
    return {};
  }

  Operation *defOp = a.getDefiningOp();
  // Accessing the alloc accesses the whole buffer.
  if (auto alloc = dyn_cast<ttng::TMEMAllocOp>(defOp)) {
    AccessRange access;
    for (uint64_t dim : alloc.getType().getShape())
      access.ranges.push_back({{0, dim}});
    return {a, std::move(access)};
  }

  // Trans and Reshape views don't change the access size.
  if (isa<ttg::MemDescTransOp, ttg::MemDescReshapeOp>(defOp)) {
    return findBufferAccess(defOp->getOperand(0));
  }

  // Subviews can reduce the access sizes.
  if (auto subview = dyn_cast<ttg::MemDescSubviewOp>(defOp)) {
    auto [alloc, parentAccess] = findBufferAccess(subview.getSrc());
    if (!alloc)
      return {};
    // Handle subview of a subview. The first `rankOffset` access sizes are
    // the same as in the parent access.
    AccessRange childAccess;
    for (auto i : llvm::seq(parentAccess.rankOffset))
      childAccess.ranges.push_back(parentAccess.ranges[i]);

    // The subview may have a smaller rank, in which case its access size is
    // just 1 for the higher dims.
    childAccess.rankOffset =
        subview.getSrc().getType().getRank() - subview.getType().getRank();
    for (auto [i, offset] : llvm::enumerate(subview.getOffsets())) {
      auto parentRange = parentAccess.ranges[i + parentAccess.rankOffset];
      if (!parentRange) {
        childAccess.ranges.push_back({});
        continue;
      }

      // If the offset is not known, then the entire dim may be accessed.
      APInt value;
      if (!matchPattern(offset, m_ConstantInt(&value))) {
        childAccess.ranges.push_back({});
        continue;
      }

      uint64_t accessStart = parentRange->start() + value.getSExtValue();
      uint64_t accessSize = 1;
      if (i >= childAccess.rankOffset)
        accessSize = subview.getType().getShape()[i - childAccess.rankOffset];
      childAccess.ranges.push_back({{accessStart, accessStart + accessSize}});
    }
    return {alloc, std::move(childAccess)};
  }

  // Subslice is a subview only on the N dimension.
  if (auto subslice = dyn_cast<ttng::TMEMSubSliceOp>(defOp)) {
    auto [alloc, parentAccess] = findBufferAccess(subslice.getSrc());
    if (!alloc)
      return {};
    if (!parentAccess.ranges[1])
      return {alloc, parentAccess};
    uint64_t mStart = parentAccess.ranges[1]->start() + subslice.getN();
    uint64_t mSize = subslice.getType().getShape()[1];
    AccessRange childAccess = parentAccess;
    childAccess.ranges[1] = {{mStart, mStart + mSize}};
    return {alloc, std::move(childAccess)};
  }

  // Unknown defining op.
  return {};
}

static bool tmemMayAlias(Value a, Value b) {
  auto [aAlloc, aRanges] = findBufferAccess(a);
  auto [bAlloc, bRanges] = findBufferAccess(b);
  // If the underlying buffer was not identified, assume mayalias.
  if (!aAlloc || !bAlloc)
    return true;
  // If the buffers are different, they don't alias.
  if (aAlloc != bAlloc)
    return false;
  // If the access ranges along any dimension are known to not overlap, then the
  // accesses don't alias.
  for (auto [aRange, bRange] : llvm::zip(aRanges.ranges, bRanges.ranges)) {
    // If either access range at this dim is unknown, we can't determine if they
    // don't overlap.
    if (!aRange || !bRange)
      continue;
    // The access ranges are known and don't overlap.
    if (!aRange->intersects(*bRange))
      return false;
  }
  return true;
}

// Sink tmem_loads as close to their use as possible to reduce register
// pressure.
static void sinkLoad(ttng::TMEMLoadOp load, ttg::ConvertLayoutOp cvt) {
  Operation *insertBefore = nullptr;
  Operation *next = cvt->getNextNode();
  while (next && !next->hasTrait<OpTrait::IsTerminator>()) {
    insertBefore = next;
    bool dep = false;
    for (auto operand : getNestedOperands(next)) {
      if (operand == cvt.getResult()) {
        dep = true;
        break;
      }
    }
    if (!isMemoryEffectFree(next)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      collectEffects(next, effects);
      for (auto effect : effects) {
        // Look for potentially aliasing write or free effects.
        if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
          continue;
        if (isa<SideEffects::DefaultResource>(effect.getResource())) {
          dep = true;
          break;
        }
        if (isa<ttng::TensorMemory>(effect.getResource()) &&
            (!effect.getValue() ||
             tmemMayAlias(effect.getValue(), load.getSrc()))) {
          dep = true;
          break;
        }
      }
    }
    if (dep)
      break;
    next = next->getNextNode();
  }
  if (insertBefore) {
    load->moveBefore(insertBefore);
    cvt->moveBefore(insertBefore);
  }
}

static void hoistStore(ttng::TMEMStoreOp store, ttg::ConvertLayoutOp cvt,
                       ttng::TMEMSubSliceOp subSlice) {
  Operation *insertAfter = nullptr;
  Operation *prev = subSlice->getPrevNode();
  while (prev) {
    insertAfter = prev;
    bool dep = false;
    for (Value result : prev->getResults()) {
      if (result == cvt.getSrc() || result == subSlice.getSrc()) {
        dep = true;
        break;
      }
    }
    if (!isMemoryEffectFree(prev)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      collectEffects(prev, effects);
      for (auto effect : effects) {
        // Look for potentially aliasing read or alloc effects.
        if (!isa<MemoryEffects::Read, MemoryEffects::Allocate>(
                effect.getEffect()))
          continue;
        if (isa<SideEffects::DefaultResource>(effect.getResource())) {
          dep = true;
          break;
        }
        if (isa<ttng::TensorMemory>(effect.getResource()) &&
            (!effect.getValue() ||
             tmemMayAlias(effect.getValue(), store.getDst()))) {
          dep = true;
          break;
        }
      }
    }
    if (dep)
      break;
    prev = prev->getPrevNode();
  }
  if (insertAfter) {
    store->moveAfter(insertAfter);
    cvt->moveAfter(insertAfter);
    subSlice->moveAfter(insertAfter);
  }
}

// clang-format off
// Converts:
//  %l = ttng.tmem_load %o : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
//  %r = tt.reshape %l : tensor<128x256xf32, #blocked> -> tensor<128x2x128xf32, #blocked4>
//  %t = tt.trans %r {order = array<i32: 0, 2, 1>} : tensor<128x2x128xf32, #blocked4> -> tensor<128x128x2xf32, #blocked5>
//  %outLHS, %outRHS = tt.split %t : tensor<128x128x2xf32, #blocked5> -> tensor<128x128xf32, #blocked2>
// To:
//  %o0 = ttng.tmem_subslice %o { N = 0 }: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
//  %outLHS = ttng.tmem_load %o0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
//  %o1 = ttng.tmem_subslice %o { N = 128 }: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
//  %outRHS = ttng.tmem_load %o1 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
// clang-format on
// This will change the layout of the destination tensor to distribute each
// slice across warps. It currently only supports simple cases where tmem can be
// sliced easily. This could be extended if needed with more powerful slicing
// support of tmem.
class TMemSplitLoadPattern : public OpRewritePattern<tt::SplitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::SplitOp splitOp,
                                PatternRewriter &rewriter) const override {
    auto src = splitOp.getSrc();
    // Skip convert layout ops.
    while (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>()) {
      src = cvt.getSrc();
    }
    // Only support splitting N dimension on the outer most.
    auto transOp = src.getDefiningOp<tt::TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto reshapeOp = transOp.getSrc().getDefiningOp<tt::ReshapeOp>();
    if (!reshapeOp)
      return failure();
    auto shape = reshapeOp.getResult().getType().getShape();
    if (shape[0] != reshapeOp.getSrc().getType().getShape()[0])
      return failure();
    auto tmemLoad = reshapeOp.getSrc().getDefiningOp<ttng::TMEMLoadOp>();
    if (!tmemLoad)
      return failure();
    // We found a tmem_load that is split on the N dimension. We can split it
    // into multiple tmem_loads.
    int mDim = getShapePerCTA(tmemLoad.getSrc().getType())[0];
    // TODO: enable other M cases. (the layout is a bit more complex).
    if (mDim != 128)
      return failure();
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();
    Value tmem = tmemLoad.getSrc();
    int numWarps = ttg::lookupNumWarps(tmemLoad);
    rewriter.setInsertionPoint(tmemLoad);
    // First slice.
    Value subSlice0 = rewriter.create<ttng::TMEMSubSliceOp>(
        tmemLoad.getLoc(), tmem, 0, splitNSize);
    Attribute distLayout = ttng::getTmemCompatibleLayout(
        mDim, splitNSize, splitOp.getOutLHS().getType(), numWarps);
    RankedTensorType newLoadType = RankedTensorType::get(
        splitOp.getOutLHS().getType().getShape(),
        splitOp.getOutLHS().getType().getElementType(), distLayout);
    auto load0 = rewriter.create<ttng::TMEMLoadOp>(tmemLoad.getLoc(),
                                                   newLoadType, subSlice0);
    auto cvt0 = rewriter.create<ttg::ConvertLayoutOp>(
        tmemLoad.getLoc(), splitOp.getOutLHS().getType(), load0);
    // Second slice.
    Value subSlice1 = rewriter.create<ttng::TMEMSubSliceOp>(
        tmemLoad.getLoc(), tmem, splitNSize, splitNSize);
    auto load1 = rewriter.create<ttng::TMEMLoadOp>(tmemLoad.getLoc(),
                                                   newLoadType, subSlice1);
    auto cvt1 = rewriter.create<ttg::ConvertLayoutOp>(
        tmemLoad.getLoc(), splitOp.getOutRHS().getType(), load1);
    rewriter.replaceOp(splitOp, {cvt0, cvt1});
    sinkLoad(load0, cvt0);
    sinkLoad(load1, cvt1);
    return success();
  }
};

class TMemStoreJoinPattern : public OpRewritePattern<ttng::TMEMStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMStoreOp storeOp,
                                PatternRewriter &b) const override {
    // Look through layout conversions.
    Value src = storeOp.getSrc();
    while (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>()) {
      src = cvt.getSrc();
    }

    // Only support joinin N dimension on the outer most.
    auto reshapeOp = src.getDefiningOp<tt::ReshapeOp>();
    if (!reshapeOp)
      return failure();
    auto shape = reshapeOp.getSrc().getType().getShape();
    if (reshapeOp.getType().getShape().front() != shape[0])
      return failure();
    auto transOp = reshapeOp.getSrc().getDefiningOp<tt::TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto joinOp = transOp.getSrc().getDefiningOp<tt::JoinOp>();
    if (!joinOp)
      return failure();

    // We found a tmem_store that is joined on the N dimension. We can split it
    // into multiple tmem_stores.
    int mDim = getShapePerCTA(storeOp.getDst().getType())[0];
    // TODO: enable other M cases. (the layout is a bit more complex).
    if (mDim != 128)
      return failure();
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();

    Location loc = storeOp.getLoc();
    Value tmem = storeOp.getDst();
    int numWarps = ttg::lookupNumWarps(storeOp);
    Value truePred = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));

    Attribute distLayout = ttng::getTmemCompatibleLayout(
        mDim, splitNSize, joinOp.getLhs().getType(), numWarps);
    auto newStoreType = RankedTensorType::get(
        joinOp.getLhs().getType().getShape(),
        joinOp.getLhs().getType().getElementType(), distLayout);

    // First slice.
    auto subSlice0 = b.create<ttng::TMEMSubSliceOp>(loc, tmem, 0, splitNSize);
    auto cvt0 =
        b.create<ttg::ConvertLayoutOp>(loc, newStoreType, joinOp.getLhs());
    auto store0 =
        b.create<ttng::TMEMStoreOp>(loc, subSlice0, cvt0.getResult(), truePred);
    // Second slice.
    auto subSlice1 =
        b.create<ttng::TMEMSubSliceOp>(loc, tmem, splitNSize, splitNSize);
    auto cvt1 =
        b.create<ttg::ConvertLayoutOp>(loc, newStoreType, joinOp.getRhs());
    auto store1 =
        b.create<ttng::TMEMStoreOp>(loc, subSlice1, cvt1.getResult(), truePred);

    hoistStore(store0, cvt0, subSlice0);
    hoistStore(store1, cvt1, subSlice1);
    b.eraseOp(storeOp);
    return success();
  }
};

// Pick an optimized tmem load layout based on its users. When there are
// multiple warpgroups tmem_load results can be distirbuted along M or N across
// the warpgroups. By default distribute along N but when there is a reduction
// along N dimension we want to distribute along M instead to avoid having to
// reduce across warps.
class TMemLoadReducePattern : public OpRewritePattern<ttng::TMEMLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMLoadOp tmemLoadOp,
                                PatternRewriter &rewriter) const override {
    int numWarps = ttg::lookupNumWarps(tmemLoadOp);
    // If there is only 1 warpgroup there is nothing to optimize as the layout
    // is already reduction friendly.
    if (numWarps != 8)
      return failure();
    auto tmemEnc = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
        tmemLoadOp.getSrc().getType().getEncoding());
    if (!tmemEnc)
      return failure();
    int M = tmemEnc.getBlockM();
    int N = tmemEnc.getBlockN();
    if (M != 128)
      return failure();
    bool foundReductionAlongN = false;
    auto filter = [&](Operation *op) {
      if (isa<ttg::ConvertLayoutOp>(op) || op->hasTrait<OpTrait::Elementwise>())
        return true;
      if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
        foundReductionAlongN = reduce.getAxis() == 1;
      }
      return false;
    };
    ForwardSliceOptions fwdOpt;
    fwdOpt.filter = filter;
    SetVector<mlir::Operation *> fwdSlices;
    getForwardSlice(tmemLoadOp.getResult(), &fwdSlices, fwdOpt);
    if (!foundReductionAlongN)
      return failure();
    // Try to split along M dimension but follow the restrictions of TMEM:
    // warp0 get M = 0, warp 1 gets M = 32, warp 2 gets M = 64, warp 3 gets
    // M = 96 warp 4 gets M = 16, warp 5 gets M = 48, warp 6 gets M = 80,
    // warp 7 gets M = 112
    RankedTensorType oldType = tmemLoadOp.getType();
    Attribute newLayout = ttg::LinearEncodingAttr::get(
        tmemLoadOp.getContext(),
        ttg::getTmemLoadLayoutSplitLongM(M, N, oldType, numWarps));
    if (newLayout == oldType.getEncoding())
      return failure();

    auto newType = RankedTensorType::get(oldType.getShape(),
                                         oldType.getElementType(), newLayout);
    tmemLoadOp.getResult().setType(newType);
    OpBuilder builder(tmemLoadOp);
    builder.setInsertionPointAfter(tmemLoadOp);
    auto cvt = builder.create<ttg::ConvertLayoutOp>(
        tmemLoadOp.getLoc(), oldType, tmemLoadOp.getResult());
    tmemLoadOp.getResult().replaceAllUsesExcept(cvt.getResult(), cvt);
    return success();
  }
};

class TritonNvidiaGPUOptimizeTMemLayoutsPass
    : public TritonNvidiaGPUOptimizeTMemLayoutsPassBase<
          TritonNvidiaGPUOptimizeTMemLayoutsPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeTMemLayoutsPassBase<
      TritonNvidiaGPUOptimizeTMemLayoutsPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns
        .add<TMemSplitLoadPattern, TMemStoreJoinPattern, TMemLoadReducePattern>(
            context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUOptimizeTMemLayoutsPass() {
  return std::make_unique<TritonNvidiaGPUOptimizeTMemLayoutsPass>();
}
