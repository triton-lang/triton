#include <algorithm>
#include <array>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/Support/ErrorHandling.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUASSIGNCTALAYOUTSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Default CTA layouts are assigned during Triton-to-TritonGPU conversion. This
// pass only gives Dot/Reduce ops their preferred CTA layout and materializes
// the boundary with ttg.convert_layout; RemoveLayoutConversions cleans up
// later.
struct DotCTASplit {
  unsigned m;
  unsigned n;
};

struct DotCTALayout {
  DotCTASplit split;
  std::array<unsigned, 2> order;
};

ttg::DistributedEncodingTrait
cloneWithCGALayout(ttg::DistributedEncodingTrait layout,
                   llvm::ArrayRef<int64_t> shape, int numWarps,
                   int threadsPerWarp, ttg::CGAEncodingAttr newCGALayout) {
  if (auto blockedLayout = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    return ttg::BlockedEncodingAttr::get(
        layout.getContext(), shape, blockedLayout.getSizePerThread(),
        blockedLayout.getOrder(), numWarps, threadsPerWarp, newCGALayout);
  }

  if (auto sliceLayout = dyn_cast<ttg::SliceEncodingAttr>(layout)) {
    return ttg::SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        cloneWithCGALayout(sliceLayout.getParent(), shape, numWarps,
                           threadsPerWarp, newCGALayout));
  }

  llvm::report_fatal_error("cloneWithCGALayout not implemented for layout");
}

Value convertValueToLayout(OpBuilder &builder, Location loc, Value value,
                           Attribute layout) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy || tensorTy.getEncoding() == layout)
    return value;

  auto newTy = tensorTy.cloneWithEncoding(layout);
  return ttg::ConvertLayoutOp::create(builder, loc, newTy, value);
}

Value cloneLoadWithLayout(OpBuilder &builder, Location loc, Value value,
                          Attribute layout, int numWarps, int threadsPerWarp) {
  Value loadValue = value;
  // If the operand is already a convert-layout chain rooted at a load:
  //
  //   %old_load = tt.load %old_ptr : ... #blocked_old
  //   %old_dot_operand = ttg.convert_layout %old_load : #blocked_old ->
  //   #dot_old
  //
  // create a sibling load using the planned CTA layout:
  //
  //   %new_ptr = ttg.convert_layout %old_ptr : #blocked_old -> #blocked_new
  //   %new_load = tt.load %new_ptr : ... #blocked_new
  //   %new_dot_operand = ttg.convert_layout %new_load : #blocked_new ->
  //   #dot_new
  //
  // This avoids inserting a cross-CTA conversion on the loaded value and leaves
  // the original load available for any other users if any.
  // TODO: Match more flexible producer patterns between the load and consumer.
  while (auto cvtOp = loadValue.getDefiningOp<ttg::ConvertLayoutOp>())
    loadValue = cvtOp.getSrc();

  Operation *loadOp = loadValue.getDefiningOp();
  if (!isa_and_nonnull<triton::LoadOp, triton::DescriptorLoadLikeOpInterface>(
          loadOp))
    return value;

  auto oldTy = cast<RankedTensorType>(loadValue.getType());
  auto oldLayout = cast<ttg::DistributedEncodingTrait>(oldTy.getEncoding());

  auto cgaLayout = ttg::getCGALayout(layout);
  auto newLoadLayout = cloneWithCGALayout(oldLayout, oldTy.getShape(), numWarps,
                                          threadsPerWarp, cgaLayout);
  if (oldLayout == newLoadLayout)
    return value;

  auto newTy = oldTy.cloneWithEncoding(newLoadLayout);

  OpBuilder loadBuilder(loadOp);
  loadBuilder.setInsertionPointAfter(loadOp);

  auto convertLoadOperand = [&](Value operand) -> Value {
    if (!operand)
      return {};
    return convertValueToLayout(loadBuilder, loadOp->getLoc(), operand,
                                newLoadLayout);
  };
  auto convertLoadedValue = [&](Value loaded) {
    return convertValueToLayout(builder, loc, loaded, layout);
  };

  if (auto scalarLoad = dyn_cast<triton::LoadOp>(loadOp)) {
    Value newPtr = convertLoadOperand(scalarLoad.getPtr());
    Value newMask = convertLoadOperand(scalarLoad.getMask());
    Value newOther = convertLoadOperand(scalarLoad.getOther());
    Operation *newLoad = triton::LoadOp::create(
                             loadBuilder, scalarLoad.getLoc(), newTy, newPtr,
                             newMask, newOther, scalarLoad.getCache(),
                             scalarLoad.getEvict(), scalarLoad.getIsVolatile())
                             .getOperation();
    newLoad->setAttrs(loadOp->getAttrs());
    return convertLoadedValue(newLoad->getResult(0));
  }

  if (isa<triton::DescriptorLoadLikeOpInterface>(loadOp)) {
    Operation *newLoad = loadBuilder.clone(*loadOp);
    newLoad->getResult(0).setType(newTy);
    return convertLoadedValue(newLoad->getResult(0));
  }

  llvm_unreachable("expected scalar or descriptor load");
}

void convertOpOperandsToLayouts(Operation *op,
                                llvm::ArrayRef<Attribute> operandLayouts,
                                int numWarps, int threadsPerWarp) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  for (auto [operand, layout] :
       llvm::zip(op->getOpOperands(), operandLayouts)) {
    Value value = cloneLoadWithLayout(builder, loc, operand.get(), layout,
                                      numWarps, threadsPerWarp);
    operand.set(convertValueToLayout(builder, loc, value, layout));
  }
}

void convertOpResultsFromLayouts(Operation *op,
                                 llvm::ArrayRef<Attribute> resultLayouts) {
  OpBuilder builder(op->getContext());
  builder.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  for (auto [result, resultLayout] :
       llvm::zip(op->getResults(), resultLayouts)) {
    if (auto originalTy = dyn_cast<RankedTensorType>(result.getType())) {
      result.setType(originalTy.cloneWithEncoding(resultLayout));
      Value converted =
          convertValueToLayout(builder, loc, result, originalTy.getEncoding());
      result.replaceAllUsesExcept(converted, converted.getDefiningOp());
    }
  }
}

DotCTALayout getDotCTALayout(int64_t m, int64_t n, unsigned numCTAs,
                             bool preferTwoCTA) {
  constexpr unsigned kPreferredChunkSize = 128;
  constexpr unsigned kMinChunkSize = 64;
  constexpr unsigned kMaxMMAv5TwoCTAsNPerCTA = 256;
  auto isLegalChunkSize = [](unsigned chunk) { return chunk >= kMinChunkSize; };

  if (preferTwoCTA) {
    for (unsigned splitM = numCTAs; splitM > 1; splitM /= 2) {
      if (numCTAs % splitM != 0)
        continue;
      unsigned splitN = numCTAs / splitM;
      unsigned mPerCTA = m / splitM;
      unsigned nPerCTA = n / splitN;
      if (isLegalChunkSize(mPerCTA) && isLegalChunkSize(nPerCTA) &&
          nPerCTA <= kMaxMMAv5TwoCTAsNPerCTA)
        return {{splitM, splitN}, {0, 1}};
    }
  }

  unsigned splitM = 1;
  unsigned splitN = numCTAs;
  // Prefer a larger M chunk, up to 128 elements, by assigning splitM first.
  // splitN gets the remaining CTAs as long as each N chunk has at least 64
  // elements.
  bool foundSplit = false;
  for (unsigned chunkM = kPreferredChunkSize;
       !foundSplit && isLegalChunkSize(chunkM); chunkM /= 2) {
    unsigned candidateM = std::clamp<unsigned>(m / chunkM, 1, numCTAs);
    for (;;) {
      if (numCTAs % candidateM == 0) {
        unsigned candidateN = numCTAs / candidateM;
        unsigned nPerCTA = n / candidateN;
        if (isLegalChunkSize(nPerCTA) &&
            nPerCTA <= kMaxMMAv5TwoCTAsNPerCTA) {
          splitM = candidateM;
          splitN = candidateN;
          foundSplit = true;
          break;
        }
      }
      if (candidateM == 1)
        break;
      --candidateM;
    }
  }

  return {{splitM, splitN}, {1, 0}};
}

bool preferTwoCTASplit(triton::DotOp dot, bool isBlackwell) {
  if (!isBlackwell || ttg::lookupNumCTAs(dot) < 2)
    return false;

  auto retTy = cast<RankedTensorType>(dot.getType());
  if (retTy.getRank() != 2)
    return false;

  Value b = dot.getB();
  while (auto cvtOp = b.getDefiningOp<ttg::ConvertLayoutOp>())
    b = cvtOp.getSrc();
  return isa_and_nonnull<triton::DescriptorLoadLikeOpInterface>(
      b.getDefiningOp());
}

void assignDotCTALayout(triton::DotOp dot, bool isBlackwell) {
  MLIRContext *ctx = dot.getContext();

  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  auto dTy = cast<RankedTensorType>(dot.getD().getType());

  auto aLayout = cast<ttg::DotOperandEncodingAttr>(aTy.getEncoding());
  auto bLayout = cast<ttg::DotOperandEncodingAttr>(bTy.getEncoding());
  auto dLayout = cast<ttg::BlockedEncodingAttr>(dTy.getEncoding());

  DotCTALayout layout = getDotCTALayout(
      dTy.getShape()[0], dTy.getShape()[1], ttg::getNumCTAs(dLayout),
      preferTwoCTASplit(dot, isBlackwell));

  OpBuilder builder(dot);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numWarps = ttg::lookupNumWarps(dot);

  auto newCGALayout = ttg::CGAEncodingAttr::fromSplitParams(
      ctx, {layout.split.m, layout.split.n}, {layout.split.m, layout.split.n},
      layout.order);
  auto newDLayout = ttg::BlockedEncodingAttr::get(
      ctx, dTy.getShape(), dLayout.getSizePerThread(), dLayout.getOrder(),
      numWarps, threadsPerWarp, newCGALayout);
  auto newALayout = ttg::DotOperandEncodingAttr::get(
      ctx, aLayout.getOpIdx(), newDLayout, aLayout.getKWidth());
  auto newBLayout = ttg::DotOperandEncodingAttr::get(
      ctx, bLayout.getOpIdx(), newDLayout, bLayout.getKWidth());

  convertOpOperandsToLayouts(dot.getOperation(),
                             {newALayout, newBLayout, newDLayout}, numWarps,
                             threadsPerWarp);
  convertOpResultsFromLayouts(dot.getOperation(), {newDLayout});
}

ttg::CGAEncodingAttr getReduceCGALayout(triton::ReduceOp reduce,
                                        RankedTensorType srcTy) {
  unsigned rank = srcTy.getRank();
  auto order = ttg::getOrder(srcTy);
  auto sizePerThread = ttg::getContigPerThread(srcTy);
  auto srcLayout = cast<ttg::DistributedEncodingTrait>(srcTy.getEncoding());

  SmallVector<unsigned> ctasPerCGA(rank, 0);
  unsigned remainingCTAs = ttg::getNumCTAs(srcLayout);
  // Keep the reduced dimension within a single CTA so reductions do not cross
  // CTAs. Assign CTAs to the remaining dimensions in layout order, bounded by
  // how many elements each CTA can cover in that dimension.
  for (int i = rank - 1; i >= 0; --i) {
    unsigned dim = order[i];
    if (dim == reduce.getAxis()) {
      ctasPerCGA[dim] = 1;
      continue;
    }

    ctasPerCGA[dim] = std::min<unsigned>(
        srcTy.getShape()[dim] / sizePerThread[dim], remainingCTAs);
    ctasPerCGA[dim] = std::max(ctasPerCGA[dim], 1u);
    remainingCTAs /= ctasPerCGA[dim];
  }

  // Put any leftover CTAs on the fastest non-reduced dimension.
  bool assignedRemainingCTAs = false;
  for (int i = rank - 1; i >= 0; --i) {
    unsigned dim = order[i];
    if (dim == reduce.getAxis())
      continue;
    ctasPerCGA[dim] *= remainingCTAs;
    assignedRemainingCTAs = true;
    break;
  }

  SmallVector<unsigned> ctaSplitNum = ctasPerCGA;
  // If numCTAs > 1 and the only dimension is the reduced dimension, the loops
  // above leave all CTAs unassigned. Put the remaining CTAs on that dimension
  // so that they all collaborate in the reduction.
  if (!assignedRemainingCTAs && remainingCTAs > 0) {
    ctasPerCGA[order[rank - 1]] *= remainingCTAs;
    ctaSplitNum[order[rank - 1]] = ctasPerCGA[order[rank - 1]];
  }

  auto ctaOrder = ttg::getCTAOrder(srcLayout);
  return ttg::CGAEncodingAttr::fromSplitParams(reduce.getContext(), ctasPerCGA,
                                               ctaSplitNum, ctaOrder);
}

void assignReduceCTALayout(triton::ReduceOp reduce) {
  MLIRContext *ctx = reduce.getContext();
  Value src = reduce.getOperand(0);
  auto srcTy = cast<RankedTensorType>(src.getType());

  auto srcLayout = cast<ttg::DistributedEncodingTrait>(srcTy.getEncoding());

  OpBuilder builder(reduce);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numWarps = ttg::lookupNumWarps(reduce);

  auto cgaLayout = getReduceCGALayout(reduce, srcTy);
  auto newSrcLayout = cloneWithCGALayout(srcLayout, srcTy.getShape(), numWarps,
                                         threadsPerWarp, cgaLayout);

  SmallVector<Attribute> operandLayouts(reduce.getNumOperands(), newSrcLayout);
  Attribute resultLayout;
  if (srcTy.getRank() > 1)
    resultLayout =
        ttg::SliceEncodingAttr::get(ctx, reduce.getAxis(), newSrcLayout);
  SmallVector<Attribute> resultLayouts(reduce.getNumResults(), resultLayout);

  convertOpOperandsToLayouts(reduce.getOperation(), operandLayouts, numWarps,
                             threadsPerWarp);
  convertOpResultsFromLayouts(reduce.getOperation(), resultLayouts);
}

struct AssignCTALayoutsPass
    : public impl::TritonNvidiaGPUAssignCTALayoutsPassBase<
          AssignCTALayoutsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    unsigned numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs == 1)
      return;

    auto targetAttr =
        mod->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
    bool isBlackwell = false;
    if (targetAttr && targetAttr.getValue().starts_with("cuda:")) {
      int computeCapability = getNVIDIAComputeCapability(mod);
      isBlackwell = computeCapability >= 100 && computeCapability < 120;
    }

    mod.walk([&](Operation *op) {
      if (auto dot = dyn_cast<triton::DotOp>(op))
        assignDotCTALayout(dot, isBlackwell);
      if (auto reduce = dyn_cast<triton::ReduceOp>(op))
        assignReduceCTALayout(reduce);
    });
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
