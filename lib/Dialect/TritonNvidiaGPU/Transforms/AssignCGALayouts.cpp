#include <algorithm>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/Support/ErrorHandling.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUASSIGNCGALAYOUTSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Default CGA layouts are assigned during Triton-to-TritonGPU conversion. This
// pass only gives Dot/Reduce ops their preferred CGA layout and materializes
// the boundary with ttg.convert_layout; RemoveLayoutConversions cleans up
// later.
struct DotCGASplit {
  unsigned m;
  unsigned n;
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

  assert(false && "cloneWithCGALayout not implemented for layout");
}

Value convertValueToLayout(OpBuilder &builder, Location loc, Value value,
                           Attribute layout) {
  auto tensorTy = cast<RankedTensorType>(value.getType());
  if (tensorTy.getEncoding() == layout)
    return value;

  auto newTy = tensorTy.cloneWithEncoding(layout);
  return ttg::ConvertLayoutOp::create(builder, loc, newTy, value);
}

void convertOpOperandsToLayouts(Operation *op,
                                llvm::ArrayRef<Attribute> operandLayouts) {
  OpBuilder builder(op);
  Location loc = op->getLoc();
  for (auto [operand, layout] :
       llvm::zip(op->getOpOperands(), operandLayouts)) {
    Value value = convertValueToLayout(builder, loc, operand.get(), layout);
    if (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>())
      convert->setDiscardableAttr("ttg.force_cga_rematerialization",
                                  builder.getUnitAttr());
    operand.set(value);
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

DotCGASplit getDotCGASplit(int64_t m, int64_t n, unsigned numCTAs) {
  constexpr unsigned kPreferredChunkSize = 128;
  constexpr unsigned kMinChunkSize = 64;
  auto isLegalChunkSize = [](unsigned chunk) { return chunk >= kMinChunkSize; };

  unsigned splitM = 1;
  unsigned splitN = numCTAs;
  // Prefer a larger M chunk, up to 128 elements, by assigning splitM first.
  // splitN gets the remaining CTAs as long as each N chunk has at least 64
  // elements.
  for (unsigned chunkM = kPreferredChunkSize; isLegalChunkSize(chunkM);
       chunkM /= 2) {
    splitM = std::clamp<unsigned>(m / chunkM, 1, numCTAs);
    splitN = numCTAs / splitM;
    if (isLegalChunkSize(n / splitN))
      break;
  }

  return {splitM, splitN};
}

void assignDotCGALayout(triton::DotOp dot) {
  MLIRContext *ctx = dot.getContext();

  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  auto dTy = cast<RankedTensorType>(dot.getD().getType());

  auto aLayout = cast<ttg::DotOperandEncodingAttr>(aTy.getEncoding());
  auto bLayout = cast<ttg::DotOperandEncodingAttr>(bTy.getEncoding());
  auto dLayout = cast<ttg::BlockedEncodingAttr>(dTy.getEncoding());

  DotCGASplit split = getDotCGASplit(dTy.getShape()[0], dTy.getShape()[1],
                                     ttg::getNumCTAs(dLayout));

  OpBuilder builder(dot);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numWarps = ttg::lookupNumWarps(dot);

  auto newCGALayout = ttg::CGAEncodingAttr::fromSplitParams(
      ctx, {split.m, split.n}, {split.m, split.n}, {1, 0});
  auto newDLayout = ttg::BlockedEncodingAttr::get(
      ctx, dTy.getShape(), dLayout.getSizePerThread(), dLayout.getOrder(),
      numWarps, threadsPerWarp, newCGALayout);
  auto newALayout = ttg::DotOperandEncodingAttr::get(
      ctx, aLayout.getOpIdx(), newDLayout, aLayout.getKWidth());
  auto newBLayout = ttg::DotOperandEncodingAttr::get(
      ctx, bLayout.getOpIdx(), newDLayout, bLayout.getKWidth());

  convertOpOperandsToLayouts(dot.getOperation(),
                             {newALayout, newBLayout, newDLayout});
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

void assignReduceCGALayout(triton::ReduceOp reduce) {
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

  convertOpOperandsToLayouts(reduce.getOperation(), operandLayouts);
  convertOpResultsFromLayouts(reduce.getOperation(), resultLayouts);
}

struct AssignCGALayoutsPass
    : public impl::TritonNvidiaGPUAssignCGALayoutsPassBase<
          AssignCGALayoutsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    unsigned numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs == 1)
      return;

    mod.walk([&](Operation *op) {
      if (auto dot = dyn_cast<triton::DotOp>(op))
        assignDotCGALayout(dot);
      if (auto reduce = dyn_cast<triton::ReduceOp>(op))
        assignReduceCGALayout(reduce);
    });
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
