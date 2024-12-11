#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

static RankedTensorType replaceEncoding(RankedTensorType oldType,
                                        Attribute newEncoding) {
  return RankedTensorType::get(oldType.getShape(), oldType.getElementType(),
                               newEncoding);
}

// This function considers a gather op in isolation and attemps to determine
// whether an optimized layout can be applied to the source and index tensors.
static void setOptimizedGatherLayout(GatherOp op) {
  RankedTensorType srcType = op.getSrc().getType();
  RankedTensorType idxType = op.getIndices().getType();

  // HACK: Heurstic to avoid a performance footgun: a layout that conflicts with
  // load coalescing will trigger a layout conversion, so try to avoid this
  // scenario by checking the order. We should replace this when layout
  // assignment is a proper constraint system.
  auto distributedItf = cast<DistributedEncodingTrait>(srcType.getEncoding());
  unsigned axis = op.getAxis();
  //if (distributedItf.getThreadOrder().front() != axis)
  //  return;

  // Determine a warp-local gather layout that minimizes the number of emitted
  // warp shuffles.
  unsigned numThreadsPerWarp =
      product<unsigned>(triton::gpu::getThreadsPerWarp(srcType.getEncoding()));
  unsigned numWarps =
      product<unsigned>(triton::gpu::getWarpsPerCTA(srcType.getEncoding()));

  // If in a gather column, each thread owns `srcSizePerThread[axis]` elements
  // in the source tensor and `idxSizePerThread[axis]` elements in the index
  // tensor (including broadcasting), then the number of index shuffles per
  // column is `srcSizePerThread[axis] * idxSizePerThread[axis]`. This is then
  // replicated over the number of columns in which a thread owns (an equal
  // number of) elements, which is `product(srcSizePerThread[i] for i != axis)`.
  //
  // Thus, the total number of index shuffles is `product(srcSizePerThread) *
  // idxSizePerThread[axis]`. Since we cannot alter the number of threads per
  // warp or the number of warps, `product(srcSizePerThread)` is just a function
  // of the shape.
  //
  // So we want to minimize `idxSizePerThread[axis]`. Note that broadcasting is
  // forbidden in the source tensor but allowed in the index tensor. Choose the
  // smallest value while still ensuring that a warp spans whole columns.
  //
  // In order to prevent broadcasting in the source tensor layout, ensure
  //
  //   sizePerThread(i) * threadsPerWarp(i) * warpsPerCTA(i) = shape(i)
  //
  // For all i != axis in the source tensor. The same relationship must hold for
  // the index tensor. This means we can't just set `idxSizePerThread[axis]` to
  // 1 and compute the rest from that. Find the smallest value where this
  // relationship is still respected.

  // We know that the layouts will be the same between the two tensors except
  // for `sizePerThread[axis]`.
  unsigned rank = srcType.getRank();
  SmallVector<unsigned> threadsPerWarp(rank);
  SmallVector<unsigned> warpsPerCTA(rank);
  SmallVector<unsigned> order;
  order.push_back(axis);

  // Minimize `sizePerThread[axis]` by putting as many theads along the axis as
  // possible, limited to the actual size of the dimension.
  unsigned maxThreadsInAxis =
      std::min<unsigned>(srcType.getDimSize(axis), numThreadsPerWarp);
  threadsPerWarp[axis] = maxThreadsInAxis;

  // Now spread them along the other dimensions. Do this according to order
  // (arbitrary).
  unsigned threadsToAlloc = numThreadsPerWarp / maxThreadsInAxis;
  for (unsigned dim : distributedItf.getThreadOrder()) {
    if (dim == axis)
      continue;
    order.push_back(dim);
    unsigned nextThreadAlloc =
        std::min<unsigned>(srcType.getDimSize(dim), threadsToAlloc);
    threadsPerWarp[dim] = nextThreadAlloc;
    threadsToAlloc /= nextThreadAlloc;
  }
  assert(llvm::none_of(threadsPerWarp, [](unsigned c) { return c == 0; }));

  // There must be one warp along the gather axis.
  warpsPerCTA[axis] = 1;
  // Allocate the remaining warps in the same manner.
  unsigned warpsToAlloc = numWarps;
  for (unsigned dim : distributedItf.getWarpOrder()) {
    if (dim == axis)
      continue;
    unsigned warpsCanFit = srcType.getDimSize(dim) / threadsPerWarp[dim];
    assert(warpsCanFit != 0);
    unsigned nextWarpAlloc = std::min<unsigned>(warpsCanFit, warpsToAlloc);
    warpsPerCTA[dim] = nextWarpAlloc;
    warpsToAlloc /= nextWarpAlloc;
  }
  assert(llvm::none_of(warpsPerCTA, [](unsigned c) { return c == 0; }));

  // Just set `sizePerThread` to 1 along other dimensions and let broadcasting
  // handling it.
  SmallVector<unsigned> sizePerThread(rank, 1);
  sizePerThread[axis] = srcType.getDimSize(axis) / threadsPerWarp[axis];

  // Overflow by broadcasting along the gather axis since this is the most
  // predictable.
  threadsPerWarp[axis] *= threadsToAlloc;
  warpsPerCTA[axis] *= warpsToAlloc;

  assert(product(threadsPerWarp) == numThreadsPerWarp);
  assert(product(warpsPerCTA) == numWarps);

  // Construct the new layout.
  MLIRContext *ctx = srcType.getContext();
  auto ctaLayout = CTALayoutAttr::get(ctx, distributedItf.getCTAsPerCGA(),
                                      distributedItf.getCTASplitNum(),
                                      distributedItf.getCTAOrder());
  auto newLayout =
      BlockedEncodingAttr::get(ctx, sizePerThread, threadsPerWarp, warpsPerCTA,
                              order, ctaLayout);

  // Update the layout on the gather op and insert conversions.
  mlir::ImplicitLocOpBuilder b(op.getLoc(), op);
  auto cvtSrc = b.create<ConvertLayoutOp>(replaceEncoding(srcType, newLayout),
                                          op.getSrc());
  auto cvtIdx = b.create<ConvertLayoutOp>(replaceEncoding(idxType, newLayout),
                                          op.getIndices());
  op.getSrcMutable().set(cvtSrc);
  op.getIndicesMutable().set(cvtIdx);

  b.setInsertionPointAfter(op);
  auto cvtOut = b.create<ConvertLayoutOp>(op.getType(), op.getResult());
  op.getResult().replaceAllUsesExcept(cvtOut, cvtOut);
  op.getResult().setType(replaceEncoding(op.getType(), newLayout));

  // Mark the layout as optimized on the op to prevent it from being changed.
  op.setEfficientLayout(true);

  // Make sure we did this right.
  assert(GatherLoweringHelper(op).isWarpLocal());
}

#define GEN_PASS_DEF_TRITONGPUOPTIMIZEGATHERLAYOUTS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {
struct OptimizeGatherLayoutsPass
    : public impl::TritonGPUOptimizeGatherLayoutsBase<
          OptimizeGatherLayoutsPass> {
  void runOnOperation() override {
    getOperation().walk(setOptimizedGatherLayout);
  }
};
} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
