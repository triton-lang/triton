#include "StreamOps.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

// Shared skeleton for building a TDM chain:
//   <buffer view> = createSingleBufferView(...)
//   <tdmOp>       = buildTDMOp(builder, view, pred)   <- caller-supplied
//   <waitOp>      = amdgpu::AsyncTDMWait(tdmOp)
//   replace uses of original load with a local_load after waitOp
static TDMChainOps createTDMAsync(
    Operation *origLoad, Value alloc, Value extractIdx,
    function_ref<Operation *(OpBuilder &, Location, Value, Value)> buildTDMOp) {
  OpBuilder builder(origLoad);
  Location loc = origLoad->getLoc();

  Value pred = arith::ConstantIntOp::create(builder, loc, 1, 32);

  auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
                      .getDefiningOp<ttg::MemDescIndexOp>();

  Operation *tdmOp = buildTDMOp(builder, loc, viewLoad, pred);

  auto waitOp = triton::amdgpu::AsyncTDMWait::create(builder, loc,
                                                     tdmOp->getResult(0), 0);

  auto maybeSharedLoad = tt::replaceUsesWithLocalLoad(
      builder, origLoad->getResult(0), viewLoad, waitOp);

  return {tdmOp, waitOp, maybeSharedLoad};
}

TDMChainOps createTDMAsyncCopy(tt::DescriptorLoadOp loadOp, Value alloc,
                               Value extractIdx) {
  return createTDMAsync(
      loadOp, alloc, extractIdx,
      [&](OpBuilder &builder, Location loc, Value view, Value pred) {
        Value desc = createUpdateTDMDescriptorOp(builder, loc, loadOp.getDesc(),
                                                 loadOp.getIndices(),
                                                 /*pred=*/pred);
        return triton::amdgpu::AsyncTDMCopyGlobalToLocalOp::create(builder, loc,
                                                                   desc, view);
      });
}

TDMChainOps createTDMAsyncGather(tt::DescriptorGatherOp gatherOp, Value alloc,
                                 Value extractIdx) {
  return createTDMAsync(
      gatherOp, alloc, extractIdx,
      [&](OpBuilder &builder, Location loc, Value view, Value pred) {
        // Convert to the TDM index layout for better gather performance;
        // insert a layout convert if the indices don't already have it.
        auto indices = gatherOp.getXOffsets();
        auto indicesType = cast<RankedTensorType>(indices.getType());
        auto idxEnc = getTDMGatherScatterIndexEncoding(gatherOp, indicesType);
        if (indicesType.getEncoding() != idxEnc) {
          auto newIdxType = RankedTensorType::get(
              indicesType.getShape(), indicesType.getElementType(), idxEnc);
          indices =
              ttg::ConvertLayoutOp::create(builder, loc, newIdxType, indices);
        }
        Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
        Value desc = createUpdateTDMDescriptorOp(
            builder, loc, gatherOp.getDesc(), {zero, gatherOp.getYOffset()},
            /*pred=*/pred);
        return triton::amdgpu::AsyncTDMGatherOp::create(builder, loc, desc,
                                                        indices, view);
      });
}

AsyncCopyChainOps createAsyncCopy(tt::LoadOp loadOp, Value alloc,
                                  Value extractIdx, int contiguity) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  // Extract local subview from shared allocation
  auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
                      .getDefiningOp<ttg::MemDescIndexOp>();

  auto copyOp = ttg::AsyncCopyGlobalToLocalOp::create(
      builder, loc, loadOp.getPtr(), viewLoad, loadOp.getMask(),
      loadOp.getOther(), loadOp.getCachePolicyAttr(), loadOp.getIsVolatile(),
      contiguity);
  auto commitOp =
      ttg::AsyncCommitGroupOp::create(builder, loc, copyOp->getResult(0));
  ttg::AsyncWaitOp waitOp =
      ttg::AsyncWaitOp::create(builder, loc, commitOp->getResult(0), 0);

  auto maybeSharedLoad = tt::replaceUsesWithLocalLoad(
      builder, loadOp->getResult(0), viewLoad, waitOp);

  return {copyOp, commitOp, waitOp, maybeSharedLoad};
}

StreamCopyChainOps createStreamCopy(tt::LoadOp loadOp, Value alloc,
                                    Value extractIdx) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  // Extract local subview from shared allocation
  auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
                      .getDefiningOp<ttg::MemDescIndexOp>();

  tt::LoadOp newLoadOp = cast<tt::LoadOp>(builder.clone(*loadOp));
  auto storeOp = ttg::LocalStoreOp::create(builder, loc, newLoadOp, viewLoad);
  auto maybeLocalLoad =
      tt::replaceUsesWithLocalLoad(builder, loadOp->getResult(0), viewLoad);

  return {newLoadOp, viewLoad, storeOp, maybeLocalLoad};
}

} // namespace mlir
