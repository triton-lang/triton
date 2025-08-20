#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp> allOps;
  funcOp->walk([&](scf::ForOp forOp) { allOps.push_back(forOp); });

  SmallVector<scf::ForOp> leafOps;
  for (scf::ForOp forOp : allOps) {
    auto searchResult = forOp.getBody()->walk(
        [](scf::ForOp) { return WalkResult::interrupt(); });
    if (!searchResult.wasInterrupted())
      leafOps.push_back(forOp);
  }
  return leafOps;
}

SmallVector<unsigned> getShapePerCTATile(RankedTensorType tensorTy) {
  auto llEnc = triton::gpu::toLinearEncoding(tensorTy);
  auto sizePerThread = llEnc.getSizePerThread();
  auto threadsPerWarp = llEnc.getThreadsPerWarp();
  auto warpsPerCTA = llEnc.getWarpsPerCTA();
  SmallVector<unsigned> shape;
  for (auto [size, thread, warp] :
       llvm::zip(sizePerThread, threadsPerWarp, warpsPerCTA)) {
    shape.push_back(size * thread * warp);
  }
  return shape;
}

} // namespace mlir::triton::AMD
