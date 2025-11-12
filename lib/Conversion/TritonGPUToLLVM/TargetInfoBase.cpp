#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {

std::optional<int>
TargetInfoBase::getWarpGroupStartThreadId(Block *block) const {
  using namespace triton::gpu;

  // Look for an enclosing `ttg.warp_specialize` op.
  while (block && block->getParentOp() &&
         !isa<WarpSpecializePartitionsOp>(block->getParentOp()))
    block = block->getParentOp()->getBlock();
  if (!block || !block->getParentOp())
    return {};

  auto partitions = cast<WarpSpecializePartitionsOp>(block->getParentOp());
  unsigned idx = block->getParent()->getRegionNumber();
  WarpSpecializeOp ws = partitions.getParentOp();
  std::optional<ArrayRef<int32_t>> startIds = ws.getWarpGroupStartIds();
  assert(startIds && "cannot get warp group ID before warp group allocation");
  int32_t warpStartId = (*startIds)[idx];
  int threadsPerWarp =
      TritonGPUDialect::getThreadsPerWarp(ws->getParentOfType<ModuleOp>());
  return warpStartId * threadsPerWarp;
}

Value TargetInfoBase::getThreadId(RewriterBase &rewriter, Location loc) const {
  Value tid =
      ::mlir::gpu::ThreadIdOp::create(rewriter, loc, ::mlir::gpu::Dimension::x);
  tid = arith::IndexCastOp::create(rewriter, loc, rewriter.getIntegerType(32),
                                   tid);

  Operation *lookupPt = &rewriter.getInsertionBlock()->front();
  int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
  int numWarps = triton::gpu::lookupNumWarps(lookupPt);
  int upperBound = numWarps * threadsPerWarp;

  TritonLLVMOpBuilder b(loc, rewriter);

  // If this is being created inside a warp specialize op, compute the relative
  // thread ID within the warp group.
  if (std::optional<int> startId =
          getWarpGroupStartThreadId(rewriter.getInsertionBlock())) {
    tid = arith::SubIOp::create(rewriter, loc, tid, b.i32_val(*startId));
  }

  assert(llvm::isPowerOf2_32(upperBound));
  // help LLVM's known bits analysis:
  tid = b.and_(tid, b.i32_val(upperBound - 1));

  return tid;
}

Value TargetInfoBase::getLaneId(RewriterBase &rewriter, Location loc) const {
  return getLaneAndWarpId(rewriter, loc).first;
}

std::pair<Value, Value> TargetInfoBase::getLaneAndWarpId(RewriterBase &rewriter,
                                                         Location loc) const {
  TritonLLVMOpBuilder b(loc, rewriter);
  Value tid = getThreadId(rewriter, loc);
  int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
  Value warpSizeVal = b.i32_val(threadsPerWarp);

  // If there is only one warp, the warp ID is always 0.
  Operation *lookupPt = &rewriter.getInsertionBlock()->front();
  Value laneId;
  Value warpId;
  if (triton::gpu::lookupNumWarps(lookupPt) == 1) {
    laneId = tid;
    warpId = b.i32_val(0);
  } else {
    laneId = b.urem(tid, warpSizeVal);
    warpId = b.udiv(tid, warpSizeVal);
  }

  return {laneId, warpId};
}

} // namespace mlir::triton
