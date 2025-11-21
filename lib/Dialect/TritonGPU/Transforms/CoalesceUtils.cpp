

#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {
BlockedEncodingAttr buildCoalescedEncoding(
    MLIRContext *context, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    Operation *op, int numWarps, int threadsPerWarp,
    triton::gpu::CTAEncodingAttr CTALayout, SmallVector<int64_t> shapePerCTA) {
  Value ptr = getMemAccessPtr(op);
  auto refTensorType = cast<RankedTensorType>(ptr.getType());

  LDBG("Considering op: " << *op);
  LLVM_DEBUG({
    DBGS() << "axis info of pointer: ";
    axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
  SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
  LDBG("order=[" << triton::join(order, ", ") << "]");

  auto matchesShape = [&refTensorType](const Value &val) {
    auto rttType = dyn_cast<RankedTensorType>(val.getType());
    return rttType && rttType.getShape() == refTensorType.getShape();
  };

  // The desired divisibility is the maximum divisibility among all dependent
  // pointers which have the same shape and order as `ptr`.
  llvm::SmallSetVector<Operation *, 32> memAccessesSameOrder;
  memAccessesSameOrder.insert(op);
  if (ptr.getDefiningOp()) {
    for (Operation *use : mlir::getSlice(op)) {
      Value val = getMemAccessPtr(use);
      if (!val || !matchesShape(val) || memAccessesSameOrder.contains(use))
        continue;
      auto currOrder = getOrderFromContiguity(
          axisInfoAnalysis.getAxisInfo(val)->getContiguity());
      if (order == currOrder) {
        LDBG("multi-root-slice: insert to memAccessesSameOrder " << *use);
        memAccessesSameOrder.insert(use);
      }
    }
  }

  LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");

  int numElems = product<int64_t>(shapePerCTA);
  int numThreads = numWarps * threadsPerWarp;

  unsigned perThread =
      getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA);
  LDBG("perThread for op: " << perThread);

  for (Operation *opSameOrder : memAccessesSameOrder) {
    if (opSameOrder == op)
      continue;
    unsigned currPerThread = getNumElementsPerThread(
        opSameOrder, order, axisInfoAnalysis, shapePerCTA);
    LDBG("perThread for opSameOrder: " << currPerThread);
    perThread = std::max(perThread, currPerThread);
  }

  perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
  LDBG("perThread: " << perThread);

  if (!dyn_cast<triton::LoadOp>(op)) {
    // For ops that can result in a global memory write, we should enforce
    // that each thread handles at most 128 bits, which is the widest
    // available vectorized store op; otherwise, the store will have "gaps"
    // in the memory write at the warp level, resulting in worse performance.
    // For loads, we can expect that the gaps won't matter due to the L1
    // cache.
    perThread = std::min<int>(
        perThread,
        getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA));
  }
  SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
  sizePerThread[order[0]] = perThread;
  return BlockedEncodingAttr::get(context, refTensorType.getShape(),
                                  sizePerThread, order, numWarps,
                                  threadsPerWarp, CTALayout);
}
} // namespace mlir::triton::gpu
