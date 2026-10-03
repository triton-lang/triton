

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

static bool isSm80FourWarpScalarByteLoad(Operation *op, int numWarps,
                                         RankedTensorType tensorType,
                                         ArrayRef<int64_t> contiguity) {
  if (!isa<triton::LoadOp>(op) || numWarps != 4 || tensorType.getRank() != 2 ||
      getElementBitWidth(tensorType) != 8 ||
      !llvm::all_of(contiguity, [](int64_t value) { return value == 1; }))
    return false;

  auto moduleOp = op->getParentOfType<ModuleOp>();
  auto targetAttr =
      moduleOp
          ? moduleOp->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName)
          : nullptr;
  return targetAttr && targetAttr.getValue().starts_with("cuda:") &&
         getNVIDIAComputeCapability(moduleOp) == 80;
}

BlockedEncodingAttr
buildCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp,
                       triton::gpu::CGAEncodingAttr cgaLayout,
                       SmallVector<int64_t> shapePerCTA) {
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

  auto matchesShape = [&refTensorType](const Value &val) {
    auto rttType = dyn_cast<RankedTensorType>(val.getType());
    return rttType && rttType.getShape() == refTensorType.getShape();
  };
  llvm::SetVector<Operation *> slice;
  if (ptr.getDefiningOp())
    slice = mlir::getSlice(op);

  if (isSm80FourWarpScalarByteLoad(op, numWarps, refTensorType, contiguity)) {
    // All-one i8 roots can inherit sizePerThread=16 from a coalesced peer
    // because their default orders tie. On SM80 with four warps this spills;
    // use the alternate order to keep the weak root scalar.
    bool borrowsWideLayout = llvm::any_of(slice, [&](Operation *use) {
      Value val = getMemAccessPtr(use);
      if (!val || use == op || !matchesShape(val))
        return false;
      auto currOrder = getOrderFromContiguity(
          axisInfoAnalysis.getAxisInfo(val)->getContiguity());
      return order == currOrder &&
             getNumElementsPerThread(use, order, axisInfoAnalysis,
                                     shapePerCTA) >= 16;
    });
    if (borrowsWideLayout)
      order = {0, 1};
  }
  LDBG("order=[" << triton::join(order, ", ") << "]");

  // The desired divisibility is the maximum divisibility among all dependent
  // pointers which have the same shape and order as `ptr`.
  llvm::SmallSetVector<Operation *, 32> memAccessesSameOrder;
  memAccessesSameOrder.insert(op);
  if (ptr.getDefiningOp()) {
    for (Operation *use : slice) {
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
  return BlockedEncodingAttr::get(op->getContext(), refTensorType.getShape(),
                                  sizePerThread, order, numWarps,
                                  threadsPerWarp, cgaLayout);
}
} // namespace mlir::triton::gpu
