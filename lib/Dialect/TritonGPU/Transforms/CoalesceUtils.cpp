

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
// Coalescing only needs membership in the bidirectional slice, not a
// topological ordering or a fresh transitive slice for every reached operation.
static llvm::SetVector<Operation *> getCoalescingSlice(Operation *root) {
  llvm::SetVector<Operation *> slice;
  slice.insert(root);
  for (unsigned i = 0; i < slice.size(); ++i) {
    Operation *op = slice[i];
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        slice.insert(def);
      } else if (auto arg = dyn_cast<BlockArgument>(operand)) {
        Operation *parent = arg.getOwner()->getParentOp();
        if (parent && !parent->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
            parent->getNumRegions() == 1 && parent->getRegion(0).hasOneBlock())
          slice.insert(parent);
      }
    }
    for (Operation *user : op->getUsers())
      slice.insert(user);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          slice.insert(&nested);
  }
  return slice;
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
  LDBG("order=[" << triton::join(order, ", ") << "]");

  auto matchesShape = [&refTensorType](const Value &val) {
    auto rttType = dyn_cast<RankedTensorType>(val.getType());
    return rttType && rttType.getShape() == refTensorType.getShape();
  };

  int numElems = product<int64_t>(shapePerCTA);
  int numThreads = numWarps * threadsPerWarp;
  unsigned limit = std::max(numElems / numThreads, 1);
  unsigned perThread =
      getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA);

  // Stores are capped by their own vector width, so other accesses cannot
  // improve their layout. Loads already at the per-thread limit also need
  // no slice analysis.
  if (isa<triton::LoadOp>(op) && perThread < limit && ptr.getDefiningOp()) {
    for (Operation *use : getCoalescingSlice(op)) {
      Value val = getMemAccessPtr(use);
      if (!val || !matchesShape(val) || use == op)
        continue;
      auto currOrder = getOrderFromContiguity(
          axisInfoAnalysis.getAxisInfo(val)->getContiguity());
      if (order != currOrder)
        continue;
      perThread = std::max(
          perThread,
          getNumElementsPerThread(use, order, axisInfoAnalysis, shapePerCTA));
      if (perThread >= limit)
        break;
    }
  }
  perThread = std::min(perThread, limit);
  SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
  sizePerThread[order[0]] = perThread;
  return BlockedEncodingAttr::get(op->getContext(), refTensorType.getShape(),
                                  sizePerThread, order, numWarps,
                                  threadsPerWarp, cgaLayout);
}
} // namespace mlir::triton::gpu
