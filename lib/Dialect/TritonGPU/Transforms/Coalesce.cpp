#include <iterator>
#include <numeric>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUCOALESCE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// Descriptor load/stores don't need to consider L1 coalescing but the
// destination layout will affect the shared memory load/store generated. So we
// still want to allow vectorization for the src/destination layout up to
// 16bytes.
static Attribute pickDescriptorLoadStoreLayout(int numWarps, int threadsPerWarp,
                                               RankedTensorType type) {
  auto shapePerCTA = triton::gpu::getShapePerCTA(type);
  int numElems = product<int64_t>(shapePerCTA);
  int numThreads = numWarps * threadsPerWarp;
  int numElemsPerThread = std::max(numElems / numThreads, 1);

  int maxVectorSize = 128 / type.getElementTypeBitWidth();

  int vectorSize = std::min(numElemsPerThread, maxVectorSize);
  SmallVector<unsigned> sizePerThread(type.getRank(), 1);
  sizePerThread.back() = vectorSize;

  SmallVector<unsigned> order =
      getMatrixOrder(type.getRank(), /*rowMajor*/ true);
  auto CTALayout = triton::gpu::getCTALayout(type.getEncoding());

  Attribute layout = triton::gpu::BlockedEncodingAttr::get(
      type.getContext(), type.getShape(), sizePerThread, order, numWarps,
      threadsPerWarp, CTALayout);
  return layout;
}

static void pickDescriptorLoadStoreLayout(
    ModuleOp moduleOp, llvm::MapVector<Operation *, Attribute> &layoutMap) {
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(moduleOp);
  moduleOp.walk([&](Operation *op) {
    int numWarps = lookupNumWarps(op);
    if (auto load = dyn_cast<DescriptorOpInterface>(op)) {
      if (load->getNumResults() == 1)
        layoutMap[op] = pickDescriptorLoadStoreLayout(
            numWarps, threadsPerWarp,
            cast<RankedTensorType>(load->getResult(0).getType()));
    }
    if (auto store = dyn_cast<DescriptorStoreLikeOpInterface>(op)) {
      layoutMap[op] = pickDescriptorLoadStoreLayout(numWarps, threadsPerWarp,
                                                    store.getSrc().getType());
    }
  });
}

struct CoalescePass : public impl::TritonGPUCoalesceBase<CoalescePass> {

  void
  emitLowPerThreadRemarksOnAxisInfo(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                    Operation *op, Value memoryAccessPtr) {
    auto mainError = op->emitRemark()
                     << "When computing coalesced encoding, only one element "
                        "per thread is assigned with information from axis "
                        "info analysis. Performance may be suboptimal.";

    llvm::SetVector<Operation *> backwardSlice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = false;
    opt.filter = [&axisInfoAnalysis](Operation *op) {
      bool allDivisibilityOne = true;
      // check if results are all with divisibility 1
      for (auto result : op->getResults()) {
        auto axisInfo = axisInfoAnalysis.getAxisInfo(result);
        auto divisibility = axisInfo->getDivisibility();
        auto divisibilityIsOne = product<int64_t>(divisibility) == 1;
        allDivisibilityOne = allDivisibilityOne && divisibilityIsOne;
      }
      return allDivisibilityOne;
    };
    getBackwardSlice(op, &backwardSlice, opt);

    auto divisibility =
        axisInfoAnalysis.getAxisInfo(memoryAccessPtr)->getDivisibility();
    auto contiguity =
        axisInfoAnalysis.getAxisInfo(memoryAccessPtr)->getContiguity();
    // check if divisibility in all dimensions is 1
    auto divisibilityIsOne = product<int64_t>(divisibility) == 1;

    auto contiguityIsOne = product<int64_t>(contiguity) == 1;

    if (divisibilityIsOne) {
      mainError.attachNote()
          << "The divisibility of the pointer is 1 in all dimensions. ";
      for (auto sliceOp : backwardSlice) {
        bool operandWithDivisibilityOne = false;
        for (auto operand : sliceOp->getOperands()) {
          auto axisInfo = axisInfoAnalysis.getAxisInfo(operand);
          if (product<int64_t>(axisInfo->getDivisibility()) == 1) {
            operandWithDivisibilityOne = true;
            break;
          }
        }
        bool resultWithDivisibilityOne = false;
        if (sliceOp->getNumResults() > 0) {
          auto lhsValue = sliceOp->getResult(0);
          auto axisInfo = axisInfoAnalysis.getAxisInfo(lhsValue);
          resultWithDivisibilityOne =
              product<int64_t>(axisInfo->getDivisibility()) == 1;
        }
        if (!operandWithDivisibilityOne && resultWithDivisibilityOne) {
          // ignore certain ops
          if (isa<triton::GetProgramIdOp>(sliceOp) ||
              isa<arith::ConstantOp>(sliceOp)) {
            continue;
          }
          mainError.attachNote(sliceOp->getLoc())
              << "Divisibility of 1 first introduced here: " << *sliceOp;
          if (isa<triton::LoadOp>(sliceOp)) {
            mainError.attachNote(sliceOp->getLoc())
                << "tt.load resets divisibility. Consider add "
                   "`tt.multiple_of` if you believe it is correct for the "
                   "data.";
          } else if (isa<arith::DivUIOp>(sliceOp) ||
                     isa<arith::DivSIOp>(sliceOp)) {
            mainError.attachNote(sliceOp->getLoc())
                << "Division resets divisibility. Consider add "
                   "`tt.multiple_of` if you believe it is correct for the "
                   "data.";
          }
          // TODO: we can add more ops here. Still looking for examples for
          // converging to 1 with GCD and multiplications.
        }
      }
    }
    if (contiguityIsOne) {
      mainError.attachNote()
          << "The contiguity of the pointer is 1 in all dimensions.";
    }
  }

  void
  setCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp,
                       llvm::MapVector<Operation *, Attribute> &layoutMap) {
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
      for (Operation *use : mlir::multiRootGetSlice(op)) {
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

    auto shapePerCTA = triton::gpu::getShapePerCTA(refTensorType);
    LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");

    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;

    unsigned perThread = getNumElementsPerThread(op, order, axisInfoAnalysis);
    LDBG("perThread for op: " << perThread);

    for (Operation *opSameOrder : memAccessesSameOrder) {
      if (opSameOrder == op)
        continue;
      unsigned currPerThread =
          getNumElementsPerThread(opSameOrder, order, axisInfoAnalysis);
      LDBG("perThread for opSameOrder: " << currPerThread);
      perThread = std::max(perThread, currPerThread);
    }
    LDBG("perThread after max: " << perThread);
    LDBG("numElems: " << numElems);
    LDBG("numThreads: " << numThreads);
    auto perThreadFromExecutionConfig = std::max(numElems / numThreads, 1);
    auto perThreadFromAxisInfo = perThread;
    if (perThreadFromAxisInfo == 1) {
      emitLowPerThreadRemarksOnAxisInfo(axisInfoAnalysis, op, ptr);
    }
    perThread =
        std::min<int>(perThreadFromAxisInfo, perThreadFromExecutionConfig);
    LDBG("perThread: " << perThread);

    if (!dyn_cast<triton::LoadOp>(op)) {
      // For ops that can result in a global memory write, we should enforce
      // that each thread handles at most 128 bits, which is the widest
      // available vectorized store op; otherwise, the store will have "gaps"
      // in the memory write at the warp level, resulting in worse performance.
      // For loads, we can expect that the gaps won't matter due to the L1
      // cache.
      perThread = std::min<int>(
          perThread, getNumElementsPerThread(op, order, axisInfoAnalysis));
    }
    SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
    sizePerThread[order[0]] = perThread;

    auto CTALayout = triton::gpu::getCTALayout(refTensorType.getEncoding());
    layoutMap[op] = triton::gpu::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp, CTALayout);
  }

  static Type getNewType(Type type, Attribute encoding) {
    RankedTensorType tensorType = cast<RankedTensorType>(type);
    return tensorType.cloneWithEncoding(encoding);
  }

  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    llvm::MapVector<Operation *, Attribute> layoutMap;
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(moduleOp);
    moduleOp.walk([&](Operation *curr) {
      Value ptr = getMemAccessPtr(curr);
      if (!ptr)
        return;
      // We only convert `tensor<tt.ptr<>>` load/store
      bool isPtrTensor = false;
      if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
        isPtrTensor = isa<PointerType>(tensorType.getElementType());
      if (!isPtrTensor)
        return;
      int numWarps = lookupNumWarps(curr);
      setCoalescedEncoding(axisInfoAnalysis, curr, numWarps, threadsPerWarp,
                           layoutMap);
    });

    // Also pick a layout for descriptor load/store ops.
    pickDescriptorLoadStoreLayout(moduleOp, layoutMap);

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    for (auto &kv : layoutMap) {
      convertDistributedOpEncoding(kv.second, kv.first);
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
