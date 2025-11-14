#include <iterator>
#include <numeric>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
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

      auto tensorType = cast<RankedTensorType>(ptr.getType());
      CTALayoutAttr ctaLayout = getCTALayout(tensorType.getEncoding());
      SmallVector<int64_t> shapePerCTA = getShapePerCTA(tensorType);
      auto layout = buildCoalescedEncoding(&getContext(), axisInfoAnalysis,
                                           curr, numWarps, threadsPerWarp,
                                           ctaLayout, shapePerCTA);
      layoutMap[curr] = layout;
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
