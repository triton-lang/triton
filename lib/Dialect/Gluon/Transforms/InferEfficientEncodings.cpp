#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gluon-infer-efficient-encodings"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONINFEREFFICIENTENCODINGSPASS
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"

namespace {
unsigned getNumElementsPerThread(Operation *op, SmallVector<unsigned> order,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 SmallVector<int64_t> shapePerCTA) {
  Value val = getMemAccessPtr(op);
  auto ty = cast<RankedTensorType>(val.getType());
  AxisInfo &valInfo = *axisInfoAnalysis.getAxisInfo(val);
  unsigned elemNumBits = getElementBitWidth(ty);
  unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
  unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
  unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
  unsigned maxContig =
      std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
  unsigned alignment = std::min(maxMultiple, maxContig);
  unsigned currPerThread = std::min(alignment, 128 / elemNumBits);
  LDBG("elemNumBytes: " << elemNumBytes
                        << ", divisibility: " << maxMultipleBytes
                        << ", contig: " << valInfo.getContiguity(order[0])
                        << ", alignment: " << alignment);
  return currPerThread;
}

ttg::CTALayoutAttr getCTALayout(mlir::MLIRContext *ctx, unsigned rank) {
  return ttg::CTALayoutAttr::getDefault(ctx, rank);
}

} // anonymous namespace

class GluonInferEfficientEncodingsPass
    : public impl::GluonInferEfficientEncodingsPassBase<
          GluonInferEfficientEncodingsPass> {
  void
  setCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp,
                       llvm::MapVector<Operation *, Attribute> &layoutMap) {

    Value ptr = getMemAccessPtr(op);
    auto refTensorType = cast<RankedTensorType>(ptr.getType());

    // LDBG("Considering op: " << *op);
    // LLVM_DEBUG({
    //     DBGS() << "axis info of pointer: ";
    //     axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
    //     llvm::dbgs() << "\n";
    // });
    llvm::outs() << "\n";
    llvm::outs() << "Considering op: " << *op << "\n";
    llvm::outs() << "axis info of pointer: ";
    axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::outs());
    llvm::outs() << "\n";

    auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
    SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
    // LDBG("order=[" << triton::join(order, ", ") << "]");
    llvm::outs() << "order=[" << triton::join(order, ", ") << "]\n";

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

    // TODO: hardcode ctaSplitNum for now. read ctaSplitNum from frontend
    // options?
    unsigned rank = refTensorType.getShape().size();
    SmallVector<unsigned> ctaSplitNum(rank, 1);
    auto shapePerCTA =
        ttg::getShapePerCTA(ctaSplitNum, refTensorType.getShape());
    // auto shapePerCTA = ttg::getShapePerCTA(refTensorType);

    // LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");
    llvm::outs() << "shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]\n";

    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;

    unsigned perThread =
        getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA);
    // LDBG("perThread for op: " << perThread);
    llvm::outs() << "perThread for op: " << perThread << "\n";

    for (Operation *opSameOrder : memAccessesSameOrder) {
      if (opSameOrder == op)
        continue;
      unsigned currPerThread = getNumElementsPerThread(
          opSameOrder, order, axisInfoAnalysis, shapePerCTA);
      // LDBG("perThread for opSameOrder: " << currPerThread);
      llvm::outs() << "perThread for opSameOrder: " << currPerThread << "\n";
      perThread = std::max(perThread, currPerThread);
    }

    perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
    // LDBG("perThread: " << perThread);
    llvm::outs() << "perThread: " << perThread << "\n";

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

    // TODO: hardcode for now. read ctaSplitNum from frontend options?
    // auto CTALayout = triton::gpu::getCTALayout(refTensorType.getEncoding());
    auto CTALayout =
        getCTALayout(&getContext(), refTensorType.getShape().size());
    layoutMap[op] = triton::gpu::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp, CTALayout);
  }

  void runOnOperation() override {
    llvm::outs() << "\n";
    llvm::outs() << "\n";

    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // 1. for every load/store with efficient encoding,
    // infer efficient encoding for ptrs

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    llvm::MapVector<Operation *, Attribute> layoutMap;
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
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

      // we only consider those with efficient encoding
      if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
        auto encoding = tensorType.getEncoding();
        if (!encoding || !isa<gluon::EfficientEncodingAttr>(encoding))
          return;
      }

      int numWarps = ttg::lookupNumWarps(curr);
      setCoalescedEncoding(axisInfoAnalysis, curr, numWarps, threadsPerWarp,
                           layoutMap);
    });

    llvm::outs() << "\n";
    llvm::outs() << "[INFERRED LAYOUTS]:\n";
    for (auto &pair : layoutMap) {
      Operation *op = pair.first;
      Attribute layout = pair.second;
      llvm::outs() << "inferred layout for op: " << *op << "\n";
      layout.print(llvm::outs());
      llvm::outs() << "\n";
    }

    llvm::outs() << "[END] Inferred layouts:\n";
    llvm::outs() << "\n";
    llvm::outs() << "\n";

    // 2. propagate upstream/downstream, raise whenever conflicts?
    llvm::outs() << "[PROPAGATED]:\n";
    llvm::outs() << "\n";
    llvm::outs() << "\n";
  }
};
} // namespace mlir::triton::gluon
