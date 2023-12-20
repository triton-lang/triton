#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include <iterator>
#include <numeric>

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

template <class T> SmallVector<unsigned, 4> argSort(const T &arr) {
  SmallVector<unsigned, 4> ret(arr.size());
  std::iota(ret.begin(), ret.end(), 0);
  std::stable_sort(ret.begin(), ret.end(),
                   [&](unsigned x, unsigned y) { return arr[x] > arr[y]; });
  return ret;
}

// Type of val can be either Tensor Pointer or Tensor.
static RankedTensorType getTensorType(const Value &val) {
  auto valType = val.getType();
  if (valType.isa<PointerType>())
    valType = valType.cast<PointerType>().getPointeeType();
  return valType.cast<RankedTensorType>();
}

unsigned getElementBitWidth(const Value &val) {
  auto tensorType = getTensorType(val);

  auto typeForMem =
      tensorType.getElementType().isa<PointerType>()
          ? tensorType.getElementType().cast<PointerType>().getPointeeType()
          : tensorType.getElementType();
  return typeForMem.getIntOrFloatBitWidth();
}

static Value getMemAccessPtr(Operation *op) {
  if (auto ld = dyn_cast<triton::LoadOp>(op))
    return ld.getPtr();
  if (auto atomic = dyn_cast<triton::AtomicRMWOp>(op))
    return atomic.getPtr();
  if (auto atomic = dyn_cast<triton::AtomicCASOp>(op))
    return atomic.getPtr();
  if (auto insert = dyn_cast<triton::gpu::InsertSliceAsyncOp>(op))
    return insert.getSrc();
  if (auto store = dyn_cast<triton::StoreOp>(op))
    return store.getPtr();
  return nullptr;
}

// TODO(Keren): integrate it into AxisInfoAnalysis
static AxisInfo getAxisInfoForTensorPointer(const Value &val) {
  auto valType = val.getType();
  // TODO(Chenggang): encoding for tensor pointers is meaningless, remove
  // these later while merging into the GitHub main
  auto ptrType = valType.cast<PointerType>();
  auto tensorTy = ptrType.getPointeeType().cast<RankedTensorType>();
  auto makeTensorPtr = getMakeTensorPtrOp(val);
  auto order = makeTensorPtr.getOrder();
  auto tileShape = triton::gpu::getShapePerCTA(tensorTy);
  size_t rank = order.size();
  auto elemSizeInBytes = tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
  SmallVector<int64_t> contiguity(rank, 1);
  SmallVector<int64_t> divisibility(rank, 1);
  SmallVector<int64_t> constancy(rank, 1);
  // The contiguity in `order[0]` is `tileShape[order[0]]`
  // The divisibility in `order[0]` is 16
  // TODO[goostavz]: confirm the legality of it
  contiguity[order[0]] = tileShape[order[0]];
  divisibility[order[0]] = 16 * 8 / elemSizeInBytes;
  return AxisInfo(contiguity, divisibility, constancy);
}

struct CoalescePass : public TritonGPUCoalesceBase<CoalescePass> {
  void
  setCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp,
                       llvm::MapVector<Operation *, Attribute> &layoutMap) {
    Value ptr = getMemAccessPtr(op);
    auto refTensorType = getTensorType(ptr);

    // Get the contiguity order of `ptr`
    SmallVector<unsigned> order;
    LDBG("op is: " << *op);
    if (ptr.getType().isa<PointerType>()) {
      auto makeTensorPtr = getMakeTensorPtrOp(ptr);
      std::copy(makeTensorPtr.getOrder().begin(),
                makeTensorPtr.getOrder().end(), std::back_inserter(order));
    } else {
      // Normal cases
      auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
      order = argSort(contiguity);
      LLVM_DEBUG({
        DBGS() << "contiguity is: ";
        for (const auto &O : contiguity) {
          llvm::dbgs() << O << " ";
        }
        llvm::dbgs() << "\n";
      });
    }
    LLVM_DEBUG({
      DBGS() << "order is: ";
      for (const auto &O : order) {
        llvm::dbgs() << O << " ";
      }
      llvm::dbgs() << "\n";
    });

    auto matchesShape = [&refTensorType](const Value &val) {
      if (val.getType() == refTensorType) {
        return true;
      }

      auto rttType = val.getType().dyn_cast<RankedTensorType>();
      return rttType ? rttType.getShape() == refTensorType.getShape() : false;
    };

    // The desired divisibility is the maximum divisibility
    // among all dependent pointers who have the same order as
    // `ptr`.
    // We only do it for normal tensors of pointers, not tensor pointers.
    llvm::SmallSetVector<Operation *, 32> memAccessesSameOrder;
    memAccessesSameOrder.insert(op);
    if (ptr.getDefiningOp()) {
      for (Operation *use : mlir::multiRootGetSlice(op)) {
        Value val = getMemAccessPtr(use);
        if (!val || !matchesShape(val) || memAccessesSameOrder.contains(use))
          continue;
        auto currOrder =
            argSort(axisInfoAnalysis.getAxisInfo(val)->getContiguity());
        if (order == currOrder) {
          LDBG("multi-root-slice: insert to memAccessesSameOrder " << *use);
          memAccessesSameOrder.insert(use);
        }
      }
    }

    auto shapePerCTA = triton::gpu::getShapePerCTA(refTensorType);
    LLVM_DEBUG({
      DBGS() << "shapePerCTA is ";
      for (const auto &O : shapePerCTA) {
        llvm::dbgs() << O << " ";
      }
      llvm::dbgs() << "\n";
    });
    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;
    int numElemsPerThread = std::max(numElems / numThreads, 1);

    // For tensor of pointers, the element to access is the pointee type;
    // while for tensor pointer type (`refTensorType` is directly the final
    // shape), the element to access is itself.
    auto typeForMem = refTensorType.getElementType().isa<PointerType>()
                          ? refTensorType.getElementType()
                                .cast<PointerType>()
                                .getPointeeType()
                          : refTensorType.getElementType();

    auto getNumElementPerThread = [&](Operation *op) {
      Value val = getMemAccessPtr(op);
      AxisInfo valInfo;
      if (val.getType().isa<PointerType>()) {
        valInfo = getAxisInfoForTensorPointer(val);
      } else {
        assert(val.getType().isa<RankedTensorType>());
        valInfo = *axisInfoAnalysis.getAxisInfo(val);
      }
      unsigned elemNumBits = getElementBitWidth(val);
      unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
      unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
      unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
      unsigned maxContig =
          std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
      unsigned alignment = std::min(maxMultiple, maxContig);
      unsigned currPerThread = std::min(alignment, 128 / elemNumBits);
      return currPerThread;
    };
    unsigned perThread = getNumElementPerThread(op);
    LDBG("perThread for op: " << perThread);
    for (Operation *opSameOrder : memAccessesSameOrder) {
      if (opSameOrder == op)
        continue;
      unsigned currPerThread = getNumElementPerThread(opSameOrder);
      LDBG("perThread for opSameOrder: " << currPerThread);
      perThread = std::max(perThread, currPerThread);
    }

    perThread = std::min<int>(perThread, numElemsPerThread);
    LDBG("perThread: " << perThread);

    if (!dyn_cast<triton::LoadOp>(op)) {
      // For ops that can result in a global memory write, we should enforce
      // that each thread handles at most 128 bits, which is the widest
      // available vectorized store op; otherwise, the store will have "gaps"
      // in the memory write at the warp level, resulting in worse performance.
      // For loads, we can expect that the gaps won't matter due to the L1
      // cache.
      unsigned elemNumBits = getElementBitWidth(ptr);
      perThread = std::min<int>(perThread, getNumElementPerThread(op));
    }
    SmallVector<unsigned, 4> sizePerThread(refTensorType.getRank(), 1);
    sizePerThread[order[0]] = perThread;

    auto CTALayout = triton::gpu::getCTALayout(refTensorType.getEncoding());
    layoutMap[op] = triton::gpu::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp, CTALayout);
  }

  static Type getNewType(Type type, Attribute encoding) {
    RankedTensorType tensorType = type.cast<RankedTensorType>();
    return RankedTensorType::get(tensorType.getShape(),
                                 tensorType.getElementType(), encoding);
  }

  void coalesceOp(Attribute encoding, Operation *op) {
    OpBuilder builder(op);
    // Convert operands
    // For load/store with tensor pointers, we don't have to change the
    // operands' type, we do this by changing the outputs' type of
    // `make_tensor_ptr`
    SmallVector<Value, 4> newArgs;
    for (auto operand : op->getOperands()) {
      auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
      if (tensorType &&
          !tensorType.getEncoding().isa<triton::gpu::SharedEncodingAttr>()) {
        Type newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = isa<triton::gpu::InsertSliceAsyncOp>(op);
      newTypes.push_back(isAsync ? t : getNewType(t, encoding));
    }

    // Construct new op with the new encoding
    Operation *newOp =
        builder.create(op->getLoc(), op->getName().getIdentifier(), newArgs,
                       newTypes, op->getAttrs());

    // Cast the results back to the original layout
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();
  }

  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    llvm::MapVector<Operation *, Attribute> layoutMap;
    moduleOp.walk([&](Operation *curr) {
      Value ptr = getMemAccessPtr(curr);
      if (!ptr)
        return;
      // We only convert `tensor<tt.ptr<>>` or `tt.ptr<tensor<>>` load/store
      bool isPtrTensor = false, isTensorPointer = false;
      if (auto tensorType = ptr.getType().dyn_cast<RankedTensorType>())
        isPtrTensor = tensorType.getElementType().isa<PointerType>();
      if (auto ptrType = ptr.getType().dyn_cast<PointerType>())
        isTensorPointer = ptrType.getPointeeType().isa<RankedTensorType>();
      if (!isPtrTensor && !isTensorPointer)
        return;
      auto mod = curr->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int threadsPerWarp =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      setCoalescedEncoding(axisInfoAnalysis, curr, numWarps, threadsPerWarp,
                           layoutMap);
    });

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    for (auto &kv : layoutMap) {
      coalesceOp(kv.second, kv.first);
    }
  }
};

std::unique_ptr<Pass> mlir::triton::gpu::createCoalescePass() {
  return std::make_unique<CoalescePass>();
}
