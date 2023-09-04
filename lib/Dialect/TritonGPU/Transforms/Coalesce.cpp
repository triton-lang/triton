#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include <iterator>
#include <numeric>

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

unsigned getElementBitWidth(const Value &val) {
  auto valType = val.getType();
  if (valType.isa<PointerType>())
    valType = valType.cast<PointerType>().getPointeeType();
  auto tensorType = valType.cast<RankedTensorType>();

  auto typeForMem =
      tensorType.getElementType().isa<PointerType>()
          ? tensorType.getElementType().cast<PointerType>().getPointeeType()
          : tensorType.getElementType();
  return typeForMem.getIntOrFloatBitWidth();
}

typedef DenseMap<Value, std::function<Type(Type)>> LayoutMap;

struct CoalescePass : public TritonGPUCoalesceBase<CoalescePass> {
  Attribute getCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 Value ptr, Operation *op, int numWarps,
                                 int threadsPerWarp) {
    auto refType = ptr.getType();
    if (refType.isa<PointerType>())
      refType = refType.cast<PointerType>().getPointeeType();
    auto refTensorType = refType.cast<RankedTensorType>();

    // TODO(Keren): integrate it into AxisInfoAnalysis
    // Get axis info
    auto queryAxisInfo = [&](const Value &val) -> AxisInfo {
      auto valType = val.getType();
      // Tensor pointer
      // TODO(Chenggang): encoding for tensor pointers is meaningless, remove
      // these later while merging into the GitHub main
      if (auto ptrType = valType.dyn_cast<PointerType>()) {
        auto tensorTy = ptrType.getPointeeType().dyn_cast<RankedTensorType>();
        assert(tensorTy);
        auto makeTensorPtr = getMakeTensorPtrOp(val);
        auto order = makeTensorPtr.getOrder();
        auto tileShape = triton::gpu::getShapePerCTA(tensorTy);
        size_t rank = order.size();
        auto elemSizeInBytes =
            tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
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
      // Normal cases
      assert(valType.isa<RankedTensorType>());
      return *axisInfoAnalysis.getAxisInfo(val);
    };

    // Get the contiguity order of `ptr`
    SmallVector<unsigned> order;
    if (auto ptrType = ptr.getType().dyn_cast<PointerType>()) {
      // Tensor pointer
      auto makeTensorPtr = getMakeTensorPtrOp(ptr);
      std::copy(makeTensorPtr.getOrder().begin(),
                makeTensorPtr.getOrder().end(), std::back_inserter(order));
    } else {
      // Normal cases
      order = argSort(queryAxisInfo(ptr).getContiguity());
    }

    auto matchesOrder = [&refTensorType](const Value &val) {
      if (val.getType() == refTensorType) {
        return true;
      }

      auto rttType = val.getType().dyn_cast<RankedTensorType>();
      if (!rttType) {
        return false;
      }
      return rttType.getShape() == refTensorType.getShape();
    };

    // The desired divisibility is the maximum divisibility
    // among all dependent pointers who have the same order as
    // `ptr`.
    // We only do it for normal tensors of pointers, not tensor pointers.
    SetVector<Value> withSameOrder;
    withSameOrder.insert(ptr);
    if (refType.isa<RankedTensorType>() && ptr.getDefiningOp()) {
      for (Operation *op : mlir::multiRootGetSlice(ptr.getDefiningOp())) {
        for (Value val : op->getResults()) {
          if (!matchesOrder(val))
            continue;
          auto currOrder =
              argSort(axisInfoAnalysis.getAxisInfo(val)->getContiguity());
          if (order == currOrder)
            withSameOrder.insert(val);
        }
      }
    }

    auto shapePerCTA = triton::gpu::getShapePerCTA(refTensorType);
    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;
    int numElemsPerThread = std::max(numElems / numThreads, 1);

    // For tensor of pointers, the element to access is the pointee type;
    // while for tensor pointer type (`refType` is directly the final shape),
    // the element to access is itself.
    auto typeForMem = refTensorType.getElementType().isa<PointerType>()
                          ? refTensorType.getElementType()
                                .cast<PointerType>()
                                .getPointeeType()
                          : refTensorType.getElementType();

    // Thread tile size depends on memory alignment
    SmallVector<unsigned, 4> sizePerThread(refTensorType.getRank(), 1);
    unsigned perThread = 1;
    for (Value val : withSameOrder) {
      auto valInfo = queryAxisInfo(val);
      unsigned elemNumBits = getElementBitWidth(val);
      unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
      unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
      unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
      unsigned maxContig =
          std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
      unsigned alignment = std::min(maxMultiple, maxContig);
      unsigned currPerThread = std::min(alignment, 128 / elemNumBits);
      perThread = std::max(perThread, currPerThread);
    }

    perThread = std::min<int>(perThread, numElemsPerThread);

    if (!dyn_cast<triton::LoadOp>(op)) {
      // For ops that can result in a global memory write, we should enforce
      // that each thread handles at most 128 bits, which is the widest
      // available vectorized store op; otherwise, the store will have "gaps"
      // in the memory write at the warp level, resulting in worse performance.
      // For loads, we can expect that the gaps won't matter due to the L1
      // cache.
      unsigned elemNumBits = getElementBitWidth(ptr);
      perThread = std::min<int>(perThread, 128 / elemNumBits);
    }
    sizePerThread[order[0]] = perThread;

    auto CTALayout = triton::gpu::getCTALayout(refTensorType.getEncoding());
    return triton::gpu::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp, CTALayout);
  }

  std::function<Type(Type)>
  getTypeConverter(ModuleAxisInfoAnalysis &axisInfoAnalysis, Value ptr,
                   Operation *op, int numWarps, int threadsPerWarp) {
    Attribute encoding = getCoalescedEncoding(axisInfoAnalysis, ptr, op,
                                              numWarps, threadsPerWarp);
    return [encoding](Type type) {
      RankedTensorType tensorType = type.cast<RankedTensorType>();
      return RankedTensorType::get(tensorType.getShape(),
                                   tensorType.getElementType(), encoding);
    };
  }

  template <class T>
  void coalesceOp(LayoutMap &layoutMap, Operation *op, Value ptr,
                  OpBuilder builder) {
    if (!layoutMap.count(ptr))
      return;

    // Convert operands
    // For load/store with tensor pointers, we don't have to change the
    // operands' type, we do this by changing the outputs' type of
    // `make_tensor_ptr`
    auto convertType = layoutMap.lookup(ptr);
    SmallVector<Value, 4> newArgs;
    for (auto operand : op->getOperands()) {
      auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
      if (tensorType &&
          !tensorType.getEncoding().isa<triton::gpu::SharedEncodingAttr>())
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), convertType(tensorType), operand));
      else
        newArgs.push_back(operand);
    }

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = std::is_same<T, triton::gpu::InsertSliceAsyncOp>::value;
      newTypes.push_back(isAsync ? t : convertType(t));
    }

    // Construct new op with the new encoding
    Operation *newOp =
        builder.create<T>(op->getLoc(), newTypes, newArgs, op->getAttrs());

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

  void coalesceMakeTensorPtrOpResult(LayoutMap &layoutMap, Operation *op,
                                     Value ptr, OpBuilder builder) {
    if (!layoutMap.count(ptr))
      return;

    // Convert result type
    auto convertType = layoutMap.lookup(ptr);
    auto ptrType = ptr.getType().cast<PointerType>();
    auto resultTensorType = convertType(ptrType.getPointeeType());
    auto newResultType =
        PointerType::get(resultTensorType, ptrType.getAddressSpace());

    // Build new operation and replace
    Operation *newOp = builder.create<MakeTensorPtrOp>(
        op->getLoc(), newResultType, op->getOperands(), op->getAttrs());
    op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
    op->erase();
  }

  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    LayoutMap layoutMap;
    moduleOp.walk([&](Operation *curr) {
      Value ptr;
      if (auto op = dyn_cast<triton::LoadOp>(curr))
        ptr = op.getPtr();
      if (auto op = dyn_cast<triton::AtomicRMWOp>(curr))
        ptr = op.getPtr();
      if (auto op = dyn_cast<triton::AtomicCASOp>(curr))
        ptr = op.getPtr();
      if (auto op = dyn_cast<triton::gpu::InsertSliceAsyncOp>(curr))
        ptr = op.getSrc();
      if (auto op = dyn_cast<triton::StoreOp>(curr))
        ptr = op.getPtr();
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
      auto convertType = getTypeConverter(axisInfoAnalysis, ptr, curr, numWarps,
                                          threadsPerWarp);
      layoutMap[ptr] = convertType;
    });

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    moduleOp.walk([&](Operation *curr) {
      OpBuilder builder(curr);
      if (auto load = dyn_cast<triton::LoadOp>(curr)) {
        coalesceOp<triton::LoadOp>(layoutMap, curr, load.getPtr(), builder);
        return;
      }
      if (auto op = dyn_cast<triton::AtomicRMWOp>(curr)) {
        coalesceOp<triton::AtomicRMWOp>(layoutMap, curr, op.getPtr(), builder);
        return;
      }
      if (auto op = dyn_cast<triton::AtomicCASOp>(curr)) {
        coalesceOp<triton::AtomicCASOp>(layoutMap, curr, op.getPtr(), builder);
        return;
      }
      if (auto load = dyn_cast<triton::gpu::InsertSliceAsyncOp>(curr)) {
        coalesceOp<triton::gpu::InsertSliceAsyncOp>(layoutMap, curr,
                                                    load.getSrc(), builder);
        return;
      }
      if (auto store = dyn_cast<triton::StoreOp>(curr)) {
        coalesceOp<triton::StoreOp>(layoutMap, curr, store.getPtr(), builder);
        return;
      }
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUCoalescePass() {
  return std::make_unique<CoalescePass>();
}
