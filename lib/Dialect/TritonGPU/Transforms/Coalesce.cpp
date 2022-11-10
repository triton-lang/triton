#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <numeric>

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct CoalescePass : public TritonGPUCoalesceBase<CoalescePass> {
  Attribute getCoalescedEncoding(AxisInfoAnalysis &axisInfo, Value ptr,
                                 int numWarps) {
    auto origType = ptr.getType().cast<RankedTensorType>();
    // Get the shape of the tensor.
    size_t rank = origType.getRank();
    AxisInfo info = axisInfo.lookupLatticeElement(ptr)->getValue();
    // Layout order in decreasing order of contiguity
    SmallVector<unsigned, 4> order(rank);
    std::iota(order.begin(), order.end(), 0);
    auto contiguity = info.getContiguity();
    std::sort(order.begin(), order.end(), [&](unsigned x, unsigned y) {
      return contiguity[x] > contiguity[y];
    });

    int numElems = product(origType.getShape());
    int numThreads = numWarps * 32;
    int numElemsPerThread = std::max(numElems / numThreads, 1);

    // Thread tile size depends on memory alignment
    SmallVector<unsigned, 4> sizePerThread(rank, 1);
    PointerType ptrType = origType.getElementType().cast<PointerType>();
    auto pointeeType = ptrType.getPointeeType();
    unsigned numBits =
        pointeeType.isa<triton::Float8Type>() ?
        8 : pointeeType.getIntOrFloatBitWidth();
    unsigned maxMultiple = info.getDivisibility(order[0]);
    unsigned maxContig = info.getContiguity(order[0]);
    unsigned alignment = std::min(maxMultiple, maxContig);
    unsigned perThread = std::min(alignment, 128 / numBits);
    sizePerThread[order[0]] = std::min<int>(perThread, numElemsPerThread);

    SmallVector<unsigned> dims(rank);
    std::iota(dims.begin(), dims.end(), 0);
    // create encoding
    Attribute encoding = triton::gpu::BlockedEncodingAttr::get(
        &getContext(), origType.getShape(), sizePerThread, order, numWarps);
    return encoding;
  }

  std::function<Type(Type)> getTypeConverter(AxisInfoAnalysis &axisInfo,
                                             Value ptr, int numWarps) {
    Attribute encoding = getCoalescedEncoding(axisInfo, ptr, numWarps);
    return [encoding](Type _type) {
      RankedTensorType type = _type.cast<RankedTensorType>();
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   encoding);
    };
  }

  template <class T>
  void coalesceOp(AxisInfoAnalysis &axisInfo, Operation *op, Value ptr,
                  OpBuilder builder) {
    RankedTensorType ty = ptr.getType().template dyn_cast<RankedTensorType>();
    if (!ty)
      return;
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    AxisInfo info = axisInfo.lookupLatticeElement(ptr)->getValue();
    auto convertType = getTypeConverter(axisInfo, ptr, numWarps);
    // convert operands
    SmallVector<Value, 4> newArgs;
    for (auto v : op->getOperands()) {
      auto vTy = v.getType().dyn_cast<RankedTensorType>();
      if (vTy && !vTy.getEncoding().isa<triton::gpu::SharedEncodingAttr>())
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), convertType(v.getType()), v));
      else
        newArgs.push_back(v);
    }
    // convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool is_async = std::is_same<T, triton::gpu::InsertSliceAsyncOp>::value;
      newTypes.push_back(is_async ? t : convertType(t));
    }
    // construct new op with the new encoding
    Operation *newOp =
        builder.create<T>(op->getLoc(), newTypes, newArgs, op->getAttrs());
    // cast the results back to the original layout
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
    Operation *op = getOperation();
    // Run axis info analysis
    AxisInfoAnalysis axisInfo(&getContext());
    axisInfo.run(op);
    OpBuilder builder(op);

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    op->walk([&](Operation *curr) {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(curr);
      if (auto load = dyn_cast<triton::LoadOp>(curr))
        coalesceOp<triton::LoadOp>(axisInfo, curr, load.ptr(), builder);
      if (auto op = dyn_cast<triton::AtomicRMWOp>(curr))
        coalesceOp<triton::AtomicRMWOp>(axisInfo, curr, op.ptr(), builder);
      if (auto op = dyn_cast<triton::AtomicCASOp>(curr))
        coalesceOp<triton::AtomicCASOp>(axisInfo, curr, op.ptr(), builder);
      if (auto load = dyn_cast<triton::gpu::InsertSliceAsyncOp>(curr))
        coalesceOp<triton::gpu::InsertSliceAsyncOp>(axisInfo, curr, load.src(),
                                                    builder);
      if (auto store = dyn_cast<triton::StoreOp>(curr))
        coalesceOp<triton::StoreOp>(axisInfo, curr, store.ptr(), builder);
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUCoalescePass() {
  return std::make_unique<CoalescePass>();
}
