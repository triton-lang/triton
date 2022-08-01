#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <numeric>

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct CoalescePass : public TritonGPUCoalesceBase<CoalescePass> {

  Attribute getCoalescedEncoding(AxisInfoAnalysis &axisInfo, Value ptr) {
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
    // Thread tile size depends on memory alignment
    SmallVector<unsigned, 4> sizePerThread(rank, 1);
    PointerType ptrType = origType.getElementType().cast<PointerType>();
    unsigned numBits = ptrType.getPointeeType().getIntOrFloatBitWidth();
    unsigned maxMultiple = info.getDivisibility(order[0]);
    unsigned maxContig = info.getContiguity(order[0]);
    unsigned alignment = std::min(maxMultiple, maxContig);
    unsigned perThread = std::min(alignment, 128 / numBits);
    sizePerThread[order[0]] = perThread;
    // create encoding
    Attribute encoding = triton::gpu::TritonGPUBlockedEncodingAttr::get(
        &getContext(), origType.getShape(), sizePerThread, order,
        this->numWarps);
    return encoding;
  }

  std::function<Type(Type)> getTypeConverter(AxisInfoAnalysis &axisInfo,
                                             Value ptr) {
    Attribute encoding = getCoalescedEncoding(axisInfo, ptr);
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
    AxisInfo info = axisInfo.lookupLatticeElement(ptr)->getValue();
    auto convertType = getTypeConverter(axisInfo, ptr);
    // convert operands
    SmallVector<Value, 4> newArgs;
    for (auto v : op->getOperands())
      newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), convertType(v.getType()), v));
    // convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes())
      newTypes.push_back(convertType(t));
    // construct new op with the new encoding
    Operation *newOp =
        builder.create<T>(op->getLoc(), newTypes, newArgs, op->getAttrs());
    // cast the results back to the original layout
    for (size_t i = 0; i < op->getNumResults(); i++) {
      auto newResult = builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), op->getResult(i).getType(), newOp->getResult(i));
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
      if (auto store = dyn_cast<triton::StoreOp>(curr))
        coalesceOp<triton::StoreOp>(axisInfo, curr, store.ptr(), builder);
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUCoalescePass() {
  return std::make_unique<CoalescePass>();
}
