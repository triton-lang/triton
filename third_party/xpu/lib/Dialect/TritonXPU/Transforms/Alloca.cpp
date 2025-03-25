//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUALLOCA
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUAllocaPass
    : public impl::TritonXPUAllocaBase<TritonXPUAllocaPass> {

public:
  using impl::TritonXPUAllocaBase<TritonXPUAllocaPass>::TritonXPUAllocaBase;

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    m.walk([&](triton::xpu::LoadOp loadOp) {
      auto loc = loadOp.getLoc();
      OpBuilder builder(loadOp);
      auto resType = loadOp.getResult().getType();
      auto gmPtrType = loadOp.getPtr().getType();
      auto lmPtrType = addrspaceCast(gmPtrType, 0);
      auto size =
          mlir::isa<RankedTensorType>(gmPtrType)
              ? product(mlir::cast<RankedTensorType>(gmPtrType).getShape())
              : 1;
      if (auto gm2lmOp =
              dyn_cast<triton::xpu::GM2LMOp>(loadOp->getPrevNode())) {
        auto allocaOp =
            builder.create<triton::xpu::AllocaOp>(loc, lmPtrType, size);

        auto operandSegmentSizesAttr =
            gm2lmOp->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
        SmallVector<int32_t> operandSegmentSizes(
            operandSegmentSizesAttr.asArrayRef());
        ++operandSegmentSizes[2]; // 0: ptr, 1: len, 2: bufPtr
        gm2lmOp->setAttr("operandSegmentSizes",
                         builder.getDenseI32ArrayAttr(operandSegmentSizes));

        gm2lmOp->insertOperands(gm2lmOp->getNumOperands(), {allocaOp});

        allocaOp->moveBefore(gm2lmOp);
      } else {
        llvm_unreachable("Only support GM2LM as previous node of load");
      }
    });

    m.walk([&](triton::xpu::StoreOp storeOp) {
      auto loc = storeOp.getLoc();
      OpBuilder builder(storeOp);
      auto resType = storeOp.getValue().getType();
      auto gmPtrType = storeOp.getPtr().getType();
      auto lmPtrType = addrspaceCast(gmPtrType, 0);
      auto size =
          mlir::isa<RankedTensorType>(gmPtrType)
              ? product(mlir::cast<RankedTensorType>(gmPtrType).getShape())
              : 1;
      if (auto lm2gmOp =
              dyn_cast<triton::xpu::LM2GMOp>(storeOp->getNextNode())) {
        auto allocaOp =
            builder.create<triton::xpu::AllocaOp>(loc, lmPtrType, size);

        auto operandSegmentSizesAttr =
            lm2gmOp->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
        SmallVector<int, 4> operandSegmentSizes(
            operandSegmentSizesAttr.asArrayRef());
        ++operandSegmentSizes[3]; // 0: ptr, 1: value, 2: len, 3: bufPtr
        lm2gmOp->setAttr("operandSegmentSizes",
                         builder.getDenseI32ArrayAttr(operandSegmentSizes));
        lm2gmOp->insertOperands(lm2gmOp->getNumOperands(), {allocaOp});
        // remove value from lm2gm
        --operandSegmentSizes[1];
        lm2gmOp->setAttr("operandSegmentSizes",
                         builder.getDenseI32ArrayAttr(operandSegmentSizes));
        lm2gmOp->eraseOperands(1);

        allocaOp->moveBefore(storeOp);
        storeOp->setOperand(0, allocaOp);
      } else {
        llvm_unreachable("Only support LM2GM as next node of store");
      }
    });

    // Move Alloca in the Front of FuncOp Body
    m.walk([&](triton::xpu::AllocaOp allocaOp) {
      // 1.Find FuncOp
      Operation *ancestorOp = allocaOp;
      while (!isa<triton::FuncOp>(ancestorOp)) {
        Block *block = ancestorOp->getBlock();
        ancestorOp = block->getParentOp();
      }
      // 2. Move alloca in the Front of the First Op in the FuncOp Body
      Operation *firstOp =
          &(*(cast<triton::FuncOp>(ancestorOp).getBody().front().begin()));
      allocaOp->moveBefore(firstOp);
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
