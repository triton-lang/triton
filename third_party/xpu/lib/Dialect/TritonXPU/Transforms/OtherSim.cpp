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

#define GEN_PASS_DEF_TRITONXPUOTHERSIM
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUOtherSim
    : public impl::TritonXPUOtherSimBase<TritonXPUOtherSim> {

public:
  using impl::TritonXPUOtherSimBase<TritonXPUOtherSim>::TritonXPUOtherSimBase;

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    bool skip = true;
    m.walk([&](triton::xpu::ReduceOp reduceOp) { skip = false; });
    if (skip) {
      return;
    }

    m.walk([&](triton::xpu::LoadOp loadOp) {
      auto loc = loadOp.getLoc();
      OpBuilder builder(loadOp);
      if (auto other = loadOp.getOther()) {
        unsigned numElems = getTotalElemsPerThread(other.getType());
        Type elemTy = getElementTypeOrSelf(other.getType());
        unsigned vecSize = 1u;
        if (auto vecType = dyn_cast<VectorType>(elemTy)) {
          vecSize = vecType.getNumElements();
        }
        int64_t _bufLen = numElems * vecSize;
        Block *block = loadOp->getBlock();
        auto gm2lmOp = loadOp.getPtr().getDefiningOp<triton::xpu::GM2LMOp>();
        auto allocaOp =
            gm2lmOp.getBufPtr().getDefiningOp<triton::xpu::AllocaOp>();
        // Create If(len < bufLen)
        auto len = gm2lmOp.getLen();
        auto lenElemTy = getElementTypeOrSelf(len);
        auto extractLen = builder.create<mlir::triton::xpu::ExtractOp>(
            loc, lenElemTy, builder.getI32IntegerAttr(0), len);
        auto bufLen = builder.create<arith::ConstantIntOp>(
            loc, _bufLen, lenElemTy.getIntOrFloatBitWidth());
        auto sltBufLen = builder.create<arith::CmpIOp>(
            loc, builder.getI1Type(), arith::CmpIPredicate::slt, extractLen,
            bufLen);
        auto ifOp = builder.create<scf::IfOp>(loc, sltBufLen,
                                              /*withElseRegion=*/false);
        ifOp->moveBefore(gm2lmOp);
        extractLen->moveBefore(ifOp);
        bufLen->moveBefore(ifOp);
        sltBufLen->moveBefore(ifOp);
        // Create Constant/Store
        if (auto otherDef = other.getDefiningOp()) {
          if (auto constOp = dyn_cast<arith::ConstantOp>(otherDef)) {
            auto newConstOp = builder.create<arith::ConstantOp>(
                loc, constOp.getType(), constOp.getValue());
            auto storeOp = builder.create<triton::xpu::StoreOp>(
                loc, allocaOp, newConstOp, Value(), Value(), -1, false);
            newConstOp->moveBefore(ifOp.thenBlock()->getTerminator());
            storeOp->moveBefore(ifOp.thenBlock()->getTerminator());
          } else if (auto vconstOp =
                         dyn_cast<triton::xpu::VConstOp>(otherDef)) {
            auto newVConstOp = builder.create<triton::xpu::VConstOp>(
                loc, vconstOp.getType(), vconstOp.getValue());
            auto storeOp = builder.create<triton::xpu::StoreOp>(
                loc, allocaOp, newVConstOp, Value(), Value(), -1, false);
            newVConstOp->moveBefore(ifOp.thenBlock()->getTerminator());
            storeOp->moveBefore(ifOp.thenBlock()->getTerminator());
          } else {
            auto storeOp = builder.create<triton::xpu::StoreOp>(
                loc, allocaOp, otherDef->getResults()[0], Value(), Value(), -1,
                false);
            storeOp->moveBefore(ifOp.thenBlock()->getTerminator());
          }
        } else {
          auto storeOp = builder.create<triton::xpu::StoreOp>(
              loc, allocaOp, other, Value(), Value(), -1, false);
          storeOp->moveBefore(ifOp.thenBlock()->getTerminator());
        }
      }
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
