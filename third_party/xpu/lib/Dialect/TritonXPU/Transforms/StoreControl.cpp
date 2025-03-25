//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO[dyq]: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-store-control"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUSTORECONTROL
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUStoreControl
    : public impl::TritonXPUStoreControlBase<TritonXPUStoreControl> {

  using impl::TritonXPUStoreControlBase<
      TritonXPUStoreControl>::TritonXPUStoreControlBase;

  void getGroupInfo(triton::xpu::ReduceOp &reduceOp, int64_t &groupSize,
                    int64_t &groupNum) {
    auto types = reduceOp.getOperandTypes();
    assert(types.size() > 1);
    for (int i = 0; i < types.size() - 1; ++i) {
      if (auto tensorType = dyn_cast<RankedTensorType>(types[i])) {
        auto clusterEncoding =
            cast<triton::xpu::ClusterLayoutAttr>(tensorType.getEncoding());
        if (i == 0) {
          groupSize = product(clusterEncoding.getCoresPerGroup());
          groupNum = product(clusterEncoding.getGroupsPerCluster());
        } else {
          assert(groupSize == product(clusterEncoding.getCoresPerGroup()));
          assert(groupNum == product(clusterEncoding.getGroupsPerCluster()));
        }
      }
    }
  }

  bool findDefChain(Operation *startOp, Operation *endOp,
                    SetVector<Operation *> &chain,
                    SetVector<Operation *> &visitedOps) {
    if (!startOp) {
      return false;
    }
    chain.insert(startOp);
    if (startOp == endOp) {
      return true;
    }
    for (auto operand : startOp->getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (!visitedOps.count(defOp)) {
        if (findDefChain(defOp, endOp, chain, visitedOps)) {
          return true;
        }
      }
    }
    chain.pop_back();
    return false;
  }

  bool hasBroadcast(Operation *startOp, Operation *endOp) {
    SetVector<Operation *> chain;
    SetVector<Operation *> visitedOps;
    findDefChain(startOp, endOp, chain, visitedOps);
    for (auto op : chain) {
      if (isa<triton::xpu::BroadcastOp, triton::SplatOp>(op)) {
        return true;
      }
    }
    return false;
  }

  bool isSameSize(triton::xpu::ReduceOp &reduceOp,
                  triton::xpu::StoreOp storeOp) {
    llvm::ArrayRef<int64_t> redResShape = {1};
    auto redRes = reduceOp.getResult()[0];
    if (auto redResTy = dyn_cast<RankedTensorType>(redRes.getType())) {
      auto sliceEncoding =
          cast<triton::gpu::SliceEncodingAttr>(redResTy.getEncoding());
      redResShape = redResTy.getShape();
    }
    llvm::ArrayRef<int64_t> storeValShape = {1};
    auto storeVal = storeOp.getValue();
    if (auto storeValTy = dyn_cast<RankedTensorType>(storeVal.getType())) {
      storeValShape = storeValTy.getShape();
    }
    if (product(redResShape) == product(storeValShape)) {
      return true;
    }
    return false;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    DenseMap<Operation *, SetVector<Operation *>> ifBodyMap;
    m.walk([&](triton::xpu::StoreOp storeOp) {
      OpBuilder builder(storeOp);
      auto loc = storeOp->getLoc();
      SetVector<Operation *> ifBodyOps;
      if (auto op = findDefOpBwd<triton::xpu::ReduceOp>(storeOp.getValue())) {
        auto reduceOp = cast<triton::xpu::ReduceOp>(op);
        ReduceOpHelper help(reduceOp);
        auto srcShape = help.getSrcShape();
        if (srcShape.size() > 1 && reduceOp.getAxis() != srcShape.size() - 1) {
          return;
        }
        if (hasBroadcast(storeOp, op) || !isSameSize(reduceOp, storeOp)) {
          return;
        }
        auto allocaOp = storeOp.getPtr().getDefiningOp();
        for (Operation *user : allocaOp->getUsers()) {
          if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMOp>(user)) {
            ifBodyOps.insert(allocaOp);
            ifBodyOps.insert(storeOp);
            ifBodyOps.insert(lm2gmOp);
          }
        }
        if (ifBodyOps.empty()) {
          return;
        }
        int64_t _groupSize = 64, _groupNum = 1;
        getGroupInfo(reduceOp, _groupSize, _groupNum);
        auto coreId =
            builder.create<triton::xpu::GetCoreIdOp>(loc, builder.getI32Type());
        auto groupSize =
            builder.create<arith::ConstantIntOp>(loc, _groupSize, 32);
        auto coreIdInsideGroup = builder.create<arith::RemSIOp>(
            loc, builder.getI32Type(), coreId, groupSize);
        auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
        auto isCoreId0InsideGroup = builder.create<arith::CmpIOp>(
            loc, builder.getI1Type(), arith::CmpIPredicate::eq,
            coreIdInsideGroup, zero);
        int64_t _usedCoreNum = _groupSize * _groupNum;
        auto usedCoreNum =
            builder.create<arith::ConstantIntOp>(loc, _usedCoreNum, 32);
        auto sltUsedCoreNum = builder.create<arith::CmpIOp>(
            loc, builder.getI1Type(), arith::CmpIPredicate::slt, coreId,
            usedCoreNum);
        auto cond = builder.create<arith::AndIOp>(
            loc, builder.getI1Type(), isCoreId0InsideGroup, sltUsedCoreNum);
        auto ifOp = builder.create<scf::IfOp>(loc, cond,
                                              /*withElseRegion=*/false);
        ifBodyMap[ifOp] = ifBodyOps;
        LLVM_DEBUG(llvm::dbgs() << "[StoreControl] GroupSize: " << _groupSize
                                << ", usedCoreNum: " << _usedCoreNum << "\n");
      }
    });
    for (auto &pair : ifBodyMap) {
      auto ifOp = cast<scf::IfOp>(pair.first);
      auto ifBodyOps = pair.second;
      for (auto op : ifBodyOps) {
        op->moveBefore(ifOp.thenBlock()->getTerminator());
      }
    }
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
