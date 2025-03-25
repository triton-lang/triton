//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-interleave"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUINTERLEAVE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUInterleave
    : public impl::TritonXPUInterleaveBase<TritonXPUInterleave> {

public:
  using impl::TritonXPUInterleaveBase<
      TritonXPUInterleave>::TritonXPUInterleaveBase;

  bool isSameSize(Value mulValue, triton::xpu::MakeRangeOp makeRangeOp) {
    auto mulValDefOp = findDefOpBwd<arith::ConstantOp>(mulValue);
    if (mulValDefOp) {
      auto constOp = cast<arith::ConstantOp>(mulValDefOp);
      auto type = constOp.getResult().getType();
      int64_t constValue = 0;
      if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
        auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
        auto elementType = tensorType.getElementType();
        if (elementType.isInteger(32)) {
          constValue = *denseAttr.getValues<int32_t>().begin();
        } else if (elementType.isInteger(64)) {
          constValue = *denseAttr.getValues<int64_t>().begin();
        } else {
          llvm_unreachable(
              "[Offset Analysis] Unsupported Element Type in ConstOp");
        }
      } else {
        constValue =
            cast<IntegerAttr>(constOp.getValue()).getValue().getZExtValue();
      }
      int64_t rangeSize = makeRangeOp.getRealSize();
      if (constValue != rangeSize) {
        return false;
      }
    } else {
      return false;
    }
    return true;
  }

  Operation *findInterleavePatternOp(Operation *lhs, Operation *rhs) {
    llvm::SetVector<Operation *> visitedDownwards, visitedUpwards;

    int mulCnt = 0;
    std::function<Operation *(Operation *)> findDownwards =
        [&](Operation *op) -> Operation * {
      if (mulCnt > 1 || (isa<arith::SubIOp>(op) || isa<arith::DivSIOp>(op))) {
        return nullptr;
      }
      if (auto muliOp = dyn_cast<arith::MulIOp>(op)) {
        mulCnt += 1;
        auto makeRangeOp = cast<triton::xpu::MakeRangeOp>(rhs);
        if (!isSameSize(muliOp.getLhs(), makeRangeOp) &&
            !isSameSize(muliOp.getRhs(), makeRangeOp)) {
          return nullptr;
        }
      }
      if (!visitedDownwards.insert(op)) {
        return nullptr;
      }
      if (isa<arith::AddIOp>(op)) {
        return op;
      }
      for (auto user : op->getUsers()) {
        if (Operation *foundOp = findDownwards(user)) {
          return foundOp;
        }
      }
      return nullptr;
    };

    std::function<bool(Operation *)> findUpwards = [&](Operation *op) -> bool {
      if (isa<arith::AddIOp>(op) || isa<arith::SubIOp>(op) ||
          isa<arith::MulIOp>(op) || isa<arith::DivSIOp>(op)) {
        return false;
      }
      if (op == rhs) {
        return true;
      }
      if (!visitedUpwards.insert(op)) {
        return false;
      }
      for (auto operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (findUpwards(defOp)) {
            return true;
          }
        }
      }
      return false;
    };

    Operation *targetOp = findDownwards(lhs);
    Operation *upStartOp = nullptr;
    if (targetOp) {
      for (auto operand : targetOp->getOperands()) {
        if (operand.getDefiningOp() &&
            !visitedDownwards.count(operand.getDefiningOp())) {
          upStartOp = operand.getDefiningOp();
        }
      }
    }
    if (upStartOp && findUpwards(upStartOp)) {
      return targetOp;
    }

    return nullptr;
  }

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    llvm::DenseMap<Operation *, Operation *> addiRangeMap;

    // 1. Get the map of AddIOp and MakeRangeOp for create InterleaveOp
    m.walk([&](triton::xpu::LoadOp loadOp) {
      auto res = loadOp.getResult();
      if (auto tensorTy = dyn_cast<RankedTensorType>(res.getType())) {
        if (tensorTy.getShape().size() == 1) {
          auto getProgramIdOp =
              findDefOpBwd<triton::GetProgramIdOp>(loadOp.getPtr());
          auto makeRangeOp =
              findDefOpBwd<triton::xpu::MakeRangeOp>(loadOp.getPtr());
          auto getNumProgramsOp =
              findDefOpBwd<triton::GetNumProgramsOp>(loadOp.getPtr());
          if (getProgramIdOp && makeRangeOp && !getNumProgramsOp) {
            if (auto addIOp =
                    findInterleavePatternOp(getProgramIdOp, makeRangeOp)) {
              addiRangeMap[addIOp] = makeRangeOp;
            }
          }
        }
      }
    });

    m.walk([&](triton::xpu::StoreOp storeOp) {
      auto val = storeOp.getValue();
      if (auto tensorTy = dyn_cast<RankedTensorType>(val.getType())) {
        if (tensorTy.getShape().size() == 1) {
          auto getProgramIdOp =
              findDefOpBwd<triton::GetProgramIdOp>(storeOp.getPtr());
          auto makeRangeOp =
              findDefOpBwd<triton::xpu::MakeRangeOp>(storeOp.getPtr());
          auto valueMakeRangeOp =
              findDefOpBwd<triton::xpu::MakeRangeOp>(storeOp.getValue());
          auto getNumProgramsOp =
              findDefOpBwd<triton::GetNumProgramsOp>(storeOp.getPtr());
          if (getProgramIdOp && makeRangeOp && !valueMakeRangeOp &&
              !getNumProgramsOp) {
            if (auto addIOp =
                    findInterleavePatternOp(getProgramIdOp, makeRangeOp))
              addiRangeMap[addIOp] = makeRangeOp;
          }
        }
      }
    });

    // 2. Remove GetProgramIdOp * BLOCK_SIZE and replace MakeRangeOp with
    // InterleaveOp for PointWise
    for (const auto &pair : addiRangeMap) {
      auto addIOp = cast<mlir::arith::AddIOp>(pair.first);
      auto makeRangeOp = cast<triton::xpu::MakeRangeOp>(pair.second);
      OpBuilder builder(makeRangeOp);
      auto loc = builder.getUnknownLoc();
      auto start = makeRangeOp.getStart();
      auto end = makeRangeOp.getEnd();
      auto idx = makeRangeOp.getLoopIndex();
      auto reduceOp = findUserOp<triton::xpu::ReduceOp>(makeRangeOp);
      // Interleaving is not suitable for ReduceOp
      if (!reduceOp && idx) {
        LLVM_DEBUG(llvm::dbgs() << "[Interleave] Hit Interleave\n");
        auto interleaveOp = builder.create<triton::xpu::InterleaveOp>(
            loc, addIOp.getType(), start, end, idx, Value());
        addIOp.getResult().replaceAllUsesWith(interleaveOp);
        addIOp.erase();
      }
    }
  }
};
} // namespace xpu
} // namespace triton
} // namespace mlir
