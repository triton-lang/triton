#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONRESOLVEAUTOENCODINGSPASS
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"

#define DEBUG_TYPE "gluon-resolve-auto-encodings"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

bool isAutoEncodingTensorType(Type ty) {
  auto tensorTy = dyn_cast<RankedTensorType>(ty);
  return tensorTy && isa<gluon::AutoEncodingAttr>(tensorTy.getEncoding());
}

LogicalResult inferAutoLayouts(FuncOp func) {
  // Disallow auto encoding accross function call boundaries
  for (auto argTy : func.getArgumentTypes()) {
    if (isAutoEncodingTensorType(argTy)) {
      return func->emitError(
          "Functions taking auto encoding must be fully inlined");
    }
  }
  for (auto resultTy : func.getResultTypes()) {
    if (isAutoEncodingTensorType(resultTy))
      return func->emitError(
          "Functions returning auto encoding must be fully inlined");
  }

  llvm::MapVector<Value, Attribute> valueToEncoding;
  llvm::PriorityWorklist<Value> worklist;

  auto updateEncoding = [&](ArrayRef<Value> values,
                            Attribute enc) -> LogicalResult {
    for (auto value : values) {
      auto [it, inserted] = valueToEncoding.insert({value, enc});
      if (!inserted) {
        if (it->second != enc) {
          auto defOp = value.getDefiningOp();
          auto op = defOp ? defOp : func;
          return op->emitOpError("Found conflicting encodings for value");
        }
      } else {
        LLVM_DEBUG({
          DBGS() << "Setting value:\n\t" << value << "\nto encoding:\n\t" << enc
                 << "\n";
        });
        worklist.insert(value);
      }
    }
    return success();
  };

  // 1. Set seed values from layout conversions
  auto res = func.walk([&](gluon::SetAutoLayoutOp op) -> WalkResult {
    auto res = updateEncoding({op.getSrc()}, op.getType().getEncoding());
    op.getResult().replaceAllUsesWith(op.getSrc());
    op->erase();
    return res;
  });

  if (res.wasInterrupted())
    return failure();

  // 2. Propagate encodings through the graph until fixed point, or conflict
  while (!worklist.empty()) {
    auto val = worklist.pop_back_val();
    auto enc = valueToEncoding[val];
    assert(enc);

    // Propagate to users
    for (OpOperand &use : val.getUses()) {
      auto op = use.getOwner();
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        auto offset = 3 * isa<scf::ForOp>(op);
        auto tiedArgs = getTiedArgs(op, use.getOperandNumber() - offset);
        if (failed(updateEncoding(tiedArgs, enc)))
          return failure();
      } else if (isa<scf::YieldOp>(op)) {
        auto tiedArgs = getTiedArgs(op, use.getOperandNumber());
        if (failed(updateEncoding(tiedArgs, enc)))
          return failure();
      } else {
        auto dstEnc = inferDstEncoding(op, enc);
        if (dstEnc) {
          if (failed(updateEncoding(llvm::to_vector_of<Value>(op->getResults()),
                                    dstEnc)))
            return failure();
        }
      }
    }

    // Propagate to defining ops
    if (auto opResult = dyn_cast<OpResult>(val)) {
      auto definingOp = opResult.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto tiedArgs = getTiedArgs(definingOp, opResult.getResultNumber());
        if (failed(updateEncoding(tiedArgs, enc)))
          return failure();
      } else {
        auto srcEncoding = inferSrcEncoding(definingOp, enc);
        if (srcEncoding) {
          llvm::SmallVector<Value> tensorOperands;
          for (auto operand : definingOp->getOperands())
            if (isa<RankedTensorType>(operand.getType()))
              tensorOperands.push_back(operand);

          if (failed(updateEncoding(tensorOperands, srcEncoding)))
            return failure();
        }
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        auto offset = isa<scf::ForOp>(parentOp);
        auto tiedArgs = getTiedArgs(parentOp, blockArg.getArgNumber() - offset);
        if (failed(updateEncoding(tiedArgs, enc)))
          return failure();
      }
    }
  }

  // 3. Transfer propagated encodings into the graph
  auto ctx = func.getContext();
  for (auto &[val, enc] : valueToEncoding) {
    auto existingTy = cast<RankedTensorType>(val.getType());
    assert(isa<gluon::AutoEncodingAttr>(existingTy.getEncoding()));
    auto ty = existingTy.cloneWithEncoding(enc);
    val.setType(ty);

    if (auto opResult = dyn_cast<OpResult>(val)) {
      if (auto constantOp = dyn_cast<arith::ConstantOp>(opResult.getOwner())) {
        auto value = cast<SplatElementsAttr>(constantOp.getValueAttr());
        auto newValue =
            SplatElementsAttr::get(ty, value.getSplatValue<Attribute>());
        constantOp.setValueAttr(newValue);
      }
    }
  }

  return success();
}

LogicalResult inferAutoLayouts(ModuleOp &mod) {
  for (auto &op : *mod.getBody()) {
    auto func = dyn_cast<FuncOp>(&op);
    if (!func)
      continue;
    if (failed(inferAutoLayouts(func)))
      return failure();
  }
  return success();
}

} // anonymous namespace

class GluonResolveAutoEncodingsPass
    : public impl::GluonResolveAutoEncodingsPassBase<
          GluonResolveAutoEncodingsPass> {
public:
  using BaseT =
      impl::GluonResolveAutoEncodingsPassBase<GluonResolveAutoEncodingsPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    // Do layout inference
    if (failed(inferAutoLayouts(m)))
      return signalPassFailure();

    // Double check we didn't miss anything
    auto res = m.walk([](Operation *op) -> WalkResult {
      for (auto resTy : op->getResultTypes()) {
        if (isAutoEncodingTensorType(resTy)) {
          return op->emitOpError("Failed to infer return type");
        }
      }
      return success();
    });
    if (res.wasInterrupted())
      return signalPassFailure();

    res = m.walk([](Block *block) -> WalkResult {
      for (auto argTy : block->getArgumentTypes()) {
        if (isAutoEncodingTensorType(argTy)) {
          return block->getParentOp()->emitError(
              "Failed to infer block argument type");
        }
      }
      return success();
    });
    if (res.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace mlir::triton::gluon
