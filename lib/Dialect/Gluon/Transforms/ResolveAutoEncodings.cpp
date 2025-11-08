#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
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

    // Set seed values from set_auto_layout ops
    llvm::MapVector<FuncOp, llvm::MapVector<Value, LayoutInfo>> funcValueEnc;
    llvm::MapVector<FuncOp, llvm::PriorityWorklist<Value>> funcWorklist;
    llvm::MapVector<FuncOp, llvm::MapVector<Attribute, uint64_t>> funcHashMemo;

    auto seeded = m.walk([&](gluon::SetAutoLayoutOp op) -> WalkResult {
      FuncOp func = op->getParentOfType<FuncOp>();
      auto layout = LayoutInfo{op.getType().getEncoding()};
      if (failed(updateEncoding({op.getSrc()}, layout, &func,
                                funcValueEnc[func], funcWorklist[func],
                                funcHashMemo[func])))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (seeded.wasInterrupted())
      return signalPassFailure();

    // Do layout inference
    if (failed(inferLayout(m, isAutoEncodingTensorType, funcValueEnc,
                           funcWorklist, funcHashMemo)))
      return signalPassFailure();

    // Cleanup set_auto_layout ops
    m.walk([&](gluon::SetAutoLayoutOp op) {
      assert(op.getSrc().getType() == op.getType());
      op.getResult().replaceAllUsesWith(op.getSrc());
      op->erase();
    });

    if (failed(doubleCheckEncodings(m, isAutoEncodingTensorType)))
      return signalPassFailure();
  }
};
} // namespace mlir::triton::gluon
