#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
LogicalResult inferAutoLayout(ModuleOp &mod) {
  for (auto &op : *mod.getBody()) {
    auto func = dyn_cast<FuncOp>(&op);
    if (!func)
      continue;

    // Set seed values from set_auto_layout ops
    llvm::SmallVector<std::pair<Value, Attribute>> seedEncodings;
    func.walk([&](gluon::SetAutoLayoutOp op) {
      seedEncodings.push_back({op.getSrc(), op.getType().getEncoding()});
    });

    if (failed(inferLayout(func, isAutoEncodingTensorType, seedEncodings)))
      return failure();

    auto walkResult = func.walk([&](triton::LinearApplyOp op) {
      if (!isAutoEncodingTensorType(op.getBases().getType()))
        return WalkResult::advance();

      // Use the canonical layout only when explicit layout inference did not
      // reach the basis or any other value in its producing graph.
      unsigned threadsPerWarp =
          static_cast<unsigned>(ttg::TritonGPUDialect::getThreadsPerWarp(mod));
      unsigned numWarps =
          static_cast<unsigned>(ttg::lookupNumWarps(op.getOperation()));
      unsigned numCTAs =
          static_cast<unsigned>(ttg::lookupNumCTAs(op.getOperation()));
      auto cgaLayout = ttg::CGAEncodingAttr::fromSplitParams(
          mod.getContext(), {numCTAs}, {1}, {0});
      auto basisEncoding = ttg::BlockedEncodingAttr::get(
          mod.getContext(), {1}, {threadsPerWarp}, {numWarps}, {0}, cgaLayout);
      llvm::SmallVector<std::pair<Value, Attribute>> basisSeedEncodings = {
          {op.getBases(), basisEncoding}};
      if (failed(inferLayout(func, isAutoEncodingTensorType,
                             basisSeedEncodings)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
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
    ModuleOp m = getOperation();

    // Do layout inference
    if (failed(inferAutoLayout(m)))
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
