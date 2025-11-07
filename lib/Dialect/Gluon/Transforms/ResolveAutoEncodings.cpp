#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

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
    // Do layout inference
    if (failed(inferLayout(m, isAutoEncodingTensorType)))
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
