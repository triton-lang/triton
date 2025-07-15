#include "Dialect/ProtonGPU/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir::triton::proton::gpu {

#define GEN_PASS_DEF_SCHEDULEBUFFERSTOREPASS
#include "Dialect/ProtonGPU/Transforms/Passes.h.inc"

struct ScheduleBufferStorePass
    : public impl::ScheduleBufferStorePassBase<ScheduleBufferStorePass> {

  using impl::ScheduleBufferStorePassBase<
      ScheduleBufferStorePass>::ScheduleBufferStorePassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = m.getContext();
    OpBuilder builder(context);

    // TODO(srir): Add support for non-inline kernels
    FuncOp func = *m.getOps<triton::FuncOp>().begin();
    auto startStoreList = llvm::SmallVector<CircularStoreOp, 8>();
    auto endStoreMap = llvm::SmallDenseMap<int, CircularStoreOp, 8>();

    func.walk([&](CircularStoreOp store) {
      if (store.getIsStart())
        startStoreList.push_back(store);
      else
        endStoreMap[store.getScopeId()] = store;
    });

    for (auto store : startStoreList) {
      int scopeId = store.getScopeId();
      auto endStore = endStoreMap[scopeId];
      if (!endStore) {
        mlir::emitError(func.getLoc(), "proton end store not found");
        signalPassFailure();
        return;
      }
      builder.setInsertionPoint(endStore);
      builder.clone(*store);
      store->erase();
    }
  }
};

} // namespace mlir::triton::proton::gpu
