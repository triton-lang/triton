#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <array>

namespace mlir {
namespace triton {
namespace instrument {

#define GEN_PASS_DEF_TRITONINSTRUMENTPREPARECONSANCAPTURES
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

namespace {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

bool hasSharedMemoryBuffers(ModuleOp mod) {
  bool result = false;
  mod.walk([&](ttg::LocalAllocOp op) { result |= op.isSharedMemoryAlloc(); });
  return result;
}

bool hasTensorMemoryBuffers(ModuleOp mod) {
  bool result = false;
  mod.walk([&](Operation *op) {
    for (Type type : op->getResultTypes()) {
      auto memDescType = dyn_cast<ttg::MemDescType>(type);
      if (!memDescType)
        continue;
      result |= isa<ttng::TensorMemorySpaceAttr>(memDescType.getMemorySpace());
    }
  });
  return result;
}

bool hasBarriers(ModuleOp mod) {
  bool result = false;
  mod.walk([&](ttg::MBarrierOpInterface op) {
    result |= !op.getBarriers().empty();
  });
  return result;
}

bool hasCpAsync(ModuleOp mod) {
  bool result = false;
  mod.walk([&](Operation *op) {
    if (isa<ttg::AsyncCopyGlobalToLocalOp, ttg::AsyncCommitGroupOp,
            ttg::AsyncWaitOp>(op))
      result = true;
  });
  return result;
}

int getNumCommitKinds(ModuleOp mod, const ConSanTargetHooks *hooks) {
  std::array<bool, tti::CommitKind::NumCommitKinds> commitKinds{};
  if (hasCpAsync(mod))
    commitKinds[tti::CommitKind::AsyncCp] = true;
  for (auto kind : hooks->getRequiredCommitKinds(mod)) {
    if (kind >= 0 && kind < tti::CommitKind::NumCommitKinds)
      commitKinds[kind] = true;
  }

  int result = 0;
  for (bool required : commitKinds)
    result += required;
  return result;
}

class PrepareConSanCaptures
    : public impl::TritonInstrumentPrepareConSanCapturesBase<
          PrepareConSanCaptures> {
public:
  using impl::TritonInstrumentPrepareConSanCapturesBase<
      PrepareConSanCaptures>::TritonInstrumentPrepareConSanCapturesBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (target.empty()) {
      mod.emitError("ConSan capture preparation requires a target hook key");
      return signalPassFailure();
    }

    auto hooks = createConSanHooks(target);
    if (!hooks) {
      mod.emitError("no ConSan hooks registered for target '") << target << "'";
      return signalPassFailure();
    }

    int numActiveMemTypes = (hasSharedMemoryBuffers(mod) ? 1 : 0) +
                            (hasTensorMemoryBuffers(mod) ? 1 : 0);
    int totalCaptures =
        tti::estimateConSanCaptureCount(numActiveMemTypes, hasBarriers(mod),
                                        getNumCommitKinds(mod, hooks.get()));
    int extraBytes = totalCaptures * tti::kCaptureSizeBytes;

    auto i32Ty = IntegerType::get(mod.getContext(), 32);
    mod.walk([&](ttg::WarpSpecializeOp ws) {
      ws->setAttr(tti::kConSanExtraCaptureBytesAttr,
                  IntegerAttr::get(i32Ty, extraBytes));
    });
  }
};

} // namespace

} // namespace instrument
} // namespace triton
} // namespace mlir
