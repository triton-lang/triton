#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetInfo.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <array>

namespace mlir {
namespace triton {
namespace instrument {

namespace {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

bool hasSharedMemoryBuffers(ModuleOp mod) {
  bool result = false;
  mod.walk([&](ttg::LocalAllocOp op) { result |= op.isSharedMemoryAlloc(); });
  // Warp specialization itself uses compiler-owned shared scratch. ConSan's
  // lock capture adds another scratch user, so reserve the shared-memory state
  // captures whenever a warp-specialized region is present.
  mod.walk([&](ttg::WarpSpecializeOp) { result = true; });
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

int getNumCommitKinds(ModuleOp mod, const ConSanTargetInfo &targetInfo) {
  std::array<bool, tti::CommitKind::NumCommitKinds> commitKinds{};
  if (hasCpAsync(mod))
    commitKinds[tti::CommitKind::AsyncCp] = true;
  for (auto kind : targetInfo.getRequiredCommitKinds(mod)) {
    if (kind >= 0 && kind < tti::CommitKind::NumCommitKinds)
      commitKinds[kind] = true;
  }

  int result = 0;
  for (bool required : commitKinds)
    result += required;
  return result;
}

} // namespace

void prepareConSanCaptures(ModuleOp mod, const ConSanTargetInfo &targetInfo,
                          bool hasClusterBarriers) {
  bool hasSharedBuffers = hasSharedMemoryBuffers(mod);
  int numActiveMemTypes =
      (hasSharedBuffers ? 1 : 0) + (hasTensorMemoryBuffers(mod) ? 1 : 0);
  int totalCaptures = tti::estimateConSanCaptureCount(
      numActiveMemTypes, hasBarriers(mod), hasClusterBarriers,
      getNumCommitKinds(mod, targetInfo),
      hasSharedBuffers && targetInfo.needsAsyncProxyFenceTracking(mod));
  int extraBytes = totalCaptures * tti::kCaptureSizeBytes;

  auto i32Ty = IntegerType::get(mod.getContext(), 32);
  mod.walk([&](ttg::WarpSpecializeOp ws) {
    ws->setAttr(tti::kConSanExtraCaptureBytesAttr,
                IntegerAttr::get(i32Ty, extraBytes));
  });
}

} // namespace instrument
} // namespace triton
} // namespace mlir
