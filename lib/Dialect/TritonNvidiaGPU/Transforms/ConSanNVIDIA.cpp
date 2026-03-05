#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

using tti::BarrierInitInfo;
using tti::BarrierWaitInfo;
using tti::MemEffectsOpInfo;
using tti::WaitOpInfo;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

class NVIDIAConSanHooks : public tti::ConSanTargetHooks {
public:
  bool isTMAOp(Operation *op) const override {
    return isa<ttng::AsyncTMACopyGlobalToLocalOp,
               ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAGatherOp,
               ttng::AsyncTMAScatterOp>(op);
  }

  bool isPostInstrumentedOp(Operation *op) const override {
    return isa<ttng::WaitBarrierOp>(op);
  }

  std::optional<BarrierInitInfo>
  getBarrierInitInfo(Operation *op) const override {
    if (auto initOp = dyn_cast<ttng::InitBarrierOp>(op))
      return BarrierInitInfo{initOp.getAlloc(), initOp.getCount()};
    return std::nullopt;
  }

  std::optional<BarrierWaitInfo>
  getBarrierWaitInfo(Operation *op) const override {
    if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op))
      return BarrierWaitInfo{waitOp.getAlloc(), waitOp.getPhase(),
                             waitOp.getPred()};
    return std::nullopt;
  }

  std::optional<WaitOpInfo> getWaitOpInfo(Operation *op) const override {
    if (auto tmaStoreWaitOp = dyn_cast<ttng::TMAStoreWaitOp>(op))
      return WaitOpInfo{tti::CommitKind::TmaStore,
                        static_cast<int>(tmaStoreWaitOp.getPendings()),
                        /*transferWrites=*/false};
    return std::nullopt;
  }

  std::optional<MemEffectsOpInfo>
  getMemEffectsOpInfo(Operation *op) const override {
    std::optional<MemEffectsOpInfo> info;
    if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = copyOp.getPred();
      info->barriers.push_back({copyOp.getBarrier(), nullptr, /*count=*/0});
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        copyOp.getResult());
    }
    if (auto storeOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
      info->commitKind = tti::CommitKind::TmaStore;
      info->implicitCommit = true;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        storeOp.getSrc());
    }
    if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = gatherOp.getPred();
      info->barriers.push_back({gatherOp.getBarrier(), nullptr, /*count=*/0});
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        gatherOp.getResult());
    }
    if (auto scatterOp = dyn_cast<ttng::AsyncTMAScatterOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::None;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        scatterOp.getSrc());
    }
    if (auto arriveOp = dyn_cast<ttng::ArriveBarrierOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = arriveOp.getPred();
      info->barriers.push_back(
          {arriveOp.getAlloc(), nullptr, (int)arriveOp.getCount()});
    }
    return info;
  }

  SmallVector<tti::CommitKind::Kind>
  getRequiredCommitKinds(ModuleOp module) const override {
    SmallVector<tti::CommitKind::Kind> kinds;
    bool needsTmaStore = false;
    bool needsWgmma = false;
    module.walk([&](Operation *op) {
      if (isa<ttng::AsyncTMACopyLocalToGlobalOp, ttng::TMAStoreWaitOp>(op))
        needsTmaStore = true;
      if (isa<ttng::WarpGroupDotOp, ttng::WarpGroupDotWaitOp>(op))
        needsWgmma = true;
    });
    if (needsWgmma)
      kinds.push_back(tti::CommitKind::Wgmma);
    if (needsTmaStore)
      kinds.push_back(tti::CommitKind::TmaStore);
    return kinds;
  }
};

void registerConSanNVIDIAHooks() {
  tti::registerConSanHooks(
      "nvidia", [] { return std::make_unique<NVIDIAConSanHooks>(); });
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
