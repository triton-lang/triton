#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"

namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;
namespace tti = mlir::triton::instrument;

using tti::BarrierInitInfo;
using tti::BarrierInvalidateInfo;
using tti::BarrierWaitInfo;
using tti::CommitKindDesc;
using tti::MemEffectsOpInfo;
using tti::WaitOpInfo;

namespace mlir {

class AMDConSanHooks : public tti::ConSanTargetHooks {
public:
  bool isTMAOp(Operation *op) const override {
    return isa<ttag::AsyncTDMCopyGlobalToLocalOp,
               ttag::AsyncTDMCopyLocalToGlobalOp>(op);
  }

  // TDM ops from the same warp complete in issue order. ConSan's thread model
  // uses one logical TDM thread per WS partition, so the outstanding-commit
  // check excludes the calling thread's own column to avoid intra-partition
  // false positives while still detecting cross-partition races.
  bool isOrderedCommitKind(tti::CommitKind::Kind kind) const override {
    return kind == tti::CommitKind::TmaStore;
  }

  std::optional<BarrierInitInfo>
  getBarrierInitInfo(Operation *op) const override {
    if (auto initOp = dyn_cast<ttag::InitBarrierOp>(op))
      return BarrierInitInfo{initOp.getBarrier(), initOp.getCount()};
    return std::nullopt;
  }

  std::optional<BarrierWaitInfo>
  getBarrierWaitInfo(Operation *op) const override {
    if (auto waitOp = dyn_cast<ttag::WaitBarrierOp>(op))
      return BarrierWaitInfo{waitOp.getBarrier(), waitOp.getPhase(),
                             /*pred=*/Value()};
    return std::nullopt;
  }

  std::optional<BarrierInvalidateInfo>
  getBarrierInvalidateInfo(Operation *op) const override {
    return std::nullopt;
  }

  std::optional<WaitOpInfo> getWaitOpInfo(Operation *op) const override {
    // AMD amdgpu::AsyncWaitOp replaces ttg::AsyncWaitOp after
    // UpdateAsyncWaitCount. Read the preserved commit-group count.
    if (auto asyncWaitOp = dyn_cast<ttag::AsyncWaitOp>(op)) {
      if (auto attr = asyncWaitOp->getAttrOfType<IntegerAttr>(
              "ttg.num_commit_groups")) {
        return WaitOpInfo{tti::CommitKind::AsyncCp, (int)attr.getInt(),
                          /*transferWrites=*/true, /*transferReads=*/false};
      }
      return std::nullopt;
    }
    if (auto tdmWaitOp = dyn_cast<ttag::AsyncTDMWait>(op)) {
      return WaitOpInfo{tti::CommitKind::TmaStore,
                        static_cast<int>(tdmWaitOp.getNum()),
                        /*transferWrites=*/true, /*transferReads=*/true};
    }
    // AMD AsyncTDMIntrinsicWait: replaces AsyncTDMWait after
    // UpdateAsyncWaitCount. Read the preserved TDM operation count.
    if (auto tdmWaitOp = dyn_cast<ttag::AsyncTDMIntrinsicWait>(op)) {
      if (auto attr =
              tdmWaitOp->getAttrOfType<IntegerAttr>("ttg.num_tdm_ops")) {
        return WaitOpInfo{tti::CommitKind::TmaStore, (int)attr.getInt(),
                          /*transferWrites=*/true, /*transferReads=*/true};
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  Value getIssuerCTAPred(ImplicitLocOpBuilder & /*b*/,
                         Operation * /*op*/) const override {
    return nullptr;
  }

  std::optional<MemEffectsOpInfo>
  getMemEffectsOpInfo(Operation *op) const override {
    auto info = ConSanTargetHooks::getMemEffectsOpInfo(op);
    if (info)
      return info;
    // AsyncTDMCopyGlobalToLocalOp: Async copy from global to shared memory.
    // When a barrier is present, TDM signals it once per warp.
    // The barrier init count must account for this (e.g. count=NUM_WARPS),
    // so ConSan decrements the shadow counter by numWarps (one per warp).
    // When no barrier is present, completion is tracked via the TDM wait
    // counter (AsyncTDMWait), modeled as CommitCount with implicitCommit.
    if (auto copyOp = dyn_cast<ttag::AsyncTDMCopyGlobalToLocalOp>(op)) {
      info.emplace();
      if (Value barrier = copyOp.getBarrier()) {
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        int numWarps = ttg::lookupNumWarps(copyOp);
        info->barriers.push_back({barrier, nullptr, numWarps});
      } else {
        info->trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
        info->commitKind = tti::CommitKind::TmaStore;
        info->implicitCommit = true;
      }
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        copyOp.getResult());
    }
    // AsyncTDMCopyLocalToGlobalOp: Async copy from shared to global memory
    // Same principles as AsyncTDMCopyGlobalToLocalOp apply.
    if (auto storeOp = dyn_cast<ttag::AsyncTDMCopyLocalToGlobalOp>(op)) {
      info.emplace();
      if (Value barrier = storeOp.getBarrier()) {
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        int numWarps = ttg::lookupNumWarps(storeOp);
        info->barriers.push_back({barrier, nullptr, numWarps});
      } else {
        info->trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
        info->commitKind = tti::CommitKind::TmaStore;
        info->implicitCommit = true;
      }
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        storeOp.getSrc());
    }
    // AMD ArriveBarrierOp: Explicit barrier arrival.
    // Arrive is per-THREAD when called explicitly (unlike TDM which
    // is per-warp). Scale by total threads in the partition so ConSan's shadow
    // barrier state matches the hardware arrival count.
    if (auto arriveOp = dyn_cast<ttag::ArriveBarrierOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      int numWarps = ttg::lookupNumWarps(arriveOp);
      auto mod = arriveOp->getParentOfType<ModuleOp>();
      int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
      int totalCount = (int)arriveOp.getCount() * numWarps * threadsPerWarp;
      info->barriers.push_back({arriveOp.getBarrier(), nullptr, totalCount});
    }
    return info;
  }

  SmallVector<CommitKindDesc> getOutstandingWriteCommitKinds() const override {
    return {{tti::CommitKind::AsyncCp, "async_copy_global_to_shared"},
            {tti::CommitKind::TmaStore, "async_tdm_global_to_shared"}};
  }

  SmallVector<CommitKindDesc> getOutstandingReadCommitKinds() const override {
    return {{tti::CommitKind::TmaStore, "async_tdm_shared_to_global"}};
  }

  SmallVector<tti::CommitKind::Kind>
  getRequiredCommitKinds(ModuleOp module) const override {
    SmallVector<tti::CommitKind::Kind> kinds;
    bool needsTdm = false;
    bool needsAsyncCp = false;
    module.walk([&](Operation *op) {
      if (isa<ttag::AsyncTDMCopyGlobalToLocalOp,
              ttag::AsyncTDMCopyLocalToGlobalOp, ttag::AsyncTDMWait,
              ttag::AsyncTDMIntrinsicWait>(op))
        needsTdm = true;
      if (isa<ttag::AsyncWaitOp>(op))
        needsAsyncCp = true;
    });
    if (needsTdm)
      kinds.push_back(tti::CommitKind::TmaStore);
    if (needsAsyncCp)
      kinds.push_back(tti::CommitKind::AsyncCp);
    return kinds;
  }
};

void registerConSanAMDHooks() {
  tti::registerConSanHooks("amd",
                           [] { return std::make_unique<AMDConSanHooks>(); });
}

} // namespace mlir
