#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

using tti::BarrierInitInfo;
using tti::BarrierInvalidateInfo;
using tti::BarrierWaitInfo;
using tti::CommitKindDesc;
using tti::MemEffectsOpInfo;
using tti::WaitOpInfo;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

namespace {

Value getLeaderCTAPredicate(ImplicitLocOpBuilder &b, uint32_t broadcastMask) {
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  Value ctaIdInGroup = arith::AndIOp::create(
      b, ctaId, arith::ConstantIntOp::create(b, broadcastMask, 32));
  return arith::CmpIOp::create(b, arith::CmpIPredicate::eq, ctaIdInGroup,
                               arith::ConstantIntOp::create(b, 0, 32));
}

} // namespace

class NVIDIAConSanHooks : public tti::ConSanTargetHooks {
public:
  bool isTMAOp(Operation *op) const override {
    return isa<ttng::AsyncTMACopyGlobalToLocalOp,
               ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAGatherOp,
               ttng::AsyncTMAScatterOp>(op);
  }

  std::optional<BarrierInitInfo>
  getBarrierInitInfo(Operation *op) const override {
    if (auto initOp = dyn_cast<ttng::InitBarrierOp>(op)) {
      auto barrierTy = initOp.getAlloc().getType();
      // Match mbarrier.init lowering: the leader barrier accounts for every CTA
      // that routes arrivals to it.
      uint32_t count = initOp.getCount() * ttg::lookupNumCTAs(op) /
                       barrierTy.getNumElements();
      return BarrierInitInfo{initOp.getAlloc(), count};
    }
    return std::nullopt;
  }

  std::optional<BarrierWaitInfo>
  getBarrierWaitInfo(Operation *op) const override {
    if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op))
      return BarrierWaitInfo{waitOp.getBarrier(), waitOp.getPhase(),
                             waitOp.getPred()};
    return std::nullopt;
  }

  std::optional<BarrierInvalidateInfo>
  getBarrierInvalidateInfo(Operation *op) const override {
    if (auto invalOp = dyn_cast<ttng::InvalBarrierOp>(op))
      return BarrierInvalidateInfo{invalOp.getAlloc()};
    return std::nullopt;
  }

  std::optional<WaitOpInfo> getWaitOpInfo(Operation *op) const override {
    if (auto tmaStoreWaitOp = dyn_cast<ttng::TMAStoreWaitOp>(op))
      return WaitOpInfo{tti::CommitKind::TmaStore,
                        static_cast<int>(tmaStoreWaitOp.getPendings()),
                        /*transferWrites=*/false, /*transferReads=*/true};
    return std::nullopt;
  }

  Value getIssuerCTAPred(ImplicitLocOpBuilder &b,
                         Operation *op) const override {
    // mask = 0 means no CTA predication.
    uint32_t mask = 0;
    auto getBarrierMask = [&](Value barrier) {
      auto barrierTy = cast<ttg::MemDescType>(barrier.getType());
      auto kBlock = StringAttr::get(op->getContext(), "block");
      return toLinearLayout(barrierTy).getFreeVariableMasks().lookup(kBlock);
    };
    if (auto initOp = dyn_cast<ttng::InitBarrierOp>(op))
      mask = getBarrierMask(initOp.getAlloc());
    if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op))
      mask = getBarrierMask(waitOp.getAlloc());
    if (auto invalOp = dyn_cast<ttng::InvalBarrierOp>(op))
      mask = getBarrierMask(invalOp.getAlloc());

    // In 2CTA tcgen05 and tmem_copy, only the even CTA in each (i, i^1) pair
    // issues the op.
    if (isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp, ttng::TCGen5CommitOp,
            ttng::TMEMCopyOp>(op) &&
        ttng::getModuleTwoCTAs(op))
      mask = 0x1;
    if (!mask)
      return nullptr;
    return getLeaderCTAPredicate(b, mask);
  }

  std::optional<MemEffectsOpInfo>
  getMemEffectsOpInfo(Operation *op) const override {
    auto info = ConSanTargetHooks::getMemEffectsOpInfo(op);
    if (info)
      return info;
    if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op)) {
      // TODO: For async TMA barriers, the barrier "arrive" corresponding to the
      // completion mechanism is modeled by barrier_expect. Individual
      // async_tma_copy ops should not decrement the barrier state, otherwise
      // multiple copies using the same barrier would incorrectly advance the
      // phase multiple times. This should be improved bu tracking the barrier
      // expected byte count, and "arriving" the barrier when the expected byte
      // count is reached.
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = expectOp.getPred();
      info->barriers.push_back({expectOp.getBarrier(), nullptr,
                                /*count=*/1,
                                MemEffectsOpInfo::BarrierTrackingMode::None});
    }
    if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        loadOp.getSrc());
    }
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        storeOp.getDst());
    }
    if (auto allocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                          allocOp.getResult());
      }
    }
    if (auto copyOp = dyn_cast<ttng::TMEMCopyOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        copyOp.getSrc(), "Src");
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        copyOp.getDst(), "Dst");
      if (copyOp.getBarrier())
        info->barriers.push_back({copyOp.getBarrier(), nullptr, 1});
    }
    if (auto mmav5Op = dyn_cast<ttng::MMAv5OpInterface>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = mmav5Op.getPredicate();
      for (auto [barrier, barrierPred] :
           llvm::zip(mmav5Op.getCompletionBarriers(),
                     mmav5Op.getCompletionBarrierPreds())) {
        info->barriers.push_back({barrier, barrierPred, 1});
      }
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        mmav5Op.getA(), "A");
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        mmav5Op.getB(), "B");
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        mmav5Op.getAccumulator(), "Acc");
      if (auto mmaScaledOp = dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
        info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                          mmaScaledOp.getAScale(), "AScale");
        info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                          mmaScaledOp.getBScale(), "BScale");
      }
    }
    if (auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = commitOp.getPred();
      info->barriers.push_back({commitOp.getBarrier(), nullptr, 1});
    }
    if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      if (wgmmaOp.getIsAsync() == true) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
        info->commitKind = tti::CommitKind::Wgmma;
        info->implicitCommit = true;
        info->barriers = {};
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getA().getType().getEncoding())) {
          info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                            wgmmaOp.getA(), "A");
        }
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getB().getType().getEncoding())) {
          info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                            wgmmaOp.getB(), "B");
        }
      }
    }
    if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = copyOp.getPred();
      info->barriers.push_back(
          {copyOp.getBarrier(), nullptr, /*count=*/0,
           MemEffectsOpInfo::BarrierTrackingMode::EffectWrites});
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
      info->barriers.push_back(
          {gatherOp.getBarrier(), nullptr, /*count=*/0,
           MemEffectsOpInfo::BarrierTrackingMode::EffectWrites});
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
          {arriveOp.getBarrier(), nullptr, (int)arriveOp.getCount()});
    }
    return info;
  }

  SmallVector<CommitKindDesc> getOutstandingReadCommitKinds() const override {
    return {{tti::CommitKind::Wgmma, "warpgroup_mma operand read"},
            {tti::CommitKind::TmaStore, "async_copy_shared_to_global"}};
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
