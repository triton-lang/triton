#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

using tti::AsyncProxyFenceInfo;
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

uint32_t getBlockBroadcastMask(Type type) {
  auto memDescTy = cast<ttg::MemDescType>(type);
  auto kBlock = StringAttr::get(type.getContext(), "block");
  return toLinearLayout(memDescTy).getFreeVariableMasks().lookup(kBlock);
}

} // namespace

class NVIDIAConSanHooks : public tti::ConSanTargetHooks {
public:
  bool isTMAOp(Operation *op) const override {
    return isa<ttng::TMAOpInterface, ttng::AsyncSharedStoreOp>(op);
  }

  bool isCLCOp(Operation *op) const override {
    return isa<ttng::CLCTryCancelOp>(op);
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

  std::optional<WaitOpInfo>
  getWaitOpInfo(Operation *op, const tti::AuxDataMap &) const override {
    if (auto tmaStoreWaitOp = dyn_cast<ttng::TMAStoreWaitOp>(op))
      return WaitOpInfo{tti::CommitKind::TmaStore,
                        static_cast<int>(tmaStoreWaitOp.getPendings()),
                        /*transferWrites=*/false, /*transferReads=*/true};
    return std::nullopt;
  }

  std::optional<AsyncProxyFenceInfo>
  getAsyncProxyFenceInfo(Operation *op) const override {
    if (auto fence = dyn_cast<ttng::FenceAsyncSharedOp>(op))
      return AsyncProxyFenceInfo{fence.getBCluster()};
    return std::nullopt;
  }

  bool needsAsyncProxyFenceTracking(ModuleOp module) const override {
    bool needed = false;
    module.walk([&](Operation *op) {
      needed |= hasSharedAccess(op, ttg::SharedKind::Async);
    });
    return needed;
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
    if (isa<ttng::BarrierExpectOp, ttng::ArriveBarrierOp>(op)) {
      std::optional<uint32_t> fromCTA;
      if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op))
        fromCTA = expectOp.getFromCTA();
      else
        fromCTA = cast<ttng::ArriveBarrierOp>(op).getFromCTA();
      if (fromCTA)
        mask = ~*fromCTA & (ttg::lookupNumCTAs(op) - 1);
    }
    if (auto loadOp = dyn_cast<ttng::TMALoadLikeOpInterface>(op)) {
      if (loadOp.getMulticast())
        mask = getBlockBroadcastMask(loadOp.getResult().getType());
    }
    if (auto storeOp = dyn_cast<ttng::TMAStoreLikeOpInterface>(op))
      mask = getBlockBroadcastMask(storeOp.getSrc().getType());
    if (isa<ttng::CLCTryCancelOp>(op) && ttg::lookupNumCTAs(op) > 1) {
      Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
      return arith::CmpIOp::create(b, arith::CmpIPredicate::eq, ctaId,
                                   arith::ConstantIntOp::create(b, 0, 32));
    }

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
    std::optional<MemEffectsOpInfo> info =
        ConSanTargetHooks::getMemEffectsOpInfo(op);
    if (!info) {
      if (!isa<ttng::BarrierExpectOp, ttng::TCGen5CommitOp,
               ttng::ArriveBarrierOp>(op))
        return std::nullopt;
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
    }

    SmallVector<std::pair<Value, StringRef>> namedOperands;
    if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op)) {
      info->pred = expectOp.getPred();
      info->barriers.push_back(
          {expectOp.getBarrier(), nullptr,
           /*count=*/1, MemEffectsOpInfo::BarrierTrackingMode::Frontier,
           /*txCount=*/static_cast<int>(expectOp.getSize())});
    }
    if (auto copyOp = dyn_cast<ttng::TMEMCopyOp>(op)) {
      namedOperands = {{copyOp.getSrc(), "Src"}, {copyOp.getDst(), "Dst"}};
    }
    if (auto mmav5Op = dyn_cast<ttng::MMAv5OpInterface>(op)) {
      info->pred = mmav5Op.getPredicate();
      for (auto [barrier, barrierPred] :
           llvm::zip(mmav5Op.getCompletionBarriers(),
                     mmav5Op.getCompletionBarrierPreds())) {
        info->barriers.push_back(
            {barrier, barrierPred, 1,
             MemEffectsOpInfo::BarrierTrackingMode::TensorCore});
      }
      namedOperands = {{mmav5Op.getA(), "A"},
                       {mmav5Op.getB(), "B"},
                       {mmav5Op.getAccumulator(), "Acc"}};
      if (auto mmaScaledOp = dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
        namedOperands.emplace_back(mmaScaledOp.getAScale(), "AScale");
        namedOperands.emplace_back(mmaScaledOp.getBScale(), "BScale");
      }
    }
    if (auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op)) {
      info->pred = commitOp.getPred();
      info->barriers.push_back(
          {commitOp.getBarrier(), nullptr, 1,
           MemEffectsOpInfo::BarrierTrackingMode::TensorCore});
    }
    if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      if (wgmmaOp.getIsAsync() == true) {
        info->trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
        info->commitKind = tti::CommitKind::Wgmma;
        info->implicitCommit = true;
        info->barriers = {};
      }
      namedOperands = {{wgmmaOp.getA(), "A"}, {wgmmaOp.getB(), "B"}};
    }
    if (auto loadOp = dyn_cast<ttng::TMALoadLikeOpInterface>(op)) {
      info->pred = loadOp.getPred();
      int txCount = tti::getMemDescLength(loadOp.getResult());
      if (loadOp.getMulticast()) {
        uint32_t resultMask =
            getBlockBroadcastMask(loadOp.getResult().getType());
        uint32_t barrierMask =
            getBlockBroadcastMask(loadOp.getBarrier().getType());
        uint32_t collapsedMask = resultMask & barrierMask;
        for (; collapsedMask; collapsedMask &= collapsedMask - 1)
          txCount *= 2;
      }
      info->barriers.push_back(
          {loadOp.getBarrier(), nullptr, /*count=*/0,
           MemEffectsOpInfo::BarrierTrackingMode::EffectWrites,
           /*txCount=*/-txCount});
    }
    if (auto storeOp = dyn_cast<ttng::AsyncSharedStoreOp>(op)) {
      info->barriers.push_back(
          {storeOp.getBarrier(), nullptr, /*count=*/0,
           MemEffectsOpInfo::BarrierTrackingMode::EffectWrites,
           /*txCount=*/
           -static_cast<int>(tti::getMemDescLength(storeOp.getDst()))});
    }
    if (auto tryCancelOp = dyn_cast<ttng::CLCTryCancelOp>(op)) {
      info->barriers.push_back(
          {tryCancelOp.getMbarrier(), nullptr, /*count=*/0,
           MemEffectsOpInfo::BarrierTrackingMode::EffectWrites,
           /*txCount=*/
           -static_cast<int>(tti::getMemDescLength(tryCancelOp.getResult()))});
    }
    if (isa<ttng::TMAStoreLikeOpInterface>(op)) {
      info->trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
      info->commitKind = tti::CommitKind::TmaStore;
      info->implicitCommit = true;
    }
    if (auto arriveOp = dyn_cast<ttng::ArriveBarrierOp>(op)) {
      info->pred = arriveOp.getPred();
      info->barriers.push_back(
          {arriveOp.getBarrier(), nullptr, (int)arriveOp.getCount()});
    }

    if (!namedOperands.empty()) {
      SmallVector<MemEffectsOpInfo::Effects> effects;
      for (auto [value, name] : namedOperands)
        for (auto it = info->operandEffects.begin();
             it != info->operandEffects.end(); ++it)
          if (it->buf == value) {
            effects.emplace_back(*it).operandName = name.str();
            break;
          }
      info->operandEffects = std::move(effects);
    }
    return info;
  }

  SmallVector<CommitKindDesc>
  getOutstandingReadCommitKinds(const tti::AuxDataMap &) const override {
    return {{tti::CommitKind::Wgmma, "warpgroup_mma operand read"},
            {tti::CommitKind::TmaStore, "async_copy_shared_to_global"}};
  }

  SmallVector<tti::CommitKind::Kind>
  getRequiredCommitKinds(ModuleOp module) const override {
    SmallVector<tti::CommitKind::Kind> kinds;
    bool needsTmaStore = false;
    bool needsWgmma = false;
    module.walk([&](Operation *op) {
      if (isa<ttng::TMAStoreLikeOpInterface, ttng::TMAStoreWaitOp>(op))
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
