#ifndef TRITONINSTRUMENT_CONSAN_TARGET_HOOKS_H
#define TRITONINSTRUMENT_CONSAN_TARGET_HOOKS_H

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace mlir::triton::instrument {

struct MemEffectsOpInfo {
  // Controls which memory effects become visible to a CTA after it waits on
  // this barrier.
  //
  // Frontier snapshots the issuing thread's current visibility frontier into
  // the barrier. A later wait publishes whatever shared/tensor memory writes
  // and reads were visible to that logical thread before the arrive/commit. Use
  // this for ordering operations whose semantics are a release of prior work.
  //
  // EffectWrites does not snapshot the whole thread frontier. Instead, it
  // attaches only the explicit write effects of this op to the barrier. A later
  // wait publishes those op-local writes and nothing else. Use this for PTX ops
  // that perform the write and also signal the barrier via
  // `mbarrier::complete_tx`.
  enum class BarrierTrackingMode {
    Frontier,
    EffectWrites,
  };
  struct Effects {
    enum RW { Read, Write } rw;
    Value buf;
    std::string operandName = "";
    uint32_t length = 0;

    Effects(RW rw, Value buf, std::string operandName = "")
        : rw(rw), buf(buf), operandName(operandName),
          length(getMemDescLength(buf)) {}
  };
  struct BarrierInfo {
    Value barrier;
    Value pred;
    int count;
    BarrierTrackingMode trackingMode = BarrierTrackingMode::Frontier;
    int txCount = 0;
  };
  enum class TrackingKind {
    None,
    Barrier,
    wgmmaCommit,
    CommitCount
  } trackingKind = TrackingKind::None;

  CommitKind::Kind commitKind = CommitKind::None;

  SmallVector<BarrierInfo> barriers;
  Value pred;
  SmallVector<Effects> operandEffects;
  bool implicitCommit = false;
};

struct BarrierInitInfo {
  Value alloc;
  uint32_t count;
};

struct BarrierWaitInfo {
  Value alloc;
  Value phase;
  Value pred;
};

struct BarrierInvalidateInfo {
  Value alloc;
};

struct WaitOpInfo {
  CommitKind::Kind commitKind;
  int pendingCount;
  bool transferWrites;
  bool transferReads;
};

struct CommitKindDesc {
  CommitKind::Kind kind;
  std::string operationDesc;
};

class ConSanTargetHooks {
public:
  virtual ~ConSanTargetHooks() = default;

  virtual bool isTMAOp(Operation *op) const = 0;

  virtual bool isCLCOp(Operation *op) const { return false; }

  virtual std::optional<BarrierInitInfo>
  getBarrierInitInfo(Operation *op) const = 0;

  virtual std::optional<BarrierWaitInfo>
  getBarrierWaitInfo(Operation *op) const = 0;

  virtual std::optional<BarrierInvalidateInfo>
  getBarrierInvalidateInfo(Operation *op) const = 0;

  virtual std::optional<WaitOpInfo> getWaitOpInfo(Operation *op) const = 0;

  virtual Value getIssuerCTAPred(ImplicitLocOpBuilder &b,
                                 Operation *op) const = 0;

  virtual std::optional<MemEffectsOpInfo>
  getMemEffectsOpInfo(Operation *op) const {
    namespace ttg = triton::gpu;
    std::optional<MemEffectsOpInfo> info;
    if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
      info->commitKind = CommitKind::AsyncCp;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        copyOp.getResult());
    }
    if (auto loadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        loadOp.getSrc());
    }
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        storeOp.getDst());
    }
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                          allocOp.getResult());
      }
    }
    return info;
  }

  // Returns commit kinds used by addWriteChecks to detect outstanding
  // write accesses to shared memory.
  virtual SmallVector<CommitKindDesc> getOutstandingWriteCommitKinds() const {
    return {{CommitKind::AsyncCp, "async_copy_global_to_shared"}};
  }

  // Returns commit kinds used by addReadChecks to detect outstanding
  // read accesses to shared memory.
  virtual SmallVector<CommitKindDesc> getOutstandingReadCommitKinds() const {
    return {};
  }

  // Returns true for commit kinds whose ops complete in issue order within a
  // warp. ConSan's thread model tracks one logical
  // thread per WS partition, so it cannot distinguish intra-warp ordering from
  // cross-warp races inside the same partition. For such kinds, the
  // outstanding-commit check excludes the calling thread's own column, avoiding
  // intra-partition false positives while still detecting cross-partition
  // races.
  virtual bool isOrderedCommitKind(CommitKind::Kind kind) const {
    return false;
  }

  virtual SmallVector<CommitKind::Kind>
  getRequiredCommitKinds(ModuleOp module) const = 0;
};

LogicalResult runConcurrencySanitizer(ModuleOp module,
                                      const ConSanTargetHooks *hooks);

using ConSanHooksFactory = std::function<std::unique_ptr<ConSanTargetHooks>()>;
void registerConSanHooks(llvm::StringRef key, ConSanHooksFactory factory);
std::unique_ptr<ConSanTargetHooks> createConSanHooks(llvm::StringRef key);

} // namespace mlir::triton::instrument

#endif // TRITONINSTRUMENT_CONSAN_TARGET_HOOKS_H
