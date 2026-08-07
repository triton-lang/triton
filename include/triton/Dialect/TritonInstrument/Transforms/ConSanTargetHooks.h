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
  //
  // TensorCore tracks only tensor-core reads and writes issued by the current
  // partition, excluding its generic visibility frontier. In practice, this
  // distinction matters for distributed shared memory: arrival on a remote
  // mbarrier is not ordered with prior remote shared-memory writes.
  enum class BarrierTrackingMode {
    Frontier,
    EffectWrites,
    TensorCore,
  };
  struct Effects {
    RW rw;
    std::optional<gpu::SharedKind> sharedKind;
    Value buf;
    std::string operandName = "";
    uint32_t length = 0;

    Effects(RW rw, Value buf, std::string operandName = "",
            std::optional<gpu::SharedKind> sharedKind = std::nullopt)
        : rw(rw), sharedKind(sharedKind), buf(buf), operandName(operandName),
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

struct AsyncProxyFenceInfo {
  bool cluster;
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

  virtual std::optional<WaitOpInfo>
  getWaitOpInfo(Operation *op, const AuxDataMap &auxData) const = 0;

  virtual std::optional<AsyncProxyFenceInfo>
  getAsyncProxyFenceInfo(Operation *op) const {
    return std::nullopt;
  }

  virtual bool needsAsyncProxyFenceTracking(ModuleOp module) const {
    return false;
  }

  virtual Value getIssuerCTAPred(ImplicitLocOpBuilder &b,
                                 Operation *op) const = 0;

  virtual std::optional<MemEffectsOpInfo>
  getMemEffectsOpInfo(Operation *op) const {
    namespace ttg = triton::gpu;
    if (getBarrierInitInfo(op) || getBarrierInvalidateInfo(op))
      return std::nullopt;
    MemEffectsOpInfo info;
    if (isa<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      info.trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
      info.commitKind = CommitKind::AsyncCp;
    } else {
      info.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
    }
    for (const auto &access : getMemoryAccesses(op)) {
      if (access.isShared(ttg::SharedKind::Barrier))
        continue;
      info.operandEffects.emplace_back(access.isWrite ? RW::Write : RW::Read,
                                       access.value, "", access.sharedKind);
    }
    if (info.operandEffects.empty())
      return std::nullopt;
    return info;
  }

  // Returns commit kinds used by addWriteChecks to detect outstanding
  // write accesses to shared memory.
  virtual SmallVector<CommitKindDesc> getOutstandingWriteCommitKinds() const {
    return {{CommitKind::AsyncCp, "async_copy_global_to_shared"}};
  }

  // Returns commit kinds used by addReadChecks to detect outstanding
  // read accesses to shared memory.
  virtual SmallVector<CommitKindDesc>
  getOutstandingReadCommitKinds(const AuxDataMap &auxData) const {
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
                                      const ConSanTargetHooks &hooks);

using ConSanHooksFactory = std::function<std::unique_ptr<ConSanTargetHooks>()>;
void registerConSanHooks(llvm::StringRef key, ConSanHooksFactory factory);
std::unique_ptr<ConSanTargetHooks> createConSanHooks(llvm::StringRef key);

} // namespace mlir::triton::instrument

#endif // TRITONINSTRUMENT_CONSAN_TARGET_HOOKS_H
