#ifndef TRITONINSTRUMENT_CONSAN_TARGET_HOOKS_H
#define TRITONINSTRUMENT_CONSAN_TARGET_HOOKS_H

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>

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
    struct StaticSharedBuffer {
      BufferRegion region;
    };
    using Buffer = std::variant<Value, StaticSharedBuffer>;

    enum RW { Read, Write } rw;
    enum class Proxy { Generic, Async } proxy;
    Buffer buffer;
    std::string operandName = "";
    uint32_t length = 0;

    Effects(RW rw, Value buf, std::string operandName = "",
            Proxy proxy = Proxy::Generic)
        : rw(rw), proxy(proxy), buffer(buf), operandName(operandName),
          length(getMemDescLength(buf)) {}

    Effects(RW rw, BufferRegion region, std::string operandName = "",
            Proxy proxy = Proxy::Generic)
        : rw(rw), proxy(proxy), buffer(StaticSharedBuffer{region}),
          operandName(operandName), length(region.length) {}
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

  virtual std::optional<WaitOpInfo> getWaitOpInfo(Operation *op) const = 0;

  virtual std::optional<AsyncProxyFenceInfo>
  getAsyncProxyFenceInfo(Operation *op) const {
    return std::nullopt;
  }

  virtual bool needsAsyncProxyFenceTracking(ModuleOp module) const {
    return false;
  }

  virtual Value getIssuerCTAPred(ImplicitLocOpBuilder &b,
                                 Operation *op) const = 0;

  // Creates the target-specific cluster rendezvous used to publish ConSan's
  // shared lock initialization. Some targets represent it as multiple ops.
  virtual SmallVector<Operation *>
  createInitClusterBarrier(ImplicitLocOpBuilder &b) const = 0;

  // For scratch-backed operations whose result is replicated across CTAs,
  // returns the CTA-id bits that identify peers sharing one scratch result.
  // The target hook also predicates instrumentation to the producer CTA.
  virtual std::optional<uint16_t>
  getScratchCTABroadcastMask(Operation *op) const {
    return std::nullopt;
  }

  virtual FailureOr<std::optional<MemEffectsOpInfo>>
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
    if (auto gatherOp = dyn_cast<ttg::LocalGatherOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        gatherOp.getSrc());
    }
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        storeOp.getDst());
    }
    if (auto scatterOp = dyn_cast<ttg::LocalScatterOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        scatterOp.getDst());
    }
    if (auto atomicOp = dyn_cast<ttg::LocalAtomicScatterRMWOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        atomicOp.getDst());
    }
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                          allocOp.getResult());
      }
    }

    // Allocation runs before ConSan and publishes allocation.size only for
    // operation-local compiler scratch, making it the authoritative generic
    // marker for an SSA-less shared-memory effect. allocation.offset alone is
    // also used by explicit buffers, virtual call frames, function scheduler
    // state, and late synthetic conversions.
    Attribute sizeAttr = op->getAttr("allocation.size");
    if (!sizeAttr)
      return info;
    auto size = dyn_cast<IntegerAttr>(sizeAttr);
    auto offset = op->getAttrOfType<IntegerAttr>("allocation.offset");
    if (!size || !offset) {
      op->emitError()
          << "compiler scratch metadata requires integer allocation.offset "
             "and allocation.size attributes";
      return failure();
    }

    int64_t offsetValue = offset.getInt();
    int64_t sizeValue = size.getInt();
    constexpr uint64_t maxSharedMemorySize =
        uint64_t{kSharedMemoryObjectMask} + 1;
    bool isValid = offsetValue >= 0 && sizeValue > 0;
    if (isValid) {
      uint64_t unsignedOffset = static_cast<uint64_t>(offsetValue);
      uint64_t unsignedSize = static_cast<uint64_t>(sizeValue);
      isValid = unsignedOffset <= maxSharedMemorySize &&
                unsignedSize <= maxSharedMemorySize - unsignedOffset;
    }
    if (!isValid) {
      op->emitError() << "invalid compiler scratch allocation metadata: offset "
                      << offsetValue << ", size " << sizeValue
                      << "; the interval must be non-empty and fit in the "
                         "24-bit shared-memory address space";
      return failure();
    }

    if (!info)
      info.emplace();
    if (info->trackingKind == MemEffectsOpInfo::TrackingKind::None)
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
    if (info->trackingKind != MemEffectsOpInfo::TrackingKind::Barrier) {
      op->emitError("compiler scratch cannot be combined with asynchronous "
                    "operation effect tracking");
      return failure();
    }
    info->operandEffects.emplace_back(
        MemEffectsOpInfo::Effects::Write,
        BufferRegion{static_cast<uint32_t>(offsetValue),
                     static_cast<uint32_t>(sizeValue)},
        "Scratch");
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
