#ifndef TRITONINSTRUMENT_CONSAN_TARGET_HOOKS_H
#define TRITONINSTRUMENT_CONSAN_TARGET_HOOKS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "llvm/ADT/APInt.h"
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
      uint32_t offset;
      uint32_t length;

      triton::BufferRegion getRegion(unsigned numCTAs) const {
        triton::BufferRegion region;
        region.baseOffset = offset;
        region.length = length;
        for (unsigned cta = 0; cta < numCTAs; ++cta)
          region.ctaAddresses.push_back(
              {cta, triton::AddressSet::fromRange(offset, length)});
        return region;
      }
    };

    RW rw;
    std::optional<gpu::SharedKind> sharedKind;
    std::variant<Value, StaticSharedBuffer> buffer;
    std::string operandName = "";
    uint32_t length = 0;

    Effects(RW rw, Value buf, std::string operandName = "",
            std::optional<gpu::SharedKind> sharedKind = std::nullopt)
        : rw(rw), sharedKind(sharedKind), buffer(buf), operandName(operandName),
          length(getMemDescLength(buf)) {}

    Effects(RW rw, StaticSharedBuffer buffer,
            std::string operandName = "Scratch")
        : rw(rw), sharedKind(gpu::SharedKind::Generic), buffer(buffer),
          operandName(operandName), length(buffer.length) {}
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

  // Publish shared-cluster state initialization through the target's native
  // cluster rendezvous, which may consist of multiple operations.
  virtual SmallVector<Operation *>
  createInitClusterBarrier(ImplicitLocOpBuilder &b) const = 0;

  // A call-frame summary cannot represent target-specific synchronization or
  // compiler scratch that crosses CTA boundaries.
  virtual bool hasUnsummarizableCalleeState(Operation *op) const {
    return false;
  }

  virtual std::optional<MemEffectsOpInfo>
  getMemEffectsOpInfo(Operation *op) const {
    namespace ttg = triton::gpu;
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

  // AMD barriers are ordinary LDS objects and have no invalidate operation.
  virtual bool barrierWritesInvalidate() const { return false; }
};

inline FailureOr<std::optional<MemEffectsOpInfo>>
getConSanMemEffectsOpInfo(const ConSanTargetHooks &hooks, Operation *op) {
  std::optional<MemEffectsOpInfo> info = hooks.getMemEffectsOpInfo(op);
  Attribute rawSize = op->getAttr("allocation.size");
  if (!rawSize)
    return info;

  auto size = dyn_cast<IntegerAttr>(rawSize);
  auto offset = dyn_cast_or_null<IntegerAttr>(op->getAttr("allocation.offset"));
  if (!size || !offset || !size.getType().isSignlessInteger() ||
      !offset.getType().isSignlessInteger()) {
    op->emitError("compiler scratch metadata requires integer "
                  "allocation.offset and allocation.size attributes");
    return failure();
  }

  constexpr uint64_t maxSharedMemorySize = uint64_t{1} << 24;
  const llvm::APInt &offsetValue = offset.getValue();
  const llvm::APInt &sizeValue = size.getValue();
  bool valid = !offsetValue.isNegative() && !sizeValue.isNegative() &&
               !sizeValue.isZero() && offsetValue.ule(maxSharedMemorySize) &&
               sizeValue.ule(maxSharedMemorySize);
  if (valid) {
    uint64_t unsignedOffset = offsetValue.getZExtValue();
    uint64_t unsignedSize = sizeValue.getZExtValue();
    valid = unsignedSize <= maxSharedMemorySize - unsignedOffset;
  }
  if (!valid) {
    InFlightDiagnostic diagnostic =
        op->emitError("invalid compiler scratch allocation metadata: offset ");
    if (offsetValue.getBitWidth() <= 64)
      diagnostic << offset.getInt();
    else
      diagnostic << offset;
    diagnostic << ", size ";
    if (sizeValue.getBitWidth() <= 64)
      diagnostic << size.getInt();
    else
      diagnostic << size;
    diagnostic << "; the interval must be non-empty and fit in the "
                  "24-bit shared-memory address space";
    return failure();
  }

  if (!info)
    info.emplace();
  if (info->trackingKind == MemEffectsOpInfo::TrackingKind::None)
    info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
  if (info->trackingKind != MemEffectsOpInfo::TrackingKind::Barrier ||
      info->implicitCommit) {
    op->emitError("compiler scratch cannot be combined with "
                  "asynchronous operation effect tracking");
    return failure();
  }
  // A ConSan write performs both read- and write-conflict checks, so it is the
  // conservative read/write summary for compiler-owned scratch.
  StringRef name = isa<CallOpInterface>(op) ? "Callee scratch" : "Scratch";
  info->operandEffects.emplace_back(
      RW::Write,
      MemEffectsOpInfo::Effects::StaticSharedBuffer{
          static_cast<uint32_t>(offsetValue.getZExtValue()),
          static_cast<uint32_t>(sizeValue.getZExtValue())},
      name.str());
  return info;
}

LogicalResult runConcurrencySanitizer(ModuleOp module,
                                      const ConSanTargetHooks &hooks);

using ConSanHooksFactory = std::function<std::unique_ptr<ConSanTargetHooks>()>;
void registerConSanHooks(llvm::StringRef key, ConSanHooksFactory factory);
std::unique_ptr<ConSanTargetHooks> createConSanHooks(llvm::StringRef key);

} // namespace mlir::triton::instrument

#endif // TRITONINSTRUMENT_CONSAN_TARGET_HOOKS_H
