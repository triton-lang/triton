#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/FunctionBuilder.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/StringMap.h"

// clang-format off
// Concurrency Sanitizer data structures:
// ConSan keeps auxilary data requied for tracking memory accesses in tensors.
// These tensors are stored as a distributed tensor or in global scratch memory.
//
// Name              | Storage | Rank/Type       | Description
// ------------------|---------|-----------------|------------
// buffers           | tensor  | <B x i64>       | Base pointers of all (sub)buffers
// barriers          | tensor  | <K x i64>       | Pointers to all individual mbarriers
// barrierStates     | scratch | <K x i64>       | Packed barrier phase (bit 0), arrival counts (bits[1..20] init, [21..40] current), and signed tx-count (bits[41..61]); zero means invalid/uninitialized
// waiting           | scratch | <K x i32>       | Two bits per thread: waiting flag bit (LSB), stored phase bit (bit 1)
// writeVisibility   | scratch | <B x i64>       | Per-buffer thread-visibility bitmask (bit i => thread i visible)
// readVisibility    | scratch | <B x T x i64>   | Per-buffer, per-thread visibility lanes (row-updated; values are bitmasks)
// writeTracking     | scratch | <B x K x i8>    | Map buffers -> barriers that track writes
// readTracking      | scratch | <B x K x i64>   | Map buffers -> barriers that track reads
// outstandingCommits
//   (async/wgmma)   | scratch | <B x T x i8>    | Number of outstanding commits per buffer/thread (2D replaces prior 1D)
// clang-format on

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

#define GEN_PASS_DEF_TRITONINSTRUMENTCONCURRENCYSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

static llvm::StringMap<ConSanHooksFactory> &getHooksRegistry() {
  static llvm::StringMap<ConSanHooksFactory> registry;
  return registry;
}

void registerConSanHooks(llvm::StringRef key, ConSanHooksFactory factory) {
  getHooksRegistry()[key] = std::move(factory);
}

std::unique_ptr<ConSanTargetHooks> createConSanHooks(llvm::StringRef key) {
  auto it = getHooksRegistry().find(key);
  if (it != getHooksRegistry().end())
    return it->second();
  return nullptr;
}

namespace {

// OpBuilder listener tracking operations added to the builder to be wrapped
// with a lock acquire/release pair.
class CriticalSectionListener : public ImplicitLocOpBuilder::Listener {
public:
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint /*previous*/) override {
    if (firstOp == nullptr) {
      firstOp = op;
    }
    lastOp = op;
  }
  void maybeWrapWithCriticalSection(ImplicitLocOpBuilder &b,
                                    AuxDataMap &auxData, Value pred) {
    Operation *_firstOp = firstOp;
    Operation *_lastOp = lastOp;
    if (firstOp != nullptr && lastOp != nullptr) {
      assert(firstOp->getParentRegion() == lastOp->getParentRegion());
      b.setInsertionPoint(_firstOp);
      tti::ExperimentalLockAcquireOp::create(b, auxData.lock.at(_firstOp).value,
                                             pred);
      b.setInsertionPointAfter(_lastOp);
      tti::ExperimentalLockReleaseOp::create(b, auxData.lock.at(_firstOp).value,
                                             pred);
    }
  }

private:
  Operation *firstOp = nullptr;
  Operation *lastOp = nullptr;
};

bool isTensorCoreOp(Operation *op) {
  return isa<ttng::MMAv5OpInterface, ttng::TCGen5CommitOp, ttng::TMEMCopyOp>(
      op);
}

std::optional<int> maybeGetPartitionIdx(Operation *op) {
  if (auto wsOp = op->getParentOfType<ttg::WarpSpecializePartitionsOp>()) {
    return op->getParentRegion()->getRegionNumber();
  }
  if (Operation *parent = op->getParentOp()) {
    return maybeGetPartitionIdx(parent);
  }
  return std::nullopt;
}

int getCurrentThread(Operation *op, const ConSanTargetHooks *hooks) {
  // Default partition is 0, other partitions are idx + 1
  int thread = maybeGetPartitionIdx(op).value_or(-1) + 1;
  if (hooks->isTMAOp(op)) {
    thread += TMA_THREAD_OFFSET;
    return thread;
  }
  if (isTensorCoreOp(op)) {
    thread += TC_THREAD_OFFSET;
    return thread;
  }
  return thread;
}

int getBaseThread(int thread) { return thread % NUM_THREADS; }

// Peer threads are the equivalent threads in the TMA, TC and normal
// thread classes.
// If a thread is a base thread, return the mask with the peers, otherwise
// return the mask with the thread itself.
uint64_t getThreadPeersMask(int thread) {
  uint64_t mask = 1ULL << thread;
  if (thread < NUM_THREADS) {
    mask |= 1ULL << (thread + TMA_THREAD_OFFSET);
    mask |= 1ULL << (thread + TC_THREAD_OFFSET);
  }
  return mask;
}

int getActiveMask(Operation *op) {
  int numParts = 1;

  if (auto wsOp = op->getParentOfType<ttg::WarpSpecializeOp>()) {
    numParts = wsOp.getPartitionRegions().size() + 1;
  }
  if (auto wsOp = op->getParentOfType<ttg::WarpSpecializePartitionsOp>()) {
    numParts = wsOp.getPartitionRegions().size() + 1;
  }
  int activeMask = 0;
  for (int i = 0; i < numParts; ++i)
    activeMask |= (1 << i);
  return activeMask;
}

Value currentCTAMask(ImplicitLocOpBuilder &b) {
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  return arith::ShLIOp::create(b, arith::ConstantIntOp::create(b, 1, 32),
                               ctaId);
}

uint16_t getBlockBroadcastMask(Value alloc) {
  auto allocTy = cast<ttg::MemDescType>(alloc.getType());
  auto kBlock = StringAttr::get(alloc.getContext(), "block");
  return toLinearLayout(allocTy).getFreeVariableMasks().lookup(kBlock);
}

Value createCTABitset(ImplicitLocOpBuilder &b, uint32_t pattern,
                      uint32_t baseMask) {
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  Value base = arith::AndIOp::create(
      b, ctaId, arith::ConstantIntOp::create(b, baseMask, 32));
  return arith::ShLIOp::create(b, arith::ConstantIntOp::create(b, pattern, 32),
                               base);
}

Value getLeaderCTA(ImplicitLocOpBuilder &b, Value barrier) {
  uint16_t broadcastMask = getBlockBroadcastMask(barrier);
  if (!broadcastMask)
    return currentCTAMask(b);
  int numCTAs = ttg::lookupNumCTAs(b);
  auto encoding = ttng::getTMAMulticastMaskEncoding(numCTAs, broadcastMask);
  return createCTABitset(b, /*pattern=*/1, encoding.fixedBits);
}

Value getRecipientCTAs(ImplicitLocOpBuilder &b, Operation *op) {
  if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op))
    return getLeaderCTA(b, expectOp.getAlloc());
  if (auto arriveOp = dyn_cast<ttng::ArriveBarrierOp>(op))
    return getLeaderCTA(b, arriveOp.getAlloc());
  if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    return getLeaderCTA(b, copyOp.getBarrier());

  SmallVector<uint16_t> broadcastMasks;
  if (auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op)) {
    broadcastMasks = ttng::getCTABroadcastMasks(ttng::getModuleTwoCTAs(op),
                                                commitOp.getDescs());
  } else if (isa<ttng::TMEMCopyOp>(op)) {
    broadcastMasks = ttng::getCTABroadcastMasks(ttng::getModuleTwoCTAs(op), {});
  } else if (auto mmaOp = dyn_cast<ttng::TCGen5MMAOp>(op)) {
    SmallVector<Value> commitDescs;
    if (mmaOp.getMulticast()) {
      if (isa<ttg::SharedEncodingTrait>(mmaOp.getA().getType().getEncoding()))
        commitDescs.push_back(mmaOp.getA());
      commitDescs.push_back(mmaOp.getB());
    }
    broadcastMasks =
        ttng::getCTABroadcastMasks(mmaOp.getTwoCtas(), commitDescs);
  } else if (auto mmaScaledOp = dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
    broadcastMasks = ttng::getCTABroadcastMasks(mmaScaledOp.getTwoCtas(), {});
  }
  if (broadcastMasks.empty())
    return currentCTAMask(b);

  int numCTAs = ttg::lookupNumCTAs(b);
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  Value recipientCTAs = arith::ConstantIntOp::create(b, 0, 32);
  // Match tcgen05_commit lowering in MMAv5.cpp: build one concrete recipient
  // bitset per descriptor, then OR those bitsets.
  for (uint16_t broadcastBits : broadcastMasks) {
    // Compute the map that goes from cta_id to lead_cta_id (fixedBits)
    // and the pattern that goes from cta_0 to its multicast group (pattern).
    auto encoding = ttng::getTMAMulticastMaskEncoding(numCTAs, broadcastBits);
    Value fixedBitsVal =
        arith::ConstantIntOp::create(b, encoding.fixedBits, 32);
    Value base = arith::AndIOp::create(b, ctaId, fixedBitsVal);
    Value patternVal = arith::ConstantIntOp::create(b, encoding.pattern, 32);
    Value descRecipientCTAs = arith::ShLIOp::create(b, patternVal, base);
    recipientCTAs = arith::OrIOp::create(b, recipientCTAs, descRecipientCTAs);
  }
  return recipientCTAs;
}

class ConcurrencySanitizerImpl {
public:
  ConcurrencySanitizerImpl(ModuleOp module, const ConSanTargetHooks *hooks)
      : module(module), hooks(hooks) {}

  void run() {
    tti::FunctionBuilder funcBuilder(module, auxData);
    auxData.populateAndPassToWarpSpecialize(module, funcBuilder, hooks);

    tt::FuncOp entryPoint = tti::getEntryPoint(module);

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());
    instrumentMemoryOperations(b, funcBuilder);
  }

private:
  void instrumentMemoryOperations(ImplicitLocOpBuilder &b,
                                  tti::FunctionBuilder &funcBuilder) {
    module.walk([&](Operation *op) {
      CriticalSectionListener listener;
      b.setListener(&listener);

      int thread = getCurrentThread(op, hooks);
      int baseThread = getBaseThread(thread);
      b.setLoc(op->getLoc());
      b.setInsertionPoint(op);
      if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op)) {
        // Place insert point after specific ops:
        // allocs - we want to
        //   check if it is not overwriting any earlier allocation, but the
        //   memref value can be referenced only after it is created.
        b.setInsertionPointAfter(op);
      }

      if (auto info = hooks->getBarrierWaitInfo(op)) {
        // For waits we want to instrument it before and after, so we do it
        // manually inside instrumentBarrierWait (disable the critical section
        // listener and return early)
        b.setListener(nullptr);
        instrumentBarrierWait(op, info->alloc, info->phase, info->pred, thread,
                              baseThread, funcBuilder);
        return;
      }

      instrumentMemEffects(b, op, thread, funcBuilder);
      b.setLoc(op->getLoc());
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        auto partitionRegions = wsOp.getPartitionRegions();
        if (!partitionRegions.empty()) {
          uint64_t destMask = 0;
          for (size_t idx = 0, e = partitionRegions.size(); idx < e; ++idx)
            destMask |= getThreadPeersMask(idx + 1);
          if (destMask) {
            for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
              funcBuilder.createCopyWriteVisibilityCall(b, thread, destMask,
                                                        nullptr, memType, op);
              funcBuilder.createCopyReadVisibilityCall(b, thread, destMask,
                                                       nullptr, memType, op);
            }
          }
        }
      }
      if (auto info = hooks->getBarrierInitInfo(op)) {
        Value pred = hooks->getIssuerCTAPred(b, op);
        funcBuilder.createVerifyBarrierCanInitCall(b, info->alloc, pred, op,
                                                   currentCTAMask(b));
        funcBuilder.createInitBarrierStateCall(b, info->alloc, info->count,
                                               pred, op);
      }
      if (auto info = hooks->getBarrierInvalidateInfo(op)) {
        Value barrier = info->alloc;
        Value pred = hooks->getIssuerCTAPred(b, op);
        funcBuilder.createVerifyBarrierInitializedCall(b, barrier, pred, op,
                                                       currentCTAMask(b));
        funcBuilder.createInvalidateBarrierStateCall(b, barrier, pred, op);
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          funcBuilder.createClearBarrierWriteTrackingCall(b, barrier, pred,
                                                          memType, op);
          funcBuilder.createClearBarrierReadTrackingCall(b, barrier, pred,
                                                         memType, op);
        }
      }
      if (auto asyncCommitGroupOp = dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
        if (!auxData.commits[CommitKind::AsyncCp].empty())
          funcBuilder.createCommitAccessesCall(b, thread, nullptr,
                                               CommitKind::AsyncCp, op);
      }
      if (auto asyncWaitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
        funcBuilder.createClearOutstandingCommitsTransferWritesCall(
            b, baseThread, getThreadPeersMask(thread), asyncWaitOp.getNum(),
            nullptr, CommitKind::AsyncCp, MemType::SHARED_MEM, op);
      }
      if (auto wgmmaWaitOp = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
        funcBuilder.createClearOutstandingCommitsTransferReadsCall(
            b, baseThread, getThreadPeersMask(thread),
            wgmmaWaitOp.getPendings(), nullptr, CommitKind::Wgmma,
            MemType::SHARED_MEM, op);
      }
      if (auto info = hooks->getWaitOpInfo(op)) {
        if (info->transferWrites && info->transferReads) {
          funcBuilder.createClearOutstandingCommitsTransferBothCall(
              b, baseThread, getThreadPeersMask(thread), info->pendingCount,
              nullptr, info->commitKind, MemType::SHARED_MEM, op);
        } else if (info->transferWrites) {
          funcBuilder.createClearOutstandingCommitsTransferWritesCall(
              b, baseThread, getThreadPeersMask(thread), info->pendingCount,
              nullptr, info->commitKind, MemType::SHARED_MEM, op);
        } else if (info->transferReads) {
          funcBuilder.createClearOutstandingCommitsTransferReadsCall(
              b, baseThread, getThreadPeersMask(thread), info->pendingCount,
              nullptr, info->commitKind, MemType::SHARED_MEM, op);
        }
      }

      listener.maybeWrapWithCriticalSection(b, auxData, nullptr);
      b.setListener(nullptr);
    });
  }

  void instrumentBarrierWait(Operation *op, Value alloc, Value phase,
                             Value pred, int thread, int baseThread,
                             tti::FunctionBuilder &funcBuilder) {
    ImplicitLocOpBuilder wb(op->getLoc(), op);
    pred = tti::maybeAnd(wb, pred, hooks->getIssuerCTAPred(wb, op));
    Value lock = auxData.lock.at(op).value;
    // Pre-wait: mark waiting threads and check for deadlock.
    tti::ExperimentalLockAcquireOp::create(wb, lock, pred);
    funcBuilder.createVerifyBarrierInitializedCall(wb, alloc, pred, op,
                                                   currentCTAMask(wb));
    funcBuilder.createSetWaitingCall(wb, alloc, baseThread, phase, pred, op);
    funcBuilder.createCheckAllActiveWaitingCall(wb, getActiveMask(op), pred,
                                                op);
    tti::ExperimentalLockReleaseOp::create(wb, lock, pred);
    // Post-wait: transfer visible writes and reads to all peer threads,
    // and clear waiting for this barrier.
    assert(!auxData.barriers.empty() &&
           "barrier descriptors must exist when instrumenting wait");
    wb.setInsertionPointAfter(op);
    tti::ExperimentalLockAcquireOp::create(wb, lock, pred);
    for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
      funcBuilder.createTransferVisibleWritesCall(
          wb, alloc, getThreadPeersMask(thread), pred, memType, op);
      funcBuilder.createTransferVisibleReadsCall(
          wb, alloc, getThreadPeersMask(thread), pred, memType, op);
    }
    funcBuilder.createClearWaitingCall(wb, alloc, baseThread, pred, op);
    tti::ExperimentalLockReleaseOp::create(wb, lock, pred);
  }

  void instrumentBarrierExpectNonLeaderArrive(
      ImplicitLocOpBuilder &b, ttng::BarrierExpectOp expectOp,
      Value nonLeaderPred, int thread, tti::FunctionBuilder &funcBuilder) {
    Value barrier = expectOp.getAlloc();
    Value recipientCTAs = getLeaderCTA(b, barrier);

    // Match BarrierOpToLLVM's cross-CTA path: non-leader CTAs contribute a
    // plain arrive of count 1 to the leader barrier. The generic barrier path
    // models the leader CTA's expect_tx.
    for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
      funcBuilder.createTrackVisibleWritesCall(
          b, barrier, thread, nonLeaderPred, memType, expectOp, recipientCTAs);
      funcBuilder.createTrackVisibleReadsCall(b, barrier, thread, nonLeaderPred,
                                              memType, expectOp, recipientCTAs);
    }
    funcBuilder.createVerifyBarrierArriveCall(
        b, barrier, /*count=*/1, nonLeaderPred, expectOp, recipientCTAs);
    funcBuilder.createUpdateBarrierStateCall(
        b, barrier, /*count=*/1, nonLeaderPred, expectOp, recipientCTAs);
  }

  void instrumentMemEffects(ImplicitLocOpBuilder &b, Operation *op, int thread,
                            tti::FunctionBuilder &funcBuilder) {
    int baseThread = getBaseThread(thread);
    std::optional<MemEffectsOpInfo> opInfo = hooks->getMemEffectsOpInfo(op);
    if (!opInfo) {
      return;
    }
    Value pred = opInfo->pred;
    // Barrier expect performs an arrive on non-leader CTAs, so we need to
    // instrument it separately before incorporating getIssuerCTAPred.
    Value issuerCTAPred = hooks->getIssuerCTAPred(b, op);
    if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op)) {
      if (issuerCTAPred) {
        Value nonLeaderPred = arith::XOrIOp::create(
            b, issuerCTAPred, arith::ConstantIntOp::create(b, 1, 1));
        nonLeaderPred = tti::maybeAnd(b, pred, nonLeaderPred);
        instrumentBarrierExpectNonLeaderArrive(b, expectOp, nonLeaderPred,
                                               thread, funcBuilder);
      }
    }
    pred = tti::maybeAnd(b, pred, issuerCTAPred);
    Value recipientCTAs = getRecipientCTAs(b, op);
    for (auto effect : opInfo->operandEffects) {
      Value buf = effect.buf;
      auto bufType = cast<ttg::MemDescType>(buf.getType());
      MemType memType = MemType::TENSOR_MEM;
      if (isa<ttg::SharedEncodingTrait>(bufType.getEncoding())) {
        memType = MemType::SHARED_MEM;
      }
      if (effect.rw == MemEffectsOpInfo::Effects::Read) {
        // For op that is reading, we only need to check if anything else
        // is writing to the same buffer.
        addWriteChecks(b, funcBuilder, op, buf, effect.length, pred, memType,
                       thread, effect.operandName, opInfo->commitKind);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier) {
          funcBuilder.createSetReadVisibilityCall(
              b, buf, effect.length, getThreadPeersMask(thread), pred, memType,
              op, recipientCTAs);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::CommitCount) {
          assert(memType == MemType::SHARED_MEM);
          funcBuilder.createStageAccessForCommitCall(b, buf, effect.length,
                                                     baseThread, pred, memType,
                                                     opInfo->commitKind, op);
        }
      }
      if (effect.rw == MemEffectsOpInfo::Effects::Write) {
        // Op is writing to the buffer, we need to check if anything else
        // is reading or writing to the same buffer.
        addWriteChecks(b, funcBuilder, op, buf, effect.length, pred, memType,
                       thread, effect.operandName, opInfo->commitKind);
        addReadChecks(b, funcBuilder, op, buf, effect.length, pred, memType,
                      thread, effect.operandName, opInfo->commitKind);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier) {
          funcBuilder.createSetWriteVisibilityCall(
              b, buf, effect.length, getThreadPeersMask(thread), pred, memType,
              op, recipientCTAs);
          funcBuilder.createClearWriteTrackingCall(b, buf, effect.length, pred,
                                                   memType, op, recipientCTAs);
          funcBuilder.createClearReadVisibilityCall(b, buf, effect.length, pred,
                                                    memType, op, recipientCTAs);
          funcBuilder.createClearReadTrackingCall(b, buf, effect.length, pred,
                                                  memType, op, recipientCTAs);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::CommitCount) {
          assert(memType == MemType::SHARED_MEM);
          funcBuilder.createStageAccessForCommitCall(b, buf, effect.length,
                                                     baseThread, pred, memType,
                                                     opInfo->commitKind, op);
        }
      }
    }
    for (const auto &barrierInfo : opInfo->barriers) {
      Value barrier = barrierInfo.barrier;
      Value combinedPred = tti::maybeAnd(b, barrierInfo.pred, pred);
      funcBuilder.createVerifyBarrierInitializedCall(b, barrier, combinedPred,
                                                     op, recipientCTAs);
      if (barrierInfo.trackingMode ==
          MemEffectsOpInfo::BarrierTrackingMode::Frontier) {
        // If the op has barriers, we treat it as a commit emitted for each
        // barrier.
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          funcBuilder.createTrackVisibleWritesCall(
              b, barrier, thread, combinedPred, memType, op, recipientCTAs);
          funcBuilder.createTrackVisibleReadsCall(
              b, barrier, thread, combinedPred, memType, op, recipientCTAs);
        }
      } else if (barrierInfo.trackingMode ==
                 MemEffectsOpInfo::BarrierTrackingMode::EffectWrites) {
        for (const auto &effect : opInfo->operandEffects) {
          if (effect.rw != MemEffectsOpInfo::Effects::Write)
            continue;
          auto bufType = cast<ttg::MemDescType>(effect.buf.getType());
          MemType memType = MemType::TENSOR_MEM;
          if (isa<ttg::SharedEncodingTrait>(bufType.getEncoding()))
            memType = MemType::SHARED_MEM;
          funcBuilder.createTrackBarrierWriteForBufferCall(
              b, barrier, effect.buf, effect.length, combinedPred, memType, op);
        }
      }
      if (barrierInfo.count > 0 || barrierInfo.txCount != 0) {
        funcBuilder.createVerifyBarrierArriveCall(
            b, barrier, barrierInfo.count, combinedPred, op, recipientCTAs,
            barrierInfo.txCount);
        funcBuilder.createUpdateBarrierStateCall(
            b, barrier, barrierInfo.count, combinedPred, op, recipientCTAs,
            barrierInfo.txCount);
      }
    }
    if (opInfo->implicitCommit) {
      assert(opInfo->trackingKind ==
             MemEffectsOpInfo::TrackingKind::CommitCount);
      funcBuilder.createCommitAccessesCall(b, baseThread, pred,
                                           opInfo->commitKind, op);
    }
  }

  void addWriteChecks(ImplicitLocOpBuilder &b,
                      tti::FunctionBuilder &funcBuilder, Operation *op,
                      Value buf, uint32_t length, Value pred, MemType memType,
                      int thread, const std::string &operandName,
                      CommitKind::Kind opCommitKind = CommitKind::None) {
    funcBuilder.createVerifyWriteVisibilityCall(b, buf, length, thread,
                                                operandName, pred, memType, op);
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM) {
      for (const auto &commitKindDesc :
           hooks->getOutstandingWriteCommitKinds()) {
        bool excludeSelf = (opCommitKind == commitKindDesc.kind &&
                            hooks->isOrderedCommitKind(opCommitKind));
        funcBuilder.createCheckOutstandingCommitsCall(
            b, buf, length, getBaseThread(thread), commitKindDesc.operationDesc,
            pred, memType, commitKindDesc.kind, op, excludeSelf);
      }
    }
  }

  void addReadChecks(ImplicitLocOpBuilder &b, tti::FunctionBuilder &funcBuilder,
                     Operation *op, Value buf, uint32_t length, Value pred,
                     MemType memType, int thread,
                     const std::string &operandName,
                     CommitKind::Kind opCommitKind = CommitKind::None) {
    funcBuilder.createVerifyReadVisibilityCall(b, buf, length, thread,
                                               operandName, pred, memType, op);
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM) {
      for (const auto &commitKindDesc :
           hooks->getOutstandingReadCommitKinds()) {
        bool excludeSelf = (opCommitKind == commitKindDesc.kind &&
                            hooks->isOrderedCommitKind(opCommitKind));
        funcBuilder.createCheckOutstandingCommitsCall(
            b, buf, length, getBaseThread(thread), commitKindDesc.operationDesc,
            pred, memType, commitKindDesc.kind, op, excludeSelf);
      }
    }
  }

  ModuleOp module;
  AuxDataMap auxData;
  const ConSanTargetHooks *hooks;
};

} // namespace

void runConcurrencySanitizer(ModuleOp module, const ConSanTargetHooks *hooks) {
  assert(hooks && "hooks must not be null");
  ConcurrencySanitizerImpl impl(module, hooks);
  impl.run();
}

class ConcurrencySanitizerPass
    : public impl::TritonInstrumentConcurrencySanitizerBase<
          ConcurrencySanitizerPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto targetAttr = module->getAttrOfType<StringAttr>(ttg::AttrTargetName);
    assert(targetAttr && "module missing ttg.target attribute");
    StringRef target = targetAttr.strref();
    StringRef key = target.starts_with("cuda:")  ? "nvidia"
                    : target.starts_with("hip:") ? "amd"
                                                 : "";
    auto hooks = createConSanHooks(key);
    assert(hooks && "no ConSan hooks registered for target");
    runConcurrencySanitizer(module, hooks.get());
  }
};

} // namespace instrument
} // namespace triton
} // namespace mlir
