#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

#define GEN_PASS_DEF_TRITONINSTRUMENTCONCURRENCYSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

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
      b.create<tti::ExperimentalLockAcquireOp>(auxData.lock[_firstOp].value,
                                               pred);
      b.setInsertionPointAfter(_lastOp);
      b.create<tti::ExperimentalLockReleaseOp>(auxData.lock[_firstOp].value,
                                               pred);
    }
  }

private:
  Operation *firstOp = nullptr;
  Operation *lastOp = nullptr;
};

bool isTMAOp(Operation *op) {
  return isa<ttng::AsyncTMACopyGlobalToLocalOp,
             ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAGatherOp,
             ttng::AsyncTMAScatterOp>(op);
}

bool isTensorCoreOp(Operation *op) {
  return isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp, ttng::TCGen5CommitOp>(
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

int getCurrentThread(Operation *op) {
  // Default partition is 0, other partitions are idx + 1
  int thread = maybeGetPartitionIdx(op).value_or(-1) + 1;
  if (isTMAOp(op)) {
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

} // namespace

class ConcurrencySanitizerPass
    : public impl::TritonInstrumentConcurrencySanitizerBase<
          ConcurrencySanitizerPass> {
public:
  void runOnOperation() override {
    module = getOperation();

    auxData.populateAndPassToWarpSpecialize(module);

    tt::FuncOp entryPoint = tti::getEntryPoint(module);

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());
    instrumentMemoryOperations(b);
  }

private:
  void instrumentMemoryOperations(ImplicitLocOpBuilder &b) {
    module.walk([&](Operation *op) {
      CriticalSectionListener listener;
      b.setListener(&listener);

      int thread = getCurrentThread(op);
      int baseThread = getBaseThread(thread);
      b.setLoc(op->getLoc());
      b.setInsertionPoint(op);
      if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp, ttng::WaitBarrierOp>(op)) {
        // Place insert point after specific ops:
        // allocs - we want to
        //   check if it is not overwriting any earlier allocation, but the
        //   memref value can be referenced only after it is created.
        // wait barriers - we can update aux data only after the wait is
        //   completed
        b.setInsertionPointAfter(op);
      }

      instrumentMemEffects(b, op, thread);

      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        auto partitionRegions = wsOp.getPartitionRegions();
        if (!partitionRegions.empty()) {
          uint64_t destMask = 0;
          for (size_t idx = 0, e = partitionRegions.size(); idx < e; ++idx)
            destMask |= getThreadPeersMask(idx + 1);
          if (destMask) {
            for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
              auto writeVis = auxData.writeVisibility[(int)memType][op];
              if (writeVis.value) {
                b.create<tti::ExperimentalCopyWriteVisibilityOp>(
                    thread, static_cast<int64_t>(destMask), writeVis.value,
                    writeVis.type, nullptr);
              }
              auto readVis = auxData.readVisibility[(int)memType][op];
              if (readVis.value) {
                b.create<tti::ExperimentalCopyReadVisibilityOp>(
                    thread, static_cast<int64_t>(destMask), readVis.value,
                    readVis.type, nullptr);
              }
            }
          }
        }
      }
      if (auto initOp = dyn_cast<ttng::InitBarrierOp>(op)) {
        if (auxData.barriers[op].value && auxData.barrierStates[op].value) {
          b.create<tti::ExperimentalInitBarrierStateOp>(
              initOp.getAlloc(), initOp.getCount(), auxData.barriers[op].value,
              auxData.barrierStates[op].value, auxData.barrierStates[op].type);
        }
      }
      if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op)) {
        // Pre-wait: mark waiting threads and check for deadlock.
        {
          CriticalSectionListener preListener;
          b.setListener(&preListener);
          b.setInsertionPoint(waitOp);
          auto pred = waitOp.getPred();
          auto barrier = waitOp.getAlloc();
          if (auxData.barriers[op].value && auxData.waiting[op].value &&
              auxData.barrierStates[op].value) {
            b.create<tti::ExperimentalSetWaitingOp>(
                barrier, baseThread, waitOp.getPhase(),
                auxData.barriers[op].value, auxData.waiting[op].value,
                auxData.waiting[op].type, pred);
            int activeMask = getActiveMask(op);

            b.create<tti::ExperimentalCheckAllActiveWaitingOp>(
                activeMask, auxData.barriers[op].value,
                auxData.waiting[op].value, auxData.waiting[op].type,
                auxData.barrierStates[op].value, auxData.barrierStates[op].type,
                pred);
          }

          preListener.maybeWrapWithCriticalSection(b, auxData, pred);
          b.setListener(&listener);
          b.setInsertionPointAfter(waitOp);
        }
        // Post-wait: transfer visible writes and reads to all peer threads,
        // and clear waiting for this barrier
        auto _barriers = auxData.barriers[op].value;
        assert(!auxData.barriers.empty());
        auto pred = waitOp.getPred();
        auto barrier = waitOp.getAlloc();
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          if (auxData.writeVisibility[(int)memType][op].value) {
            // Transfer visible writes and reads to all peer threads
            uint64_t peerMask = getThreadPeersMask(thread);
            b.create<tti::ExperimentalTransferVisibleWritesOp>(
                barrier, peerMask, _barriers,
                auxData.writeVisibility[(int)memType][op].value,
                auxData.writeVisibility[(int)memType][op].type,
                auxData.writeTracking[(int)memType][op].value,
                auxData.writeTracking[(int)memType][op].type, pred);
            b.create<tti::ExperimentalTransferVisibleReadsOp>(
                barrier, peerMask, _barriers,
                auxData.readVisibility[(int)memType][op].value,
                auxData.readVisibility[(int)memType][op].type,
                auxData.readTracking[(int)memType][op].value,
                auxData.readTracking[(int)memType][op].type, pred);
          }
        }
        if (auxData.barriers[op].value && auxData.waiting[op].value) {
          b.create<tti::ExperimentalClearWaitingOp>(
              barrier, baseThread, auxData.barriers[op].value,
              auxData.waiting[op].value, auxData.waiting[op].type, pred);
        }
      }
      if (auto asyncCommitGroupOp = dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
        b.create<tti::ExperimentalCommitAccessesOp>(
            thread, auxData.asyncCpCommits[op].value,
            auxData.asyncCpCommits[op].type, nullptr);
      }
      if (auto asyncWaitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
        b.create<tti::ExperimentalClearOutstandingCommitsTransferWritesOp>(
            thread, getThreadPeersMask(thread), asyncWaitOp.getNum(),
            auxData.asyncCpCommits[op].value, auxData.asyncCpCommits[op].type,
            auxData.writeVisibility[(int)MemType::SHARED_MEM][op].value,
            auxData.writeVisibility[(int)MemType::SHARED_MEM][op].type,
            nullptr);
      }
      if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        if (wgmmaOp.getIsAsync() == true) {
          // Add commit (implicit in ttgir) after staging wgmma's operand for
          // read
          b.create<tti::ExperimentalCommitAccessesOp>(
              thread, auxData.wgmmaCommits[op].value,
              auxData.wgmmaCommits[op].type, nullptr);
        }
      }
      if (auto wgmmaWaitOp = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
        b.create<tti::ExperimentalClearOutstandingCommitsTransferReadsOp>(
            thread, getThreadPeersMask(thread), wgmmaWaitOp.getPendings(),
            auxData.wgmmaCommits[op].value, auxData.wgmmaCommits[op].type,
            auxData.readVisibility[(int)MemType::SHARED_MEM][op].value,
            auxData.readVisibility[(int)MemType::SHARED_MEM][op].type, nullptr);
      }
      listener.maybeWrapWithCriticalSection(b, auxData, nullptr);
      b.setListener(nullptr);
    });
  }

  struct MemEffectsOpInfo {
    struct Effects {
      enum class RW { Read, Write } rw;
      Value buf;
      std::string operandName = "";
    };
    struct BarrierInfo {
      Value barrier;
      Value pred;
      int count;
    };
    enum class TrackingKind {
      None,
      Barrier,
      wgmmaCommit,
      asyncCpCommit
    } trackingKind = TrackingKind::None;
    SmallVector<BarrierInfo> barriers;
    Value pred;
    SmallVector<Effects> operandEffects;
  };

  void instrumentMemEffects(ImplicitLocOpBuilder &b, Operation *op,
                            int thread) {
    std::optional<MemEffectsOpInfo> opInfo = getMemEffectsOpInfo(op);
    if (!opInfo) {
      return;
    }
    auto _barriers = auxData.barriers[op].value;
    Value pred = opInfo->pred;
    auto combinePredicates = [&](Value barrierPred) -> Value {
      if (barrierPred && pred) {
        return b.create<arith::AndIOp>(b.getLoc(), barrierPred, pred);
      }
      return barrierPred ? barrierPred : pred;
    };
    for (auto effect : opInfo->operandEffects) {
      Value buf = effect.buf;
      auto bufType = cast<ttg::MemDescType>(buf.getType());
      MemType memType = MemType::TENSOR_MEM;
      if (isa<ttg::SharedEncodingTrait>(bufType.getEncoding())) {
        memType = MemType::SHARED_MEM;
      }
      auto _buffers = auxData.buffers[(int)memType][op].value;

      if (effect.rw == MemEffectsOpInfo::Effects::RW::Read) {
        // For op that is reading, we only need to check if anything else
        // is writing to the same buffer.
        addWriteChecks(b, op, buf, pred, memType, thread, effect.operandName);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier &&
            _barriers) {
          b.create<tti::ExperimentalSetReadVisibilityOp>(
              buf, getThreadPeersMask(thread), _buffers,
              auxData.readVisibility[(int)memType][op].value,
              auxData.readVisibility[(int)memType][op].type, pred);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::wgmmaCommit) {
          assert(isa<ttng::WarpGroupDotOp>(op));
          assert(memType == MemType::SHARED_MEM);
          b.create<tti::ExperimentalStageAccessForCommitOp>(
              buf, thread, _buffers, auxData.wgmmaCommits[op].value,
              auxData.wgmmaCommits[op].type, pred);
        }
        assert(opInfo->trackingKind !=
               MemEffectsOpInfo::TrackingKind::asyncCpCommit);
      }
      if (effect.rw == MemEffectsOpInfo::Effects::RW::Write) {
        // Op is writing to the buffer, we need to check if anything else
        // is reading or writing to the same buffer.
        addWriteChecks(b, op, buf, pred, memType, thread, effect.operandName);
        addReadChecks(b, op, buf, pred, memType, thread, effect.operandName);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier &&
            _barriers) {
          b.create<tti::ExperimentalSetWriteVisibilityOp>(
              buf, getThreadPeersMask(thread), _buffers,
              auxData.writeVisibility[(int)memType][op].value,
              auxData.writeVisibility[(int)memType][op].type, pred);
          b.create<tti::ExperimentalClearWriteTrackingOp>(
              buf, _buffers, auxData.writeTracking[(int)memType][op].value,
              auxData.writeTracking[(int)memType][op].type, pred);
          b.create<tti::ExperimentalClearReadVisibilityOp>(
              buf, _buffers, auxData.readVisibility[(int)memType][op].value,
              auxData.readVisibility[(int)memType][op].type, pred);
          b.create<tti::ExperimentalClearReadTrackingOp>(
              buf, _buffers, auxData.readTracking[(int)memType][op].value,
              auxData.readTracking[(int)memType][op].type, pred);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::asyncCpCommit) {
          assert(memType == MemType::SHARED_MEM);
          b.create<tti::ExperimentalStageAccessForCommitOp>(
              buf, thread, _buffers, auxData.asyncCpCommits[op].value,
              auxData.asyncCpCommits[op].type, pred);
        }
        assert(opInfo->trackingKind !=
               MemEffectsOpInfo::TrackingKind::wgmmaCommit);
      }
    }
    for (const auto &barrierInfo : opInfo->barriers) {
      Value barrier = barrierInfo.barrier;
      Value combinedPred = combinePredicates(barrierInfo.pred);
      // If the op has barriers, we treat it as a commit emitted for each
      // barrier.
      for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
        if (!auxData.writeVisibility[(int)memType][op].value) {
          continue;
        }
        b.create<tti::ExperimentalTrackVisibleWritesOp>(
            barrier, thread, _barriers,
            auxData.writeVisibility[(int)memType][op].value,
            auxData.writeVisibility[(int)memType][op].type,
            auxData.writeTracking[(int)memType][op].value,
            auxData.writeTracking[(int)memType][op].type, combinedPred);
        b.create<tti::ExperimentalTrackVisibleReadsOp>(
            barrier, thread, _barriers,
            auxData.readVisibility[(int)memType][op].value,
            auxData.readVisibility[(int)memType][op].type,
            auxData.readTracking[(int)memType][op].value,
            auxData.readTracking[(int)memType][op].type, combinedPred);
      }
      if (auxData.barriers[op].value && auxData.barrierStates[op].value &&
          barrierInfo.count > 0) {
        b.create<tti::ExperimentalVerifyBarrierArriveOp>(
            barrier, barrierInfo.count, auxData.barriers[op].value,
            auxData.barrierStates[op].value, auxData.barrierStates[op].type,
            combinedPred);
        b.create<tti::ExperimentalUpdateBarrierStateOp>(
            barrier, barrierInfo.count, auxData.barriers[op].value,
            auxData.barrierStates[op].value, auxData.barrierStates[op].type,
            combinedPred);
      }
    }
  }

  void addWriteChecks(ImplicitLocOpBuilder &b, Operation *op, Value buf,
                      Value pred, MemType memType, int thread,
                      std::string operandName) {
    auto buffers = auxData.buffers[(int)memType][op].value;
    if (!auxData.barriers.empty()) {
      StringAttr operandNameAttr = b.getStringAttr(operandName);
      b.create<tti::ExperimentalVerifyWriteVisibilityOp>(
          buf, thread, buffers, auxData.writeVisibility[(int)memType][op].value,
          auxData.writeVisibility[(int)memType][op].type, operandNameAttr,
          pred);
    }
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM && auxData.asyncCpCommits[op].value) {
      b.create<tti::ExperimentalCheckOutstandingCommitsOp>(
          buf, buffers, auxData.asyncCpCommits[op].value,
          auxData.asyncCpCommits[op].type, "async_copy_global_to_shared", pred);
    }
  }

  void addReadChecks(ImplicitLocOpBuilder &b, Operation *op, Value buf,
                     Value pred, MemType memType, int thread,
                     std::string operandName) {
    auto buffers = auxData.buffers[(int)memType][op].value;
    if (!auxData.barriers.empty()) {
      StringAttr operandNameAttr = b.getStringAttr(operandName);
      b.create<tti::ExperimentalVerifyReadVisibilityOp>(
          buf, thread, buffers, auxData.readVisibility[(int)memType][op].value,
          auxData.readVisibility[(int)memType][op].type, operandNameAttr, pred);
    }
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM && auxData.wgmmaCommits[op].value) {
      b.create<tti::ExperimentalCheckOutstandingCommitsOp>(
          buf, buffers, auxData.wgmmaCommits[op].value,
          auxData.wgmmaCommits[op].type, "warpgroup_mma operand read", pred);
    }
  }

  std::optional<MemEffectsOpInfo> getMemEffectsOpInfo(Operation *op) {
    std::optional<MemEffectsOpInfo> info;
    if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = copyOp.getPred();
      info->barriers.push_back({copyOp.getBarrier(), nullptr, 1});
      info->operandEffects.push_back(
          {.rw = MemEffectsOpInfo::Effects::RW::Write,
           .buf = copyOp.getResult()});
    }
    if (auto storeOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::None;
      info->operandEffects.push_back(
          {.rw = MemEffectsOpInfo::Effects::RW::Read, .buf = storeOp.getSrc()});
    }
    if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = gatherOp.getPred();
      info->barriers.push_back({gatherOp.getBarrier(), nullptr, 1});
      info->operandEffects.push_back(
          {.rw = MemEffectsOpInfo::Effects::RW::Write,
           .buf = gatherOp.getResult()});
    }
    if (auto scatterOp = dyn_cast<ttng::AsyncTMAScatterOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::None;
      info->operandEffects.push_back({.rw = MemEffectsOpInfo::Effects::RW::Read,
                                      .buf = scatterOp.getSrc()});
    }
    if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::asyncCpCommit;
      info->operandEffects.push_back(
          {.rw = MemEffectsOpInfo::Effects::RW::Write,
           .buf = copyOp.getResult()});
    }
    if (auto loadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.push_back(
          {.rw = MemEffectsOpInfo::Effects::RW::Read, .buf = loadOp.getSrc()});
    }
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.push_back(
          {.rw = MemEffectsOpInfo::Effects::RW::Write,
           .buf = storeOp.getDst()});
    }
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        info->operandEffects.push_back(
            {.rw = MemEffectsOpInfo::Effects::RW::Write,
             .buf = allocOp.getResult()});
      }
    }
    if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.push_back(
          {.rw = MemEffectsOpInfo::Effects::RW::Read, .buf = loadOp.getSrc()});
    }
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.push_back(
          {.rw = MemEffectsOpInfo::Effects::RW::Write,
           .buf = storeOp.getDst()});
    }
    if (auto allocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        info->operandEffects.push_back(
            {.rw = MemEffectsOpInfo::Effects::RW::Write,
             .buf = allocOp.getResult()});
      }
    }
    if (auto mmav5Op = dyn_cast<ttng::TCGen5MMAOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = mmav5Op.getPred();
      for (auto [barrier, barrierPred] :
           llvm::zip(mmav5Op.getBarriers(), mmav5Op.getBarrierPreds())) {
        info->barriers.push_back({barrier, barrierPred, 1});
      }
      info->operandEffects.push_back({.rw = MemEffectsOpInfo::Effects::RW::Read,
                                      .buf = mmav5Op.getA(),
                                      .operandName = "A"});
      info->operandEffects.push_back({.rw = MemEffectsOpInfo::Effects::RW::Read,
                                      .buf = mmav5Op.getB(),
                                      .operandName = "B"});
      info->operandEffects.push_back(
          {.rw = MemEffectsOpInfo::Effects::RW::Write,
           .buf = mmav5Op.getAccumulator(),
           .operandName = "Acc"});
    }
    if (auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = commitOp.getPred();
      info->barriers.push_back({commitOp.getBarrier(), nullptr, 1});
    }
    if (auto arriveOp = dyn_cast<ttng::ArriveBarrierOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = arriveOp.getPred();
      info->barriers.push_back(
          {arriveOp.getAlloc(), nullptr, (int)arriveOp.getCount()});
    }
    if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      if (wgmmaOp.getIsAsync() == true) {
        info = {.trackingKind = MemEffectsOpInfo::TrackingKind::wgmmaCommit,
                .barriers = {}};
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getA().getType().getEncoding())) {
          info->operandEffects.emplace_back(MemEffectsOpInfo::Effects{
              .rw = MemEffectsOpInfo::Effects::RW::Read,
              .buf = wgmmaOp.getA(),
              .operandName = "A"});
        }
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getB().getType().getEncoding())) {
          info->operandEffects.emplace_back(MemEffectsOpInfo::Effects{
              .rw = MemEffectsOpInfo::Effects::RW::Read,
              .buf = wgmmaOp.getB(),
              .operandName = "B"});
        }
      }
    }
    return info;
  }

  ModuleOp module;
  AuxDataMap auxData;
};

} // namespace instrument
} // namespace triton
} // namespace mlir
