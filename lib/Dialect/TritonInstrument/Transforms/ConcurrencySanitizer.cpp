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

      if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op)) {
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
      }
      if (auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op)) {
        auto _barriers = auxData.barriers[op].value;
        for (MemType memType : {MemType::TENSOR_MEM, MemType::SHARED_MEM}) {
          b.create<tti::ExperimentalTrackVisibleWritesOp>(
              commitOp.getBarrier(), thread, _barriers,
              auxData.writeVisibility[(int)memType][op].value,
              auxData.writeVisibility[(int)memType][op].type,
              auxData.writeTracking[(int)memType][op].value,
              auxData.writeTracking[(int)memType][op].type, commitOp.getPred());
          b.create<tti::ExperimentalTrackVisibleReadsOp>(
              commitOp.getBarrier(), thread, _barriers,
              auxData.readVisibility[(int)memType][op].value,
              auxData.readVisibility[(int)memType][op].type,
              auxData.readTracking[(int)memType][op].value,
              auxData.readTracking[(int)memType][op].type, commitOp.getPred());
        }
      }
      if (auto arriveOp = dyn_cast<ttng::ArriveBarrierOp>(op)) {
        auto _barriers = auxData.barriers[op].value;
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          if (auxData.writeVisibility[(int)memType][op].value) {
            b.create<tti::ExperimentalTrackVisibleWritesOp>(
                arriveOp.getAlloc(), thread, _barriers,
                auxData.writeVisibility[(int)memType][op].value,
                auxData.writeVisibility[(int)memType][op].type,
                auxData.writeTracking[(int)memType][op].value,
                auxData.writeTracking[(int)memType][op].type,
                arriveOp.getPred());
            b.create<tti::ExperimentalTrackVisibleReadsOp>(
                arriveOp.getAlloc(), thread, _barriers,
                auxData.readVisibility[(int)memType][op].value,
                auxData.readVisibility[(int)memType][op].type,
                auxData.readTracking[(int)memType][op].value,
                auxData.readTracking[(int)memType][op].type,
                arriveOp.getPred());
          }
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
    enum class TrackingKind {
      None,
      Barrier,
      wgmmaCommit,
      asyncCpCommit
    } trackingKind = TrackingKind::None;
    SmallVector<std::tuple<Value, Value>> barriersAndPreds;
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
      for (auto [barrier, pred] : opInfo->barriersAndPreds) {
        // If the op has barriers, we treat it as a commit emitted for each
        // barrier.
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          if (!auxData.writeVisibility[(int)memType][op].value) {
            continue;
          }
          if (pred && opInfo->pred) {
            pred = b.create<arith::AndIOp>(opInfo->pred, pred);
          }
          b.create<tti::ExperimentalTrackVisibleWritesOp>(
              barrier, thread, _barriers,
              auxData.writeVisibility[(int)memType][op].value,
              auxData.writeVisibility[(int)memType][op].type,
              auxData.writeTracking[(int)memType][op].value,
              auxData.writeTracking[(int)memType][op].type, pred);
          b.create<tti::ExperimentalTrackVisibleReadsOp>(
              barrier, thread, _barriers,
              auxData.readVisibility[(int)memType][op].value,
              auxData.readVisibility[(int)memType][op].type,
              auxData.readTracking[(int)memType][op].value,
              auxData.readTracking[(int)memType][op].type, pred);
        }
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
      info = {.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier,
              .barriersAndPreds = {{copyOp.getBarrier(), nullptr}},
              .pred = copyOp.getPred(),
              .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Write,
                                  .buf = copyOp.getResult()}}};
    }
    if (auto storeOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      info = {.trackingKind = MemEffectsOpInfo::TrackingKind::None,
              .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Read,
                                  .buf = storeOp.getSrc()}}};
    }
    if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
      info = {.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier,
              .barriersAndPreds = {{gatherOp.getBarrier(), nullptr}},
              .pred = gatherOp.getPred(),
              .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Write,
                                  .buf = gatherOp.getResult()}}};
    }
    if (auto scatterOp = dyn_cast<ttng::AsyncTMAScatterOp>(op)) {
      info = {.trackingKind = MemEffectsOpInfo::TrackingKind::None,
              .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Read,
                                  .buf = scatterOp.getSrc()}}};
    }
    if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      info = {.trackingKind = MemEffectsOpInfo::TrackingKind::asyncCpCommit,
              .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Write,
                                  .buf = copyOp.getResult()}}};
    }
    if (auto loadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
      info = {.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier,
              .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Read,
                                  .buf = loadOp.getSrc()}}};
    }
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
      info = {.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier,
              .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Write,
                                  .buf = storeOp.getDst()}}};
    }
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info = {.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier,
                .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Write,
                                    .buf = allocOp.getResult()}}};
      }
    }
    if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      info = {.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier,
              .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Read,
                                  .buf = loadOp.getSrc()}}};
    }
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      info = {.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier,
              .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Write,
                                  .buf = storeOp.getDst()}}};
    }
    if (auto allocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info = {.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier,
                .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Write,
                                    .buf = allocOp.getResult()}}};
      }
    }
    if (auto mmav5Op = dyn_cast<ttng::TCGen5MMAOp>(op)) {
      info = {.trackingKind = MemEffectsOpInfo::TrackingKind::Barrier,
              .barriersAndPreds = llvm::to_vector(
                  llvm::zip(mmav5Op.getBarriers(), mmav5Op.getBarrierPreds())),
              .pred = mmav5Op.getPred(),
              .operandEffects = {{.rw = MemEffectsOpInfo::Effects::RW::Read,
                                  .buf = mmav5Op.getA(),
                                  .operandName = "A"},
                                 {.rw = MemEffectsOpInfo::Effects::RW::Read,
                                  .buf = mmav5Op.getB(),
                                  .operandName = "B"},
                                 {.rw = MemEffectsOpInfo::Effects::RW::Write,
                                  .buf = mmav5Op.getAccumulator(),
                                  .operandName = "Acc"}}};
    }
    if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      if (wgmmaOp.getIsAsync() == true) {
        info = {.trackingKind = MemEffectsOpInfo::TrackingKind::wgmmaCommit};
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
