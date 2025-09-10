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

bool isTMAOp(Operation *op) {
  return isa<ttng::AsyncTMACopyGlobalToLocalOp,
             ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAGatherOp,
             ttng::AsyncTMAScatterOp>(op);
}

bool isTensorCoreOp(Operation *op) {
  return isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp,
             ttng::TCGen5CommitOp>(op);
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
SmallVector<int, 3> getThreadPeers(int thread) {
  int baseThread = thread % NUM_THREADS;
  return {baseThread, baseThread + TMA_THREAD_OFFSET,
          baseThread + TC_THREAD_OFFSET};
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
    assert(entryPoint);

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());
    instrumentMemoryOperations(b);
  }

private:
  void addWriteChecks(ImplicitLocOpBuilder &b, Operation *op, Value buf,
                      Value pred, MemType memType, bool hwPipelined,
                      int thread) {
    auto buffers = auxData.buffers[(int)memType][op];
    if (!auxData.barriers.empty()) {
      b.create<tti::ExperimentalVerifyWriteVisibilityOp>(
          buf, thread, buffers, auxData.writeVisibility[(int)memType][op],
          auxData.writeVisibilityType[(int)memType], pred);
    }
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM && auxData.asyncCpCommits[op]) {
      b.create<tti::ExperimentalCheckOutstandingCommitsOp>(
          buf, buffers, auxData.asyncCpCommits[op], auxData.asyncCpCommitsType,
          "async_copy_global_to_shared", pred);
    }
  }

  void addReadChecks(ImplicitLocOpBuilder &b, Operation *op, Value buf,
                     Value pred, MemType memType, int thread) {
    auto buffers = auxData.buffers[(int)memType][op];
    if (!auxData.barriers.empty()) {
      b.create<tti::ExperimentalVerifyReadVisibilityOp>(
          buf, thread, buffers, auxData.readVisibility[(int)memType][op],
          auxData.readVisibilityType[(int)memType], pred);
    }
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM && auxData.wgmmaCommits[op]) {
      b.create<tti::ExperimentalCheckOutstandingCommitsOp>(
          buf, buffers, auxData.wgmmaCommits[op], auxData.wgmmaCommitsType,
          "warpgroup_mma operand read", pred);
    }
  }

  struct MemEffects {
    enum class RW { Read, Write } rw;
    enum class TrackingKind {
      None,
      Barrier,
      wgmmaCommit,
      asyncCpCommit
    } trackingKind = TrackingKind::None;
    Value buf;
    SmallVector<std::tuple<Value, Value>> barriersAndPreds;
    bool hwPipelined = false;
    Value pred;
  };

  SmallVector<MemEffects> getMemEffects(Operation *op) {
    SmallVector<MemEffects> effects;
    if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = copyOp.getResult(),
                     .barriersAndPreds = {{copyOp.getBarrier(), nullptr}},
                     .pred = copyOp.getPred()});
    }
    if (auto storeOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      effects.emplace_back(MemEffects{
          .rw = MemEffects::RW::Read,
          .trackingKind = MemEffects::TrackingKind::None, // async tma writes
                                                          // not modelled yet
          .buf = storeOp.getSrc()});
    }
    if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = gatherOp.getResult(),
                     .barriersAndPreds = {{gatherOp.getBarrier(), nullptr}},
                     .pred = gatherOp.getPred()});
    }
    if (auto scatterOp = dyn_cast<ttng::AsyncTMAScatterOp>(op)) {
      effects.emplace_back(MemEffects{
          .rw = MemEffects::RW::Read,
          .trackingKind = MemEffects::TrackingKind::None, // async tma writes
                                                          // not modelled yet
          .buf = scatterOp.getSrc(),
      });
    }
    if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write,
                     .trackingKind = MemEffects::TrackingKind::asyncCpCommit,
                     .buf = copyOp.getResult()});
    }
    if (auto loadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Read,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = loadOp.getSrc()});
    }
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = storeOp.getDst()});
    }
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (allocOp.getSrc()) {
        effects.emplace_back(MemEffects{.rw = MemEffects::RW::Write,
                                        .buf = allocOp.getResult()});
      }
    }
    if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Read, .buf = loadOp.getSrc()});
    }
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write, .buf = storeOp.getDst()});
    }
    if (auto allocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
      if (allocOp.getSrc()) {
        effects.emplace_back(MemEffects{.rw = MemEffects::RW::Write,
                                        .buf = allocOp.getResult()});
      }
    }
    if (auto mmav5Op = dyn_cast<ttng::TCGen5MMAOp>(op)) {
      SmallVector<std::tuple<Value, Value>> barriersAndPreds = llvm::to_vector(
          llvm::zip(mmav5Op.getBarriers(), mmav5Op.getBarrierPreds()));

      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Read,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = mmav5Op.getA(),
                     .barriersAndPreds = barriersAndPreds,
                     .pred = mmav5Op.getPred()});

      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Read,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = mmav5Op.getB(),
                     .barriersAndPreds = barriersAndPreds,
                     .pred = mmav5Op.getPred()});

      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = mmav5Op.getAccumulator(),
                     .barriersAndPreds = barriersAndPreds,
                     .hwPipelined = true,
                     .pred = mmav5Op.getPred()});
    }
    if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      if (wgmmaOp.getIsAsync() == true) {
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getA().getType().getEncoding())) {
          effects.emplace_back(
              MemEffects{.rw = MemEffects::RW::Read,
                         .trackingKind = MemEffects::TrackingKind::wgmmaCommit,
                         .buf = wgmmaOp.getA()});
        }
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getB().getType().getEncoding())) {
          effects.emplace_back(
              MemEffects{.rw = MemEffects::RW::Read,
                         .trackingKind = MemEffects::TrackingKind::wgmmaCommit,
                         .buf = wgmmaOp.getB()});
        }
      }
    }
    return effects;
  }

  void instrumentMemoryOperations(ImplicitLocOpBuilder &b) {
    module.walk([&](Operation *op) {
      class Listener : public ImplicitLocOpBuilder::Listener {
      public:
        void notifyOperationInserted(Operation *op,
                                     OpBuilder::InsertPoint previous) override {
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
            b.create<tti::ExperimentalLockAcquireOp>(auxData.lock[_firstOp],
                                                     pred);
            b.setInsertionPointAfter(_lastOp);
            b.create<tti::ExperimentalLockReleaseOp>(auxData.lock[_firstOp],
                                                     pred);
          }
        }

      private:
        Operation *firstOp = nullptr;
        Operation *lastOp = nullptr;
      } listener;
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
      SmallVector<MemEffects> effects = getMemEffects(op);
      if (!effects.empty()) {
        auto _barriers = auxData.barriers[op];
        for (MemEffects effect : effects) {
          Value buf = effect.buf;
          auto bufType = cast<ttg::MemDescType>(buf.getType());
          MemType memType = MemType::TENSOR_MEM;
          if (isa<ttg::SharedEncodingTrait>(bufType.getEncoding())) {
            memType = MemType::SHARED_MEM;
          }
          auto _buffers = auxData.buffers[(int)memType][op];
          if (effect.rw == MemEffects::RW::Read) {
            // For op that is reading, we only need to check if anything else
            // is writing to the same buffer.
            addWriteChecks(b, op, buf, effect.pred, memType, effect.hwPipelined,
                           thread);
            if (effect.trackingKind == MemEffects::TrackingKind::Barrier) {
              b.create<tti::ExperimentalSetReadVisibilityOp>(
                  buf, thread, _buffers,
                  auxData.readVisibility[(int)memType][op],
                  auxData.readVisibilityType[(int)memType], effect.pred);
              for (auto [barrier, pred] : effect.barriersAndPreds) {
                if (pred && effect.pred) {
                  pred = b.create<arith::AndIOp>(effect.pred, pred);
                }
                b.create<tti::ExperimentalTrackVisibleReadsOp>(
                    barrier, thread, _barriers,
                    auxData.readVisibility[(int)memType][op],
                    auxData.readVisibilityType[(int)memType],
                    auxData.readTracking[(int)memType][op],
                    auxData.readTrackingType[(int)memType], pred);
              }
            }
            if (effect.trackingKind == MemEffects::TrackingKind::wgmmaCommit) {
              assert(isa<ttng::WarpGroupDotOp>(op));
              assert(memType == MemType::SHARED_MEM);
              b.create<tti::ExperimentalStageAccessForCommitOp>(
                  buf, _buffers, auxData.wgmmaCommits[op],
                  auxData.wgmmaCommitsType, effect.pred);
            }
            assert(effect.trackingKind !=
                   MemEffects::TrackingKind::asyncCpCommit);
          }
          if (effect.rw == MemEffects::RW::Write) {
            // Op is writing to the buffer, we need to check if anything else
            // is reading or writing to the same buffer.
            addWriteChecks(b, op, buf, effect.pred, memType, effect.hwPipelined,
                           thread);
            addReadChecks(b, op, buf, effect.pred, memType, thread);
            if (effect.trackingKind == MemEffects::TrackingKind::Barrier) {
              b.create<tti::ExperimentalSetWriteVisibilityOp>(
                  buf, thread, _buffers,
                  auxData.writeVisibility[(int)memType][op],
                  auxData.writeVisibilityType[(int)memType], effect.pred);
              b.create<tti::ExperimentalClearWriteTrackingOp>(
                  buf, _buffers, auxData.writeTracking[(int)memType][op],
                  auxData.writeTrackingType[(int)memType], effect.pred);
              b.create<tti::ExperimentalClearReadVisibilityOp>(
                  buf, _buffers, auxData.readVisibility[(int)memType][op],
                  auxData.readVisibilityType[(int)memType], effect.pred);
              b.create<tti::ExperimentalClearReadTrackingOp>(
                  buf, _buffers, auxData.readTracking[(int)memType][op],
                  auxData.readTrackingType[(int)memType], effect.pred);
              for (auto [barrier, pred] : effect.barriersAndPreds) {
                if (pred && effect.pred) {
                  pred = b.create<arith::AndIOp>(effect.pred, pred);
                }
                b.create<tti::ExperimentalTrackVisibleWritesOp>(
                    barrier, thread, _barriers,
                    auxData.writeVisibility[(int)memType][op],
                    auxData.writeVisibilityType[(int)memType],
                    auxData.writeTracking[(int)memType][op],
                    auxData.writeTrackingType[(int)memType], pred);
                b.create<tti::ExperimentalTrackVisibleReadsOp>(
                    barrier, thread, _barriers,
                    auxData.readVisibility[(int)memType][op],
                    auxData.readVisibilityType[(int)memType],
                    auxData.readTracking[(int)memType][op],
                    auxData.readTrackingType[(int)memType], pred);
              }
            }
            if (effect.trackingKind ==
                MemEffects::TrackingKind::asyncCpCommit) {
              assert(memType == MemType::SHARED_MEM);
              b.create<tti::ExperimentalStageAccessForCommitOp>(
                  buf, _buffers, auxData.asyncCpCommits[op],
                  auxData.asyncCpCommitsType, effect.pred);
            }
            assert(effect.trackingKind !=
                   MemEffects::TrackingKind::wgmmaCommit);
          }
        }
      }

      if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op)) {
        auto _barriers = auxData.barriers[op];
        assert(!auxData.barriers.empty());
        auto pred = waitOp.getPred();
        auto barrier = waitOp.getAlloc();
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          if (auxData.writeVisibility[(int)memType][op]) {
            // Transfer visible writes and reads to all peer threads
            for (int peer : getThreadPeers(thread)) {
              b.create<tti::ExperimentalTransferVisibleWritesOp>(
                  barrier, peer, _barriers,
                  auxData.writeVisibility[(int)memType][op],
                  auxData.writeVisibilityType[(int)memType],
                  auxData.writeTracking[(int)memType][op],
                  auxData.writeTrackingType[(int)memType], pred);
              b.create<tti::ExperimentalTransferVisibleReadsOp>(
                  barrier, peer, _barriers,
                  auxData.readVisibility[(int)memType][op],
                  auxData.readVisibilityType[(int)memType],
                  auxData.readTracking[(int)memType][op],
                  auxData.readTrackingType[(int)memType], pred);
            }
          }
        }
      }
      if (auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op)) {
        auto _barriers = auxData.barriers[op];
        for (MemType memType : {MemType::TENSOR_MEM, MemType::SHARED_MEM}) {
          b.create<tti::ExperimentalTrackVisibleWritesOp>(
              commitOp.getBarrier(), thread, _barriers,
              auxData.writeVisibility[(int)memType][op],
              auxData.writeVisibilityType[(int)memType],
              auxData.writeTracking[(int)memType][op],
              auxData.writeTrackingType[(int)memType], commitOp.getPred());
          b.create<tti::ExperimentalTrackVisibleReadsOp>(
              commitOp.getBarrier(), thread, _barriers,
              auxData.readVisibility[(int)memType][op],
              auxData.readVisibilityType[(int)memType],
              auxData.readTracking[(int)memType][op],
              auxData.readTrackingType[(int)memType], commitOp.getPred());
        }
      }
      if (auto arriveOp = dyn_cast<ttng::ArriveBarrierOp>(op)) {
        auto _barriers = auxData.barriers[op];
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          if (auxData.writeVisibility[(int)memType][op]) {
            b.create<tti::ExperimentalTrackVisibleWritesOp>(
                arriveOp.getAlloc(), thread, _barriers,
                auxData.writeVisibility[(int)memType][op],
                auxData.writeVisibilityType[(int)memType],
                auxData.writeTracking[(int)memType][op],
                auxData.writeTrackingType[(int)memType], arriveOp.getPred());
            b.create<tti::ExperimentalTrackVisibleReadsOp>(
                arriveOp.getAlloc(), thread, _barriers,
                auxData.readVisibility[(int)memType][op],
                auxData.readVisibilityType[(int)memType],
                auxData.readTracking[(int)memType][op],
                auxData.readTrackingType[(int)memType], arriveOp.getPred());
          }
        }
      }
      if (auto asyncCommitGroupOp = dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
        b.create<tti::ExperimentalCommitAccessesOp>(
            auxData.asyncCpCommits[op], auxData.asyncCpCommitsType, nullptr);
      }
      if (auto asyncWaitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
        b.create<tti::ExperimentalClearOutstandingCommitsOp>(
            auxData.asyncCpCommits[op], auxData.asyncCpCommitsType,
            asyncWaitOp.getNum(), nullptr);
      }
      if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        if (wgmmaOp.getIsAsync() == true) {
          // Add commit (implicit in ttgir) after staging wgmma's operand for
          // read
          b.create<tti::ExperimentalCommitAccessesOp>(
              auxData.wgmmaCommits[op], auxData.wgmmaCommitsType, nullptr);
        }
      }
      if (auto wgmmaWaitOp = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
        b.create<tti::ExperimentalClearOutstandingCommitsOp>(
            auxData.wgmmaCommits[op], auxData.wgmmaCommitsType,
            wgmmaWaitOp.getPendings(), nullptr);
      }
      listener.maybeWrapWithCriticalSection(b, auxData, nullptr);
      b.setListener(nullptr);
    });
  }

  ModuleOp module;
  AuxDataMap auxData;
};

} // namespace instrument
} // namespace triton
} // namespace mlir
