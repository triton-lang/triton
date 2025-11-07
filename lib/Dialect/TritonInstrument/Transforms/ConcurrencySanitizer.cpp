#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/FunctionBuilder.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

// clang-format off
// Concurrency Sanitizer data structures:
// ConSan keeps auxilary data requied for tracking memory accesses in tensors.
// These tensors are stored as a distributed tensor or in global scratch memory.
//
// Name              | Storage | Rank/Type       | Description
// ------------------|---------|-----------------|------------
// buffers           | tensor  | <B x i64>       | Base pointers of all (sub)buffers
// barriers          | tensor  | <K x i64>       | Pointers to all individual mbarriers
// barrierStates     | scratch | <K x i32>       | Packed barrier phase (bit 0) and arrival counts (bits[1..8] init, [9..16] current)
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
      tti::ExperimentalLockAcquireOp::create(b, auxData.lock[_firstOp].value,
                                             pred);
      b.setInsertionPointAfter(_lastOp);
      tti::ExperimentalLockReleaseOp::create(b, auxData.lock[_firstOp].value,
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
    tti::FunctionBuilder funcBuilder(module, auxData);
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
              auto writeVis = auxData.writeVisibility[(int)memType][op];
              if (writeVis.value) {
                funcBuilder.createCopyWriteVisibilityCall(b, thread, destMask,
                                                          nullptr, memType, op);
              }
              auto readVis = auxData.readVisibility[(int)memType][op];
              if (readVis.value) {
                funcBuilder.createCopyReadVisibilityCall(b, thread, destMask,
                                                         nullptr, memType, op);
              }
            }
          }
        }
      }
      if (auto initOp = dyn_cast<ttng::InitBarrierOp>(op)) {
        if (auxData.barriers[op].value && auxData.barrierStates[op].value) {
          funcBuilder.createInitBarrierStateCall(b, initOp.getAlloc(),
                                                 initOp.getCount(), initOp);
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

            funcBuilder.createSetWaitingCall(b, barrier, baseThread,
                                             waitOp.getPhase(), pred, waitOp);
            funcBuilder.createCheckAllActiveWaitingCall(b, getActiveMask(op),
                                                        pred, waitOp);
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
            funcBuilder.createTransferVisibleWritesCall(
                b, barrier, getThreadPeersMask(thread), pred, memType, op);
            funcBuilder.createTransferVisibleReadsCall(
                b, barrier, getThreadPeersMask(thread), pred, memType, op);
          }
        }
        if (auxData.barriers[op].value && auxData.waiting[op].value) {
          funcBuilder.createClearWaitingCall(b, barrier, baseThread, pred,
                                             waitOp);
        }
      }
      if (auto asyncCommitGroupOp = dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
        funcBuilder.createCommitAccessesCall(b, thread, nullptr,
                                             auxData.asyncCpCommits[op], op);
      }
      if (auto asyncWaitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
        funcBuilder.createClearOutstandingCommitsTransferWritesCall(
            b, thread, getThreadPeersMask(thread), asyncWaitOp.getNum(),
            nullptr, auxData.asyncCpCommits[op],
            auxData.writeVisibility[(int)MemType::SHARED_MEM][op], op);
      }
      if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        if (wgmmaOp.getIsAsync() == true) {
          // Add commit (implicit in ttgir) after staging wgmma's operand for
          // read
          funcBuilder.createCommitAccessesCall(b, thread, nullptr,
                                               auxData.wgmmaCommits[op], op);
        }
      }
      if (auto wgmmaWaitOp = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
        funcBuilder.createClearOutstandingCommitsTransferReadsCall(
            b, thread, getThreadPeersMask(thread), wgmmaWaitOp.getPendings(),
            nullptr, auxData.wgmmaCommits[op],
            auxData.readVisibility[(int)MemType::SHARED_MEM][op], op);
      }
      listener.maybeWrapWithCriticalSection(b, auxData, nullptr);
      b.setListener(nullptr);
    });
  }

  struct MemEffectsOpInfo {
    struct Effects {
      enum RW { Read, Write } rw;
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

  void instrumentMemEffects(ImplicitLocOpBuilder &b, Operation *op, int thread,
                            tti::FunctionBuilder &funcBuilder) {
    std::optional<MemEffectsOpInfo> opInfo = getMemEffectsOpInfo(op);
    if (!opInfo) {
      return;
    }
    auto _barriers = auxData.barriers[op].value;
    Value pred = opInfo->pred;
    auto combinePredicates = [&](Value barrierPred) -> Value {
      if (barrierPred && pred) {
        return arith::AndIOp::create(b, b.getLoc(), barrierPred, pred);
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
      auto buffersVT = auxData.buffers[(int)memType][op];

      if (effect.rw == MemEffectsOpInfo::Effects::Read) {
        // For op that is reading, we only need to check if anything else
        // is writing to the same buffer.
        addWriteChecks(b, funcBuilder, op, buf, pred, memType, thread,
                       effect.operandName);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier &&
            _barriers) {
          funcBuilder.createSetReadVisibilityCall(
              b, buf, getThreadPeersMask(thread), pred, memType, op);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::wgmmaCommit) {
          assert(isa<ttng::WarpGroupDotOp>(op));
          assert(memType == MemType::SHARED_MEM);
          funcBuilder.createStageAccessForCommitCall(
              b, buf, thread, pred, buffersVT, auxData.wgmmaCommits[op], op);
        }
        assert(opInfo->trackingKind !=
               MemEffectsOpInfo::TrackingKind::asyncCpCommit);
      }
      if (effect.rw == MemEffectsOpInfo::Effects::Write) {
        // Op is writing to the buffer, we need to check if anything else
        // is reading or writing to the same buffer.
        addWriteChecks(b, funcBuilder, op, buf, pred, memType, thread,
                       effect.operandName);
        addReadChecks(b, funcBuilder, op, buf, pred, memType, thread,
                      effect.operandName);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier &&
            _barriers) {
          funcBuilder.createSetWriteVisibilityCall(
              b, buf, getThreadPeersMask(thread), pred, memType, op);
          funcBuilder.createClearWriteTrackingCall(b, buf, pred, memType, op);
          funcBuilder.createClearReadVisibilityCall(b, buf, pred, memType, op);
          funcBuilder.createClearReadTrackingCall(b, buf, pred, memType, op);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::asyncCpCommit) {
          assert(memType == MemType::SHARED_MEM);
          funcBuilder.createStageAccessForCommitCall(
              b, buf, thread, pred, buffersVT, auxData.asyncCpCommits[op], op);
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
        funcBuilder.createTrackVisibleWritesCall(b, barrier, thread,
                                                 combinedPred, memType, op);
        funcBuilder.createTrackVisibleReadsCall(b, barrier, thread,
                                                combinedPred, memType, op);
      }
      if (auxData.barriers[op].value && auxData.barrierStates[op].value &&
          barrierInfo.count > 0) {
        funcBuilder.createVerifyBarrierArriveCall(b, barrier, barrierInfo.count,
                                                  combinedPred, op);
        funcBuilder.createUpdateBarrierStateCall(b, barrier, barrierInfo.count,
                                                 combinedPred, op);
      }
    }
  }

  void addWriteChecks(ImplicitLocOpBuilder &b,
                      tti::FunctionBuilder &funcBuilder, Operation *op,
                      Value buf, Value pred, MemType memType, int thread,
                      const std::string &operandName) {
    auto buffersVT = auxData.buffers[(int)memType][op];
    if (!auxData.barriers.empty()) {
      funcBuilder.createVerifyWriteVisibilityCall(b, buf, thread, operandName,
                                                  pred, memType, op);
    }
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM && auxData.asyncCpCommits[op].value) {
      funcBuilder.createCheckOutstandingCommitsCall(
          b, buf, thread, "async_copy_global_to_shared", pred, buffersVT,
          auxData.asyncCpCommits[op], op);
    }
  }

  void addReadChecks(ImplicitLocOpBuilder &b, tti::FunctionBuilder &funcBuilder,
                     Operation *op, Value buf, Value pred, MemType memType,
                     int thread, const std::string &operandName) {
    auto buffersVT = auxData.buffers[(int)memType][op];
    if (!auxData.barriers.empty()) {
      funcBuilder.createVerifyReadVisibilityCall(b, buf, thread, operandName,
                                                 pred, memType, op);
    }
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM && auxData.wgmmaCommits[op].value) {
      funcBuilder.createCheckOutstandingCommitsCall(
          b, buf, thread, "warpgroup_mma operand read", pred, buffersVT,
          auxData.wgmmaCommits[op], op);
    }
  }

  std::optional<MemEffectsOpInfo> getMemEffectsOpInfo(Operation *op) {
    std::optional<MemEffectsOpInfo> info;
    if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = copyOp.getPred();
      info->barriers.push_back({copyOp.getBarrier(), nullptr, 1});
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Write,
                                      /*.buf =*/copyOp.getResult()});
    }
    if (auto storeOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::None;
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Read,
                                      /*.buf =*/storeOp.getSrc()});
    }
    if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = gatherOp.getPred();
      info->barriers.push_back({gatherOp.getBarrier(), nullptr, 1});
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Write,
                                      /*.buf =*/gatherOp.getResult()});
    }
    if (auto scatterOp = dyn_cast<ttng::AsyncTMAScatterOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::None;
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Read,
                                      /*.buf =*/scatterOp.getSrc()});
    }
    if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::asyncCpCommit;
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Write,
                                      /*.buf =*/copyOp.getResult()});
    }
    if (auto loadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Read,
                                      /*.buf =*/loadOp.getSrc()});
    }
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Write,
                                      /*.buf =*/storeOp.getDst()});
    }
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        info->operandEffects.push_back(
            {/*.rw =*/MemEffectsOpInfo::Effects::Write,
             /*.buf =*/allocOp.getResult()});
      }
    }
    if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Read,
                                      /*.buf =*/loadOp.getSrc()});
    }
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Write,
                                      /*.buf =*/storeOp.getDst()});
    }
    if (auto allocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        info->operandEffects.push_back(
            {/*.rw =*/MemEffectsOpInfo::Effects::Write,
             /*.buf =*/allocOp.getResult()});
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
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Read,
                                      /*.buf =*/mmav5Op.getA(),
                                      /*.operandName =*/"A"});
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Read,
                                      /*.buf =*/mmav5Op.getB(),
                                      /*.operandName =*/"B"});
      info->operandEffects.push_back({/*.rw =*/MemEffectsOpInfo::Effects::Write,
                                      /*.buf =*/mmav5Op.getAccumulator(),
                                      /*.operandName =*/"Acc"});
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
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::wgmmaCommit;
        info->barriers = {};
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getA().getType().getEncoding())) {
          info->operandEffects.emplace_back(MemEffectsOpInfo::Effects{
              /*.rw =*/MemEffectsOpInfo::Effects::Read,
              /*.buf =*/wgmmaOp.getA(),
              /*.operandName =*/"A"});
        }
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getB().getType().getEncoding())) {
          info->operandEffects.emplace_back(MemEffectsOpInfo::Effects{
              /*.rw =*/MemEffectsOpInfo::Effects::Read,
              /*.buf =*/wgmmaOp.getB(),
              /*.operandName =*/"B"});
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
