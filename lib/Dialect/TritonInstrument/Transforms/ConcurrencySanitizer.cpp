#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/FunctionBuilder.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

#include <array>

// clang-format off
// Concurrency Sanitizer data structures:
// ConSan keeps auxilary data requied for tracking memory accesses in tensors.
// These tensors are stored as a distributed tensor or in global scratch memory.
//
// Name              | Storage | Rank/Type       | Description
// ------------------|---------|-----------------|------------
// buffers           | tensor  | <B x i64>       | Base pointers of all (sub)buffers
// barriers          | tensor  | <K x i64>       | Pointers to all individual mbarriers
// barrierStates     | scratch | <K x i32>       | Packed barrier phase (bit 0) and arrival counts (bits[1..8] init, [9..16] current); zero means invalid/uninitialized
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
namespace ttn = mlir::triton::nvgpu;

#define GEN_PASS_DEF_TRITONINSTRUMENTCONCURRENCYSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

namespace {

// OpBuilder listener tracking operations added to the builder to be wrapped
// with a lock acquire/release pair.
class CriticalSectionListener : public ImplicitLocOpBuilder::Listener {
public:
  explicit CriticalSectionListener(Region *trackedRegion = nullptr)
      : trackedRegion(trackedRegion) {}

  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint /*previous*/) override {
    if (trackedRegion && op->getParentRegion() != trackedRegion)
      return;
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
  Region *trackedRegion = nullptr;
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

bool isTwoCTATCGenCommitOp(Operation *op) {
  auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op);
  return commitOp && ttng::getModuleTwoCTAs(commitOp);
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

uint16_t getCTABroadcastMask(Type type) {
  auto memDescType = cast<ttg::MemDescType>(type);
  auto kBlock = StringAttr::get(type.getContext(), "block");
  return static_cast<uint16_t>(
      ttg::toLinearLayout(memDescType).getFreeVariableMasks().lookup(kBlock));
}

int getCTAGroupSize(uint16_t ctaMaskBits) {
  return 1 << llvm::popcount(ctaMaskBits);
}

Value orMasks(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  if (lhs && rhs)
    return arith::OrIOp::create(b, lhs, rhs);
  return lhs ? lhs : rhs;
}

Value combinePredicates(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  if (lhs && rhs)
    return arith::AndIOp::create(b, b.getLoc(), lhs, rhs);
  return lhs ? lhs : rhs;
}

// Get the mask of the CTAs in the same broadcasting group as the current CTA.
// Same as the logic in createTMAMulticastMask
Value broadcastingGroupCTAMask(ImplicitLocOpBuilder &b,
                               uint16_t broadcastBits) {
  int numCTAs = ttg::lookupNumCTAs(b.getInsertionBlock()->getParentOp());
  if (numCTAs == 1)
    return arith::ConstantIntOp::create(b, 1, 32);

  int blockBits = llvm::Log2_32(numCTAs);
  uint32_t fixedBits = (~broadcastBits) & (numCTAs - 1);
  uint32_t pattern = 1;
  for (int i = 0; i < blockBits; ++i) {
    if ((fixedBits & (1u << i)) == 0)
      pattern |= (pattern << (1u << i));
  }

  Value currentCTA = ttn::ClusterCTAIdOp::create(b, b.getLoc());
  Value fixedBitsVal = arith::ConstantIntOp::create(b, fixedBits, 32);
  Value patternVal = arith::ConstantIntOp::create(b, pattern, 32);
  Value base = arith::AndIOp::create(b, currentCTA, fixedBitsVal);
  return arith::ShLIOp::create(b, patternVal, base);
}

// Same as the logic in getLeaderCTAPredicate.
Value leadCTAPredicate(ImplicitLocOpBuilder &b, uint16_t broadcastBits) {
  if (!broadcastBits)
    return Value();
  Value ctaId = ttn::ClusterCTAIdOp::create(b, b.getLoc());
  Value broadcastBitsVal = arith::ConstantIntOp::create(b, broadcastBits, 32);
  Value zero = arith::ConstantIntOp::create(b, 0, 32);
  return arith::CmpIOp::create(
      b, arith::CmpIPredicate::eq,
      arith::AndIOp::create(b, ctaId, broadcastBitsVal), zero);
}

Value ctaGroupMask(ImplicitLocOpBuilder &b, bool twoCTAs) {
  return broadcastingGroupCTAMask(b, twoCTAs ? 1 : 0);
}

Value mmaCommitCTAMask(ImplicitLocOpBuilder &b, ValueRange descs,
                       bool twoCTAs) {
  Value mask;
  for (Value desc : descs) {
    uint16_t broadcastBits = getCTABroadcastMask(desc.getType());
    if (twoCTAs)
      broadcastBits |= 1;
    if (broadcastBits)
      mask = orMasks(b, mask, broadcastingGroupCTAMask(b, broadcastBits));
  }
  if (!mask && twoCTAs)
    mask = broadcastingGroupCTAMask(b, 1);
  return mask ? mask : broadcastingGroupCTAMask(b, 0);
}

uint32_t getMemDescLength(Value buf) {
  auto memDescType = cast<ttg::MemDescType>(buf.getType());
  if (isa<ttg::SharedEncodingTrait>(memDescType.getEncoding())) {
    unsigned elSize = memDescType.getElementType().getIntOrFloatBitWidth() / 8;
    return static_cast<uint32_t>(product(memDescType.getShape()) * elSize);
  }
  if (isa<ttng::TensorMemorySpaceAttr>(memDescType.getMemorySpace())) {
    return ttng::getTmemAllocSizes(memDescType).numCols;
  }
  llvm_unreachable("Unsupported memory space for memdesc");
}

} // namespace

class ConcurrencySanitizerPass
    : public impl::TritonInstrumentConcurrencySanitizerBase<
          ConcurrencySanitizerPass> {
public:
  void runOnOperation() override {
    module = getOperation();
    tti::FunctionBuilder funcBuilder(module, auxData);
    auxData.populateAndPassToWarpSpecialize(module, funcBuilder);

    tt::FuncOp entryPoint = tti::getEntryPoint(module);

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());
    instrumentMemoryOperations(b, funcBuilder);
  }

private:
  void instrumentMemoryOperations(ImplicitLocOpBuilder &b,
                                  tti::FunctionBuilder &funcBuilder) {
    tti::getEntryPoint(module).walk([&](Operation *op) {
      CriticalSectionListener listener(op->getParentRegion());
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

      bool skipBarrierInitializedCheck = false;
      if (isTwoCTATCGenCommitOp(op)) {
        auto commitOp = cast<ttng::TCGen5CommitOp>(op);
        b.setListener(nullptr);
        std::optional<MemEffectsOpInfo> commitInfo = getMemEffectsOpInfo(b, op);
        b.setListener(&listener);
        assert(commitInfo && commitInfo->barriers.size() == 1 &&
               "tcgen5 commit must have a single tracked barrier");
        Value pred = combinePredicates(b, commitInfo->pred,
                                       commitInfo->representativePred);
        const auto &barrierInfo = commitInfo->barriers.front();
        funcBuilder.createVerifyBarrierInitializedCall(
            b, barrierInfo.barrier, pred, commitOp, barrierInfo.verifyCtaMask);
        skipBarrierInitializedCheck = true;
        b.setInsertionPointAfter(op);
      }

      instrumentMemEffects(b, op, thread, funcBuilder,
                           skipBarrierInitializedCheck, &listener);
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
      if (auto initOp = dyn_cast<ttng::InitBarrierOp>(op)) {
        b.setListener(nullptr);
        Value barrier = initOp.getAlloc();
        uint16_t ctaMaskBits = getCTABroadcastMask(barrier.getType());
        Value barrierMask = broadcastingGroupCTAMask(b, ctaMaskBits);
        int count = initOp.getCount() * getCTAGroupSize(ctaMaskBits);
        b.setListener(&listener);
        funcBuilder.createVerifyBarrierCanInitCall(b, barrier, initOp,
                                                   barrierMask);
        funcBuilder.createInitBarrierStateCall(b, barrier, count, initOp,
                                               barrierMask);
      }
      if (auto invalOp = dyn_cast<ttng::InvalBarrierOp>(op)) {
        b.setListener(nullptr);
        Value barrier = invalOp.getAlloc();
        uint16_t ctaMaskBits = getCTABroadcastMask(barrier.getType());
        Value ctaMask = broadcastingGroupCTAMask(b, ctaMaskBits);
        Value leaderPred = leadCTAPredicate(b, ctaMaskBits);
        b.setListener(&listener);
        funcBuilder.createVerifyBarrierInitializedCall(b, barrier, leaderPred,
                                                       invalOp, ctaMask);
        funcBuilder.createInvalidateBarrierStateCall(b, barrier, invalOp,
                                                     ctaMask);
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          funcBuilder.createClearBarrierWriteTrackingCall(
              b, barrier, leaderPred, memType, invalOp, ctaMask);
          funcBuilder.createClearBarrierReadTrackingCall(
              b, barrier, leaderPred, memType, invalOp, ctaMask);
        }
      }
      if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op)) {
        // Pre-wait: mark waiting threads and check for deadlock.
        {
          CriticalSectionListener preListener(waitOp->getParentRegion());
          b.setInsertionPoint(waitOp);
          auto pred = waitOp.getPred();
          auto barrier = waitOp.getAlloc();
          b.setListener(nullptr);
          uint16_t ctaMaskBits = getCTABroadcastMask(barrier.getType());
          Value barrierMask = broadcastingGroupCTAMask(b, ctaMaskBits);
          b.setListener(&preListener);
          funcBuilder.createVerifyBarrierInitializedCall(b, barrier, pred,
                                                         waitOp, barrierMask);
          funcBuilder.createSetWaitingCall(b, barrier, baseThread,
                                           waitOp.getPhase(), pred, waitOp);
          funcBuilder.createCheckAllActiveWaitingCall(
              b, getActiveMask(op), pred, waitOp, barrierMask);

          preListener.maybeWrapWithCriticalSection(b, auxData, pred);
          b.setListener(&listener);
          b.setInsertionPointAfter(waitOp);
        }
        // Post-wait: transfer visible writes and reads to all peer threads,
        // and clear waiting for this barrier
        assert(!auxData.barriers.empty() &&
               "barrier descriptors must exist when instrumenting wait");
        auto pred = waitOp.getPred();
        auto barrier = waitOp.getAlloc();
        b.setListener(nullptr);
        uint16_t barrierCtaMaskBits = getCTABroadcastMask(barrier.getType());
        Value clearWaitingPred =
            combinePredicates(b, pred, leadCTAPredicate(b, barrierCtaMaskBits));

        constexpr std::array<MemType, 2> transferMemTypes = {
            MemType::SHARED_MEM, MemType::TENSOR_MEM};
        for (MemType memType : transferMemTypes) {
          SmallVector<uint16_t> transferCtaMaskBits;
          for (Value dep : waitOp.getDeps()) {
            auto memDescType = dyn_cast<ttg::MemDescType>(dep.getType());
            if (!memDescType)
              continue;
            bool isShared =
                isa<ttg::SharedEncodingTrait>(memDescType.getEncoding());
            if ((memType == MemType::SHARED_MEM && isShared) ||
                (memType == MemType::TENSOR_MEM &&
                 isa<ttng::TensorMemorySpaceAttr>(
                     memDescType.getMemorySpace()))) {
              uint16_t bits = getCTABroadcastMask(memDescType);
              if (!llvm::is_contained(transferCtaMaskBits, bits))
                transferCtaMaskBits.push_back(bits);
            }
          }
          if (transferCtaMaskBits.empty())
            transferCtaMaskBits.push_back(barrierCtaMaskBits);
          b.setListener(&listener);
          for (uint16_t bits : transferCtaMaskBits) {
            Value transferMask = broadcastingGroupCTAMask(b, bits);
            Value transferPred =
                combinePredicates(b, pred, leadCTAPredicate(b, bits));
            // Transfer visible writes and reads to all peer threads.
            funcBuilder.createTransferVisibleWritesCall(
                b, barrier, getThreadPeersMask(thread), transferPred, memType,
                op, transferMask, transferMask);
            funcBuilder.createTransferVisibleReadsCall(
                b, barrier, getThreadPeersMask(thread), transferPred, memType,
                op, transferMask, transferMask);
          }
          b.setListener(nullptr);
        }
        funcBuilder.createClearWaitingCall(b, barrier, baseThread,
                                           clearWaitingPred, waitOp);
      }
      if (isa<ttng::ClusterWaitOp, ttng::ClusterBarrierOp>(op)) {
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          funcBuilder.createClusterSyncWritesCall(b, memType, op);
          funcBuilder.createClusterSyncReadsCall(b, memType, op);
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
      if (auto tmaStoreWaitOp = dyn_cast<ttng::TMAStoreWaitOp>(op)) {
        funcBuilder.createClearOutstandingCommitsTransferReadsCall(
            b, baseThread, getThreadPeersMask(thread),
            tmaStoreWaitOp.getPendings(), nullptr, CommitKind::TmaStore,
            MemType::SHARED_MEM, op);
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
      uint32_t length = 0;
      Value ctaMask;

      Effects(RW rw, Value buf, std::string operandName = "",
              Value ctaMask = {})
          : rw(rw), buf(buf), operandName(operandName),
            length(getMemDescLength(buf)), ctaMask(ctaMask) {}
    };
    struct BarrierInfo {
      Value barrier;
      Value pred;
      int count;
      Value ctaMask;
      Value verifyCtaMask;
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
    Value representativePred;
    SmallVector<Effects> operandEffects;
    bool implicitCommit = false;
  };

  void instrumentMemEffects(ImplicitLocOpBuilder &b, Operation *op, int thread,
                            tti::FunctionBuilder &funcBuilder,
                            bool skipBarrierInitializedCheck,
                            ImplicitLocOpBuilder::Listener *listener) {
    int baseThread = getBaseThread(thread);
    b.setListener(nullptr);
    std::optional<MemEffectsOpInfo> opInfo = getMemEffectsOpInfo(b, op);
    b.setListener(listener);
    if (!opInfo)
      return;
    Value pred = combinePredicates(b, opInfo->pred, opInfo->representativePred);
    auto combineBarrierPredicates = [&](Value barrierPred) -> Value {
      return combinePredicates(b, barrierPred, pred);
    };
    for (auto effect : opInfo->operandEffects) {
      Value buf = effect.buf;
      Value ctaMask = effect.ctaMask;
      auto bufType = cast<ttg::MemDescType>(buf.getType());
      MemType memType = MemType::TENSOR_MEM;
      if (isa<ttg::SharedEncodingTrait>(bufType.getEncoding())) {
        memType = MemType::SHARED_MEM;
      }
      if (effect.rw == MemEffectsOpInfo::Effects::Read) {
        // For op that is reading, we only need to check if anything else
        // is writing to the same buffer.
        addWriteChecks(b, funcBuilder, op, buf, effect.length, pred, memType,
                       thread, effect.operandName, ctaMask);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier) {
          funcBuilder.createSetReadVisibilityCall(b, buf, effect.length,
                                                  getThreadPeersMask(thread),
                                                  pred, memType, op, ctaMask);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::CommitCount) {
          assert(memType == MemType::SHARED_MEM);
          funcBuilder.createStageAccessForCommitCall(
              b, buf, effect.length, baseThread, pred, memType,
              opInfo->commitKind, op, ctaMask);
        }
      }
      if (effect.rw == MemEffectsOpInfo::Effects::Write) {
        // Op is writing to the buffer, we need to check if anything else
        // is reading or writing to the same buffer.
        addWriteChecks(b, funcBuilder, op, buf, effect.length, pred, memType,
                       thread, effect.operandName, ctaMask);
        addReadChecks(b, funcBuilder, op, buf, effect.length, pred, memType,
                      thread, effect.operandName, ctaMask);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier) {
          funcBuilder.createSetWriteVisibilityCall(b, buf, effect.length,
                                                   getThreadPeersMask(thread),
                                                   pred, memType, op, ctaMask);
          funcBuilder.createClearWriteTrackingCall(b, buf, effect.length, pred,
                                                   memType, op, ctaMask);
          funcBuilder.createClearReadVisibilityCall(b, buf, effect.length, pred,
                                                    memType, op, ctaMask);
          funcBuilder.createClearReadTrackingCall(b, buf, effect.length, pred,
                                                  memType, op, ctaMask);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::CommitCount) {
          assert(memType == MemType::SHARED_MEM);
          funcBuilder.createStageAccessForCommitCall(
              b, buf, effect.length, baseThread, pred, memType,
              opInfo->commitKind, op, ctaMask);
        }
      }
    }
    for (const auto &barrierInfo : opInfo->barriers) {
      Value barrier = barrierInfo.barrier;
      Value combinedPred = combineBarrierPredicates(barrierInfo.pred);
      Value barrierMask = barrierInfo.ctaMask;
      Value verifyMask = barrierInfo.verifyCtaMask;
      if (!skipBarrierInitializedCheck) {
        funcBuilder.createVerifyBarrierInitializedCall(b, barrier, combinedPred,
                                                       op, verifyMask);
      }
      // If the op has barriers, we treat it as a commit emitted for each
      // barrier.
      for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
        funcBuilder.createTrackVisibleWritesCall(
            b, barrier, thread, combinedPred, memType, op, barrierMask);
        funcBuilder.createTrackVisibleReadsCall(
            b, barrier, thread, combinedPred, memType, op, barrierMask);
      }
      if (barrierInfo.count > 0) {
        funcBuilder.createVerifyBarrierArriveCall(
            b, barrier, barrierInfo.count, combinedPred, op, barrierMask);
        funcBuilder.createUpdateBarrierStateCall(b, barrier, barrierInfo.count,
                                                 combinedPred, op, barrierMask);
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
                      Value ctaMask) {
    funcBuilder.createVerifyWriteVisibilityCall(
        b, buf, length, thread, operandName, pred, memType, op, ctaMask);
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM) {
      funcBuilder.createCheckOutstandingCommitsCall(
          b, buf, length, getBaseThread(thread), "async_copy_global_to_shared",
          pred, memType, CommitKind::AsyncCp, op, ctaMask);
    }
  }

  void addReadChecks(ImplicitLocOpBuilder &b, tti::FunctionBuilder &funcBuilder,
                     Operation *op, Value buf, uint32_t length, Value pred,
                     MemType memType, int thread,
                     const std::string &operandName, Value ctaMask) {
    funcBuilder.createVerifyReadVisibilityCall(
        b, buf, length, thread, operandName, pred, memType, op, ctaMask);
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM) {
      funcBuilder.createCheckOutstandingCommitsCall(
          b, buf, length, getBaseThread(thread), "warpgroup_mma operand read",
          pred, memType, CommitKind::Wgmma, op, ctaMask);
      funcBuilder.createCheckOutstandingCommitsCall(
          b, buf, length, getBaseThread(thread), "async_copy_shared_to_global",
          pred, memType, CommitKind::TmaStore, op, ctaMask);
    }
  }

  std::optional<MemEffectsOpInfo> getMemEffectsOpInfo(ImplicitLocOpBuilder &b,
                                                      Operation *op) {
    std::optional<MemEffectsOpInfo> info;
    auto localMask = [&]() { return broadcastingGroupCTAMask(b, 0); };
    if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op)) {
      // TODO: For async TMA barriers, the barrier "arrive" corresponding to the
      // completion mechanism is modeled by barrier_expect. Individual
      // async_tma_copy ops should not decrement the barrier state, otherwise
      // multiple copies using the same barrier would incorrectly advance the
      // phase multiple times. This should be improved bu tracking the barrier
      // expected byte count, and "arriving" the barrier when the expected byte
      // count is reached.
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = expectOp.getPred();
      Value barrierMask = broadcastingGroupCTAMask(
          b, getCTABroadcastMask(expectOp.getAlloc().getType()));
      info->barriers.push_back({expectOp.getAlloc(), nullptr, /*count=*/1,
                                barrierMask, barrierMask});
    }
    if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      Value ctaMask =
          copyOp.getMulticast()
              ? broadcastingGroupCTAMask(
                    b, getCTABroadcastMask(copyOp.getResult().getType()))
              : localMask();
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = copyOp.getPred();
      if (copyOp.getMulticast())
        info->representativePred = leadCTAPredicate(
            b, getCTABroadcastMask(copyOp.getResult().getType()));
      Value barrierMask = broadcastingGroupCTAMask(
          b, getCTABroadcastMask(copyOp.getBarrier().getType()));
      // Only track visible accesses against the barrier; do not update the
      // barrier state here (see BarrierExpectOp handling above).
      info->barriers.push_back(
          {copyOp.getBarrier(), nullptr, /*count=*/0, ctaMask, barrierMask});
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        copyOp.getResult(), "", ctaMask);
    }
    if (auto storeOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
      info->commitKind = CommitKind::TmaStore;
      info->implicitCommit = true;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        storeOp.getSrc(), "", localMask());
    }
    if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = gatherOp.getPred();
      // Only track visible accesses against the barrier; do not update the
      // barrier state here (see BarrierExpectOp handling above).
      info->barriers.push_back(
          {gatherOp.getBarrier(), nullptr, /*count=*/0, localMask(),
           broadcastingGroupCTAMask(
               b, getCTABroadcastMask(gatherOp.getBarrier().getType()))});
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        gatherOp.getResult(), "", localMask());
    }
    if (auto scatterOp = dyn_cast<ttng::AsyncTMAScatterOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::None;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        scatterOp.getSrc(), "", localMask());
    }
    if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
      info->commitKind = CommitKind::AsyncCp;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        copyOp.getResult(), "", localMask());
    }
    if (auto loadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        loadOp.getSrc(), "", localMask());
    }
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        storeOp.getDst(), "", localMask());
    }
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                          allocOp.getResult(), "", localMask());
      }
    }
    if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                        loadOp.getSrc(), "", localMask());
    }
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                        storeOp.getDst(), "", localMask());
    }
    if (auto allocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
      if (allocOp.getSrc()) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
        info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Write,
                                          allocOp.getResult(), "", localMask());
      }
    }
    if (auto mmav5Op = dyn_cast<ttng::TCGen5MMAOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = mmav5Op.getPredicate();
      bool twoCTAs = mmav5Op.getTwoCtas();
      if (twoCTAs)
        info->representativePred = leadCTAPredicate(b, /*broadcastBits=*/1);
      SmallVector<Value> commitDescs;
      if (mmav5Op.getMulticast()) {
        if (isa<ttg::SharedEncodingTrait>(
                mmav5Op.getA().getType().getEncoding()))
          commitDescs.push_back(mmav5Op.getA());
        commitDescs.push_back(mmav5Op.getB());
      }
      Value commitMask = mmaCommitCTAMask(b, commitDescs, twoCTAs);
      for (auto [barrier, barrierPred] :
           llvm::zip(mmav5Op.getCompletionBarriers(),
                     mmav5Op.getCompletionBarrierPreds())) {
        Value verifyMask =
            broadcastingGroupCTAMask(b, getCTABroadcastMask(barrier.getType()));
        info->barriers.push_back(
            {barrier, barrierPred, /*count=*/1, commitMask, verifyMask});
      }
      info->operandEffects.emplace_back(
          MemEffectsOpInfo::Effects::Read, mmav5Op.getA(), "A",
          broadcastingGroupCTAMask(
              b, getCTABroadcastMask(mmav5Op.getA().getType())));
      info->operandEffects.emplace_back(
          MemEffectsOpInfo::Effects::Read, mmav5Op.getB(), "B",
          broadcastingGroupCTAMask(
              b, getCTABroadcastMask(mmav5Op.getB().getType())));
      info->operandEffects.emplace_back(
          MemEffectsOpInfo::Effects::Write, mmav5Op.getAccumulator(), "Acc",
          broadcastingGroupCTAMask(
              b, getCTABroadcastMask(mmav5Op.getAccumulator().getType())));
    }
    if (auto mmav5Op = dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = mmav5Op.getPredicate();
      bool twoCTAs = mmav5Op.getTwoCtas();
      if (twoCTAs)
        info->representativePred = leadCTAPredicate(b, /*broadcastBits=*/1);
      Value commitMask = mmaCommitCTAMask(b, ValueRange{}, twoCTAs);
      for (auto [barrier, barrierPred] :
           llvm::zip(mmav5Op.getCompletionBarriers(),
                     mmav5Op.getCompletionBarrierPreds())) {
        Value verifyMask =
            broadcastingGroupCTAMask(b, getCTABroadcastMask(barrier.getType()));
        info->barriers.push_back(
            {barrier, barrierPred, /*count=*/1, commitMask, verifyMask});
      }
      info->operandEffects.emplace_back(
          MemEffectsOpInfo::Effects::Read, mmav5Op.getA(), "A",
          broadcastingGroupCTAMask(
              b, getCTABroadcastMask(mmav5Op.getA().getType())));
      info->operandEffects.emplace_back(
          MemEffectsOpInfo::Effects::Read, mmav5Op.getB(), "B",
          broadcastingGroupCTAMask(
              b, getCTABroadcastMask(mmav5Op.getB().getType())));
      info->operandEffects.emplace_back(
          MemEffectsOpInfo::Effects::Write, mmav5Op.getAccumulator(), "Acc",
          broadcastingGroupCTAMask(
              b, getCTABroadcastMask(mmav5Op.getAccumulator().getType())));
    }
    if (auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = commitOp.getPred();
      bool twoCTAs = ttng::getModuleTwoCTAs(op);
      if (twoCTAs)
        info->representativePred = leadCTAPredicate(b, /*broadcastBits=*/1);
      Value commitMask = mmaCommitCTAMask(b, commitOp.getDescs(), twoCTAs);
      Value verifyMask = broadcastingGroupCTAMask(
          b, getCTABroadcastMask(commitOp.getBarrier().getType()));
      info->barriers.push_back({commitOp.getBarrier(), nullptr, /*count=*/1,
                                commitMask, verifyMask});
    }
    if (auto arriveOp = dyn_cast<ttng::ArriveBarrierOp>(op)) {
      info.emplace();
      info->trackingKind = MemEffectsOpInfo::TrackingKind::Barrier;
      info->pred = arriveOp.getPred();
      Value barrierMask = broadcastingGroupCTAMask(
          b, getCTABroadcastMask(arriveOp.getAlloc().getType()));
      info->barriers.push_back({arriveOp.getAlloc(), nullptr,
                                (int)arriveOp.getCount(), barrierMask,
                                barrierMask});
    }
    if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      if (wgmmaOp.getIsAsync() == true) {
        info.emplace();
        info->trackingKind = MemEffectsOpInfo::TrackingKind::CommitCount;
        info->commitKind = CommitKind::Wgmma;
        info->implicitCommit = true;
        info->barriers = {};
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getA().getType().getEncoding())) {
          info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                            wgmmaOp.getA(), "A", localMask());
        }
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getB().getType().getEncoding())) {
          info->operandEffects.emplace_back(MemEffectsOpInfo::Effects::Read,
                                            wgmmaOp.getB(), "B", localMask());
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
