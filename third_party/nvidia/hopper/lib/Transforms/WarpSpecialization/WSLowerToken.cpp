#include "Utility.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <set>

#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace ttnvws = ::mlir::triton::nvws;
namespace mlir {

#define DEBUG_TYPE "tritongpu-warp-spec-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Lower to use GetCanonicalWarpIdOp.
// In Hopper, each task is a warpgroup consisting of 4 warps.
static const int WARPS_PER_TASK = 4;
static const int THREADS_PER_TASK = 128;

Value getMBarrierPhaseBit(OpBuilder &builder, Operation *op,
                          bool emptyBarrier) {
  auto loc = op->getLoc();
  assert(isa<ttnvws::ProducerAcquireOp>(op) || isa<ttnvws::ConsumerWaitOp>(op));
  Value curPhase;
  if (auto acq = dyn_cast<ttnvws::ProducerAcquireOp>(op))
    curPhase = acq.getPhase();
  else if (auto wait = dyn_cast<ttnvws::ConsumerWaitOp>(op))
    curPhase = wait.getPhase();
  if (emptyBarrier) {
    // curPhase = curPhase xor True for emptyBarrier.
    Value _1_1b = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    curPhase = builder.create<mlir::arith::XOrIOp>(loc, curPhase, _1_1b);
  }
  LLVM_DEBUG(curPhase.dump());
  return curPhase;
}

void processProducerAcquireOp(OpBuilder &builder, ttnvws::ProducerAcquireOp op,
                              Value bufferEmpty) {
  auto loc = op.getLoc();
  Value phase = getMBarrierPhaseBit(builder, op, true);
  auto i32Ty = builder.getIntegerType(32);
  phase = builder.create<arith::ExtUIOp>(loc, i32Ty, phase);
  auto waitOp = builder.create<ttng::WaitBarrierOp>(loc, bufferEmpty, phase);
  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(waitOp, getAsyncTaskIds(op.getOperation()));
}

void processProducerCommitOp(OpBuilder &builder, ttnvws::ProducerCommitOp op,
                             Value bufferFull, ttnvws::TokenLoadType loadType,
                             unsigned fullCnt) {
  auto loc = op.getLoc();
  ttng::ArriveBarrierOp arriveOp;

  if (loadType == ttnvws::TokenLoadType::TMALoadOp) {
    // Get the count from the barriers: trace the local_alloc for the barrier
    // then find the count from init_barrier
    arriveOp = builder.create<ttng::ArriveBarrierOp>(loc, bufferFull, fullCnt);
  } else {
    assert(false);
  }

  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(arriveOp, getAsyncTaskIds(op.getOperation()));
}

void processConsumerWaitOp(OpBuilder &builder, ttnvws::ConsumerWaitOp op,
                           Value bufferFull) {
  auto loc = op.getLoc();
  Value phase = getMBarrierPhaseBit(builder, op, false);
  auto i32Ty = builder.getIntegerType(32);
  phase = builder.create<arith::ExtUIOp>(loc, i32Ty, phase);
  auto waitOp = builder.create<ttng::WaitBarrierOp>(loc, bufferFull, phase);
  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(waitOp, getAsyncTaskIds(op.getOperation()));
}

void processConsumerReleaseOp(OpBuilder &builder, ttnvws::ConsumerReleaseOp op,
                              Value bufferEmpty, int numCTAs,
                              unsigned emptyCnt) {
  auto loc = op.getLoc();
  auto arriveOp =
      builder.create<ttng::ArriveBarrierOp>(loc, bufferEmpty, emptyCnt);
  assert(op.getOperation()->hasAttr("async_task_id"));
  setAsyncTaskIds(arriveOp, getAsyncTaskIds(op.getOperation()));
}

void lowerTokenOperations(Operation *parentOp, int numCTAs,
                          int numConsumerGroups) {
  SmallVector<Operation *> deprecatedOps;
  SmallVector<Operation *> deprecatedTokenOps;
  DenseSet<Operation *> warpSpecOps;
  DenseMap<Operation *, Value> tokenToFull;
  DenseMap<Operation *, Value> tokenToEmpty;
  parentOp->walk([&](ttnvws::CreateTokenOp createTokenOp) {
    ttnvws::TokenLoadType loadType = createTokenOp.getLoadType();
    MLIRContext *context = createTokenOp.getContext();
    OpBuilder builder(createTokenOp);
    Location loc = createTokenOp.getLoc();

    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);
    auto barrierCTALayout =
        ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
        context, 1, 1, 1, {0}, barrierCTALayout);
    Type barrierMemDescType = ttg::MemDescType::get(
        {createTokenOp.getNumBuffers(), 1}, builder.getI64Type(),
        barrierEncoding, sharedMemorySpace,
        /*mutableMemory=*/true);
    Type singleBarrierMemDescType =
        ttg::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                              sharedMemorySpace, /*mutableMemory=*/true);
    // These are created prior to warp_specialize.
    Value bufferFullArray = builder.create<mlir::triton::gpu::LocalAllocOp>(
        loc, barrierMemDescType, Value());
    Value bufferEmptyArray = builder.create<mlir::triton::gpu::LocalAllocOp>(
        loc, barrierMemDescType, Value());
    tokenToFull[createTokenOp.getOperation()] = bufferFullArray;
    tokenToEmpty[createTokenOp.getOperation()] = bufferEmptyArray;

    unsigned bufferFullCount =
        loadType == ttnvws::TokenLoadType::TMALoadOp ? 1 : THREADS_PER_TASK;
    unsigned bufferEmptyCount = THREADS_PER_TASK;
    for (unsigned i = 0; i < createTokenOp.getNumBuffers(); i++) {
      Value idx = builder.create<arith::ConstantIntOp>(loc, i, 32);
      Value barrierFullView = builder.create<ttg::MemDescIndexOp>(
          loc, singleBarrierMemDescType, bufferFullArray, idx);
      // EmptyView is used for ConsumerRelease and ProducerAcquire.
      // FullView is for ConsumerWait and ProducerCommit.
      builder.create<ttng::InitBarrierOp>(loc, barrierFullView,
                                          bufferFullCount);

      Value barrierEmptyView = builder.create<ttg::MemDescIndexOp>(
          loc, singleBarrierMemDescType, bufferEmptyArray, idx);
      builder.create<ttng::InitBarrierOp>(loc, barrierEmptyView,
                                          bufferEmptyCount);
    }

    assert(numCTAs == 1 && "remote CTA is not supported yet");
    builder.create<mlir::gpu::BarrierOp>(loc);

    // Helper function for extracting one index from bufferFullArray.
    auto extractBufferFull = [&](Location loc, Value idx) -> Value {
      return builder.create<ttg::MemDescIndexOp>(loc, singleBarrierMemDescType,
                                                 bufferFullArray, idx);
    };

    // Helper function for extracting one index from bufferEmptyArray.
    auto extractBufferEmpty = [&](Location loc, Value idx) -> Value {
      return builder.create<ttg::MemDescIndexOp>(loc, singleBarrierMemDescType,
                                                 bufferEmptyArray, idx);
    };
    auto handleOneUser = [&](Operation *user) -> bool {
      // Here builder is at the user, make sure usage of values outside of
      // warp_specialize is via capture if user is in a partition region.
      // We need bufferFullArray and bufferEmptyArray.
      if (auto op = dyn_cast<ttnvws::ProducerAcquireOp>(user)) {
        Value bufferEmpty = extractBufferEmpty(loc, op.getIdx());
        auto pOp = user->getParentOp();
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferEmpty.getDefiningOp(), getAsyncTaskIds(user));
        processProducerAcquireOp(builder, op, bufferEmpty);
        deprecatedOps.push_back(user);
        return true;
      } else if (auto op = dyn_cast<ttnvws::ProducerCommitOp>(user)) {
        Value bufferFull = extractBufferFull(loc, op.getIdx());
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferFull.getDefiningOp(), getAsyncTaskIds(user));
        processProducerCommitOp(builder, op, bufferFull, loadType,
                                bufferFullCount);
        deprecatedOps.push_back(user);
        return true;
      } else if (auto op = dyn_cast<ttnvws::ConsumerWaitOp>(user)) {
        Value bufferFull = extractBufferFull(loc, op.getIdx());
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferFull.getDefiningOp(), getAsyncTaskIds(user));
        processConsumerWaitOp(builder, op, bufferFull);
        deprecatedOps.push_back(user);
        return true;
      } else if (auto op = dyn_cast<ttnvws::ConsumerReleaseOp>(user)) {
        Value bufferEmpty = extractBufferEmpty(loc, op.getIdx());
        assert(user->hasAttr("async_task_id"));
        setAsyncTaskIds(bufferEmpty.getDefiningOp(), getAsyncTaskIds(user));
        processConsumerReleaseOp(builder, op, bufferEmpty, numCTAs,
                                 bufferEmptyCount);
        deprecatedOps.push_back(user);
        return true;
      }
      return false;
    };

    // Process token users: ProducerAcquireOp, ProducerCommitOp, ConsumerWaitOp,
    // and ConsumerReleaseOp.
    for (OpOperand &use : createTokenOp.getResult().getUses()) {
      Operation *user = use.getOwner();
      auto loc = user->getLoc();
      builder.setInsertionPoint(user);
      bool handled = handleOneUser(user);
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
        unsigned opndNum = use.getOperandNumber();
        // Handle the regions. Trace uses of the argument corresponding to the
        // captured value.
        for (Region *region : wsOp.getPartitionRegions()) {
          LDBG("-- region " << region->getNumArguments());
          auto tArg = region->getArgument(opndNum);
          for (Operation *tUser : tArg.getUsers()) {
            builder.setInsertionPoint(tUser);
            // Use of TokenOp via capture of warp_specialize.
            handleOneUser(tUser);
          }
        }
        warpSpecOps.insert(user);
      } else if (!handled) {
        llvm_unreachable("Unexpected user of token");
      }
    }

    deprecatedTokenOps.push_back(createTokenOp);
  });
  for (auto op : deprecatedOps) {
    LLVM_DEBUG({
      LDBG("erasing deprecatedOps");
      op->dump();
    });
    op->erase();
  }
  unsigned tokenRemoval = 0;
  // Map from tokenOp to bufferFullArray, bufferEmptyArray.
  // If a tokenOp is used by warp_specialize, remove it and add
  // buffer[Full|Empty]Array.

  for (auto op : deprecatedTokenOps) {
    LLVM_DEBUG({
      LDBG("erasing deprecatedOps");
      op->dump();
    });
    ++tokenRemoval;
    if (auto tokenOp = dyn_cast<ttnvws::CreateTokenOp>(op)) {
      // Check to see if it is used by warpSpec. If yes, eraseOperand and
      // eraseArgument.
      for (OpOperand &use : llvm::make_early_inc_range(tokenOp->getUses())) {
        Operation *user = use.getOwner();
        if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(user)) {
          unsigned opndNum = use.getOperandNumber();
          LDBG("wsOp user numOperands: " << wsOp->getNumOperands() << " idx "
                                         << opndNum);

          LLVM_DEBUG({
            LDBG("prior to erasing " << tokenRemoval);
            parentOp->dump();
          });
          wsOp->eraseOperand(opndNum);
          Value empty = tokenToEmpty[op];
          Value full = tokenToFull[op];
          wsOp->insertOperands(wsOp.getNumOperands(), full);
          wsOp->insertOperands(wsOp.getNumOperands(), empty);
          // Handle the regions.
          for (Region *region : wsOp.getPartitionRegions()) {
            LDBG("-- region " << region->getNumArguments());
            auto tArg = region->getArgument(opndNum);
            for (Operation *tUser : tArg.getUsers()) {
              LLVM_DEBUG({
                LDBG("user for arg");
                tUser->dump();
              });
            }
            region->eraseArgument(opndNum);
            BlockArgument arg =
                region->addArgument(full.getType(), full.getLoc());
            replaceAllUsesInRegionWith(full, arg, *region);
            BlockArgument arg2 =
                region->addArgument(empty.getType(), empty.getLoc());
            replaceAllUsesInRegionWith(empty, arg2, *region);
          }
        }
      }
    }
    op->erase();
  }

  assert(numCTAs == 1 && "remote CTA is not supported yet");
  LLVM_DEBUG({
    LDBG("after lowering");
    parentOp->dump();
  });
}

void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups) {
  ModuleOp mod = funcOp.getOperation()->getParentOfType<ModuleOp>();
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);

  // lowerGetAsyncTaskIdOp(mod, numConsumerGroups);
  lowerTokenOperations(mod, numCTAs, numConsumerGroups);
}

} // namespace mlir
