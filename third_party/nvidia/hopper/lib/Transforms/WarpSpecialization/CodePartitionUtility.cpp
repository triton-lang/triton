#include "CodePartitionUtility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-utility"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Check to see if op is enclosed under ifOp.
bool enclosing(scf::IfOp ifOp, Operation *op) {
  return ifOp->isProperAncestor(op);
}

bool enclosing(scf::ForOp forOp, Operation *op) {
  return forOp->isProperAncestor(op);
}

// Check to see if there is no outer loop that is enclosed under ifOp.
bool immediateEnclosing(scf::IfOp ifOp, Operation *subOp) {
  auto pOp = subOp->getParentOfType<scf::ForOp>();
  if (!pOp)
    return true;
  return !enclosing(ifOp, pOp.getOperation());
}

// Control Ops can be replaced during the pass, but channel srcOp/dstOp should
// be valid.
static bool needAccumCntForReuse(Operation *ctrlOp, ReuseGroup *group) {
  // Goes through each channel in the ResuseGroup, check srcOp and dstOp to
  // see if it is inside ctrlOp.
  for (auto *ch : group->channels) {
    if (auto forOp = dyn_cast<scf::ForOp>(ctrlOp)) {
      if (enclosing(forOp, ch->getSrcOp()))
        return true;
      if (enclosing(forOp, ch->getDstOp()))
        return true;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(ctrlOp)) {
      if (enclosing(ifOp, ch->getSrcOp()))
        return true;
      if (enclosing(ifOp, ch->getDstOp()))
        return true;
    }
  }
  return false;
}

// Return number of AccumCnts for the given ctrlOp. We need one for each nested
// region that contains a channel. Also add accumCnt for each ReuseGroup. We can
// use a simplify pass later on to remove redundant accumCnt.
unsigned getAccumCnts(Operation *ctrlOp,
                      const DenseSet<Operation *> &regionsWithChannels,
                      ReuseConfig *config) {
  unsigned cnt = 0;
  LDBG("getAccumCnts: " << ctrlOp);
  for (auto *op : regionsWithChannels) {
    LDBG("-- getAccumCnts: " << ctrlOp << " regionsWithChannels " << op);
    if (ctrlOp == op) {
      ++cnt;
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(ctrlOp)) {
      if (enclosing(forOp, op))
        ++cnt;
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(ctrlOp)) {
      if (enclosing(ifOp, op))
        ++cnt;
      continue;
    }
    llvm_unreachable("region op other than If/For is not supported");
  }
  if (!config)
    return cnt;
  // Go through each ReuseGroup, and see if we need accumCnt for the given
  // ctrlOp. We need one for a given ReuseGroup when ctrlOp encloses an op from
  // the ReuseGroup.
  for (auto &group : config->groups)
    if (needAccumCntForReuse(ctrlOp, &group))
      ++cnt;
  return cnt;
}

// Figure out the argument index for parentForOp, associated with either
// ctrlOp or with the reuse group. For the latter, we ignore ctrlOp,
// get numbers of arguments for unique channels in parentForOp, then
// decide accumCnts for reuse groups. When reuseGroupIdx is negative,
// we find the argument index associated with unique channels inside
// ctrlOp.
unsigned getAccumArgIdx(scf::ForOp parentForOp, Operation *ctrlOp,
                        const DenseSet<Operation *> &regionsWithChannels,
                        ReuseConfig *config, int reuseGroupIdx) {
  if (reuseGroupIdx >= 0) {
    auto cnts = getAccumCnts(parentForOp, regionsWithChannels, nullptr);
    for (unsigned idx = 0; idx < reuseGroupIdx; ++idx) {
      if (needAccumCntForReuse(parentForOp.getOperation(),
                               config->getGroup(idx)))
        ++cnts;
      return cnts;
    }
  }
  // Walk parentForOp in preorder.
  unsigned preOrderId = 0, ctrlId = 0;
  bool found = false;
  parentForOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    // This will walk parentForOp.
    if (subOp == ctrlOp) {
      ctrlId = preOrderId;
      found = true;
    }
    for (auto *op : regionsWithChannels) {
      if (op == subOp) {
        LDBG("getAccumArgIdx: saw ctrlOp enclosing channel " << subOp);
        ++preOrderId;
      }
    }
  });
  assert(found && "error in getAccumArgIdx");
  LDBG("getAccumArgIdx: " << parentForOp.getOperation() << " " << ctrlOp << " "
                          << ctrlId);
  return ctrlId;
}

// Find channels of reuse group that are inside regionOp. If the channel is
// directly in regionOp, add the channel's DstOp, otherwise add the region Op
// that is directly in regionOp and encloses the channel.
void getReuseChannels(ReuseGroup *group, Operation *regionOp,
                      SmallVector<Operation *> &chList) {
  // Goes through body of regionOp, if the body op is a regionOp, check
  // to see if it contains a channel in the reuse group.
  if (auto ifOp = dyn_cast<scf::IfOp>(regionOp)) {
    for (Operation &op : ifOp.thenBlock()->getOperations()) {
      if (isa<scf::ForOp>(&op) || isa<scf::IfOp>(&op)) {
        if (needAccumCntForReuse(&op, group))
          chList.push_back(&op);
      } else {
        // Check if op is dstOp of a channel in reuse group. Assume srcOp and
        // dstOp has the same enclosing parentOp.
        for (auto *ch : group->channels) {
          if (&op == ch->getDstOp())
            chList.push_back(&op);
        }
      }
    }
    return;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(regionOp)) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (isa<scf::ForOp>(&op) || isa<scf::IfOp>(&op)) {
        if (needAccumCntForReuse(&op, group))
          chList.push_back(&op);
      } else {
        // Check if op is dstOp of a channel in reuse group. Assume srcOp and
        // dstOp has the same enclosing parentOp.
        for (auto *ch : group->channels) {
          if (&op == ch->getDstOp())
            chList.push_back(&op);
        }
      }
    }
    return;
  }
  assert(false);
}

// regionOp must contains channels in config[idx].
unsigned getReuseAccumArgIdx(Operation *regionOp,
                             const DenseSet<Operation *> &regionsWithChannels,
                             ReuseConfig *config, int reuseGroupIdx) {
  auto cnts = getAccumCnts(regionOp, regionsWithChannels, nullptr);
  unsigned argIdx = 0;
  assert(reuseGroupIdx >= 0 && reuseGroupIdx < config->getGroupSize());
  for (unsigned idx = 0; idx < reuseGroupIdx; ++idx) {
    if (needAccumCntForReuse(regionOp, config->getGroup(reuseGroupIdx)))
      ++argIdx;
  }
  return cnts + argIdx;
}

// Compute and return the buffer index and phase for a given accumulate count.
std::pair<Value, Value> getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                                             Location loc, Value accumCnt,
                                             unsigned numBuffers) {
  Value numBuffersVal =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, numBuffers, 32);
  numBuffersVal = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
      loc, builder.getI64Type(), numBuffersVal);
  // Calculate accumCnt / numBuffers
  // initBufferIdx = accumCnt - accumCnt / numBuffers * numBuffers
  // initPhase = (accumCnt / numBuffers) & 1
  Value bufferIdx = builder.createWithAsyncTaskIds<arith::DivUIOp>(
      loc, accumCnt, numBuffersVal);
  Value initBufferIdx = builder.createWithAsyncTaskIds<arith::SubIOp>(
      loc, accumCnt,
      builder.createWithAsyncTaskIds<arith::MulIOp>(loc, bufferIdx,
                                                    numBuffersVal));
  initBufferIdx = builder.createWithAsyncTaskIds<arith::TruncIOp>(
      loc, builder.getI32Type(), initBufferIdx);

  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
  bufferIdx =
      builder.createWithAsyncTaskIds<arith::AndIOp>(loc, bufferIdx, one);
  Value initPhase = builder.createWithAsyncTaskIds<arith::TruncIOp>(
      loc, builder.getI1Type(), bufferIdx);
  return {initBufferIdx, initPhase};
}

// Get the current accumulation count for the given op within its immediate
// scope.
// ForA (accumForA, accumIfA, accumForB, accumIfB)
//   IfA (accumIfA, accumForB)
//     Channel A --> uses ForA.arg[accumIfA]
//     ForB (accumForB)
//       Channel B --> uses ForB.arg[accumForB]
//   ThenYield ForA.arg[accumIfA] + 1, ForB.res[accumForB]
//   ElseYield ForA.arg[accumIfA], ForA.arg[accumForB]
//   ForC (accumForC, accumIfB)
//     IfB
//       Channel C --> uses ForC.arg[accumIfB]
//     ThenYield ForC.arg[accumIfB] + 1
//     ElseYield ForC.arg[accumIfB]
//   Channel D --> uses ForA.arg[accumForA]
Value getAccumCount(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                    const DenseSet<Operation *> &regionsWithChannels,
                    ReuseConfig *config, int reuseGroupIdx) {
  auto parentForOp = op->getParentOfType<scf::ForOp>();
  auto *pOp = op->getParentOp();
  // Get parentForOp.arg[pOp]
  unsigned tSize = parentForOp.getBody()->getArguments().size();
  unsigned parentTCnts = getAccumCnts(parentForOp, regionsWithChannels, config);
  unsigned accumArgId = getAccumArgIdx(parentForOp, pOp, regionsWithChannels,
                                       config, reuseGroupIdx);
  Value accumCnt =
      parentForOp.getBody()->getArgument(tSize - parentTCnts + accumArgId);

  LDBG("getAccumCount: parentForOp " << parentForOp.getOperation() << " pOp "
                                     << pOp << " " << tSize << " "
                                     << parentTCnts << " " << accumArgId);
  return accumCnt;
}

void getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                          unsigned numBuffers,
                          const DenseSet<Operation *> &regionsWithChannels,
                          Value &bufferIdx, Value &phase, ReuseConfig *config,
                          int reuseGroupIdx) {
  Value accumCnt =
      getAccumCount(builder, op, regionsWithChannels, config, reuseGroupIdx);
  std::tie(bufferIdx, phase) =
      getBufferIdxAndPhase(builder, op->getLoc(), accumCnt, numBuffers);
}

Value getBarrierForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                 Value barrierAlloc, Value bufferIdx) {
  auto context = barrierAlloc.getContext();
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType barrierTy = ttg::MemDescType::get(
      {1}, builder.getI64Type(),
      cast<ttg::MemDescType>(barrierAlloc.getType()).getEncoding(),
      sharedMemorySpace,
      /*mutableMemory=*/true);

  // Create barrierForTMA from barrierAlloc.
  return builder.createWithAsyncTaskIds<ttg::MemDescSubviewOp>(
      barrierAlloc.getLoc(), barrierTy, barrierAlloc,
      ArrayRef<Value>({bufferIdx}));
}

} // namespace mlir
