#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"

/*
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
//#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Tools/Sys/GetEnv.hpp"
*/
#include "CodePartitionUtility.h"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "tritongpu-warp-spec-utility"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Check to see if op is enclosed under ifOp.
bool enclosing(scf::IfOp ifOp, Operation *op) {
  auto pOp = op->getParentOfType<scf::IfOp>();
  while (pOp) {
    if (pOp == ifOp)
      return true;
    pOp = pOp->getParentOfType<scf::IfOp>();
  }
  return false;
}

bool enclosing(scf::ForOp forOp, Operation *op) {
  auto pOp = op->getParentOfType<scf::ForOp>();
  while (pOp) {
    if (pOp == forOp)
      return true;
    pOp = pOp->getParentOfType<scf::ForOp>();
  }
  return false;
}

bool channelWithReuse(Operation *dstOp,
                      SmallVector<Operation *> &opsWithBufferReuse) {
  for (auto *op : opsWithBufferReuse) {
    if (dstOp == op) {
      return true;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      if (enclosing(forOp, dstOp))
        return true;
    if (auto ifOp = dyn_cast<scf::IfOp>(op))
      if (enclosing(ifOp, dstOp))
        return true;
  }
  return false;
}

// opsWithChannels: ctrl ops with channels directly under
void excludeChannelsWithReuse(const DenseSet<Operation *> &opsWithChannels,
                              SmallVector<Operation *> &opsWithBufferReuse,
                              DenseSet<Operation *> &excludeReuse) {
  for (auto *dstOp : opsWithChannels) {
    if (!channelWithReuse(dstOp, opsWithBufferReuse))
      excludeReuse.insert(dstOp);
  }
}

// Check to see if there is no outer loop that is enclosed under ifOp.
bool immediateEnclosing(scf::IfOp ifOp, Operation *subOp) {
  auto pOp = subOp->getParentOfType<scf::ForOp>();
  if (!pOp)
    return true;
  return !enclosing(ifOp, pOp.getOperation());
}

// Return true if the IfOp contains a ForOp that is in opsWithBufferReuse.
// We want to support reuse between channels in a loop and channels in a IfOp.
bool needAccumulatedLoopCntForReuse(
    scf::IfOp ifOp, SmallVector<Operation *> &opsWithBufferReuse) {
  bool needAccum = false;
  ifOp.walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    for (auto tOp : opsWithBufferReuse) {
      if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
        // For the case of ifOp contains forOp, which contains subOp, no need to
        // generate accumLoopCount for ifOp.
        if (subOp == tOp && immediateEnclosing(ifOp, tOp)) {
          needAccum = true;
          break;
        }
      } else {
        if (subOp == tOp) {
          needAccum = true;
          break;
        }
      }
    }
  });
  return needAccum;
}

// Return number of AccumCnts for the given ctrlOp. Add a single
// AccumCnt for all channels under opsWithBufferReuse and it will be the
// last AccumCnt.
unsigned getAccumCnts(Operation *ctrlOp,
                      const DenseSet<Operation *> &opsWithChannels,
                      SmallVector<Operation *> &opsWithBufferReuse) {
  unsigned cnt = 0;
  // Add a single count for all channels under opsWithBufferReuse.
  DenseSet<Operation *> excludeReuse;
  excludeChannelsWithReuse(opsWithChannels, opsWithBufferReuse, excludeReuse);
  LDBG("getAccumCnts: " << ctrlOp);
  for (auto *op : opsWithBufferReuse) {
    LDBG("-- getAccumCnts: " << ctrlOp << " opsWithBufferReuse " << op);
  }
  for (auto *op : excludeReuse) {
    LDBG("-- getAccumCnts: " << ctrlOp << " excludeReuse " << op);
    if (ctrlOp == op) {
      ++cnt;
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(ctrlOp))
      if (enclosing(forOp, op))
        ++cnt;
    if (auto ifOp = dyn_cast<scf::IfOp>(ctrlOp))
      if (enclosing(ifOp, op))
        ++cnt;
  }
  if (auto tIf = dyn_cast<scf::IfOp>(ctrlOp))
    if (needAccumulatedLoopCntForReuse(tIf, opsWithBufferReuse))
      ++cnt;
  if (dyn_cast<scf::ForOp>(ctrlOp))
    if (opsWithBufferReuse.size() > 1)
      ++cnt;
  return cnt;
}

// Assume parentForOp has accumCnt for the specified ctrlOp. For channels with
// reuse, use getReuseAccumCntArg.
unsigned getAccumArgIdx(scf::ForOp parentForOp, Operation *ctrlOp,
                        const DenseSet<Operation *> &opsWithChannels,
                        SmallVector<Operation *> &opsWithBufferReuse) {
  DenseSet<Operation *> excludeReuse;
  excludeChannelsWithReuse(opsWithChannels, opsWithBufferReuse, excludeReuse);
  // Walk parentForOp in preorder.
  unsigned preOrderId = 0, ctrlId = 0;
  bool found = false;
  parentForOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    // This will walk parentForOp.
    if (subOp == ctrlOp) {
      ctrlId = preOrderId;
      found = true;
    }
    for (auto *op : excludeReuse) {
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
// Right now, we only support a limited form of buffer reuse. We only allow
// reuses among a list of parallel control ops. And we will add a single
// AccumCnt as the last argument.
Value getAccumCount(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                    const DenseSet<Operation *> &opsWithChannels,
                    SmallVector<Operation *> &opsWithBufferReuse) {
  auto parentForOp = op->getParentOfType<scf::ForOp>();
  auto *pOp = op->getParentOp();
  // Get parentForOp.arg[pOp]
  unsigned accumArgId;
  unsigned tSize = parentForOp.getBody()->getArguments().size();
  unsigned parentTCnts =
      getAccumCnts(parentForOp, opsWithChannels, opsWithBufferReuse);
  Value accumCnt;
  bool partOfReuse = false;
  if (opsWithBufferReuse.size() > 1) {
    partOfReuse = channelWithReuse(op, opsWithBufferReuse);
  }
  if (opsWithBufferReuse.size() > 1 && partOfReuse) {
    // Check to see if the op is inside opsWithBufferReuse.
    accumCnt = parentForOp.getBody()->getArguments().back();
    accumArgId = parentTCnts - 1;
  } else {
    accumArgId =
        getAccumArgIdx(parentForOp, pOp, opsWithChannels, opsWithBufferReuse);
    accumCnt =
        parentForOp.getBody()->getArgument(tSize - parentTCnts + accumArgId);
  }

  LDBG("getAccumCount: parentForOp " << parentForOp.getOperation() << " pOp "
                                     << pOp << " " << tSize << " "
                                     << parentTCnts << " " << accumArgId);
  return accumCnt;
}

void getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                          unsigned numBuffers,
                          const DenseSet<Operation *> &opsWithChannels,
                          Value &bufferIdx, Value &phase,
                          SmallVector<Operation *> &opsWithBufferReuse) {
  Value accumCnt =
      getAccumCount(builder, op, opsWithChannels, opsWithBufferReuse);
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
