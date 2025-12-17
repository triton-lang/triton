#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-wgmma-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// Returns whether the dot is such that:
// 1. The LHS comes from registers and
// 1.1  The LHS is defined inside the loop
// 1.2. The LHS does not come from another dot
// For these dots, we assume that we cannot rewrite their
// operands until the previous dot has finished
static bool rsDotNeedsWait(Operation *dot, scf::ForOp forOp) {
  auto dotOp = dyn_cast<ttng::WarpGroupDotOp>(dot);
  if (!dotOp)
    return false;
  auto a = dotOp.getA();
  if (!isa<RankedTensorType>(a.getType())) {
    return false;
  }
  if (forOp.isDefinedOutsideOfLoop(a)) {
    return false;
  }
  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(a.getDefiningOp())) {
    return !isa<ttg::NvidiaMmaEncodingAttr>(
        cvt.getSrc().getType().getEncoding());
  }
  return true;
}

/// Find the minimum number of async_commit_group ops between the wait
/// and the associated async_commit_group. This can be safely used as the wait
/// number.
static int minNumInterleavedCommitOps(Operation *waitOp) {
  auto countCommitsBetween = [](Operation *op1, Operation *op2) {
    int count = 0;
    for (auto op = op1; op != op2; op = op->getNextNode()) {
      if (isa<ttg::AsyncCommitGroupOp>(op))
        count++;
      // Intentionally skip block ops' children. This will give us
      // convervatively low number of insert ops.
    }
    return count;
  };

  int minCommitNumber = INT_MAX;

  // DFS the def chain of the extract op to find the insert op. On each path
  // we calculate the number of async_commit. Then we select the minimum number
  // of async_commit ops among all the paths.
  std::function<int(Value, Operation *, int)> minOverHistories =
      [&](Value val, Operation *sinkOp, int thisHistorySum) -> int {
    if (Operation *defOp = val.getDefiningOp()) {
      thisHistorySum += countCommitsBetween(defOp->getNextNode(), sinkOp);
      minCommitNumber = std::min(minCommitNumber, thisHistorySum);
      return minCommitNumber;
    }
    if (auto arg = mlir::dyn_cast<BlockArgument>(val)) {
      Block *block = arg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

      // Failed to track, return 0 conservatively.
      if (!forOp)
        return 0;

      Operation *firstForInst = &*forOp.getBody()->begin();
      int insertsBetween = countCommitsBetween(firstForInst, sinkOp);
      thisHistorySum += insertsBetween;
      if (thisHistorySum >= minCommitNumber)
        return minCommitNumber;

      // get the value assigned to the argument coming from outside the loop
      Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
      int min1 = minOverHistories(incomingVal, forOp, thisHistorySum);

      // get the value assigned to the argument coming from the previous
      // iteration
      Operation *yieldOp = block->getTerminator();
      Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
      int min2 = minOverHistories(prevVal, yieldOp, thisHistorySum);
      return std::min(std::min(min1, min2), minCommitNumber);
    }
    // Failed to track, return 0 conservatively.
    return 0;
  };

  if (waitOp->getNumOperands() != 1)
    return 0;
  Value val = waitOp->getOperand(0);
  // If the value resides in a region other than the region of the wait op, then
  // the wait op must be in some nested region. Measure the number of commits
  // between the definition value and the parent op.
  // TODO: We could measure commits in nested regions along the path if
  // necessary.
  while (waitOp->getParentRegion() != val.getParentRegion())
    waitOp = waitOp->getParentOp();
  int minCommits = minOverHistories(val, waitOp, 0);
  return minCommits;
}

/// Update wait op number by analyzing the number of async_commit_group ops
/// along all paths.
void mlir::triton::updateWaits(ModuleOp module) {
  llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
  module.walk([&](ttg::AsyncWaitOp waitOp) {
    int minNumCommits = minNumInterleavedCommitOps(waitOp);
    waitOp.setNum(minNumCommits);
    waitOps.insert(waitOp);
  });
  tt::combineRedundantWaitOps(waitOps);
}

// Add the given values as operands of the given wait, and replace all uses of
// the values with the wait.  Also adds related MemDesc's to the wait.
//
// Threading %a through the wait transforms
//
//   %a = <...>
//   (%x', %y') = ttng.async_wait %x, %y
//   %b = fn(%a)
//
// into
//
//   %a = <...>
//   (%x', %y', %a') = ttng.async_wait %x, %y, %a
//   %b = fn(%a')
//
// The wait must dominate all uses of the elements of `values`.
//
// In addition to adding each value from `values` to the wait, this function
// also adds some MemDesc's to the wait.  The idea is that if you have
//
//   %alloc = ttg.local_alloc ...
//   %a = ttng.warp_group_dot %alloc
//   %a1 = ttng.warp_group_dot_wait %a
//
// then we want the wait to depend on %alloc as well as %a.  This extends the
// live range of %alloc, so that it won't be destroyed until after the dot is
// waited on.
//
// Specifically, this function finds all warp_group_dot ops that elements of
// `values` depend on.  Then it adds the MemDesc operands of those dots to the
// wait.
static void threadValuesThroughWait(ttng::WarpGroupDotWaitOp wait,
                                    MutableArrayRef<Value> values) {
  IRRewriter builder(wait.getContext());
  builder.setInsertionPoint(wait);

  // Operands are only added to the wait through this function, so we can have
  // the invariant that the wait has no duplicates.  This makes things a bit
  // easier below.
  size_t origNumOperands = wait.getNumOperands();
  SetVector<Value> newOperands(wait.getOperands().begin(),
                               wait.getOperands().end());
  assert(newOperands.size() == origNumOperands &&
         "Wait op has duplicate operands.");

  newOperands.insert(values.begin(), values.end());

  // Find memdefs depended on by `values` through async dot ops.
  SmallVector<ttng::WarpGroupDotOp> asyncDots;
  for (Value v : values) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *op) {
      if (auto dot = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        asyncDots.push_back(dot);
        return false;
      }
      return op->getBlock() == wait->getBlock();
    };
    SetVector<Operation *> slice;
    (void)getBackwardSlice(v, &slice, options);
  }

  for (ttng::WarpGroupDotOp dot : asyncDots) {
    for (Value operand : dot.getOperands()) {
      if (isa<ttg::MemDescType>(operand.getType())) {
        newOperands.insert(operand);
      }
    }
  }

  // We can't use replaceWithNewOp because we're changing the number of return
  // values in the operation.
  auto newWait = ttng::WarpGroupDotWaitOp::create(
      builder, wait.getLoc(), llvm::to_vector(newOperands), wait.getPendings());

  auto dominatedByNewWait = [&](OpOperand &operand) {
    auto opInThisBlock =
        newWait->getBlock()->findAncestorOpInBlock(*operand.getOwner());
    return opInThisBlock && newWait->isBeforeInBlock(opInThisBlock);
  };
  for (int i = 0; i < origNumOperands; i++) {
    Value operand = wait.getResult(i);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceAllUsesWith(newWait.getResult(i));
  }
  for (int i = origNumOperands; i < newOperands.size(); i++) {
    Value operand = newWait.getOperand(i);
    if (!isa<ttg::MemDescType>(operand.getType()))
      operand.replaceUsesWithIf(newWait.getResult(i), dominatedByNewWait);
  }
  wait->erase();
}

// Split the LHS of a RSWGMMADot operation into multiple
// tensors of size MxnewK via SplitOps
SmallVector<Value> splitLhs(OpBuilder &builder,
                            TypedValue<RankedTensorType> lhs, int64_t newK) {
  auto loc = lhs.getLoc();
  auto type = lhs.getType();
  auto rank = type.getRank();
  auto shape = to_vector(type.getShape());
  auto nSplits = shape.back() / newK;
  assert(nSplits > 1);
  // Reshape K == 2x..x2xnewK
  shape.pop_back();
  for (int i = 1; i < nSplits; i *= 2) {
    shape.push_back(2);
  }
  shape.push_back(newK);
  lhs = tt::ReshapeOp::create(builder, loc, shape, lhs);
  // We want to split first the slowest running dim, then the second slowest,
  // etc.
  auto transOrder = to_vector(llvm::seq<int>(rank - 1));
  transOrder.push_back(shape.size() - 1);
  llvm::append_range(transOrder, llvm::reverse(llvm::seq(
                                     rank - 1, (int64_t)shape.size() - 1)));
  lhs = tt::TransOp::create(builder, loc, lhs, transOrder);
  // We split recursively
  SmallVector<Value> curr;
  SmallVector<Value> ret = {lhs};
  for (int i = 1; i < nSplits; i *= 2) {
    curr = ret;
    ret.clear();
    for (auto v : curr) {
      auto split = tt::SplitOp::create(builder, loc, v);
      ret.push_back(split.getResult(0));
      ret.push_back(split.getResult(1));
    }
  }

  auto mmav3Type =
      type.clone(cast<RankedTensorType>(ret.front().getType()).getShape());
  // Convert the LHS to mmav3 layout
  for (auto &v : ret) {
    v = ttg::ConvertLayoutOp::create(builder, loc, mmav3Type, v);
    // These convert_layout ops are noops by construction
    assert(isNoop(v.getDefiningOp()));
  }
  assert(ret.size() == nSplits);
  return ret;
}

// Split the RHS of a RSWGMMADot operation into multiple multiple
// tensors of size newKxN via MemDescSubslice
SmallVector<Value> splitRhs(OpBuilder &builder,
                            TypedValue<ttg::MemDescType> rhs, int64_t newK) {
  auto loc = rhs.getLoc();
  auto type = rhs.getType();
  auto rank = type.getRank();
  auto kDim = rank - 2;
  auto nSplits = type.getShape()[kDim] / newK;
  auto shape = llvm::to_vector(type.getShape());
  shape[kDim] = newK;
  SmallVector<int32_t> offsets(rank, 0);
  auto newType = ttg::MemDescType::get(
      shape, type.getElementType(), type.getEncoding(), type.getMemorySpace(),
      /*isMutable=*/false, type.getAllocShape());
  SmallVector<Value> ret;
  for (int i = 0; i < nSplits; i++) {
    offsets[kDim] = i * newK;
    Value newSmem =
        ttg::MemDescSubsliceOp::create(builder, loc, newType, rhs, offsets);
    ret.push_back(newSmem);
  }
  return ret;
}

std::vector<ttng::WarpGroupDotOp> splitRSDot(ttng::WarpGroupDotOp dotOp) {
  // Splits a wgmma(tensor, shmem) MxK, KxN -> MxN into
  // along K into multiple wgmma(tensor, shmem) Mx16, 16xN -> MxN
  // where 16 is the instruction size
  if (!isa<RankedTensorType>(dotOp.getA().getType())) {
    return {dotOp};
  }

  auto a = cast<TypedValue<RankedTensorType>>(dotOp.getA());
  auto b = cast<TypedValue<ttg::MemDescType>>(dotOp.getB());
  auto origK = a.getType().getShape().back();
  auto newK = cast<ttg::NvidiaMmaEncodingAttr>(dotOp.getType().getEncoding())
                  .getInstrShape()[2];
  auto numSplits = origK / newK;
  // Nothing to split
  if (numSplits <= 1) {
    return {dotOp};
  }

  assert(origK % newK == 0 && "origK must be divisible by newK");
  auto builder = OpBuilder(dotOp);
  auto loc = dotOp.getLoc();
  auto lhss = splitLhs(builder, a, newK);
  auto rhss = splitRhs(builder, b, newK);
  assert(lhss.size() == numSplits && "lhs must have the same number of splits");
  assert(rhss.size() == numSplits && "rhs must have the same number of splits");

  Value useC = dotOp.getUseC();
  Value C = dotOp.getC();
  auto numImpreciseAccLeft = dotOp.getMaxNumImpreciseAcc();
  std::vector<ttng::WarpGroupDotOp> dots;
  for (int i = 0; i < numSplits; i++) {
    //  2**30 is to prevent the subtile from adding
    // extra imprecise accumulator, See WGMMA.cpp
    auto take = std::min(numImpreciseAccLeft, newK);
    uint32_t numImpreciseAcc = (take == newK) ? (1u << 30) : take;
    numImpreciseAccLeft -= take;

    auto dot = ttng::WarpGroupDotOp::create(
        builder, loc, dotOp.getType(), lhss[i], rhss[i], C, useC,
        dotOp.getInputPrecision(), numImpreciseAcc, dotOp.getIsAsync());
    dots.push_back(dot);
    C = dot.getResult();
    useC = {};
  }
  dotOp.replaceAllUsesWith(dots.back().getResult());
  dotOp.erase();
  return dots;
}

// Apply splitRSDot to all dots in the input list.
llvm::MapVector<Operation *, int>
splitRSDots(const llvm::MapVector<Operation *, int> &dots) {
  llvm::MapVector<Operation *, int> ret;
  for (auto [dot, iterArgIdx] : dots) {
    auto newDots = splitRSDot(cast<ttng::WarpGroupDotOp>(dot));
    for (auto newDot : newDots) {
      ret.insert({newDot, iterArgIdx});
    }
  }
  return ret;
}

// Determines whether a given MMAv3 dot op, represented as ttng.warp_group_dot,
// needs a wait immediately after it.
//
// In PTX, MMAv3 exists only as an asynchronous op.  In Triton, we can represent
// MMAv3 ops as either ttng.warp_group_dot {isAsync=True} or ttng.warp_group_dot
// {isAsync=False}.  But even if we use ttng.warp_group_dot {isAsync=True}, the
// conservative thing is to make a dot "effectively synchronous" by inserting a
// `ttng.warp_group_dot_wait {pendings=0}` right after it.
//
// We can omit the wait and create a "properly async" dot if all of the
// following are true.
//
//  1. All operands that touch shared memory are multi-buffered, i.e. can't read
//     an incomplete value while it's being written asynchronously by a load.
//     1a. If operand A is in registers, these registers cannot be updated
//     inside
//         the loop.
//         **Exception** if the operand is produced by a preceding WGMMA,
//         then this op can be properly async. Either the f16 shortcut is
//         possible and the WGMMA's can run back-to-back (see rule 3 below), or
//         elementwise truncate is needed, in which case the preceding WGMMA is
//         not async and a WarpGroupDotWait is inserted right after, which
//         guarantees exclusive access to the operand registers.
//
//  2. If the dot is used by any op in the loop, it must be used under an `if`,
//     and will be synced with a `wait 0` at the beginning of the `if` block.
//
//  3. During iteration i, between the start of the loop up until the first
//     `ttng.warp_group_dot_wait {pendings=0}` op, the result of the dot from
//     iteration i-1 is consumed only by other MMAv3 dots as the `c` operand.
//
//     This is safe because the following pseudo-PTX is valid:
//
//        %accum = warp_group_dot %a1, %b1, %c1
//        %accum = warp_group_dot %a2, %b2, %accum
//
//     That is, the second async dot can use the result of the first one without
//     an intervening wait.  However, the only operation that can legally read
//     %accum before the wait is another warp_group_dot, and this only works for
//     the `c` operand, not `a` or `b`.  See
//     https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
//     (ttng::WarpGroupDotOp corresponds to wgmma.fence followed by one or more
//     wgmma.async ops, so our understanding is that the two
//     ttng::WarpGroupDotOps don't have to correspond to wgmma.async ops with
//     the same shapes as specified in the docs, because there's an intervening
//     fence.)
//
// If the op can be properly async, this function returns the index of the dot
// in the loop's iter_args.  (Rule (2) above ensures this is well-defined.)
//
static std::optional<int> dotCanBeProperlyAsync(ttng::WarpGroupDotOp dotOp,
                                                scf::ForOp forOp) {
  LDBG("Considering whether to make MMAv3 dot properly async: " << dotOp);

  auto checkOperand = [&](Value operand) {
    // We can always make RSGEMM async s long as the RHS can be multi-buffered
    if (isa<RankedTensorType>(operand.getType())) {
      return true;
    }
    // If it's a shmem operand, it must either be defined outside the loop, or
    // come from an MemDescIndex op.  Only ConvertLayout and view ops are
    // allowed in between.
    Value transitiveOperand = operand;
    while (isa_and_nonnull<ttg::ConvertLayoutOp, ttg::MemDescTransOp,
                           ttg::MemDescReshapeOp, ttg::MemDescSubsliceOp>(
               transitiveOperand.getDefiningOp()) ||
           isa<BlockArgument>(transitiveOperand)) {
      auto blockArg = dyn_cast<BlockArgument>(transitiveOperand);
      if (blockArg && blockArg.getOwner() == forOp.getBody()) {
        transitiveOperand =
            cast<scf::YieldOp>(blockArg.getOwner()->getTerminator())
                .getOperand(blockArg.getArgNumber() - 1);
      } else if (Operation *def = transitiveOperand.getDefiningOp()) {
        transitiveOperand = def->getOperand(0);
      }
    }
    return forOp.isDefinedOutsideOfLoop(transitiveOperand) ||
           transitiveOperand.getDefiningOp<ttg::MemDescIndexOp>();
  };

  // Rule 1: All shmem operands are multi-buffered.
  // We don't have to call checkOperand on getC() because it's always in
  // registers, never in shmem.
  assert(isa<ttg::NvidiaMmaEncodingAttr>(dotOp.getC().getType().getEncoding()));
  if (!checkOperand(dotOp.getA()) || !checkOperand(dotOp.getB())) {
    LDBG("Can't make dot async because shmem operands aren't multi-buffered");
    return std::nullopt;
  }

  // Rule 2: The dot cannot be unconditionally used by any op in the loop.
  // Uses under `if` are allowed, as can be explicitly synced with a `wait 0`.
  int iterArgIdx = -1;
  Value iterArg = nullptr;
  SmallVector<std::pair<Operation *, int>> queue;
  for (auto &use : dotOp->getUses()) {
    queue.push_back({use.getOwner(), use.getOperandNumber()});
  }
  while (!queue.empty()) {
    auto [user, argIdx] = queue.pop_back_val();
    if (user->getParentOp() == forOp) {
      // We support noops in between the dot and the yield
      if (isNoop(user)) {
        for (auto &use : user->getResult(0).getUses()) {
          queue.push_back({use.getOwner(), use.getOperandNumber()});
        }
        continue;
      }
      if (isa<scf::YieldOp>(user)) {
        if (iterArg) {
          // The dot is used by the loop's yield, but we can't have any other
          // uses.
          LDBG("Can't make dot async because dot is used by multiple ops in "
               "the loop.");
          return std::nullopt;
        }
        iterArgIdx = argIdx;
        iterArg = forOp.getRegionIterArg(argIdx);
        continue;
      }
      LDBG("Can't make dot async because dot is unconditionally used in the "
           "loop.");
      return std::nullopt;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp())) {
      if (isa<scf::YieldOp>(user)) {
        // The result is returned by the if, follow it further.
        auto uses = ifOp.getResult(argIdx).getUses();
        for (auto &use : uses) {
          queue.push_back({use.getOwner(), use.getOperandNumber()});
        }
      }
    } else {
      return std::nullopt;
    }
  }
  // Rule 2.1: We don't make the dot async if the accumulator is not fp32.
  if (!dotOp.getC().getType().getElementType().isF32()) {
    LDBG("Can't make dot async because the accumulator is not fp32");
    return std::nullopt;
  }

  // Rule 3a: Check that every use of the dot’s result (iterArg) eventually
  // reaches a WarpGroupDotOp (with use index 2), possibly after passing through
  // a chain of noops
  std::function<bool(OpOperand &)> isTransitivelyWarpGroupDot =
      [&](OpOperand &use) -> bool {
    Operation *user = use.getOwner();
    if (isa<ttng::WarpGroupDotOp>(user))
      return use.getOperandNumber() == 2;
    if (isNoop(user))
      return llvm::all_of(user->getResult(0).getUses(),
                          isTransitivelyWarpGroupDot);
    return false;
  };

  if (llvm::all_of(iterArg.getUses(), isTransitivelyWarpGroupDot))
    return iterArgIdx;

  // Rule 3b: Are all users of the dot's result from iteration i-1 after the
  // first `warp_group_dot_wait {pendings=0}` op?  If so, the dot can be
  // properly async, but we have to thread its result from iteration i-1 through
  // the wait.
  auto waitOps = forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>();
  auto firstWaitOpIter = llvm::find_if(
      waitOps, [&](auto waitOp) { return waitOp.getPendings() == 0; });
  if (firstWaitOpIter != waitOps.end() &&
      llvm::all_of(iterArg.getUsers(), [&](Operation *user) {
        assert(forOp->isAncestor(user));
        while (user->getParentOp() != forOp) {
          user = user->getParentOp();
        }
        return (*firstWaitOpIter)->isBeforeInBlock(user);
      })) {
    LDBG("MMAv3 dot can be properly async because it follows a "
         "warp_group_dot_wait "
         "{pendings=0}.\n"
         << "  wait: " << *firstWaitOpIter << "\n"
         << "  dot: " << dotOp);
    threadValuesThroughWait(*firstWaitOpIter, {iterArg});
    return iterArgIdx;
  }

  LDBG("Can't make dot async because its result from i-1 is used by "
       "something other than another MMAv3 dot as the `c` operand.");
  return std::nullopt;
}

// If necessary, insert a dot-wait inside the loop, waiting for the results of
// the properly-async dots from iteration i-1 to complete.  (We pipeline to
// depth 2, so there are at most 2 copies of each warp_group_dot in flight at a
// time.)
//
// We can skip inserting the wait if we have a `warp_group_dot_wait
// {pendings=0}` somewhere in the loop.  To see why, consider:
//
//   warp_group_dot
//   warp_group_dot; wait 0  // synchronous dot
//   warp_group_dot
//   warp_group_dot
//
// In this example, there are three properly-async dots, so we'd normally put
// `wait 3` at the end of the loop, meaning "wait until there are 3 or fewer
// pending async dots".  But note that when this iteration of the loop
// completes, there are only *two* pending async dots from this iteration, so
// this wait would do nothing.  This is true in general, no matter where the
// `wait 0` appears.
static void insertAsyncWarpGroupDotWaitInLoop(
    scf::ForOp forOp,
    const llvm::MapVector<Operation *, int /*iterArgIdx*/> &properlyAsyncDots) {
  if (properlyAsyncDots.empty())
    return;

  if (llvm::any_of(forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>(),
                   [](auto wait) { return wait.getPendings() == 0; })) {
    return;
  }

  // Insert waits before the users of the properly async dots other than loop
  // yield.
  for (auto asyncDot : llvm::make_first_range(properlyAsyncDots)) {
    DenseMap<Block *, SmallVector<OpOperand *>> blockToUses;
    for (auto &use : asyncDot->getUses()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
        continue;
      }

      auto block = use.getOwner()->getBlock();
      blockToUses[block].push_back(&use);
    }

    for (auto [block, uses] : blockToUses) {
      // Insert a wait before the first use in the block
      std::sort(uses.begin(), uses.end(), [](OpOperand *lhs, OpOperand *rhs) {
        Operation *lhsOp = lhs->getOwner();
        Operation *rhsOp = rhs->getOwner();
        return lhsOp->isBeforeInBlock(rhsOp);
      });

      // If a wgmma uses the same accumulator registers, it will be implicitly
      // pipelined by the hardware and doesn't need a wait.
      auto firstUse =
          std::find_if_not(uses.begin(), uses.end(), [](OpOperand *operand) {
            return (isa<ttng::WarpGroupDotOp>(operand->getOwner()) &&
                    operand->getOperandNumber() == 2);
          });
      if (firstUse == uses.end()) {
        continue;
      }

      OpBuilder builder((*firstUse)->getOwner());
      auto newWait = ttng::WarpGroupDotWaitOp::create(
          builder, asyncDot->getLoc(), ArrayRef<Value>{}, 0);

      SmallVector<Value> users;
      for (; firstUse != uses.end(); ++firstUse) {
        users.push_back((*firstUse)->get());
      }
      threadValuesThroughWait(newWait, users);
    }
  }

  for (auto asyncDot : llvm::make_first_range(properlyAsyncDots)) {
    // If the dot takes the LHS on registers i, we add a wait for the number
    // of properly async dots in the loop minus one.
    // This makes sure that the dot will wait until itself from the previous
    // iteration has completed, as to avoid rewriting the registers.
    if (!rsDotNeedsWait(asyncDot, forOp))
      continue;

    OpBuilder builder(asyncDot);
    builder.setInsertionPointAfter(asyncDot);
    auto newWait = ttng::WarpGroupDotWaitOp::create(
        builder, asyncDot->getLoc(), ArrayRef<Value>{},
        properlyAsyncDots.size() - 1);
    SmallVector<Value> waitOperands = {asyncDot->getResult(0)};
    threadValuesThroughWait(newWait, waitOperands);
  }

  // Add the wait right after the last properly-async dot.  This only needs to
  // wait for all properly-async dots from the i-1'th iteration to complete, IOW
  // we wait until there are most `asyncDots.size()` dots in flight.
  //
  // (You might want to put the wait at the end of the loop instead of right
  // after the last dot, but there could be a load into shmem between the last
  // async dot and the end of the loop, and that could clobber memory being used
  // by a dot.)
  IRRewriter builder(forOp.getContext());
  auto lastAsyncDot = properlyAsyncDots.back().first;
  // If the last dot is an RS dot, we don't need to insert a wait
  // as we have already inserted a wait(properlyAsyncDots.size() - 1)
  if (rsDotNeedsWait(lastAsyncDot, forOp)) {
    return;
  }
  builder.setInsertionPointAfter(lastAsyncDot);
  auto wait = ttng::WarpGroupDotWaitOp::create(builder, lastAsyncDot->getLoc(),
                                               /*inputs=*/ArrayRef<Value>{},
                                               properlyAsyncDots.size());

  // Thread the results of the async dots through the wait.
  SmallVector<Value> addlWaitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    addlWaitOperands.push_back(asyncDot->getResult(0));
  }
  threadValuesThroughWait(wait, addlWaitOperands);
}

// Convert MMAv3 ttng::WarpGroupDotOps {isAsync = False} (i.e. Hopper wgmma)
// into ttng::WarpGroupDotOps {isAsync = True} and insert
// ttng::WarpGroupDotWaitOps as necessary.
//
// We assume we have space for each dot to be pipelined to depth 2, i.e. each
// dot op in the loop can have at most 2 warp_group_dot ops in flight at once.
// (Each warp_group_dot op usually corresponds to a series of wgmma.async ops.)
void triton::asyncLaunchDots(scf::ForOp forOp) {
  LDBG("Original loop:\n" << *forOp);

  // First, change every MMAv3 ttng.warp_group_dot {isAsync=false}
  // into ttng.warp_group_dot {isAsync=true}.
  // The rest of this function is concerned with inserting
  // ttng.warp_group_dot_wait ops in the appropriate places.
  //
  // We call those dots that don't need to be followed immediately by a `wait 0`
  // "properly async", or sometimes just "async".
  //
  // For each dot, determine whether it can be properly async, or if it needs a
  // sync immediately after.  If it can be properly async, we know its only use
  // is in the loop's `yield` statement; asyncDots maps the op to its index in
  // the yield op.
  IRRewriter builder(forOp.getContext());
  llvm::MapVector<Operation *, int /*iterArgIdx*/> properlyAsyncDots;
  for (auto WarpGroupDotOp : forOp.getBody()->getOps<ttng::WarpGroupDotOp>()) {
    WarpGroupDotOp.setIsAsync(true);
    if (auto iterArgIdx = dotCanBeProperlyAsync(WarpGroupDotOp, forOp)) {
      properlyAsyncDots[WarpGroupDotOp] = *iterArgIdx;
    } else {
      builder.setInsertionPointAfter(WarpGroupDotOp);
      auto wait = ttng::WarpGroupDotWaitOp::create(
          builder, WarpGroupDotOp.getLoc(), ArrayRef<Value>{},
          /*pendings=*/0);
      SmallVector<Value> waitOperands = {WarpGroupDotOp.getResult()};
      threadValuesThroughWait(wait, waitOperands);
    }
  }

  if (properlyAsyncDots.empty()) {
    LDBG("No properly async dots.");
    return;
  }

  // Split RS dots into dots with K = 16 (the instruction size of MMAv3)
  // If we split them in nSplit dots, we will be able to keep nSplit-1 dots
  // in flight at a time.
  // We just do it if there is no wait 0 in the loop, as otherwise the split
  // just creates unnecessary commits and arrives.
  if (llvm::all_of(forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>(),
                   [](auto wait) { return wait.getPendings() != 0; })) {
    properlyAsyncDots = splitRSDots(properlyAsyncDots);
  }

  // Next, insert a wait inside the loop.  We pipeline to depth 2, so the third
  // iteration's set of asynchronous dots (and their corresponding async copies
  // from global to shmem) can't start until the first iteration's set has
  // completed.
  insertAsyncWarpGroupDotWaitInLoop(forOp, properlyAsyncDots);

  // Finally, insert a wait after the loop, waiting for dots from the final
  // iteration of the loop.
  SmallVector<Value> waitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    waitOperands.push_back(forOp.getResult(iterArgIdx));
  }

  // Insert a wait(0) before the first use outside the loop
  auto curBlock = forOp->getBlock();
  Operation *firstUse = nullptr;
  for (auto accVal : waitOperands) {
    for (auto user : accVal.getUsers()) {
      auto target = curBlock->findAncestorOpInBlock(*user);
      if (!target)
        continue;
      if (!firstUse || target->isBeforeInBlock(firstUse))
        firstUse = target;
    }
  }

  if (firstUse) {
    builder.setInsertionPoint(firstUse);
  } else {
    builder.setInsertionPoint(curBlock->getTerminator());
  }
  auto WarpGroupDotWaitAfterLoop = ttng::WarpGroupDotWaitOp::create(
      builder, forOp.getLoc(), ArrayRef<Value>{}, 0);
  threadValuesThroughWait(WarpGroupDotWaitAfterLoop, waitOperands);
}
