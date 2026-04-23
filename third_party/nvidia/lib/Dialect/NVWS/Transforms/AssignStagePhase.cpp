/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "Utilities.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

#define DEBUG_TYPE "nvws-assign-stage-phase"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSASSIGNSTAGEPHASE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
namespace {

struct AssignStagePhase {
  enum class AccessKind { Store, Read };
  struct PhaseKey {
    int partitionId;
    unsigned semaphoreOrder;
    Value semaphore;

    bool operator==(const PhaseKey &other) const {
      return partitionId == other.partitionId && semaphore == other.semaphore;
    }

    bool operator<(const PhaseKey &other) const {
      if (partitionId != other.partitionId)
        return partitionId < other.partitionId;
      return semaphoreOrder < other.semaphoreOrder;
    }
  };
  using OrderedPhaseKeys = std::set<PhaseKey>;

  struct State {
    Value stage;                      // shared stage index (per buffer group)
    std::map<PhaseKey, Value> phases; // phase values by (pid, sema)
    Value token;                      // token used for stage propagation
  };

  SetVector<Value> groupSemaphores;
  bool useSinglePhaseForGroup = false;
  SetVector<int> allGroupPartitionIds;  // all partition IDs across all acquires
  DenseMap<Value, Value> initialPhases; // initial phase by semaphore
  DenseMap<std::pair<Operation *, Value>, int> tokToStagePosMap;

  int getDepth() const {
    assert(!groupSemaphores.empty());
    return cast<SemaphoreType>(groupSemaphores.front().getType())
        .getNumStages();
  }

  // --- Single-phase eligibility analysis ------------------------------------

  // Recursively walks the warp-specialized loop and simulates one iteration of
  // the release/acquire ring.
  //
  // Proof idea:
  //   Single-phase is safe only if, within one logical loop iteration, we
  //   never revisit the same semaphore in the same partition at the same
  //   virtual stage. A duplicate means two acquires would alias the same
  //   single-phase slot without an intervening stage advance, so we must fall
  //   back to multiphase.
  //
  // State we track:
  //   - `virtualStage`: logical stage number. It advances only when an acquire
  //     is immediately followed by a fresh write, because that is the case
  //     where the ring moves to the next stage.
  //   - `seen`: keys `(semaphore, partition, virtualStage)` already visited in
  //     this simulated iteration.
  bool walkBlockForEligibility(Block *block, int &virtualStage,
                               DenseSet<std::tuple<Value, int, int>> &seen) {
    for (auto &op : *block) {
      if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(&op)) {
        if (!groupSemaphores.contains(acquireOp.getSemaphore()))
          continue;
        if (isFirstUseFreshWriteAfterAcquire(acquireOp))
          virtualStage++;
        // Two acquires of the same semaphore at same vs but different
        // partitions are concurrent (one mbarrier wait) → not a dup.
        // Acquires inside the warp-specialized loop are expected to carry a
        // single partition id. Unannotated acquires are treated separately
        // with pid=-1.
        int pid = -1;
        if (hasPartition(&op)) {
          auto partitionIds = getPartitionIds(&op);
          assert(partitionIds.size() == 1 &&
                 "expected acquire in ttg.ws to have exactly one partition");
          pid = partitionIds.front();
        }
        if (!seen.insert({acquireOp.getSemaphore(), pid, virtualStage}).second)
          return false; // duplicate → multiphase
      } else if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        if (!walkBlockForEligibility(forOp.getBody(), virtualStage, seen))
          return false;
      } else if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        // Check both branches. An acquire in either branch counts.
        int stageBeforeIf = virtualStage;
        auto seenBeforeIf = seen;
        if (!walkBlockForEligibility(ifOp.thenBlock(), virtualStage, seen))
          return false;
        if (ifOp.elseBlock()) {
          int stageAfterThen = virtualStage;
          auto seenAfterThen = seen;
          virtualStage = stageBeforeIf;
          seen = seenBeforeIf;
          if (!walkBlockForEligibility(ifOp.elseBlock(), virtualStage, seen))
            return false;
          // Merge: conservative (max) virtual_stage, union of seen.
          virtualStage = std::max(stageAfterThen, virtualStage);
          seen.insert(seenAfterThen.begin(), seenAfterThen.end());
        }
      }
    }
    return true;
  }

  // Returns true if the whole semaphore group can use single-phase.
  //
  // We prove this by finding the warp-specialized loop that drives the group
  // and running `walkBlockForEligibility` on its body. That walk tracks the
  // logical stage progression and rejects any path that revisits the same
  // `(semaphore, partition, virtualStage)` triple.
  //
  // This decision is group-wide, not per semaphore: all semaphores in the
  // group participate in the same release/acquire ring, so they must all use
  // the same phase scheme.
  bool computeSinglePhaseEligibility() {
    // depth==1 → always single-phase (one stage, nothing to cycle).
    if (getDepth() == 1)
      return true;

    // Find the first group acquire inside a warp-specialized loop.
    scf::ForOp wsLoop;
    for (Value sema : groupSemaphores) {
      for (Operation *user : sema.getDefiningOp()->getUsers()) {
        auto acquireOp = dyn_cast<SemaphoreAcquireOp>(user);
        if (!acquireOp)
          continue;
        auto forOp = acquireOp->getParentOfType<scf::ForOp>();
        if (!forOp)
          continue;
        wsLoop = getOuterWSLoop(forOp);
        if (wsLoop)
          break;
      }
      if (wsLoop)
        break;
    }

    // No warp-specialized loop → conservative, use multiphase.
    if (!wsLoop)
      return false;

    // Walk loop body recursively, tracking virtual_stage.
    // Key: (semaphore, partition_id, virtual_stage).
    DenseSet<std::tuple<Value, int, int>> seen;
    int virtualStage = 0;
    if (!walkBlockForEligibility(wsLoop.getBody(), virtualStage, seen))
      return false;

    // Must have at least one advance per iteration.
    if (virtualStage == 0)
      return false;

    return true;
  }

  AssignStagePhase(ArrayRef<SemaphoreCreateOp> semaOps) {
    std::set<int> sortedPartitionIds;
    for (auto semaOp : semaOps) {
      groupSemaphores.insert(semaOp.getResult());
      for (auto user : semaOp->getUsers()) {
        if (isa<SemaphoreAcquireOp>(user) && hasPartition(user)) {
          auto ids = getPartitionIds(user);
          sortedPartitionIds.insert(ids.begin(), ids.end());
        }
      }
    }
    allGroupPartitionIds =
        SetVector<int>(sortedPartitionIds.begin(), sortedPartitionIds.end());
    if (allGroupPartitionIds.empty())
      allGroupPartitionIds.insert(0);
  }

  unsigned getSemaphoreOrder(Value semaphore) const {
    auto it = llvm::find(groupSemaphores, semaphore);
    assert(it != groupSemaphores.end());
    return std::distance(groupSemaphores.begin(), it);
  }

  PhaseKey getPhaseKey(int partitionId, Value semaphore) const {
    return PhaseKey{partitionId, getSemaphoreOrder(semaphore), semaphore};
  }

  // --- useD analysis --------------------------------------------------------

  Value remapUseDThroughIf(Value useD, scf::IfOp ifOp, bool takeThen) const {
    auto result = dyn_cast<OpResult>(useD);
    if (!result || result.getOwner() != ifOp)
      return useD;
    unsigned idx = result.getResultNumber();
    return takeThen ? ifOp.thenYield()->getOperand(idx)
                    : ifOp.elseYield()->getOperand(idx);
  }

  std::optional<bool> tryResolveUseD(Value useD,
                                     const DenseMap<Value, bool> &facts,
                                     DenseSet<Value> &visiting) const {
    if (!useD || !visiting.insert(useD).second)
      return {};
    auto guard = llvm::scope_exit([&] { visiting.erase(useD); });

    if (auto it = facts.find(useD); it != facts.end())
      return it->second;

    APInt constant;
    if (matchPattern(useD, m_ConstantInt(&constant)))
      return !constant.isZero();

    if (auto ifOp = useD.getDefiningOp<scf::IfOp>()) {
      if (auto cond = tryResolveUseD(ifOp.getCondition(), facts, visiting)) {
        useD = remapUseDThroughIf(useD, ifOp, *cond);
        return tryResolveUseD(useD, facts, visiting);
      }
    } else if (auto xoriOp = useD.getDefiningOp<arith::XOrIOp>()) {
      auto lhs = tryResolveUseD(xoriOp.getLhs(), facts, visiting);
      auto rhs = tryResolveUseD(xoriOp.getRhs(), facts, visiting);
      if (lhs && rhs)
        return *lhs != *rhs;
    }

    return {};
  }

  bool tokenCanReachAcquire(Value token, Value acquireToken,
                            DenseSet<Value> &visiting) const {
    if (!token || !visiting.insert(token).second)
      return false;
    auto guard = llvm::scope_exit([&] { visiting.erase(token); });

    if (token == acquireToken) {
      return true;
    } else if (auto blockArg = dyn_cast<BlockArgument>(token)) {
      auto forOp = cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      int idx = static_cast<int>(blockArg.getArgNumber()) - 1;
      Value initToken = forOp.getInitArgs()[idx];
      Value yieldedToken = forOp.getYieldedValues()[idx];
      return tokenCanReachAcquire(initToken, acquireToken, visiting) ||
             tokenCanReachAcquire(yieldedToken, acquireToken, visiting);
    } else {
      auto result = cast<OpResult>(token);
      if (auto forOp = dyn_cast<scf::ForOp>(result.getOwner())) {
        Value yieldedToken = forOp.getYieldedValues()[result.getResultNumber()];
        return tokenCanReachAcquire(yieldedToken, acquireToken, visiting);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(result.getOwner())) {
        unsigned idx = result.getResultNumber();
        Value thenToken = ifOp.thenYield()->getOperand(idx);
        Value elseToken = ifOp.elseYield()->getOperand(idx);
        return tokenCanReachAcquire(thenToken, acquireToken, visiting) ||
               tokenCanReachAcquire(elseToken, acquireToken, visiting);
      }
    }

    return false;
  }

  bool proveUseDIsFalse(Value useD, Value token, SemaphoreAcquireOp acquireOp,
                        DenseMap<Value, bool> &facts,
                        DenseSet<std::pair<Value, Value>> &visiting) const {
    // Backtrack the token to the specific acquire being classified. Each token
    // step chooses a unique if/for predecessor, and `useD` is remapped through
    // that same edge before the next recursive step.
    // If the token disappeared or we revisited the same (useD, token) state,
    // this proof path cannot make progress; bail out conservatively.
    if (!token || !visiting.insert({useD, token}).second)
      return false;
    auto guard = llvm::scope_exit([&] { visiting.erase({useD, token}); });

    // Once token backtracking reaches the acquire we are proving against, the
    // remaining work is to resolve the current `useD` on that acquire path.
    if (token == acquireOp.getToken()) {
      // `useD` may already be a constant or reducible from facts recorded while
      // choosing earlier if-branches.
      if (DenseSet<Value> useDVisiting;
          auto boolValue = tryResolveUseD(useD, facts, useDVisiting)) {
        return !*boolValue;
        // If `useD` is still loop-carried at the acquire, remap it to the loop
        // init edge, because the acquire-selected path corresponds to the first
        // iteration value of that loop-carried boolean.
      } else if (auto blockArg = dyn_cast<BlockArgument>(useD)) {
        auto forOp = cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
        useD = forOp.getInitArgs()[blockArg.getArgNumber() - 1];
        return proveUseDIsFalse(useD, token, acquireOp, facts, visiting);
        // Same idea for a `scf.for` result: once token is at the acquire, the
        // relevant `useD` value is the init operand feeding that loop result.
      } else if (auto result = dyn_cast<OpResult>(useD)) {
        if (auto forOp = dyn_cast<scf::ForOp>(result.getOwner())) {
          useD = forOp.getInitArgs()[result.getResultNumber()];
          return proveUseDIsFalse(useD, token, acquireOp, facts, visiting);
        }
      }
      // No further acquire-local simplification is available, so we cannot
      // prove that `useD` is false on this path.
      return false;
      // A loop-carried token block argument means the acquire lineage came
      // either from the loop init edge or from the previous iteration yield.
      // Pick that predecessor and remap `useD` through the same loop edge.
    } else if (auto blockArg = dyn_cast<BlockArgument>(token)) {
      auto forOp = cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      int idx = blockArg.getArgNumber() - 1;
      Value initToken = forOp.getInitArgs()[idx];
      Value yieldedToken = forOp.getYieldedValues()[idx];
      DenseSet<Value> reachesVisiting{token};
      bool reachesInit = tokenCanReachAcquire(initToken, acquireOp.getToken(),
                                              reachesVisiting);
      Value nextToken = reachesInit ? initToken : yieldedToken;
      // `useD` only follows this loop choice if it is defined by the same
      // `scf.for`. In that case, remap it through the init or yielded edge
      // chosen for the token lineage; otherwise leave it unchanged.
      idx = -1;
      if (auto blockArg = dyn_cast<BlockArgument>(useD)) {
        if (blockArg.getOwner()->getParentOp() == forOp)
          idx = blockArg.getArgNumber() - 1;
      } else if (auto result = dyn_cast<OpResult>(useD)) {
        if (result.getOwner() == forOp)
          idx = result.getResultNumber();
      }
      if (idx >= 0) {
        useD = reachesInit ? forOp.getInitArgs()[idx]
                           : forOp.getYieldedValues()[idx];
      }
      return proveUseDIsFalse(useD, nextToken, acquireOp, facts, visiting);
    } else {
      auto result = cast<OpResult>(token);
      // A `scf.for` result continues from the loop yield; follow that yielded
      // token and remap `useD` through the same yielded loop edge.
      if (auto forOp = dyn_cast<scf::ForOp>(result.getOwner())) {
        token = forOp.getYieldedValues()[result.getResultNumber()];
        // Only a `useD` value produced by this same `scf.for` should be
        // rewritten here. If it is loop-carried or a loop result of this op,
        // follow the yielded edge selected for the token; otherwise leave it.
        int idx = -1;
        if (auto blockArg = dyn_cast<BlockArgument>(useD)) {
          if (blockArg.getOwner()->getParentOp() == forOp)
            idx = blockArg.getArgNumber() - 1;
        } else if (auto result = dyn_cast<OpResult>(useD)) {
          if (result.getOwner() == forOp)
            idx = result.getResultNumber();
        }
        if (idx >= 0)
          useD = forOp.getYieldedValues()[idx];
        return proveUseDIsFalse(useD, token, acquireOp, facts, visiting);
        // An `scf.if` result continues from exactly one branch on the acquire
        // lineage. Record which branch was chosen, then remap both token and
        // `useD` through that branch before recursing.
      } else if (auto ifOp = dyn_cast<scf::IfOp>(result.getOwner())) {
        unsigned idx = result.getResultNumber();
        Value thenToken = ifOp.thenYield()->getOperand(idx);
        Value elseToken = ifOp.elseYield()->getOperand(idx);
        DenseSet<Value> reachesVisiting{result};
        bool takeThen = tokenCanReachAcquire(thenToken, acquireOp.getToken(),
                                             reachesVisiting);
        facts[ifOp.getCondition()] = takeThen;
        token = takeThen ? thenToken : elseToken;
        useD = remapUseDThroughIf(useD, ifOp, takeThen);
        return proveUseDIsFalse(useD, token, acquireOp, facts, visiting);
      }
    }

    // Any other token producer is outside the supported acquire-lineage model,
    // so the proof must stop conservatively.
    return false;
  }

  bool mmaAccIsUpdate(MMAv5OpInterface mmaOp, SemaphoreAcquireOp acquireOp,
                      Value currentToken) const {
    // Facts recorded while we backtrack the token lineage. When token
    // selection proves we came through a specific if-branch, the if condition
    // becomes known on this path and can be used to fold the corresponding
    // `useD`.
    DenseMap<Value, bool> facts;
    DenseSet<std::pair<Value, Value>> visiting;
    // Walk backward from the current buffer token to the acquire being
    // classified, and keep `useD` aligned with the same control-flow choices.
    //
    // If token backtracking chooses:
    //   - a loop init edge vs yielded edge
    //   - a then edge vs else edge
    // we remap `useD` through that exact edge and try to prove it is false.
    //
    // Returning true means: along the acquire-selected token lineage, `useD`
    // is provably false, so this MMA is a fresh overwrite rather than an
    // update of prior accumulator state.
    return !proveUseDIsFalse(mmaOp.useAccumulator(), currentToken, acquireOp,
                             facts, visiting);
  }

  // --- Op matching ----------------------------------------------------------

  SemaphoreAcquireOp getAcquireOp(Operation *op) {
    if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(op)) {
      if (groupSemaphores.contains(acquireOp.getSemaphore()))
        return acquireOp;
    }
    return {};
  }

  SemaphoreBufferOp getTrackedBufferOp(Operation *op,
                                       ArrayRef<Value> trackedTokens) const {
    if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(op))
      if (groupSemaphores.contains(bufferOp.getSemaphore()))
        if (llvm::is_contained(trackedTokens, bufferOp.getToken()))
          return bufferOp;
    return {};
  }

  SemaphoreReleaseOp getTrackedReleaseOp(Operation *op,
                                         ArrayRef<Value> trackedTokens) const {
    if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(op))
      if (groupSemaphores.contains(releaseOp.getSemaphore()))
        if (llvm::is_contained(trackedTokens, releaseOp.getToken()))
          return releaseOp;
    return {};
  }
  // --- Access classification -----------------------------------------------

  std::optional<AccessKind>
  classifyTrackedBufferUse(Operation *op,
                           const SetVector<Value> &trackedBuffers,
                           SemaphoreAcquireOp acquireOp, Value currentToken) {
    auto contains = [&](Value value) { return trackedBuffers.contains(value); };

    if (auto localLoadOp = dyn_cast<LocalLoadOp>(op);
        localLoadOp && contains(localLoadOp.getSrc())) {
      return AccessKind::Read;
    } else if (auto localStoreOp = dyn_cast<LocalStoreOp>(op);
               localStoreOp && contains(localStoreOp.getDst())) {
      return AccessKind::Store;
    } else if (auto tmemLoadOp = dyn_cast<TMEMLoadOp>(op);
               tmemLoadOp && contains(tmemLoadOp.getSrc())) {
      return AccessKind::Read;
    } else if (auto tmemStoreOp = dyn_cast<TMEMStoreOp>(op);
               tmemStoreOp && contains(tmemStoreOp.getDst())) {
      return AccessKind::Store;
    } else if (auto descLoadOp = dyn_cast<nvws::DescriptorLoadOp>(op);
               descLoadOp && contains(descLoadOp.getResult())) {
      return AccessKind::Store;
    } else if (auto descGatherOp = dyn_cast<nvws::DescriptorGatherOp>(op);
               descGatherOp && contains(descGatherOp.getResult())) {
      return AccessKind::Store;
    } else if (auto scaledMmaOp = dyn_cast<TCGen5MMAScaledOp>(op);
               scaledMmaOp && (contains(scaledMmaOp.getAScale()) ||
                               contains(scaledMmaOp.getBScale()))) {
      return AccessKind::Read;
    } else if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
      if (mmaOp && (contains(mmaOp.getA()) || contains(mmaOp.getB())))
        return AccessKind::Read;
      if (mmaOp && contains(mmaOp.getAccumulator()))
        // `mmaAccIsUpdate` answers whether the accumulator operand is
        // consumed as old state (`Read`) or treated as a fresh overwrite
        // (`Store`).
        return mmaAccIsUpdate(mmaOp, acquireOp, currentToken)
                   ? AccessKind::Read
                   : AccessKind::Store;
    } else if (llvm::any_of(op->getOperands(),
                            [&](Value operand) { return contains(operand); })) {
      // for lit test, any custom op with _load/_store
      StringRef opName = op->getName().getStringRef();
      if (opName.ends_with("_load"))
        return AccessKind::Read;
      if (opName.ends_with("_store"))
        return AccessKind::Store;
    }
    return {};
  }

  static Operation *
  findFirstTrackedEventInBlock(ArrayRef<Value> trackedValues, Block *block,
                               Operation *cursorOp = nullptr) {
    Operation *eventOp = nullptr;
    for (Value trackedValue : trackedValues) {
      for (Operation *candidate : trackedValue.getUsers()) {
        while (candidate && candidate->getBlock() != block)
          candidate = candidate->getParentOp();
        if (!candidate)
          continue;
        if (cursorOp && !cursorOp->isBeforeInBlock(candidate))
          continue;
        if (!eventOp || candidate->isBeforeInBlock(eventOp))
          eventOp = candidate;
      }
    }
    return eventOp;
  }

  std::optional<AccessKind>
  mergeContinuationAccess(std::optional<AccessKind> lhs,
                          std::optional<AccessKind> rhs) const {
    if (lhs == AccessKind::Read || rhs == AccessKind::Read)
      return AccessKind::Read;
    if (lhs == AccessKind::Store || rhs == AccessKind::Store)
      return AccessKind::Store;
    return {};
  }

  std::optional<AccessKind>
  mergeBranchAccess(std::optional<AccessKind> lhs,
                    std::optional<AccessKind> rhs) const {
    if (!lhs && !rhs)
      return {};
    if (!lhs || !rhs)
      return AccessKind::Read;
    return (*lhs == AccessKind::Store && *rhs == AccessKind::Store)
               ? std::optional<AccessKind>(AccessKind::Store)
               : std::optional<AccessKind>(AccessKind::Read);
  }

  std::optional<AccessKind> classifyFirstAccessAfterBufferOp(
      Block *block, Operation *cursorOp, SetVector<Value> &trackedBuffers,
      SemaphoreAcquireOp acquireOp, Value currentToken) {
    // Pick the earliest buffer event visible in this block after the previous
    // event we already processed. Real buffer users are either in this block
    // directly, or nested under an if/for region whose parent op is the next
    // control-flow boundary we must enter first.
    // `cursorOp` is the previous event we already handled in this block. Skip
    // it so same-block continuation resumes at the next buffer event.
    Operation *eventOp = findFirstTrackedEventInBlock(
        trackedBuffers.getArrayRef(), block, cursorOp);

    if (!eventOp)
      return {};

    if (auto access = classifyTrackedBufferUse(eventOp, trackedBuffers,
                                               acquireOp, currentToken)) {
      return access;
    } else if (auto forOp = dyn_cast<scf::ForOp>(eventOp)) {
      // The tracked buffer lineage may be accessed inside the loop body
      // and/or later after the loop in the same block. Since buffer values
      // themselves are not loop-carried, these are competing first-access
      // continuations.
      auto bodyBuffers = trackedBuffers;
      auto bodyAccess = classifyFirstAccessAfterBufferOp(
          forOp.getBody(), nullptr, bodyBuffers, acquireOp, currentToken);
      auto afterLoopAccess = classifyFirstAccessAfterBufferOp(
          block, eventOp, trackedBuffers, acquireOp, currentToken);
      return mergeContinuationAccess(bodyAccess, afterLoopAccess);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(eventOp)) {
      // First try to prove a branch-local access in each branch. Only
      // branches that still have no access after local recursion fall through
      // and keep looking after the if in the same block.
      auto thenBuffers = trackedBuffers;
      auto thenAccess = classifyFirstAccessAfterBufferOp(
          ifOp.thenBlock(), nullptr, thenBuffers, acquireOp, currentToken);

      auto elseBuffers = trackedBuffers;
      auto elseAccess = ifOp.elseBlock()
                            ? classifyFirstAccessAfterBufferOp(
                                  ifOp.elseBlock(), nullptr, elseBuffers,
                                  acquireOp, currentToken)
                            : std::nullopt;

      if (!thenAccess || !elseAccess) {
        auto afterIfAccess = classifyFirstAccessAfterBufferOp(
            block, eventOp, trackedBuffers, acquireOp, currentToken);
        if (!thenAccess)
          thenAccess = afterIfAccess;
        if (!elseAccess)
          elseAccess = afterIfAccess;
      }
      return mergeBranchAccess(thenAccess, elseAccess);
    } else if (eventOp->hasTrait<OpTrait::MemDescViewTrait>()) {
      // View ops do not access the buffer. They only extend the tracked alias
      // set, so continue searching after the view itself.
      SetVector<Value> viewBuffers;
      for (Value operand : eventOp->getOperands()) {
        if (!trackedBuffers.contains(operand))
          continue;
        for (Value result : eventOp->getResults()) {
          if (isa<MemDescType>(result.getType()))
            viewBuffers.insert(result);
        }
        break;
      }
      if (!viewBuffers.empty()) {
        trackedBuffers.insert(viewBuffers.begin(), viewBuffers.end());
        return classifyFirstAccessAfterBufferOp(block, eventOp, trackedBuffers,
                                                acquireOp, currentToken);
      }
    }

    return classifyFirstAccessAfterBufferOp(block, eventOp, trackedBuffers,
                                            acquireOp, currentToken);
  }

  // Follow the lineage of one acquire token until it reaches a
  // semaphore.buffer or release. Token uses are discovered from def-use. Real
  // token users are:
  //   - nvws.semaphore.buffer / nvws.semaphore.release
  //   - scf.for when the token is passed through iter_args
  //   - scf.yield when the token is yielded out of a region
  std::optional<AccessKind>
  classifyFirstAccessAfterAcquireOp(Value trackedToken, Block *block,
                                    SemaphoreAcquireOp acquireOp) {
    if (!block || !trackedToken)
      return {};

    // Pick the earliest token event visible in this block. A real token user
    // is either in this block directly, or nested under an if/for region
    // whose parent op is the earliest control-flow boundary we must enter
    // first.
    Operation *eventOp = findFirstTrackedEventInBlock({trackedToken}, block);
    if (!eventOp)
      return {};

    if (auto ifOp = dyn_cast<scf::IfOp>(eventOp)) {
      // The token is used in one or both branches. Follow both branch-local
      // continuations and merge the first access they prove.
      auto thenAccess = classifyFirstAccessAfterAcquireOp(
          trackedToken, ifOp.thenBlock(), acquireOp);
      auto elseAccess = ifOp.elseBlock()
                            ? classifyFirstAccessAfterAcquireOp(
                                  trackedToken, ifOp.elseBlock(), acquireOp)
                            : std::nullopt;
      return mergeBranchAccess(thenAccess, elseAccess);
    } else if (auto forOp = dyn_cast<scf::ForOp>(eventOp)) {
      // The token is loop-carried. Remap it to the body iter arg and let the
      // body recursion handle either a direct access or a yielded
      // continuation.
      Value bodyToken = trackedToken;
      if (auto pos = findValuePosInRange(forOp.getInitArgs(), trackedToken))
        bodyToken = forOp.getRegionIterArgs()[*pos];
      return classifyFirstAccessAfterAcquireOp(bodyToken, forOp.getBody(),
                                               acquireOp);
    } else if (auto bufferOp = getTrackedBufferOp(eventOp, {trackedToken})) {
      std::optional<AccessKind> merged = AccessKind::Store;
      for (Value result : bufferOp->getResults()) {
        SetVector<Value> trackedBuffer;
        trackedBuffer.insert(result);
        auto access = classifyFirstAccessAfterBufferOp(
            bufferOp->getBlock(), bufferOp.getOperation(), trackedBuffer,
            acquireOp, trackedToken);
        assert(access && "token-derived buffer must reach a real use");
        merged = mergeBranchAccess(merged, access);
      }
      return merged;
    } else if (getTrackedReleaseOp(eventOp, {trackedToken})) {
      return {};
    } else {
      assert(isa<scf::YieldOp>(eventOp));

      // find the token result from the parent op
      Value parentToken;
      for (auto [opnd, ret] : llvm::zip(block->getTerminator()->getOperands(),
                                        block->getParentOp()->getResults())) {
        if (opnd == trackedToken) {
          parentToken = ret;
          break;
        }
      }

      Operation *parentOp = block->getParentOp();
      if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        // Yielded token from an if-region continues as the matching
        // if-result.
        return classifyFirstAccessAfterAcquireOp(parentToken, ifOp->getBlock(),
                                                 acquireOp);
      } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        // Yielded token from a loop body may be observed on the next
        // iteration and/or after the loop through the for-result; both
        // continuations contribute to the first-access decision.
        Value nextToken;
        for (auto [yielded, iterArg] :
             llvm::zip(cast<scf::YieldOp>(block->getTerminator()).getOperands(),
                       forOp.getRegionIterArgs())) {
          if (yielded == trackedToken) {
            nextToken = iterArg;
            break;
          }
        }
        assert(nextToken && "token not found in yield");
        auto nextAccess =
            classifyFirstAccessAfterAcquireOp(nextToken, block, acquireOp);
        auto exitAccess = classifyFirstAccessAfterAcquireOp(
            parentToken, forOp->getBlock(), acquireOp);
        return mergeContinuationAccess(nextAccess, exitAccess);
      }
    }

    return {};
  }

  bool isFirstUseFreshWriteAfterAcquire(SemaphoreAcquireOp acquireOp) {
    auto access = classifyFirstAccessAfterAcquireOp(
        acquireOp.getToken(), acquireOp->getBlock(), acquireOp);
    return access && *access == AccessKind::Store;
  }

  // --- Stage and Phase computation -----------------------------------------
  struct SemaphoreUseSummary {
    bool hasStageUse = false;
    OrderedPhaseKeys acquiredPhaseKeys;
  };

  SemaphoreUseSummary analyzeTrackedRegionUse(Block *block,
                                              scf::YieldOp yieldOp,
                                              ValueRange parentResults,
                                              ValueRange from, ValueRange to,
                                              SetVector<Value> &trackedTokens) {
    SetVector<Value> inputTokens(trackedTokens);
    // Enter the region scope by remapping tracked parent values to the names
    // visible inside the region.
    SetVector<Value> regionTrackedTokens;
    for (Value inputToken : inputTokens) {
      Value regionTrackedToken = inputToken;
      if (auto pos = findValuePosInRange(from, inputToken))
        regionTrackedToken = to[*pos];
      regionTrackedTokens.insert(regionTrackedToken);
    }
    // Analyze the region and update `regionTrackedTokens` in place to the
    // tracked tokens that remain at the end of the region.
    auto summary = analyzeSemaphoreUseInBlockImpl(block, regionTrackedTokens);
    // Translate the updated end-of-region tracked tokens back to the token
    // names visible after the region and add tracked region results from
    // values yielded out of the region.
    SetVector<Value> trackedTokensAfterRegion;
    for (Value inputToken : inputTokens) {
      Value remappedInputToken = inputToken;
      if (auto pos = findValuePosInRange(from, inputToken))
        remappedInputToken = to[*pos];
      if (regionTrackedTokens.contains(remappedInputToken))
        trackedTokensAfterRegion.insert(inputToken);
    }
    for (auto [opnd, ret] : llvm::zip(yieldOp.getOperands(), parentResults)) {
      if (regionTrackedTokens.contains(opnd))
        trackedTokensAfterRegion.insert(ret);
    }
    trackedTokens = std::move(trackedTokensAfterRegion);
    return summary;
  }

  SemaphoreUseSummary
  analyzeSemaphoreUseInBlockImpl(Block *block,
                                 SetVector<Value> &trackedTokens) {
    SemaphoreUseSummary summary;

    for (auto &op : *block) {
      if (auto acquireOp = getAcquireOp(&op)) {
        summary.hasStageUse = true;
        auto partitionIds = hasPartition(acquireOp) ? getPartitionIds(acquireOp)
                                                    : allGroupPartitionIds;
        for (int pid : partitionIds)
          summary.acquiredPhaseKeys.insert(
              getPhaseKey(pid, acquireOp.getSemaphore()));
      } else if (getTrackedBufferOp(&op, trackedTokens.getArrayRef())) {
        summary.hasStageUse = true;
      } else if (auto releaseOp =
                     getTrackedReleaseOp(&op, trackedTokens.getArrayRef())) {
        summary.hasStageUse = true;
        trackedTokens.remove(releaseOp.getToken());
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        auto nestedSummary = analyzeTrackedRegionUse(
            forOp.getBody(),
            cast<scf::YieldOp>(forOp.getBody()->getTerminator()),
            forOp.getResults(), forOp.getInitArgs(), forOp.getRegionIterArgs(),
            trackedTokens);
        summary.hasStageUse = summary.hasStageUse || nestedSummary.hasStageUse;
        summary.acquiredPhaseKeys.insert(
            nestedSummary.acquiredPhaseKeys.begin(),
            nestedSummary.acquiredPhaseKeys.end());
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        auto thenTrackedTokens = trackedTokens;
        auto thenSummary = analyzeTrackedRegionUse(
            ifOp.thenBlock(), ifOp.thenYield(), ifOp.getResults(), ValueRange(),
            ValueRange(), thenTrackedTokens);
        auto nestedSummary = thenSummary;

        if (!ifOp.elseBlock()) {
          trackedTokens = std::move(thenTrackedTokens);
        } else {
          auto elseTrackedTokens = trackedTokens;
          auto elseSummary = analyzeTrackedRegionUse(
              ifOp.elseBlock(), ifOp.elseYield(), ifOp.getResults(),
              ValueRange(), ValueRange(), elseTrackedTokens);
          trackedTokens = std::move(thenTrackedTokens);
          trackedTokens.insert(elseTrackedTokens.begin(),
                               elseTrackedTokens.end());
          nestedSummary = mergeSemaphoreUseSummaries(thenSummary, elseSummary);
        }
        summary.hasStageUse = summary.hasStageUse || nestedSummary.hasStageUse;
        summary.acquiredPhaseKeys.insert(
            nestedSummary.acquiredPhaseKeys.begin(),
            nestedSummary.acquiredPhaseKeys.end());
      }
    }

    return summary;
  }

  SemaphoreUseSummary analyzeSemaphoreUseInBlock(Block *block, Value token) {
    SetVector<Value> trackedTokens;
    if (token)
      trackedTokens.insert(token);
    return analyzeSemaphoreUseInBlockImpl(block, trackedTokens);
  }

  SemaphoreUseSummary
  mergeSemaphoreUseSummaries(const SemaphoreUseSummary &lhs,
                             const SemaphoreUseSummary &rhs) const {
    SemaphoreUseSummary summary;
    summary.hasStageUse = lhs.hasStageUse || rhs.hasStageUse;
    summary.acquiredPhaseKeys.insert(lhs.acquiredPhaseKeys.begin(),
                                     lhs.acquiredPhaseKeys.end());
    summary.acquiredPhaseKeys.insert(rhs.acquiredPhaseKeys.begin(),
                                     rhs.acquiredPhaseKeys.end());
    return summary;
  }

  Value getPhase(State &state, PhaseKey key) {
    auto [it, inserted] =
        state.phases.try_emplace(key, initialPhases.at(key.semaphore));
    return it->second;
  }

  // Infer partition IDs for a yield argumend value.
  SetVector<int> inferPartitionIds(Value arg, int fallbackPartitionId) {
    SetVector<int> argIds;
    if (auto defOp = arg.getDefiningOp()) {
      if (defOp->getNumRegions() == 0) {
        if (hasPartition(defOp))
          argIds = getPartitionIds(defOp);
      } else if (auto pos = findValuePosInRange(defOp->getResults(), arg)) {
        if (hasPartition(defOp)) {
          auto outputs = getPartitionOutputs(defOp);
          if (*pos < outputs.size())
            argIds = outputs[*pos];
        }
      }
    } else {
      for (auto user : arg.getUsers()) {
        if (isa<scf::YieldOp>(user))
          continue;
        if (hasPartition(user)) {
          auto ids = getPartitionIds(user);
          argIds.insert(ids.begin(), ids.end());
        }
      }
    }
    if (argIds.empty())
      argIds.insert(fallbackPartitionId);
    return argIds;
  }

  void assignStateInForOp(scf::ForOp forOp, State &state) {
    Value newTok;
    if (auto pos = findValuePosInRange(forOp.getInitArgs(), state.token)) {
      newTok = forOp.getRegionIterArgs()[*pos];
    }
    // find uses of arefs in forOp body
    auto summary = analyzeSemaphoreUseInBlock(forOp.getBody(), newTok);
    if (!summary.hasStageUse)
      return;

    SmallVector<Value> extraIterArgs;
    llvm::MapVector<int, Value *> tokenRefs;
    if (auto pos = findValuePosInRange(forOp.getInitArgs(), state.token)) {
      // keep reference of the token position to latest token value
      // we will need it update with the value returned from forOp
      tokenRefs[*pos] = &state.token;
      // update token value with iter argument
      state.token = forOp.getRegionIterArgs()[*pos];
    }

    extraIterArgs.push_back(state.stage);
    for (PhaseKey key : summary.acquiredPhaseKeys) {
      extraIterArgs.push_back(getPhase(state, key));
    }

    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();

    assert(hasPartition(forOp));
    auto forOpIds = getPartitionIds(forOp);
    auto forOpOutputsIds = getPartitionOutputs(forOp);
    forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

    state.stage = forOp.getRegionIterArgs()[nArgs];
    for (auto [i, key] : llvm::enumerate(summary.acquiredPhaseKeys))
      state.phases[key] = forOp.getRegionIterArgs()[nArgs + 1 + i];

    auto stateInBlock = assignStateInBlock(forOp.getBody(), state);

    // update yieldOp to return new indexes
    SmallVector<Value> extraYieldArgs;
    // associate token with stage positional argument in the iterArgs &
    // yieldOp we will need this in propagateStage function that will assign
    // stage to arefBuffer and arefExit ops
    extraYieldArgs.push_back(stateInBlock.stage);
    for (PhaseKey key : summary.acquiredPhaseKeys)
      extraYieldArgs.push_back(getPhase(stateInBlock, key));
    appendToForOpYield(forOp, extraYieldArgs);
    tokToStagePosMap[{forOp, state.token}] = nArgs;
    tokToStagePosMap[{forOp.getBody()->getTerminator(), stateInBlock.token}] =
        nArgs;

    // Annotate stage with all group partition IDs.
    forOpIds.insert(allGroupPartitionIds.begin(), allGroupPartitionIds.end());
    forOpOutputsIds.push_back(SetVector<int>(allGroupPartitionIds.begin(),
                                             allGroupPartitionIds.end()));
    // Annotate phase values with per-key partition IDs.
    for (auto [i, key] : llvm::enumerate(summary.acquiredPhaseKeys)) {
      auto argIds = inferPartitionIds(extraYieldArgs[1 + i], key.partitionId);
      forOpIds.insert(argIds.begin(), argIds.end());
      forOpOutputsIds.push_back(argIds);
    }
    setPartition(forOp, forOpIds);
    setPartitionOutputs(forOp, forOpOutputsIds);

    state.stage = forOp.getResult(nArgs);
    for (auto [i, key] : llvm::enumerate(summary.acquiredPhaseKeys))
      state.phases[key] = forOp.getResult(nArgs + 1 + i);
    for (auto [idx, tokenRef] : tokenRefs)
      *tokenRef = forOp.getResult(idx);
  }

  void assignStateInIfOp(scf::IfOp ifOp, State &state) {
    auto thenSummary =
        analyzeSemaphoreUseInBlock(ifOp.thenBlock(), state.token);
    if (ifOp.elseBlock())
      thenSummary = mergeSemaphoreUseSummaries(
          thenSummary,
          analyzeSemaphoreUseInBlock(ifOp.elseBlock(), state.token));
    if (!thenSummary.hasStageUse)
      return;

    SmallVector<Type> extraIfResults;
    extraIfResults.push_back(state.stage.getType());
    for (PhaseKey key : thenSummary.acquiredPhaseKeys)
      extraIfResults.push_back(getPhase(state, key).getType());

    OpBuilder builder(ifOp);
    size_t nResults = ifOp.getResults().size();
    auto newIfOp = replaceIfOpWithNewSignature(builder, ifOp, extraIfResults);

    auto thenState = assignStateInBlock(newIfOp.thenBlock(), state);
    auto elseState = newIfOp.elseBlock()
                         ? assignStateInBlock(newIfOp.elseBlock(), state)
                         : state;

    auto thenYieldOp = newIfOp.thenYield();
    auto elseYieldOp = newIfOp.elseYield();

    llvm::MapVector<int, Value *> tokenRefs;
    if (auto pos =
            findValuePosInRange(thenYieldOp->getOperands(), state.token)) {
      tokenRefs[*pos] = &state.token;
    }
    if (auto pos =
            findValuePosInRange(elseYieldOp->getOperands(), state.token)) {
      tokenRefs[*pos] = &state.token;
    }
    tokToStagePosMap[{newIfOp.thenYield(), thenState.token}] =
        thenYieldOp.getNumOperands();
    tokToStagePosMap[{newIfOp.elseYield(), elseState.token}] =
        elseYieldOp.getNumOperands();

    thenYieldOp->insertOperands(thenYieldOp.getNumOperands(), thenState.stage);
    elseYieldOp->insertOperands(elseYieldOp.getNumOperands(), elseState.stage);
    for (PhaseKey key : thenSummary.acquiredPhaseKeys) {
      thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                  getPhase(thenState, key));
      elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                  getPhase(elseState, key));
    }

    assert(hasPartition(ifOp));
    auto ifOpIds = getPartitionIds(ifOp);
    auto ifOpOutputsIds = getPartitionOutputs(ifOp);
    ifOp.erase();

    // Stage: all group partition IDs.
    ifOpIds.insert(allGroupPartitionIds.begin(), allGroupPartitionIds.end());
    ifOpOutputsIds.push_back(SetVector<int>(allGroupPartitionIds.begin(),
                                            allGroupPartitionIds.end()));
    // Phase: per-key partition IDs.
    for (PhaseKey key : thenSummary.acquiredPhaseKeys) {
      SetVector<int> phaseIds;
      for (Value arg : {getPhase(thenState, key), getPhase(elseState, key)}) {
        auto ids = inferPartitionIds(arg, key.partitionId);
        phaseIds.insert(ids.begin(), ids.end());
      }
      ifOpOutputsIds.push_back(phaseIds);
    }

    setPartition(newIfOp, ifOpIds);
    setPartitionOutputs(newIfOp, ifOpOutputsIds);

    state.stage = newIfOp.getResult(nResults);
    for (auto [i, key] : llvm::enumerate(thenSummary.acquiredPhaseKeys))
      state.phases[key] = newIfOp.getResult(nResults + 1 + i);
    for (auto [idx, tokenRef] : tokenRefs)
      *tokenRef = newIfOp.getResult(idx);
  }

  State assignStateInBlock(Block *block, State state) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto acquireOp = getAcquireOp(&op)) {
        ImplicitLocOpBuilder b(acquireOp.getLoc(), acquireOp);
        auto wsTag = getWarpSpecializeTag(&op);
        auto stageCluster = getStageCluster(&op);

        std::optional<SetVector<int>> phasePids, stagePids;
        if (hasPartition(&op)) {
          phasePids = getPartitionIds(&op);
          // Stage is shared across the whole semaphore group, so partitioned
          // acquires produce stage arithmetic visible to all group
          // partitions.
          stagePids = allGroupPartitionIds;
        }

        auto createInto = [&](std::optional<SetVector<int>> pids, auto opTy,
                              auto... args) {
          using ty = decltype(opTy);
          auto op = triton::gpu::createInto<ty>(
              b, b.getLoc(), pids, stageCluster,
              std::forward<decltype(args)>(args)...);
          if (wsTag)
            setWarpSpecializeTag(op, *wsTag);
          return op;
        };
        auto createIntoStage = [&](auto opTy, auto... args) {
          return createInto(stagePids, opTy,
                            std::forward<decltype(args)>(args)...);
        };
        auto createIntoPhase = [&](auto opTy, auto... args) {
          return createInto(phasePids, opTy,
                            std::forward<decltype(args)>(args)...);
        };

        // Stage update.
        Value rawStage = state.stage;
        Value acquireStage = rawStage;
        bool advanceStage = isFirstUseFreshWriteAfterAcquire(acquireOp);
        if (advanceStage) {
          auto nextStage =
              createIntoStage(arith::AddIOp{}, rawStage,
                              createIntoStage(arith::ConstantIntOp{}, 1, 32));
          auto stageWrapped = createIntoStage(
              arith::CmpIOp{}, arith::CmpIPredicate::eq, nextStage,
              createIntoStage(arith::ConstantIntOp{}, getDepth(), 32));
          auto zero = createIntoStage(arith::ConstantIntOp{}, 0, 32);
          auto wrappedStage =
              createIntoStage(arith::SelectOp{}, stageWrapped, zero, nextStage);
          acquireStage = wrappedStage;
        }
        state.stage = acquireStage;
        acquireOp.getStageMutable().assign(acquireStage);
        state.token = acquireOp.getToken();

        // Phase update. Internal phase state stays group-specific, but the
        // acquire itself always receives the final parity bit consumed by
        // mbarrier.wait.
        for (int pid : allGroupPartitionIds) {
          if (hasPartition(&op) &&
              !llvm::is_contained(getPartitionIds(&op), pid))
            continue;
          PhaseKey key = getPhaseKey(pid, acquireOp.getSemaphore());
          Value phaseState = getPhase(state, key);
          Value acquirePhase = phaseState;
          if (useSinglePhaseForGroup) {
            auto nextPhase =
                createIntoPhase(arith::XOrIOp{}, phaseState,
                                createIntoPhase(arith::ConstantIntOp{}, 1, 32));
            auto zero = createIntoPhase(arith::ConstantIntOp{}, 0, 32);
            auto phaseWrapped = createIntoPhase(
                arith::CmpIOp{}, arith::CmpIPredicate::eq, acquireStage, zero);
            phaseState = createIntoPhase(arith::SelectOp{}, phaseWrapped,
                                         nextPhase, phaseState);
            acquirePhase = phaseState;
          } else {
            auto phaseBit = createIntoPhase(
                arith::ShLIOp{}, createIntoPhase(arith::ConstantIntOp{}, 1, 32),
                acquireStage);
            phaseState = createIntoPhase(arith::XOrIOp{}, phaseState, phaseBit);
            acquirePhase =
                createIntoPhase(arith::ShRUIOp{}, phaseState, acquireStage);
            acquirePhase =
                createIntoPhase(arith::AndIOp{}, acquirePhase,
                                createIntoPhase(arith::ConstantIntOp{}, 1, 32));
          }
          state.phases[key] = phaseState;
          acquireOp.getPhaseMutable().assign(acquirePhase);
        }
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignStateInForOp(forOp, state);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignStateInIfOp(ifOp, state);
      }
    }

    return state;
  }

  void propagateStage(Value token, Value stage,
                      DenseSet<Operation *> &visited) {
    for (auto &tokUse : token.getUses()) {
      auto owner = tokUse.getOwner();
      if (visited.contains(owner))
        continue;
      visited.insert(owner);
      if (auto stageOp = dyn_cast<SemaphoreStageInterface>(owner)) {
        if (auto blk = dyn_cast<BlockArgument>(stage)) {
          assert(hasPartition(stageOp));
          auto stageOpIds = getPartitionIds(stageOp);
          auto forOp = cast<scf::ForOp>(blk.getOwner()->getParentOp());
          auto pos = findValuePosInRange(forOp.getRegionIterArgs(), stage);
          assert(pos);

          // update op partitions
          assert(hasPartition(forOp));
          auto forOpIds = getPartitionIds(forOp);
          forOpIds.insert(stageOpIds.begin(), stageOpIds.end());
          setPartition(forOp, forOpIds);

          auto forOpOutputsIds = getPartitionOutputs(forOp);
          // Widen only the stage slot (phases are handled separately)
          forOpOutputsIds[*pos].insert(stageOpIds.begin(), stageOpIds.end());
          setPartitionOutputs(forOp, forOpOutputsIds);
        }
        stageOp.setStage(stage);
      } else if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        auto tokPos = tokUse.getOperandNumber() - forOp.getNumControlOperands();
        auto iterTok = forOp.getRegionIterArg(tokPos);
        auto stagePos = tokToStagePosMap.at({forOp, iterTok});
        propagateStage(iterTok, forOp.getRegionIterArgs()[stagePos], visited);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        auto tokPos = tokUse.getOperandNumber();
        auto stagePos = tokToStagePosMap.at({yieldOp, token});
        auto parentOp = yieldOp->getParentOp();
        propagateStage(parentOp->getResult(tokUse.getOperandNumber()),
                       parentOp->getResult(stagePos), visited);
      }
    }
  }

  static LogicalResult run(ArrayRef<SemaphoreCreateOp> semaOps) {
    if (semaOps.empty())
      return success();

    // Compute single-phase eligibility per buffer group.
    AssignStagePhase impl(semaOps);
    bool singlePhaseEligible = impl.computeSinglePhaseEligibility();
    impl.useSinglePhaseForGroup = singlePhaseEligible;

    // Insert after the last semaOp so all semaphores are defined.
    ImplicitLocOpBuilder b(semaOps.back()->getLoc(), semaOps.back());
    b.setInsertionPointAfter(semaOps.back());

    State initState;
    auto firstSemaOp = semaOps.front();
    int depth = cast<SemaphoreType>(firstSemaOp.getType()).getNumStages();
    initState.stage = arith::ConstantIntOp::create(b, depth - 1, 32);
    // Per-semaphore initial phases:
    // single-phase:  isReleased=true  -> 0, isReleased=false -> 1
    // multiphase:    isReleased=true  -> 0, isReleased=false -> -1
    for (auto semaOp : semaOps) {
      uint32_t initPhase = semaOp.getIsReleased() ? 0x00000000u : 0xFFFFFFFFu;
      if (singlePhaseEligible) {
        initPhase = semaOp.getIsReleased() ? 0x00000000u : 0x00000001u;
      }
      impl.initialPhases[semaOp.getResult()] =
          arith::ConstantIntOp::create(b, static_cast<int64_t>(initPhase), 32);
    }
    impl.assignStateInBlock(firstSemaOp->getBlock(), initState);

    // Propagate stage to release/buffer ops via token chain.
    for (auto semaOp : semaOps) {
      for (auto user : semaOp->getUsers()) {
        if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(user)) {
          DenseSet<Operation *> visited;
          impl.propagateStage(acquireOp.getToken(), acquireOp.getStage(),
                              visited);
        }
      }
    }

    // Verify: all acquires must have stage/phase assigned by the main walk.
    // All release/buffer ops must have stage set via propagateStage.
    for (auto semaOp : semaOps) {
      for (auto user : semaOp->getUsers()) {
        if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(user)) {
          assert(acquireOp.getStage() &&
                 "acquire missing stage after assign-stage-phase");
          assert(acquireOp.getPhase() &&
                 "acquire missing phase after assign-stage-phase");
        }
        if (auto stageOp = dyn_cast<SemaphoreStageInterface>(user))
          assert(stageOp.getStage() &&
                 "release/buffer missing stage after propagation");
      }
    }

    return success();
  }
};

void updateOutputWithDefaultPartition(Operation *op, int pos) {
  auto opIds = getPartitionIds(op);
  opIds.insert(0);
  setPartition(op, opIds);

  auto opOutputsIds = getPartitionOutputs(op);
  opOutputsIds[pos].insert(0);
  setPartitionOutputs(op, opOutputsIds);
}

void visitBackwardSlice(scf::ForOp wsLoop, Value value,
                        std::function<void(Operation *)> callback,
                        DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      if (forOp->hasAttr(kWarpSpecializeAttrName))
        return;
      auto pos = findValuePosInRange(forOp.getRegionIterArgs(), value);
      assert(pos);
      visitBackwardSlice(wsLoop, forOp.getInitArgs()[*pos], callback, visited);
    }
  } else if (auto defOp = value.getDefiningOp();
             isa<scf::IfOp, scf::ForOp>(defOp)) {
    auto pos = findValuePosInRange(defOp->getResults(), value);
    assert(pos);
    updateOutputWithDefaultPartition(defOp, *pos);
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      visitBackwardSlice(wsLoop, ifOp.thenYield()->getOperand(*pos), callback,
                         visited);
      if (ifOp.elseBlock())
        visitBackwardSlice(wsLoop, ifOp.elseYield()->getOperand(*pos), callback,
                           visited);
      visitBackwardSlice(wsLoop, ifOp.getCondition(), callback, visited);
    } else {
      auto forOp = cast<scf::ForOp>(defOp);
      visitBackwardSlice(wsLoop,
                         forOp.getBody()->getTerminator()->getOperand(*pos),
                         callback, visited);
      // visit control operands of for-op
      for (int idx = 0; idx < forOp.getNumControlOperands(); ++idx) {
        auto control = forOp.getOperand(idx);
        visitBackwardSlice(wsLoop, control, callback, visited);
      }
    }
  } else if (wsLoop.getBody()->findAncestorOpInBlock(*defOp)) {
    callback(defOp);
    for (auto operand : defOp->getOperands()) {
      visitBackwardSlice(wsLoop, operand, callback, visited);
    }
  }
}

// Remove loop-invariant iter_args: if yield operand == iter_arg (block
// arg), the value never changes across iterations. Replace uses with
// the init value and rebuild the loop without those iter_args. Only clean up
// loops that are part of a warp-specialized region: the root
// `tt.warp_specialize` loop itself and any nested loops beneath it.
void removeLoopInvariantIterArgs(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp> loops;
  funcOp.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr(kWarpSpecializeAttrName))
      return;
    forOp.walk([&](scf::ForOp nestedForOp) { loops.push_back(nestedForOp); });
  });

  for (scf::ForOp forOp : loops) {
    auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yieldOp)
      continue;

    SmallVector<unsigned> toRemove;
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
      if (yieldOp.getOperand(i) == forOp.getRegionIterArg(i))
        toRemove.push_back(i);
    }
    if (toRemove.empty())
      continue;

    DenseSet<unsigned> removeSet(toRemove.begin(), toRemove.end());

    for (unsigned i : toRemove) {
      Value initVal = forOp.getInitArgs()[i];
      forOp.getRegionIterArg(i).replaceAllUsesWith(initVal);
      forOp.getResult(i).replaceAllUsesWith(initVal);
    }

    SmallVector<Value> newInitArgs;
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
      if (!removeSet.count(i))
        newInitArgs.push_back(forOp.getInitArgs()[i]);
    }

    unsigned numInductionVars = forOp.getNumInductionVars();
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (int j = static_cast<int>(toRemove.size()) - 1; j >= 0; --j) {
      yieldOp->eraseOperand(toRemove[j]);
      forOp.getBody()->eraseArgument(toRemove[j] + numInductionVars);
    }

    OpBuilder builder(forOp);
    auto newForOp =
        scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                           forOp.getUpperBound(), forOp.getStep(), newInitArgs);
    newForOp->setAttrs(forOp->getAttrs());
    if (forOp->hasAttr(kPartitionOutputsAttrName)) {
      SmallVector<SetVector<int>> newOutputs;
      for (auto [i, outputIds] : llvm::enumerate(getPartitionOutputs(forOp))) {
        if (!removeSet.count(i))
          newOutputs.push_back(outputIds);
      }
      if (newOutputs.empty())
        newForOp->removeAttr(kPartitionOutputsAttrName);
      else
        setPartitionOutputs(newForOp, newOutputs);
    }
    newForOp.getBody()->erase();
    newForOp.getRegion().getBlocks().splice(
        newForOp.getRegion().getBlocks().begin(),
        forOp.getRegion().getBlocks());

    unsigned newIdx = 0;
    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      if (!removeSet.count(i))
        forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(newIdx++));
    }

    forOp.erase();
  }
}

LogicalResult assignStagePhase(triton::FuncOp funcOp) {
  SmallVector<SemaphoreCreateOp> semaOps;
  funcOp.walk([&](SemaphoreCreateOp op) { semaOps.push_back(op); });

  // Keep processing order deterministic and scoped by backing buffer.
  llvm::MapVector<Value, SmallVector<SemaphoreCreateOp>> semaGroups;
  for (auto semaOp : semaOps) {
    semaGroups[semaOp.getBuffers().front()].push_back(semaOp);
  }

  for (auto &it : semaGroups) {
    if (failed(AssignStagePhase::run(it.second)))
      return failure();
  }

  auto callback = [&](Operation *op) {
    if (!isa<scf::YieldOp, scf::IfOp, scf::ForOp, triton::ReduceOp>(op)) {
      assert(hasPartition(op));
      auto partitionIds = getPartitionIds(op);
      partitionIds.insert(0);
      setPartition(op, partitionIds);
    }
  };

  funcOp.walk([&](scf::ForOp forOp) {
    DenseSet<Value> visited;
    if (forOp->hasAttr(kWarpSpecializeAttrName)) {
      for (auto result : forOp.getResults()) {
        if (isa<IntegerType, FloatType>(result.getType()) &&
            !result.use_empty()) {
          auto arg = forOp.getBody()->getTerminator()->getOperand(
              result.getResultNumber());
          bool assignDefaultPartition =
              llvm::any_of(result.getUsers(), [&](Operation *user) {
                return !hasPartition(user) ||
                       (isa<scf::ForOp>(user) && hasWarpSpecializeTag(user));
              });
          if (assignDefaultPartition) {
            updateOutputWithDefaultPartition(forOp, result.getResultNumber());
            visitBackwardSlice(forOp, arg, callback, visited);
          }
        }
      }
    }
  });

  // The stage/phase threading above can introduce loop-carried values that are
  // only forwarded unchanged. Trim them before downstream warp-specialization
  // and pipelining passes see the rewritten loops.
  removeLoopInvariantIterArgs(funcOp);
  return success();
}

// ----------------------------------------------------------------------------

} // anonymous namespace

class NVWSAssignStagePhase
    : public impl::NVWSAssignStagePhaseBase<NVWSAssignStagePhase> {
public:
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    m.walk([&](triton::FuncOp funcOp) {
      if (failed(assignStagePhase(funcOp)))
        signalPassFailure();
    });
  }
};

} // namespace triton
} // namespace mlir
