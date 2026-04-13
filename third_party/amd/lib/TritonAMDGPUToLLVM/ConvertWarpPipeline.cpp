/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "convert-warp-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTWARPPIPELINE
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

// construct a virtual block from each pipeline cluster
// block contains its buffer R/W information.
static BlockInfo buildBlockInfoFromBlock(Block *block, Allocation *allocation) {
  BlockInfo info; // running fact for this block
  for (Operation &opRef : *block) {
    Operation *op = &opRef;
    if (auto mei = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effs;
      mei.getEffects(effs);
      for (auto &eff : effs) {
        if (Value v = eff.getValue()) {
          for (auto bufId : allocation->getAllBufferIdsWithAliases(v)) {
            if (bufId == Allocation::InvalidBufferId)
              continue;
            auto interval = allocation->getAllocatedInterval(bufId);
            auto slice = AllocationSlice(v, interval, bufId);
            if (isa<MemoryEffects::Write>(eff.getEffect()))
              info.syncWriteSlices[slice].insert(op);
            else if (isa<MemoryEffects::Read>(eff.getEffect()))
              info.syncReadSlices[slice].insert(op);
          }
        }
      }
    }
  }
  return info;
}

// Pairwise dependency analysis between pipeline clusters.
// For each src → next pair, checks whether their memory intervals overlap.
// If so, marks `bars[barrierLoc] = true` to indicate a fence is needed.
//
// When `circular` is true (loop pipelines), indices wrap around modulo
// numClusters so that the last cluster feeds back to the first.
// When false (flat pipelines), indices are strictly linear.
static void analyzePipelineDependencies(ArrayRef<BlockInfo> clusterInfo,
                                        SmallVectorImpl<bool> &bars,
                                        Allocation *allocation, bool circular) {
  int numClusters = clusterInfo.size();
  for (int offset = 0; offset < numClusters; offset++) {
    for (int src = 0; src < numClusters; src++) {
      int next, barrierLoc;
      if (circular) {
        next = (src + 2 + offset) % numClusters;
        barrierLoc = (src + 1 + offset) % numClusters;
      } else {
        next = src + 2 + offset;
        barrierLoc = src + 1 + offset;
        if (next >= numClusters || barrierLoc >= numClusters)
          continue;
      }

      auto isSynced = [&]() -> bool {
        if (circular) {
          for (int idx = (src + 1) % numClusters; idx != src;
               idx = (idx + 1) % numClusters) {
            if (bars[idx])
              return true;
            if (idx == barrierLoc)
              break;
          }
        } else {
          for (int idx = src + 1; idx <= barrierLoc; idx++)
            if (bars[idx])
              return true;
        }
        return false;
      };
      if (isSynced())
        continue;

      const bool needFence = clusterInfo[src].isIntersected(
          clusterInfo[next], mlir::triton::AMD::membarFilter, allocation);
      if (needFence) {
        bars[barrierLoc] = true;
        LDBG("cluster " << src << " need fence to " << next
                        << " placing barrier at " << barrierLoc);
      }
    }
  }
}

static void emitClusterBarrier(OpBuilder &r, Location loc, bool needLocal) {
  ROCDL::SchedBarrier::create(r, loc, 0);
  if (needLocal)
    mlir::triton::gpu::BarrierOp::create(r, loc, triton::gpu::AddrSpace::Local);
  else
    ROCDL::SBarrierOp::create(r, loc);
  ROCDL::SchedBarrier::create(r, loc, 0);
}

static void emitClusterPriority(OpBuilder &r, Location loc,
                                Operation *clusterOp, bool anyHasPriority) {
  if (auto intAttr = clusterOp->getAttrOfType<IntegerAttr>(
          "triton.warp_pipeline.priority")) {
    ROCDL::SetPrioOp::create(r, loc, intAttr.getInt());
  } else if (anyHasPriority) {
    // Reset to default when other stages use priority.
    ROCDL::SetPrioOp::create(r, loc, 0);
  }
}

// Emit pre-barrier, thread-ID partitioning, and phase-shift cond_barrier.
// Returns warpLow (for reconverge) and warpHigh (consumed by phase shift).
static std::pair<Value, Value>
emitPipelinePrelude(OpBuilder &b, Location loc, int threadsPerPipelineGroup) {
  mlir::triton::gpu::BarrierOp::create(b, loc, triton::gpu::AddrSpace::Local);

  auto i32ty = b.getIntegerType(32);
  auto workIDX = ROCDL::ThreadIdXOp::create(b, loc, i32ty);
  auto constZero = arith::ConstantIntOp::create(b, loc, 0, 32);
  auto constWarpSize =
      arith::ConstantIntOp::create(b, loc, threadsPerPipelineGroup, 32);
  auto warpIDX = arith::DivSIOp::create(b, loc, workIDX, constWarpSize);
  auto warpLow = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                       warpIDX, constZero);
  auto warpHigh = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                        warpIDX, constZero);
  mlir::triton::amdgpu::CondBarrierOp::create(b, loc, warpHigh);

  return {warpLow, warpHigh};
}

// Emit priority reset and reconverge cond_barrier after a pipeline.
static void emitPipelinePostlude(OpBuilder &b, Location loc,
                                 bool anyHasPriority, Value warpLow) {
  if (anyHasPriority)
    ROCDL::SetPrioOp::create(b, loc, 0);
  mlir::triton::amdgpu::CondBarrierOp::create(b, loc, warpLow);
}

class ConvertPipelinedForPattern : public OpRewritePattern<scf::ForOp> {
public:
  ConvertPipelinedForPattern(MLIRContext *ctx, ModuleAllocation &moduleAlloc,
                             int threadsPerPipelineGroup)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2),
        moduleAllocation(moduleAlloc),
        threadsPerPipelineGroup(threadsPerPipelineGroup) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only handle loops that the frontend marked with pipelined_for.
    if (!forOp->getAttr("triton.warp_pipeline.pipelined_for"))
      return rewriter.notifyMatchFailure(forOp, "no pipelined_for");
    forOp->removeAttr("triton.warp_pipeline.pipelined_for");

    // Look up allocation info as in original pass.
    auto func = forOp->getParentOfType<mlir::triton::FuncOp>();
    Allocation *allocation = moduleAllocation.getFuncData(func);
    if (!allocation)
      return rewriter.notifyMatchFailure(forOp, "no Allocation for function");

    if (failed(emitPipelinedFor(rewriter, forOp.getLoc(), forOp, allocation,
                                threadsPerPipelineGroup)))
      return failure();

    return success();
  }

private:
  LogicalResult emitPipelinedFor(PatternRewriter &b, Location loc,
                                 scf::ForOp forOp, Allocation *allocation,
                                 int threadsPerPipelineGroup) const {
    // 1. Pre-barrier, thread partitioning, and phase shift.
    b.setInsertionPoint(forOp);
    auto [warpLow, warpHigh] =
        emitPipelinePrelude(b, loc, threadsPerPipelineGroup);

    // 2. Collect existing barrier information.
    // Scanning the loop body and classifying each consecutive block of
    // operations into a pipeline cluster (one cluster per execute_region).
    // While doing this, we also detect any pre-existing barriers located
    // between clusters.  These barriers may come from prefetch patterns, and
    // must be preserved, but only at valid cluster boundaries.
    SmallVector<Block *> clusterBlocks;
    SmallVector<Operation *> clusterOps;
    SmallVector<bool> bars;
    std::map<int, Operation *> existingBarrierMap;
    Operation *terminatorOp = nullptr;

    for (auto &op : *forOp.getBody()) {
      if (auto exeOp = dyn_cast<scf::ExecuteRegionOp>(op)) {
        // Fail conversion with executeRegion from unkown source.
        if (exeOp->getAttr("triton.warp_pipeline.stage") == nullptr)
          return failure();
        exeOp.setNoInline(false);
        clusterOps.push_back(&op);
        clusterBlocks.push_back(&exeOp->getRegion(0).front());
        bars.push_back(false);
      } else if (isa<ROCDL::BarrierOp, gpu::BarrierOp, triton::gpu::AsyncWaitOp,
                     triton::amdgpu::AsyncWaitOp, triton::amdgpu::AsyncTDMWait,
                     triton::amdgpu::AsyncTDMIntrinsicWait>(op)) {
        int currCluster = clusterBlocks.size();
        // Reject if multiple barriers appear without an intervening cluster.
        // This is functionally valid but may cause unpredictable timing. Users
        // should insert a dummy cluster explicitly if a pipeline bubble is
        // required.
        // Also only allow ops which waits local memory,
        // e.g., s_barrier is NOT allowed.
        if (existingBarrierMap.find(currCluster) != existingBarrierMap.end())
          return failure();
        existingBarrierMap[currCluster] = &op;
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        terminatorOp = &op;
      } else { // Fail conversion if any other op found outside of the cluster.
        return failure();
      }
    }

    SmallVector<BlockInfo> clusterInfo;
    for (auto cb : clusterBlocks)
      clusterInfo.push_back(buildBlockInfoFromBlock(cb, allocation));
    int numClusters = clusterInfo.size();

    // Check if any cluster has explicit priority.
    bool anyHasPriority = llvm::any_of(clusterOps, [](Operation *op) {
      return op->hasAttr("triton.warp_pipeline.priority");
    });
    LDBG("total clusters : " << numClusters);

    // Normally, we don't expect a pipelined loop begins with a barrier
    // but sometimes required by memory prefetching pattern.
    auto topBar = existingBarrierMap.find(0);
    auto bottomBar = existingBarrierMap.find(numClusters);
    if (bottomBar != existingBarrierMap.end()) {
      if (topBar != existingBarrierMap.end())
        return failure(); // Unreachable
      existingBarrierMap[0] = bottomBar->second;
      existingBarrierMap.erase(bottomBar);
    }

    // 3. Circular dependency analysis (wrap-around for loop pipelines).
    analyzePipelineDependencies(clusterInfo, bars, allocation,
                                /*circular=*/true);

    // 4. Materializing final cluster-scope barriers.  For each cluster index:
    //  • If there is a pre-existing barrier at that location, we wrap it with
    //    sched_barriers so that backend scheduling cannot move operations
    //    across it.
    //  • If no barrier exists but `bars[i]` is true, we insert a new cluster
    //    barrier (SchedBarrier + Local/SBarrier + SchedBarrier).
    //    The “local” variant is chosen when cluster-to-cluster memory
    //    dependence requires local-scope synchronization.
    //  • Cluster 0 is a special case: if no top-of-loop barrier existed,
    //    the first cluster barrier must be inserted just before the loop’s
    //    terminator, forming the wrap-around dependency.
    for (int i = 0; i < numClusters; i++) {
      if (auto exBar = existingBarrierMap.find(i);
          exBar != existingBarrierMap.end()) {
        auto exBarOp = exBar->second;
        b.setInsertionPoint(exBarOp);
        emitClusterPriority(b, loc, clusterOps[i], anyHasPriority);
        ROCDL::SchedBarrier::create(b, loc, 0);
        b.setInsertionPointAfter(exBarOp);
        ROCDL::SchedBarrier::create(b, loc, 0);
      } else {
        b.setInsertionPoint(clusterOps[i]);
        // The first one wraps back to the last of the loop
        if (i == 0 && topBar == existingBarrierMap.end()) {
          // Extra setprio needed before the loop for the first cluster
          b.setInsertionPoint(forOp);
          emitClusterPriority(b, loc, clusterOps[i], anyHasPriority);
          // inserts just before yield (=End of the loop).
          b.setInsertionPoint(terminatorOp);
        }
        emitClusterPriority(b, loc, clusterOps[i], anyHasPriority);
        emitClusterBarrier(b, loc, /*needLocal=*/bars[i]);
      }
    }

    // Post-loop priority reset and reconverge.
    b.setInsertionPointAfter(forOp);
    emitPipelinePostlude(b, loc, anyHasPriority, warpLow);
    return success();
  }

  ModuleAllocation &moduleAllocation;
  int threadsPerPipelineGroup;
};

class InlineWarpPipelineExecuteRegionPattern
    : public OpRewritePattern<scf::ExecuteRegionOp> {
public:
  InlineWarpPipelineExecuteRegionPattern(MLIRContext *ctx)
      : OpRewritePattern<scf::ExecuteRegionOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp exec,
                                PatternRewriter &rewriter) const override {
    if (exec->getAttr("no_inline"))
      return rewriter.notifyMatchFailure(exec, "explicit no_inline");

    // Only inline the stages created by the warp-pipeline frontend.
    if (!exec->getAttr("triton.warp_pipeline.stage"))
      return rewriter.notifyMatchFailure(exec, "not a warp-pipeline stage");

    // Make sure this pattern is applied after transforming pipelined forOp
    if (auto forOp = dyn_cast<scf::ForOp>(exec->getParentOp()))
      if (forOp->getAttr("triton.warp_pipeline.pipelined_for"))
        return rewriter.notifyMatchFailure(exec,
                                           "parent forOp not converted yet");

    // Expect a single-block region.
    Region &reg = exec.getRegion();
    if (!llvm::hasSingleElement(reg))
      return rewriter.notifyMatchFailure(exec, "expected single-block region");

    // Inline region.
    Block *block = &reg.front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.inlineBlockBefore(block, exec, {});
    rewriter.replaceOp(exec, results);
    rewriter.eraseOp(terminator);

    return success();
  }
};

// Process a flat (non-loop) sequence of warp-pipeline execute_regions.
// Unlike the loop case there is no wrap-around: dependencies are strictly
// linear from the first stage to the last.
//
// Emitted IR:
//   ttg.barrier local               (pre-barrier)
//   <thread ID arith>
//   cond_barrier(warpHigh)           (phase shift)
//   [s_setprio P0]
//   execute_region { stage 0 }
//   [s_setprio P1]  sched+barrier    (cluster barrier)
//   execute_region { stage 1 }
//   ...
//   [s_setprio 0]
//   cond_barrier(warpLow)            (reconverge)
//
static void emitPipelinedFlat(SmallVector<scf::ExecuteRegionOp> &clusterOps,
                              Allocation *allocation,
                              int threadsPerPipelineGroup) {
  Location loc = clusterOps.front().getLoc();
  OpBuilder b(clusterOps.front().getContext());
  int numClusters = clusterOps.size();

  // 1. Pre-barrier and phase shift before the first execute_region.
  b.setInsertionPoint(clusterOps.front());
  auto [warpLow, warpHigh] =
      emitPipelinePrelude(b, loc, threadsPerPipelineGroup);

  // 2. Dependency analysis — linear, no wrap-around.
  SmallVector<Block *> clusterBlocks;
  SmallVector<bool> bars(numClusters, false);

  for (auto exec : clusterOps) {
    exec.setNoInline(false);
    clusterBlocks.push_back(&exec->getRegion(0).front());
  }

  SmallVector<BlockInfo> clusterInfo;
  for (auto *cb : clusterBlocks)
    clusterInfo.push_back(buildBlockInfoFromBlock(cb, allocation));

  bool anyHasPriority = llvm::any_of(clusterOps, [](scf::ExecuteRegionOp op) {
    return op->hasAttr("triton.warp_pipeline.priority");
  });

  // Linear dependency analysis (no wrap-around for flat pipelines).
  analyzePipelineDependencies(clusterInfo, bars, allocation,
                              /*circular=*/false);

  // 3. Materialize cluster barriers.
  //    Cluster 0 gets only its priority (inserted after cond_barrier above).
  //    Clusters 1..N get priority + cluster barrier, unless a pre-existing
  //    barrier op (e.g., async_wait) already exists between the clusters —
  //    in that case, wrap it with sched_barriers instead of adding a new one.
  emitClusterPriority(b, loc, clusterOps[0], anyHasPriority);

  for (int i = 1; i < numClusters; i++) {
    Operation *existingBarrier = nullptr;
    for (Operation *op = clusterOps[i - 1]->getNextNode();
         op && op != clusterOps[i].getOperation(); op = op->getNextNode()) {
      if (isa<ROCDL::BarrierOp, gpu::BarrierOp, triton::gpu::AsyncWaitOp,
              triton::amdgpu::AsyncWaitOp, triton::amdgpu::AsyncTDMWait,
              triton::amdgpu::AsyncTDMIntrinsicWait>(op)) {
        existingBarrier = op;
        break;
      }
    }

    if (existingBarrier) {
      b.setInsertionPoint(existingBarrier);
      emitClusterPriority(b, loc, clusterOps[i], anyHasPriority);
      ROCDL::SchedBarrier::create(b, loc, 0);
      b.setInsertionPointAfter(existingBarrier);
      ROCDL::SchedBarrier::create(b, loc, 0);
    } else {
      b.setInsertionPoint(clusterOps[i]);
      emitClusterPriority(b, loc, clusterOps[i], anyHasPriority);
      emitClusterBarrier(b, loc, /*needLocal=*/bars[i]);
    }
  }

  // 4. Post-sequence reconverge.
  b.setInsertionPointAfter(clusterOps.back());
  emitPipelinePostlude(b, loc, anyHasPriority, warpLow);
}

// Walk the module for flat warp-pipeline execute_region sequences
// (produced by WarpPipeliner::createFlatPipeline) and emit phase-shift
// barriers around them.
static void processUnrolledPipelineRegions(ModuleOp m,
                                           ModuleAllocation &moduleAllocation,
                                           int threadsPerPipelineGroup) {
  auto isIgnorable = [](Operation *op) {
    return isa<ROCDL::BarrierOp, gpu::BarrierOp, triton::gpu::AsyncWaitOp,
               triton::amdgpu::AsyncWaitOp, triton::amdgpu::AsyncTDMWait,
               triton::amdgpu::AsyncTDMIntrinsicWait>(op);
  };

  m.walk([&](triton::FuncOp funcOp) {
    Allocation *allocation = moduleAllocation.getFuncData(funcOp);
    if (!allocation)
      return;

    for (Block &block : funcOp.getBody()) {
      // Collect contiguous sequences of flat warp-pipeline execute_regions,
      // splitting at any non-ignorable, non-pipeline op.
      SmallVector<SmallVector<scf::ExecuteRegionOp>> sequences;
      SmallVector<scf::ExecuteRegionOp> current;

      for (auto &op : block) {
        if (auto exec = dyn_cast<scf::ExecuteRegionOp>(&op)) {
          if (exec->hasAttr("triton.warp_pipeline.stage") &&
              !isa<scf::ForOp>(exec->getParentOp())) {
            current.push_back(exec);
            continue;
          }
        }
        if (isIgnorable(&op))
          continue;
        if (!current.empty()) {
          sequences.push_back(std::move(current));
          current.clear();
        }
      }
      if (!current.empty())
        sequences.push_back(std::move(current));

      for (auto &seq : sequences) {
        if (seq.size() < 2)
          continue;
        LDBG("processing flat pipeline with " << seq.size() << " stages");
        emitPipelinedFlat(seq, allocation, threadsPerPipelineGroup);
      }
    }
  });
}

// Check if the wrap-around cluster barrier of a converted pipelined loop
// includes a local memory fence (ttg.barrier local).  The wrap-around barrier
// is the last cluster barrier emitted just before the scf.yield terminator:
//   [s_setprio]  sched_barrier  ttg.barrier_local|s_barrier  sched_barrier
//   yield
static bool hasLocalFenceAtWrapAround(scf::ForOp forOp) {
  auto *yieldOp = forOp.getBody()->getTerminator();
  if (!yieldOp)
    return false;
  Operation *op = yieldOp->getPrevNode();
  if (!op || !isa<ROCDL::SchedBarrier>(op))
    return false;
  op = op->getPrevNode();
  if (!op)
    return false;
  if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op))
    return barrier.hasLocal();
  return false;
}

// Eliminate redundant conditional barriers between consecutive warp-pipelined
// regions.  When loop 1's wrap-around barrier already includes a local fence,
// the phase shift naturally carries over into the next pipeline: the post-loop
// reconverge and pre-pipeline phase shift cancel, and the intervening
// pre-barrier is redundant because membar will not need to insert a barrier
// (the wrap-around fence already resolved all pending LDS writes).
//
// The "next pipeline" can be either another scf.for or a flat (unrolled)
// pipeline represented as a sequence of scf.execute_region ops.
//
// Before:                              After:
//   scf.for { loop 1 }                  scf.for { loop 1 }
//   [s_setprio 0]                       [s_setprio 0]
//   cond_barrier(warpLow)   ← erase    <thread ID arith> (dead, cleaned later)
//   ttg.barrier local       ← erase    [s_setprio P]
//   <thread ID arith>                   scf.for / execute_region { pipeline 2 }
//   cond_barrier(warpHigh)  ← erase
//   [s_setprio P]
//   scf.for / execute_region { pipeline 2 }
//
static void eliminateRedundantCondBarriers(ModuleOp m) {
  SmallVector<Operation *> toErase;

  m.walk([&](triton::FuncOp funcOp) {
    for (Block &block : funcOp.getBody()) {
      SmallVector<triton::amdgpu::CondBarrierOp> condBarriers;
      for (auto &op : block)
        if (auto cb = dyn_cast<triton::amdgpu::CondBarrierOp>(&op))
          condBarriers.push_back(cb);

      for (size_t i = 0; i + 1 < condBarriers.size(); i++) {
        auto postLoopCB = condBarriers[i];
        auto preLoopCB = condBarriers[i + 1];

        // The post-loop cond_barrier must be preceded by a scf.for
        // (possibly with an intervening s_setprio reset).
        Operation *prev = postLoopCB->getPrevNode();
        if (prev && isa<ROCDL::SetPrioOp>(prev))
          prev = prev->getPrevNode();
        auto prevFor = dyn_cast_or_null<scf::ForOp>(prev);
        if (!prevFor)
          continue;

        // The pre-loop cond_barrier must be followed by a warp-pipelined
        // scf.for or a flat pipeline execute_region (possibly with an
        // intervening s_setprio).
        Operation *next = preLoopCB->getNextNode();
        if (next && isa<ROCDL::SetPrioOp>(next))
          next = next->getNextNode();
        bool nextIsPipeline = isa_and_nonnull<scf::ForOp>(next) ||
                              (isa_and_nonnull<scf::ExecuteRegionOp>(next) &&
                               next->hasAttr("triton.warp_pipeline.stage"));
        if (!nextIsPipeline)
          continue;

        if (!hasLocalFenceAtWrapAround(prevFor))
          continue;

        // Find the ttg.barrier local (pre-barrier) between the two
        // cond_barriers.
        triton::gpu::BarrierOp preBarrier = nullptr;
        for (Operation *op = postLoopCB->getNextNode(); op && op != preLoopCB;
             op = op->getNextNode()) {
          if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(op)) {
            if (barrier.hasLocal()) {
              preBarrier = barrier;
              break;
            }
          }
        }
        if (!preBarrier)
          continue;

        LDBG("eliminating redundant barriers between back-to-back loops");
        toErase.push_back(postLoopCB);
        toErase.push_back(preBarrier);
        toErase.push_back(preLoopCB);
        i++;
      }
    }
  });

  for (auto *op : llvm::reverse(toErase))
    op->erase();
}

struct ConvertWarpPipeline
    : public mlir::triton::impl::ConvertWarpPipelineBase<ConvertWarpPipeline> {

public:
  ConvertWarpPipeline(StringRef arch)
      : ConvertWarpPipelineBase<ConvertWarpPipeline>() {
    this->arch = arch.str();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    mlir::triton::AMD::TargetInfo targetInfo(arch.getValue());
    size_t partitionSize = targetInfo.getSharedMemoryPartitionSize();
    ModuleAllocation moduleAllocation(
        m, triton::defaultAllocationAnalysisScratchSizeFn, partitionSize);

    if (targetInfo.getISAFamily() == mlir::triton::AMD::ISAFamily::Unknown) {
      m.emitError("unsupported target: '") << arch.getValue() << "'";
      return signalPassFailure();
    }
    // Thread count of one warp-pipeline group.
    // A block runs on 4 SIMDs with 2 warps per SIMD. Warp-pipelining splits
    // these warps into two groups (one warp per SIMD) that execute different
    // stages at different times.
    int threadsPerPipelineGroup = targetInfo.getWarpSize() * 4;

    RewritePatternSet patternFor(&getContext());
    RewritePatternSet patternInline(&getContext());
    patternFor.add<ConvertPipelinedForPattern>(&getContext(), moduleAllocation,
                                               threadsPerPipelineGroup);
    patternInline.add<InlineWarpPipelineExecuteRegionPattern>(&getContext());

    if (failed(applyPatternsGreedily(m, std::move(patternFor))))
      signalPassFailure();

    // Flat (unrolled) pipeline regions are still wrapped in execute_regions
    // with no_inline=true from WarpPipeliner.  Process them before inlining.
    processUnrolledPipelineRegions(m, moduleAllocation,
                                   threadsPerPipelineGroup);

    // Must run after patternFor and flat processing (all regions converted,
    // barriers inserted) but before patternInline (inlining execute_regions
    // would flatten the IR and obscure the cond_barrier adjacency we rely on).
    eliminateRedundantCondBarriers(m);

    if (failed(applyPatternsGreedily(m, std::move(patternInline))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::AMD {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertWarpPipelinePass(StringRef arch) {
  return std::make_unique<ConvertWarpPipeline>(arch);
}
} // namespace mlir::triton::AMD
