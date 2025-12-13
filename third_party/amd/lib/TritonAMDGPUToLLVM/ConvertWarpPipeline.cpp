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
          for (auto bufId : allocation->getBufferIds(v)) {
            if (bufId == Allocation::InvalidBufferId)
              continue;
            auto interval = allocation->getAllocatedInterval(bufId);
            auto slice = AllocationSlice(v, interval);
            if (isa<MemoryEffects::Write>(eff.getEffect()))
              info.syncWriteSlices[slice].insert(op);
            else if (isa<MemoryEffects::Read>(eff.getEffect()))
              info.syncWriteSlices[slice].insert(op);
          }
        }
      }
    }
  }
  return info;
}

static void emitClusterBarrier(PatternRewriter &r, Location loc,
                               bool needLocal) {
  ROCDL::SchedBarrier::create(r, loc, 0);
  if (needLocal)
    mlir::triton::gpu::LocalBarrierOp::create(r, loc);
  else
    ROCDL::SBarrierOp::create(r, loc);
  ROCDL::SchedBarrier::create(r, loc, 0);
}

class ConvertPipelinedForPattern : public OpRewritePattern<scf::ForOp> {
public:
  ConvertPipelinedForPattern(MLIRContext *ctx, ModuleAllocation &moduleAlloc)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2),
        moduleAllocation(moduleAlloc) {}

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

    if (failed(emitPipelinedFor(rewriter, forOp.getLoc(), forOp, allocation)))
      return failure();

    return success();
  }

private:
  LogicalResult emitPipelinedFor(PatternRewriter &b, Location loc,
                                 scf::ForOp forOp,
                                 Allocation *allocation) const {
    // 1. Insert conditional branch first,
    b.setInsertionPoint(forOp);
    // Set barrier before starting the loop. This resolves any outstanding
    // synchronization before beginning the specialized asymmetric
    // synchronization.
    auto preBarrier = gpu::BarrierOp::create(b, loc);

    // Insert condbarrier::second_half before starting the loop
    // FIXME : correctly calculate numbers per the arch
    auto i32ty = b.getIntegerType(32);
    auto workIDX = ROCDL::ThreadIdXOp::create(b, loc, i32ty);
    auto constZero = arith::ConstantIntOp::create(b, loc, 0, 32);
    auto constWarpSize = arith::ConstantIntOp::create(b, loc, 256, 32);
    auto warpIDX = arith::DivSIOp::create(b, loc, workIDX, constWarpSize);
    auto warpLow = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                         warpIDX, constZero);
    auto warpHigh = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                          warpIDX, constZero);

    mlir::triton::amdgpu::CondBarrierOp::create(b, loc, warpHigh);

    // Insert condbarrier::first_half after the end of the loop
    b.setInsertionPointAfter(forOp);
    mlir::triton::amdgpu::CondBarrierOp::create(b, loc, warpLow);

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
                     triton::amdgpu::AsyncTDMWait>(op)) {
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
        bars.push_back(false);
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

    // 3. Performing pairwise dependency analysis between clusters.  For each
    // src → next pair (with wrap-around), we check whether their memory
    // intervals overlap.  If so, a fence/barrier must be inserted at the
    // boundary cluster (barrierLoc).  The analysis is expressed as a
    // circular traversal so that pipeline stages form a ring.
    // • `bars[i] = true` marks that a new cluster barrier must be inserted
    //   before cluster i.
    // • Existing barriers override or satisfy required fences, so we do not
    //   insert duplicates.
    for (int offset = 0; offset < numClusters; offset++) {
      for (int src = 0; src < numClusters; src++) {
        const int next = (src + 2 + offset) % numClusters;
        const int barrierLoc = (src + 1 + offset) % numClusters;
        LDBG("Inspecting src:" << src << " to next:" << next);
        // Check if any existing barrier sits between src and barrierIdx
        auto isSynced = [&]() -> bool {
          for (int idx = (src + 1) % numClusters; idx != src;
               idx = (idx + 1) % numClusters) {
            if (bars[idx])
              return true;
            if (idx == barrierLoc)
              break;
          }
          return false;
        };
        // Skip if dependency is already resolved.
        if (isSynced()) {
          LDBG("already synced");
          continue;
        }
        const bool needFence = clusterInfo[src].isIntersected(
            clusterInfo[next], mlir::triton::AMD::membarFilter);
        // insert fence/barrier in front of this cluster
        LDBG("need fence?: " << needFence);
        if (needFence) {
          bars[barrierLoc] = true;
          LDBG("cluster " << src << " need fence to " << next
                          << " placing barrier at " << barrierLoc);
        }
      }
    }

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
        ROCDL::SchedBarrier::create(b, loc, 0);
        b.setInsertionPointAfter(exBarOp);
        ROCDL::SchedBarrier::create(b, loc, 0);
      } else {
        b.setInsertionPoint(clusterOps[i]);
        // The first one wraps back to the last of the loop
        if (i == 0 && topBar == existingBarrierMap.end())
          // inserts just before yield (=End of the loop).
          b.setInsertionPoint(terminatorOp);
        emitClusterBarrier(b, loc, /*needLocal=*/bars[i]);
      }
    }
    return success();
  }

  ModuleAllocation &moduleAllocation;
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

struct ConvertWarpPipeline
    : public mlir::triton::impl::ConvertWarpPipelineBase<ConvertWarpPipeline> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    ModuleAllocation moduleAllocation(m);

    RewritePatternSet patternFor(&getContext());
    RewritePatternSet patternInline(&getContext());
    patternFor.add<ConvertPipelinedForPattern>(&getContext(), moduleAllocation);
    patternInline.add<InlineWarpPipelineExecuteRegionPattern>(&getContext());

    if (failed(applyPatternsGreedily(m, std::move(patternFor))))
      signalPassFailure();
    if (failed(applyPatternsGreedily(m, std::move(patternInline))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::AMD {
std::unique_ptr<OperationPass<ModuleOp>> createConvertWarpPipelinePass() {
  return std::make_unique<ConvertWarpPipeline>();
}
} // namespace mlir::triton::AMD
