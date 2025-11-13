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
            if (isa<MemoryEffects::Write>(eff.getEffect()))
              info.syncWriteIntervals[interval].insert(op);
            else if (isa<MemoryEffects::Read>(eff.getEffect()))
              info.syncReadIntervals[interval].insert(op);
          }
        }
      }
    }
  }
  return info;
}

static void emitClusterBarrier(PatternRewriter &rewriter, Location loc,
                               bool needLocal) {
  rewriter.create<ROCDL::SchedBarrier>(loc, 0);
  if (needLocal)
    rewriter.create<mlir::triton::gpu::LocalBarrierOp>(loc);
  else
    rewriter.create<ROCDL::SBarrierOp>(loc);
  rewriter.create<ROCDL::SchedBarrier>(loc, 0);
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
  LogicalResult emitPipelinedFor(PatternRewriter &builder, Location loc,
                                 scf::ForOp forOp,
                                 Allocation *allocation) const {
    // 1. Insert conditional branch first,
    builder.setInsertionPointAfter(forOp);
    // Set barrier before starting the loop. This resolves any outstanding
    // synchronization before beginning the specialized asymmetric
    // synchronization.
    auto preBarrier = builder.create<gpu::BarrierOp>(loc);
    preBarrier->moveBefore(forOp);
    builder.setInsertionPointAfter(preBarrier);

    // Insert condbarrier::second_half before starting the loop
    // FIXME : correctly calculate numbers per the arch
    auto i32ty = builder.getIntegerType(32);
    auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
    auto constZero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    auto constWarpSize = builder.create<arith::ConstantIntOp>(loc, 256, 32);
    auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWarpSize);
    auto warpLow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 warpIDX, constZero);
    auto warpHigh = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  warpIDX, constZero);

    builder.create<mlir::triton::amdgpu::CondBarrierOp>(loc, warpHigh);

    // Insert condbarrier::first_half after the end of the loop
    builder.setInsertionPointAfter(forOp);
    builder.create<mlir::triton::amdgpu::CondBarrierOp>(loc, warpLow);

    // 2. Collect existing barrier information.
    SmallVector<Block *> clusterBlocks;
    SmallVector<Operation *> clusterOps;
    SmallVector<bool> bars;
    std::map<int, Operation *> existingBarrierMap;
    Operation *terminatorOp = nullptr;

    for (auto &op : *forOp.getBody()) {
      if (auto exeOp = dyn_cast<scf::ExecuteRegionOp>(op)) {
        // fail conversion with executeRegion from unkown source.
        if (exeOp->getAttr("triton.warp_pipeline.stage") == nullptr)
          return failure();
        exeOp.setNoInline(false);
        clusterOps.push_back(&op);
        clusterBlocks.push_back(&exeOp->getRegion(0).front());
        bars.push_back(false);
      } else if (isa<ROCDL::BarrierOp, ROCDL::SBarrierOp,
                     triton::gpu::AsyncWaitOp, triton::amdgpu::AsyncTDMWait>(
                     op)) {
        int currCluster = clusterBlocks.size();
        if (existingBarrierMap.find(currCluster) != existingBarrierMap.end())
          return failure(); // Unreachable
        existingBarrierMap[currCluster] = &op;
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        terminatorOp = &op;
      } else { // fail conversion if any other op found out out of the cluster.
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

    // 3. Dependency check from node 'src' to 'next'
    for (int offset = 0; offset < numClusters; offset++) {
      for (int src = 0; src < numClusters; src++) {
        const int next = (src + 2 + offset) % numClusters;
        const int barrierLoc = (src + 1 + offset) % numClusters;

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
        if (isSynced())
          continue;

        const bool needFence =
            clusterInfo[src].isIntersected(clusterInfo[next], nullptr);
        // insert fence/barrier in front of this cluster
        if (needFence) {
          bars[barrierLoc] = true;
          LDBG("cluster " << src << " need fence to " << next
                          << " placing barrier at " << barrierLoc);
        }
      }
    }

    // 4. Finally insert cluster barriers.
    for (int i = 0; i < numClusters; i++) {
      if (auto exBar = existingBarrierMap.find(i);
          exBar != existingBarrierMap.end()) {
        auto exBarOp = exBar->second;
        if (bars[i]) {
          builder.setInsertionPointAfter(exBarOp);
          emitClusterBarrier(builder, loc, /*needLocal=*/true);
          if (!isa<triton::gpu::AsyncWaitOp, triton::amdgpu::AsyncTDMWait>(
                  exBarOp))
            builder.eraseOp(exBarOp);
        } else { // wrap with sched barrier
          builder.setInsertionPoint(exBarOp);
          builder.create<ROCDL::SchedBarrier>(loc, 0);
          builder.setInsertionPointAfter(exBarOp);
          builder.create<ROCDL::SchedBarrier>(loc, 0);
        }
      } else {
        builder.setInsertionPoint(clusterOps[i]);
        // The first one wraps back to the last of the loop
        if (i == 0 && topBar == existingBarrierMap.end())
          // inserts just before yield (=End of the loop).
          builder.setInsertionPoint(terminatorOp);
        emitClusterBarrier(builder, loc, /*needLocal=*/bars[i]);
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

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertPipelinedForPattern>(&getContext(), moduleAllocation);
    patterns.add<InlineWarpPipelineExecuteRegionPattern>(&getContext());

    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::AMD {
std::unique_ptr<OperationPass<ModuleOp>> createConvertWarpPipelinePass() {
  return std::make_unique<ConvertWarpPipeline>();
}
} // namespace mlir::triton::AMD
