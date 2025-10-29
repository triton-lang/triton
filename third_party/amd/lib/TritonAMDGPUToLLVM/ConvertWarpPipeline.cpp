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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
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

class ConvertWarpPipeline
    : public mlir::triton::impl::ConvertWarpPipelineBase<ConvertWarpPipeline> {

  void emitClusterBarrier(OpBuilder &b, Location loc, bool needLocal) {
    b.create<ROCDL::SchedBarrier>(loc, 0);
    if (needLocal)
      b.create<mlir::triton::gpu::LocalBarrierOp>(loc);
    else
      b.create<ROCDL::SBarrierOp>(loc);
    b.create<ROCDL::SchedBarrier>(loc, 0);
  }

  void emitPipelinedFor(OpBuilder &builder, Location loc, scf::ForOp forOp,
                        Allocation *allocation) {

    // 1. Insert conditional branch first,
    builder.setInsertionPointAfter(forOp);
    // Set barrier before starting the loop. This resolves any outstanding
    // synchronization before beginning the specialized asymmetric
    // synchronization.
    auto preBarrier = builder.create<gpu::BarrierOp>(loc);
    preBarrier->moveBefore(forOp);
    builder.setInsertionPointAfter(preBarrier);

    // Insert condbarrier::second_half before starting the loop
    // FIXME : correctly calculate numbers by the given num_warps.
    auto i32ty = builder.getIntegerType(32);
    auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
    auto constZero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    auto constWarpSize = builder.create<arith::ConstantIntOp>(loc, 256, 32);
    auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWarpSize);
    auto warpLow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 warpIDX, constZero);
    auto warpHigh = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  warpIDX, constZero);

    auto condBarrierHigh =
        builder.create<mlir::triton::amdgpu::CondBarrierOp>(loc, warpHigh);

    // Insert condbarrier::first_half after the end of the loop
    builder.setInsertionPointAfter(forOp);
    auto condBarrierLow =
        builder.create<mlir::triton::amdgpu::CondBarrierOp>(loc, warpLow);

    SmallVector<Block *> clusterBlocks;
    SmallVector<Operation *> clusterOps;
    SmallVector<bool> bars;
    std::map<int, Operation *> existingBarrierMap;
    Operation *terminatorOp;

    // 2. Collect existing barrier information.
    for (auto &op : *forOp.getBody()) {
      if (auto exeOp = dyn_cast<scf::ExecuteRegionOp>(op)) {
        // fail conversion with executeRegion from unkown source.
        if (exeOp->getAttr("triton.warp_pipeline.stage") == nullptr)
          return;
        exeOp.setNoInline(false);
        clusterOps.push_back(&op);
        clusterBlocks.push_back(&exeOp->getRegion(0).front());
        bars.push_back(false);
      } else if (isa<ROCDL::BarrierOp, ROCDL::SBarrierOp,
                     triton::gpu::AsyncWaitOp, triton::amdgpu::AsyncTDMWait>(
                     op)) {
        int currCluster = clusterBlocks.size();
        if (existingBarrierMap.find(currCluster) != existingBarrierMap.end())
          return; // Unreachable

        existingBarrierMap[currCluster] = &op;
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        terminatorOp = &op;
      } else // fail conversion if any other op found out out of the cluster.
        return;
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
        return; // Unreachable
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
        if (needFence) {
          // insert fence/barrier before this cluster
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
        if (bars[i]) {
          auto exBarOp = exBar->second;
          builder.setInsertionPointAfter(exBarOp);
          emitClusterBarrier(builder, loc, true);
          if (!isa<triton::gpu::AsyncWaitOp, triton::amdgpu::AsyncTDMWait>(
                  exBarOp))
            exBarOp->erase();
        } // else do nothing.
      } else {
        builder.setInsertionPoint(clusterOps[i]);
        // The first one wraps back to the last of the loop
        if (i == 0 && topBar == existingBarrierMap.end())
          // inserts just before yield.
          builder.setInsertionPoint(terminatorOp);
        emitClusterBarrier(builder, loc, bars[i]);
      }
    }
  }

public:
  ConvertWarpPipeline() : ConvertWarpPipelineBase<ConvertWarpPipeline>() {}

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m);
    ModuleAllocation moduleAllocation(m);

    for (auto funcOp : m.getOps<mlir::triton::FuncOp>()) {
      Allocation *allocation = moduleAllocation.getFuncData(funcOp);
      funcOp.walk([&](scf::ForOp forOp) {
        if (auto totalStages =
                forOp->getAttr("triton.warp_pipeline.total_stages")) {
          Location loc = forOp.getLoc();
          emitPipelinedFor(builder, loc, forOp, allocation);
        }
      });
    }
  }
};

} // namespace

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>> createConvertWarpPipelinePass() {
  return std::make_unique<ConvertWarpPipeline>();
}

} // namespace mlir::triton::AMD
