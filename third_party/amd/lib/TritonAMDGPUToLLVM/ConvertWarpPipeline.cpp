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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

//#include "Allocation.h"
#include "triton/Analysis/Membar.h"
//#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
//#include "mlir/Interfaces/ControlFlowInterfaces.h"



#define DEBUG_TYPE "convert-warp-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTWARPPIPELINE
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

static BlockInfo buildBlockInfoFromBlock(Block *block,
                                         Allocation *allocation) {
  BlockInfo info; // running fact for this block

  for (Operation &opRef : *block) {
    Operation *op = &opRef;

    // Existing CTA barrier: clear any pending deps.
    if (isa<gpu::BarrierOp>(op)) {
      LDBG("synced by barrier ...");
      info.sync();
      continue;
    }

    //BlockInfo cur; // facts contributed by this single op

    // Inter-procedural summary: calls contribute their callee summary.
    if (isa<triton::CallOp>(op)) {
      // assert 0
    } else {
      // Intra-procedural memory effects tied to concrete Values.
      if (auto mei = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effs;
        mei.getEffects(effs);
        for (auto &eff : effs) {
          //op->dump();
          if (Value v = eff.getValue()) {
            for (auto bufId : allocation->getBufferIds(v)) {
              if (bufId == Allocation::InvalidBufferId) continue;
              auto interval = allocation->getAllocatedInterval(bufId);
              if (isa<MemoryEffects::Write>(eff.getEffect())) {
                info.syncWriteIntervals[interval].insert(op);
              } else if (isa<MemoryEffects::Read>(eff.getEffect())) {
                info.syncReadIntervals[interval].insert(op);
              }
              //info.dump();
            }
          }
        }
      }
    }

    // Merge this op's fact into the running block fact.
    //info.join(cur);
  }
  //LDBG("returning ...");
  //info.dump();
  return info;
}


class ConvertWarpPipeline
    : public mlir::triton::impl::ConvertWarpPipelineBase<ConvertWarpPipeline> {

void emitClusterBarrier(OpBuilder &b, Location loc){
  b.create<ROCDL::SchedBarrier>(loc, 0);
  b.create<ROCDL::SBarrierOp>(loc);
  b.create<ROCDL::SchedBarrier>(loc, 0);
}

    
void emitPipelinedFor(OpBuilder &builder, Location loc, scf::ForOp forOp, Allocation *allocation) {

  //insert cond branch first,
  builder.setInsertionPointAfter(forOp);
  // Set barrier before starting the loop. This resolves any remaining required
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

  // duplicate condBarrier for lead_stages
  auto condBarrierHigh =
      builder.create<mlir::triton::amdgpu::CondBarrierOp>(loc, warpHigh);

  // Insert condbarrier::first_half after the end of the loop
  builder.setInsertionPointAfter(forOp);
  auto condBarrierLow = builder.create<mlir::triton::amdgpu::CondBarrierOp>(loc, warpLow);

  // in case a loop begins with a barrier
  bool barrierAtTop = false;
  
  SmallVector<Block *> clusterBlocks;

  //insert barrier whe.n needed.
  bool needBarrier = false;
  for (auto &op : *forOp.getBody()) {
    if(auto exeOp = dyn_cast<scf::ExecuteRegionOp>(op)) {
      clusterBlocks.push_back(&exeOp->getRegion(0).front());
      if (needBarrier){
        //insert a barrier
        auto prevOp = exeOp->getPrevNode();
        builder.setInsertionPointAfter(prevOp);
        emitClusterBarrier(builder, loc);
      }
      else{
        needBarrier = true;
      }
    }
    // skip inserting a barrier if there's one already
    else if (isa<ROCDL::BarrierOp, ROCDL::SBarrierOp, triton::gpu::AsyncWaitOp>(op))
      needBarrier = false;
    else if(auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      if (needBarrier && !barrierAtTop){
        //insert a barrier
        auto prevOp = yieldOp->getPrevNode();
        builder.setInsertionPointAfter(prevOp);
        emitClusterBarrier(builder, loc);
      }
    }

  }


  SmallVector<BlockInfo> clusterInfo;
  for (auto cb: clusterBlocks)
    clusterInfo.push_back(buildBlockInfoFromBlock(cb, allocation));

  //bool needFence = clusterInfo[0].isIntersected(clusterInfo[1], nullptr);
  LDBG("cluster dependency analysis");
  int numClusters = clusterInfo.size();
  LDBG("total clusters : " << numClusters);

  bool bars[numClusters];
  for (int i=0; i<numClusters; i++)
    bars[i] = false;

  for (int j=0; j<numClusters; j++){
    //clusterInfo[i].dump();
    for (int i=0; i<numClusters; i++){
      
      int next = (i+2+j)%numClusters;
      //LDBG("checking " << i << " to "<< next);
      int curr = (i+1)%numClusters;
      bool synced = false;
      while (curr != i && curr != next){
        if (bars[curr]){
          synced = true;
          break;
        }
        curr = (curr+1)%numClusters;
      }
      // also if next can already have a fence 
      if (bars[next])
        synced = true;

      // synced between i and j, no need to check.
      if (synced)
        continue;

      bool needFence = clusterInfo[i].isIntersected(clusterInfo[next], nullptr);
      if (needFence){
        // insert fence/barrier before this cluster
        bars[next] = true;
        
        LDBG("cluster " << i << " need fence to "<< next);
        clusterInfo[i].dump();
        clusterInfo[next].dump();
      }
      for (int i=0; i<numClusters; i++)
        LDBG("bars [" << i << "] = "<< bars[i]);
    }
  }

}

public:
  ConvertWarpPipeline()
      : ConvertWarpPipelineBase<ConvertWarpPipeline>() {

  }

  void runOnOperation() override {

    LDBG("cluster dependency analysis open");

    ModuleOp m = getOperation();
    OpBuilder builder(m);
    ModuleAllocation moduleAllocation(m);
    
    for (auto funcOp : m.getOps<mlir::triton::FuncOp>()) {
      Allocation *allocation = moduleAllocation.getFuncData(funcOp);
      funcOp.walk([&](scf::ForOp forOp) {
        if (auto totalStages = forOp->getAttr("total_stages")){
          Location loc = forOp.getLoc();
          emitPipelinedFor(builder, loc, forOp, allocation);
        }
      });
    }
    LDBG("cluster dependency analysis close");
  }
};

} // namespace

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertWarpPipelinePass() {
  return std::make_unique<ConvertWarpPipeline>();
}

} // namespace mlir::triton::AMD

