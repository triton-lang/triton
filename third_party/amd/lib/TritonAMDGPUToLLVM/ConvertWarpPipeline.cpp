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

#define DEBUG_TYPE "convert-warp-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTWARPPIPELINE
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

class ConvertWarpPipeline
    : public mlir::triton::impl::ConvertWarpPipelineBase<ConvertWarpPipeline> {

void emitClusterBarrier(OpBuilder &b, Location loc){
  b.create<ROCDL::SchedBarrier>(loc, 0);
  b.create<ROCDL::SBarrierOp>(loc);
  b.create<ROCDL::SchedBarrier>(loc, 0);
}

    
void emitPipelinedFor(OpBuilder &builder, Location loc, scf::ForOp forOp) {

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
  
  //insert barrier when needed.
  bool needBarrier = false;
  for (auto &op : *forOp.getBody()) {
    if(auto exeOp = dyn_cast<scf::ExecuteRegionOp>(op)) {
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
}

public:
  ConvertWarpPipeline()
      : ConvertWarpPipelineBase<ConvertWarpPipeline>() {

  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m);
  
    for (auto funcOp : m.getOps<mlir::triton::FuncOp>()) {
      funcOp.walk([&](scf::ForOp forOp) {
        if (auto totalStages = forOp->getAttr("total_stages")){
          Location loc = forOp.getLoc();
          emitPipelinedFor(builder, loc, forOp);
        }
      });
    }
  }
};

} // namespace

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertWarpPipelinePass() {
  return std::make_unique<ConvertWarpPipeline>();
}

} // namespace mlir::triton::AMD

