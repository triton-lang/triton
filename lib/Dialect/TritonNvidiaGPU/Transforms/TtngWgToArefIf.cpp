#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WSUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

#include <iostream>
#include <memory>
#include <optional>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritonnvidiagpu-ttng-wg-to-aref-if"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
namespace ttng = triton::nvidia_gpu;

class TritonNvidiaGPUTtngWgToArefIf
    : public TritonNvidiaGPUTtngWgToArefIfBase<TritonNvidiaGPUTtngWgToArefIf> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    // for each ttng.warp_group, generate an ifOp to wrap around ttng.warp_group
    //   ttng.wapr_group start_warp(n) num_warps(m)  { .. }
    //     becomes
    //   scf.if (wid >= n && wid < n + m) {
    //        ttng.wapr_group start_warp(n) num_warps(m)  { .. }
    //   }

    SmallVector<ttng::WarpGroupOp> wgOps = findWarpGroupOps(m);
    assert(wgOps.size() > 1);
    OpBuilder builder(wgOps[0]);

    for (auto wgOp : wgOps) {
      auto loc = wgOp.getLoc();

      // generate if (wgid >= start_warp && wgid < start_warp + num_warps)
      // Note: use nvvm dialect here
      builder.setInsertionPointAfter(wgOp);
      auto threadIdX =
          builder.create<NVVM::ThreadIdXOp>(loc, builder.getI32Type());
      auto c32 = builder.create<arith::ConstantIntOp>(loc, 32, 32);
      auto wgid = builder.create<arith::DivSIOp>(loc, threadIdX, c32);
      auto cond1 = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, wgid,
          builder.create<arith::ConstantIntOp>(loc, wgOp.getStartWarp(), 32));
      auto cond2 = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, wgid,
          builder.create<arith::ConstantIntOp>(
              loc, wgOp.getStartWarp() + wgOp.getNumWarps(), 32));
      auto cond = builder.create<arith::AndIOp>(loc, cond1, cond2);
      auto arefIfOp =
          builder.create<scf::IfOp>(loc, cond, /*withElseRegion=*/false);
      arefIfOp->setAttr(ATTR_WS_AREF_IF, builder.getUnitAttr());

      // move wgOp to thenRegion
      wgOp->moveBefore(arefIfOp.thenYield());
    }

  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUTtngWgToArefIfPass() {
  return std::make_unique<TritonNvidiaGPUTtngWgToArefIf>();
}
