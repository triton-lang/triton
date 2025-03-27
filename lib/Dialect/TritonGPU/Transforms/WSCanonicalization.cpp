#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <set>

#include "mlir/IR/OperationSupport.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

#define DEBUG_TYPE "tritongpu-warp-spec-canonicalization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_TRITONGPUWSCANONICALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUWSCanonicalization
    : public impl::TritonGPUWSCanonicalizationBase<
          TritonGPUWSCanonicalization> {
public:
  using impl::TritonGPUWSCanonicalizationBase<
      TritonGPUWSCanonicalization>::TritonGPUWSCanonicalizationBase;

  void runOnOperation() override {
    if (numConsumerGroups == 0)
      return;

    // Find top-level ifOp that specializes warps. Such ifOp has the following
    // form:
    //  %51 = ttng.get_canonical_warp_id
    //  %52 = arith.divui %51, %c4_i32
    //  %53 = arith.cmpi eq, %52, %c0_i32
    //  scf.if %53 {
    //   ...
    //  }
    DenseMap<scf::IfOp, AsyncTaskId> ifOpToTaskId;
    getOperation()->walk([&](scf::IfOp ifOp) {
      // Skip ifOp that has more than one region
      if (ifOp.elseBlock())
        return;

      // Get the condition of scf.if
      Value cond = ifOp.getCondition();
      auto cmpOp = cond.getDefiningOp<arith::CmpIOp>();
      if (!cmpOp || cmpOp.getPredicate() != arith::CmpIPredicate::eq)
        return;

      // Ensure the LHS of comparison is from an arith.divui
      Value divResult = cmpOp.getLhs();
      auto divOp = divResult.getDefiningOp<arith::DivUIOp>();
      if (!divOp)
        return;

      // Ensure the RHS of comparison is a constant
      auto warpGroupId = cmpOp.getRhs().getDefiningOp<arith::ConstantIntOp>();
      if (!warpGroupId)
        return;

      // Ensure the divisor is 4
      auto divisorCst = divOp.getRhs().getDefiningOp<arith::ConstantIntOp>();
      if (!divisorCst || divisorCst.value() != 4)
        return;

      // Ensure the dividend is from ttng.get_canonical_warp_id
      Value warpId = divOp.getLhs();
      auto warpOp = warpId.getDefiningOp<ttng::GetCanonicalWarpIdOp>();
      if (!warpOp)
        return;

      // Al conditions matc
      LLVM_DEBUG({
        LDBG("Warp specialization region:");
        ifOp.dump();
      });

      auto asyncTaskIds = getAsyncTaskIds(ifOp);
      assert(asyncTaskIds.size() == 1 && "Expecting one async task id");
      auto taskId = asyncTaskIds[0];
      assert(taskId == warpGroupId.value() &&
             "Expecting task id to match warp group id");
      ifOpToTaskId[ifOp] = taskId;
    });

    // Fix up the async task ids for each op in the specialized region
    for (const auto &item : ifOpToTaskId) {
      auto ifOp = item.first;
      auto taskId = item.second;
      SmallVector<AsyncTaskId> regionTaskIds = {taskId};
      ifOp->walk([&](Operation *op) {
        // Fix up the async task ids
        if (getAsyncTaskIds(op) != regionTaskIds) {
          LLVM_DEBUG({
            LDBG("Fixing up async task ids to  " << taskId << " for ");
            op->dump();
          });
          setAsyncTaskIds(op, regionTaskIds);
        }
      });
    }
  }
};
} // namespace gpu
} // namespace triton
} // namespace mlir
