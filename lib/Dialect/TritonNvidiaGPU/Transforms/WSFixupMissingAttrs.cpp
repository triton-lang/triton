#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace mlir {

namespace ttng = triton::nvidia_gpu;

namespace {

class TritonGPUWSFixupMissingAttrsPass
    : public TritonGPUWSFixupMissingAttrsBase<
          TritonGPUWSFixupMissingAttrsPass> {
public:
  TritonGPUWSFixupMissingAttrsPass() = default;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!ttng::TritonNvidiaGPUDialect::getWSSupportedAttr(mod))
      return;
    OpBuilder builder(mod);
    mod->walk([&](mlir::triton::FuncOp funcOp) {
      for (Operation &op : funcOp.getBody().front().getOperations()) {
        if (!isa<scf::IfOp>(&op))
          continue;
        auto agentIds = getAgentIds(&op);
        if (agentIds.size() != 1)
          continue;
        Block *roleIdBlock = nullptr;
        op.walk<WalkOrder::PreOrder>([&](Operation *subOp) {
          setAgentIds(subOp, agentIds);
          // Find the outermost common block that has roleId.
          // The below implementation assumes that:
          // - all lock/unlock ops are in the same block (denoted as B).
          // - there is always one scf.if op in the front of `B` which has
          //   role id attached.
          // The above assumptions are maintained by WSMutex pass currently.
          if (!roleIdBlock && isa<scf::IfOp>(subOp) && getWSRoleId(subOp))
            roleIdBlock = subOp->getBlock();
        });
        if (!roleIdBlock)
          continue;
        int roleId = 0;
        for (Operation &roleOp : roleIdBlock->getOperations()) {
          auto optionalRoleId = getWSRoleId(&roleOp);
          if (!optionalRoleId) {
            setRoleId(&roleOp, roleId);
          } else {
            roleId = *optionalRoleId;
          }
          roleOp.walk([&](Operation *subOp) { setRoleId(subOp, roleId); });
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTritonNvidiaGPUWSFixupMissingAttrs() {
  return std::make_unique<TritonGPUWSFixupMissingAttrsPass>();
}

} // namespace mlir
