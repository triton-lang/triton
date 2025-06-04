#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
namespace {
constexpr const char *syncedViaAsyncWaitAttrName =
    "ttg.amdgpu.syncedViaAsyncWait";

// Traverses the def-chain including control flow of the token and returns true
// if all defining operations are an AsyncWait
bool comesFromAsyncWait(Value token) {
  if (auto defOp = token.getDefiningOp()) {
    return isa<triton::gpu::AsyncWaitOp>(defOp);
  }

  auto blockArg = dyn_cast<BlockArgument>(token);
  // If the token has no defining op and is not an BlockArgument bail out
  if (!blockArg) {
    return false;
  }

  auto block = blockArg.getOwner();
  auto argId = blockArg.getArgNumber();

  auto destOperandFromAsyncWait = [argId](auto &&operands) {
    assert(argId < operands.size());
    return comesFromAsyncWait(operands[argId]);
  };

  // Check all predecessor block's terminator and follow the passed value at
  // argId to see if they are immediately an AsyncWait.
  for (auto *pred : block->getPredecessors()) {
    auto terminator = pred->getTerminator();
    if (auto br = dyn_cast<BranchOpInterface>(terminator)) {
      for (auto successor : llvm::enumerate(br->getSuccessors())) {
        if (block != successor.value())
          continue;
        auto operands = br.getSuccessorOperands(successor.index());
        if (!destOperandFromAsyncWait(operands))
          return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

// Returns true if one of the operands is a LocalLoad synced via AsyncWait.
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2) {
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::amdgpu::BufferLoadToLocalOp>(op);
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    return localLoad && isSyncedViaAsyncWait(localLoad);
  };

  // Early return if neither or both operands are an AsyncLoad
  if (isAsyncLoad(op1) == isAsyncLoad(op2)) {
    return false;
  }

  return isLocalLoadWithAsyncWaitToken(op1) ||
         isLocalLoadWithAsyncWaitToken(op2);
};
} // namespace

void annotateLocalLoadsSyncedViaAsyncWait(ModuleOp mod) {
  SmallVector<triton::gpu::LocalLoadOp> localLoads;
  mod->walk([&](triton::gpu::LocalLoadOp localLoadOp) {
    localLoads.emplace_back(localLoadOp);
  });

  auto *ctx = mod->getContext();
  for (auto &loadOp : localLoads) {
    auto token = loadOp.getToken();
    bool isSyncedViaAsyncWait = token && comesFromAsyncWait(token);
    loadOp->setAttr(syncedViaAsyncWaitAttrName,
                    BoolAttr::get(ctx, isSyncedViaAsyncWait));
  }
}

bool isSyncedViaAsyncWait(triton::gpu::LocalLoadOp localLoadOp) {
  auto attr = localLoadOp->getAttr(syncedViaAsyncWaitAttrName);
  if (!attr) {
    localLoadOp.emitRemark("has no async sync information attached to it which "
                           "might negatively affect performance. Run "
                           "annotateLocalLoadSyncedViaAsyncWait first");
    return false;
  }
  return cast<BoolAttr>(attr).getValue();
}

bool membarFilter(Operation *op1, Operation *op2) {
  return filterAsyncLocalLoadsDependencies(op1, op2);
}
} // namespace mlir::triton::AMD
