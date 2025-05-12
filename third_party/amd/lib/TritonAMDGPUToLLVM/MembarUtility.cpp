#include "third_party/amd/include/TritonAMDGPUToLLVM/MembarUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
namespace {

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
    if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
      if (!destOperandFromAsyncWait(br.getDestOperands()))
        return false;
    } else if (auto condBr = dyn_cast<cf::CondBranchOp>(terminator)) {
      if (condBr.getTrueDest() == block) {
        if (!destOperandFromAsyncWait(condBr.getTrueDestOperands()))
          return false;
      }
      if (condBr.getFalseDest() == block) {
        if (!destOperandFromAsyncWait(condBr.getFalseDestOperands()))
          return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

// Returns true if one of the operands is a LocalLoad synced via AsyncWait.
bool filterAsyncLocalLoadsDeppendencies(Operation *op1, Operation *op2) {
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::amdgpu::BufferLoadToLocalOp>(op);
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    if (!localLoad)
      return false;
    Value token = localLoad.getToken();
    if (!token || !comesFromAsyncWait(token))
      return false;
    return true;
  };

  // Early return if neither or both operands are an AsyncLoad
  if (isAsyncLoad(op1) == isAsyncLoad(op2)) {
    return false;
  }

  return isLocalLoadWithAsyncWaitToken(op1) ||
         isLocalLoadWithAsyncWaitToken(op2);
};
} // namespace

bool membarFilter(Operation *op1, Operation *op2) {
  return filterAsyncLocalLoadsDeppendencies(op1, op2);
}
} // namespace mlir::triton::AMD
