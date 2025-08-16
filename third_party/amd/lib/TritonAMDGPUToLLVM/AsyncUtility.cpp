#include "AsyncUtility.h"

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TargetInfo.h"
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
} // namespace

void annotateLocalLoadsSyncedViaAsyncWait(ModuleOp mod) {
  SmallVector<triton::gpu::LocalLoadOp> localLoads;
  mod->walk([&](triton::gpu::LocalLoadOp localLoadOp) {
    localLoads.emplace_back(localLoadOp);
  });

  auto *ctx = mod->getContext();
  for (auto &loadOp : localLoads) {
    auto token = loadOp.getToken();
    if (loadOp->hasAttr(syncedViaAsyncWaitAttrName))
      continue;

    bool isSyncedViaAsyncWait = token && comesFromAsyncWait(token);
    loadOp->setAttr(syncedViaAsyncWaitAttrName,
                    BoolAttr::get(ctx, isSyncedViaAsyncWait));
  }
}

bool isSyncedViaAsyncWait(Operation *op) {
  assert(op);

  auto attr = op->getAttr(syncedViaAsyncWaitAttrName);
  if (!attr) {
    op->emitRemark("has no async sync information attached to it which "
                   "might negatively affect performance. Run "
                   "annotateLocalLoadSyncedViaAsyncWait first");
    return false;
  }
  return cast<BoolAttr>(attr).getValue();
}

namespace {
LLVM::AliasScopeDomainAttr getLoadScopeDomain(MLIRContext *ctx) {
  Builder b(ctx);
  return b.getAttr<LLVM::AliasScopeDomainAttr>(
      b.getStringAttr("amdgpu.AsyncOps"),
      b.getStringAttr(
          "Domain to hold alias scopes to specify aliasing information between "
          "AsyncCopyGlobalToLocal, BufferLoadToLocal and LocalLoad ops"));
}

LLVM::AliasScopeAttr getAsyncCopyScope(MLIRContext *ctx) {
  Builder b(ctx);
  auto name = b.getStringAttr("amdgpu.AsyncCopies");
  auto desc = b.getStringAttr(
      "Scope containing all AsyncCopyGlobalToLocal and BufferLoadToLocal ops");
  return b.getAttr<LLVM::AliasScopeAttr>(name, getLoadScopeDomain(ctx), desc);
}

LLVM::AliasScopeAttr getLoadCopyScope(MLIRContext *ctx) {
  Builder b(ctx);
  auto name = b.getStringAttr("amdgpu.LocalLoads");
  auto desc = b.getStringAttr("Scope containing all LocalLoad ops");
  return b.getAttr<LLVM::AliasScopeAttr>(name, getLoadScopeDomain(ctx), desc);
}
} // namespace

void addAsyncCopyAliasScope(LLVM::AliasAnalysisOpInterface directToLdsOp) {
  auto ctx = directToLdsOp->getContext();
  Builder b(ctx);
  directToLdsOp.setAliasScopes(b.getArrayAttr(getAsyncCopyScope(ctx)));
}

void addLocalLoadNoAliasScope(triton::gpu::LocalLoadOp localLoadOp,
                              LLVM::AliasAnalysisOpInterface llLoadOp) {
  if (!isSyncedViaAsyncWait(localLoadOp))
    return;

  return addLocalLoadNoAliasScope(llLoadOp);
}

void addLocalLoadNoAliasScope(LLVM::AliasAnalysisOpInterface llLoadOp) {
  auto ctx = llLoadOp->getContext();

  // Do not alias with AsyncCopies
  auto noAliasScopes = ArrayAttr::get(ctx, getAsyncCopyScope(ctx));
  llLoadOp.setNoAliasScopes(noAliasScopes);

  // Add to different scope as ops without any scope alias with everything
  auto aliasScopes = ArrayAttr::get(ctx, getLoadCopyScope(ctx));
  llLoadOp.setAliasScopes(aliasScopes);
}

unsigned
fitToValidDirectToLdsVecSize(unsigned maxVecSize, unsigned elemBitwidth,
                             const triton::AMD::TargetInfo &targetInfo) {
  while (maxVecSize > 0 && !targetInfo.supportsDirectToLdsLoadBitWidth(
                               maxVecSize * elemBitwidth)) {
    maxVecSize /= 2;
  }
  return maxVecSize;
}

} // namespace mlir::triton::AMD
