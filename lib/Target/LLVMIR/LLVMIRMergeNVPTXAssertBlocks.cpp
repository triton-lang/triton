#include "LLVMPasses.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

namespace llvm {
PreservedAnalyses MergeNVPTXAssertBlocksPass::run(Function &function,
                                                  FunctionAnalysisManager &) {
  const Function *report = function.getParent()->getFunction("__assertfail");
  if (!report || !report->isDeclaration() || !report->doesNotReturn())
    return PreservedAnalyses::all();

  // A statically unrolled or pipelined kernel can check the same assertion
  // hundreds of times. Share its terminal reporting block across those edges.
  // Matching the complete call preserves the message, file, line and ABI.
  DenseMap<size_t, SmallVector<CallInst *, 1>> seen;
  SmallVector<BasicBlock *> redundant;
  for (BasicBlock &block : function) {
    if (block.isEntryBlock() || block.hasAddressTaken() || block.size() != 2)
      continue;
    auto *call = dyn_cast<CallInst>(&block.front());
    auto *end = dyn_cast<UnreachableInst>(block.getTerminator());
    if (!end || !call || call->getCalledFunction() != report ||
        call->hasOperandBundles() || call->isConvergent() ||
        call->hasMetadataOtherThanDebugLoc() ||
        end->hasMetadataOtherThanDebugLoc() || call->hasDbgRecords() ||
        end->hasDbgRecords())
      continue;
    hash_code key =
        hash_combine(call->getDebugLoc().get(), end->getDebugLoc().get());
    for (const Use &arg : call->args())
      key = hash_combine(key, arg.get());
    auto &candidates = seen[static_cast<size_t>(key)];
    auto found = llvm::find_if(candidates, [&](const CallInst *other) {
      return call->isIdenticalTo(other) &&
             call->getDebugLoc() == other->getDebugLoc() &&
             end->getDebugLoc() ==
                 other->getParent()->getTerminator()->getDebugLoc();
    });
    if (found == candidates.end()) {
      candidates.push_back(call);
      continue;
    }
    block.replaceAllUsesWith((*found)->getParent());
    redundant.push_back(&block);
  }
  for (BasicBlock *block : redundant)
    block->eraseFromParent();
  return redundant.empty() ? PreservedAnalyses::all()
                           : PreservedAnalyses::none();
}
} // namespace llvm
