#include "LLVMPasses.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static bool runOnFunction(Function &F) {
  bool Changed = false;
  for (BasicBlock &BB : F) {
    for (Instruction &inst : BB) {
      if (CallInst *callInst = dyn_cast<CallInst>(&inst)) {
        if (callInst->getCalledFunction()->getName() == "__triton_block_cse") {
          callInst->replaceAllUsesWith(callInst->getArgOperand(0));
          callInst->eraseFromParent();
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

PreservedAnalyses RemoveBlockCSEPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  bool b = runOnFunction(F);
  return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
