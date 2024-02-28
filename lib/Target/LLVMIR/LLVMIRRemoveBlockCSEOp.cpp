#include "LLVMPasses.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static bool runOnFunction(Function &F) {
  bool Changed = false;
  SmallVector<CallInst *, 16> ToErase;
  for (BasicBlock &BB : F) {
    for (Instruction &Inst : BB) {
      if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
        if (CI->getCalledFunction() == nullptr) {
          continue;
        }
        if (CI->getCalledFunction()->getName() == "__triton_block_cse") {
          CI->replaceAllUsesWith(CI->getArgOperand(0));
          ToErase.push_back(CI);
          Changed = true;
        }
      }
    }
  }
  for (CallInst *callInst : ToErase) {
    callInst->eraseFromParent();
  }
  return Changed;
}

PreservedAnalyses RemoveBlockCSEPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  bool b = runOnFunction(F);
  return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
