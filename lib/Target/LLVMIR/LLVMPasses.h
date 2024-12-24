#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

// Pass to pre-process LLVM IR before optimization and break up phi of struct.
// Breaking up those phis into elementary types allows better optimizations
// downstream.
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static StringRef name() { return "BreakStructPhiNodesPass"; }
};

// Pass to pre-process LLVM IR before optimization to rewrite certain inline asm
// instructions to make for slightly better patched code.:
struct InlineAsmRewritePass : PassInfoMixin<InlineAsmRewritePass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static StringRef name() { return "InlineAsmRewritePass"; }
};
} // namespace llvm
