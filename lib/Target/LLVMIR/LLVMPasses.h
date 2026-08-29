#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

// Pass to pre-process LLVM IR before optimization and break up phi of struct.
// Breaking up those phis into elementary types allows better optimizations
// downstream.
struct BreakStructPhiNodesPass
    : OptionalPassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static StringRef name() { return "BreakStructPhiNodesPass"; }
};

// Re-form vector additions when every lane of a vector multiply is extracted
// and added to the same scalar constant. Run after vectorization so NVPTX can
// lower the multiply-add pairs to packed operations.
struct VectorizeExtractedAddsPass
    : OptionalPassInfoMixin<VectorizeExtractedAddsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static StringRef name() { return "VectorizeExtractedAddsPass"; }
};

// Split FP32 arithmetic pairs with only one demanded lane on NVPTX. Run after
// vectorization, followed by InstSimplify to clean up extracts and repacking.
struct ScalarizePackedFOpsPass
    : OptionalPassInfoMixin<ScalarizePackedFOpsPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static StringRef name() { return "ScalarizePackedFOpsPass"; }
};

} // namespace llvm
