
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <deque>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

/// Dialect plugin registration mechanism.
/// Observe that it also allows to register passes.
/// Necessary symbol to register the dialect plugin.
// extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
// mlirGetDialectPluginInfo() {
//   return {MLIR_PLUGIN_API_VERSION, "Standalone", LLVM_VERSION_STRING,
//           [](DialectRegistry *registry) {
//             registry->insert<mlir::standalone::StandaloneDialect>();
//             mlir::standalone::registerPasses();
//           }};
// }

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUHELLOEXTENSION
#include "Passes.h.inc"

namespace {

class HelloExtension : public OpRewritePattern<DotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    return success();
  }
};

} // anonymous namespace

struct HelloExtensionPass : public impl::TritonGPUHelloExtensionBase<HelloExtensionPass> {
  void runOnOperation()
    // override
    {
    // MLIRContext *context = &getContext();
    // ModuleOp m = getOperation();
    // RewritePatternSet decomposePatterns(context);
    // decomposePatterns.add<HelloExtension>(context);
    // if (applyPatternsGreedily(m, std::move(decomposePatterns)).failed()) {
    //   signalPassFailure();
    // }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir


inline void registerStandaloneSwitchBarFoo() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::gpu::createTritonGPUHelloExtension();
  });
}

inline void registerPasses() {
  registerStandaloneSwitchBarFoo();
}


/// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "HelloExtensionPlugin", LLVM_VERSION_STRING,
          []() { registerPasses(); }};
}
