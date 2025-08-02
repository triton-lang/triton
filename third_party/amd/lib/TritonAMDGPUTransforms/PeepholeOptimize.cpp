#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "triton/Analysis/Utility.h"

using namespace mlir::triton;

namespace mlir::triton::AMD {

void populatePeepholeOptimizationPatterns(RewritePatternSet &patterns,
                                          const AMD::TargetInfo &targetInfo,
                                          PatternBenefit benefit);
}

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDPEEPHOLEOPTIMIZE
#include "TritonAMDGPUTransforms/Passes.h.inc"

struct TritonAMDPeepholeOptimize
    : impl::TritonAMDPeepholeOptimizeBase<TritonAMDPeepholeOptimize> {
  using Base::Base;

  void runOnOperation() override {
    auto mod = getOperation();
    AMD::TargetInfo targetInfo(this->archGenerationName.getValue());
    if (targetInfo.getISAFamily() == AMD::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '")
          << this->archGenerationName.getValue() << "'";
      return signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    AMD::populatePeepholeOptimizationPatterns(patterns, targetInfo, 1);
    (void)applyPatternsGreedily(mod, std::move(patterns));
  }
};

} // namespace mlir

