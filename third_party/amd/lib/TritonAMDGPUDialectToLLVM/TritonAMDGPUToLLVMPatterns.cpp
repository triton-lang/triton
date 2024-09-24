#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace mlir::triton::AMD {
void populateTritonAMDGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit) {
  // TODO: Insert TrtionAMDGPU dialect patterns.
}
} // namespace mlir::triton::AMD
