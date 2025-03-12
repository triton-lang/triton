#include "third_party/amd/include/TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace mlir::triton::AMD {
void populateTritonAMDGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit) {
  populateExtractSliceOpToLLVMPatterns(typeConverter, patterns, benefit);
  MLIRContext *ctx = &typeConverter.getContext();
  populateInThreadTransposeOpToTTGPatterns(ctx, patterns, benefit);
}
} // namespace mlir::triton::AMD
