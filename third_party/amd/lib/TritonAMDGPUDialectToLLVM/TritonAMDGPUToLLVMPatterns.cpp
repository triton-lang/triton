#include "third_party/amd/include/TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace mlir::triton::AMD {
void populateTritonAMDGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const AMD::TargetInfo &targetInfo,
                                        PatternBenefit benefit) {
  populateExtractSliceOpToLLVMPatterns(typeConverter, patterns, benefit);
  populateInThreadTransposeOpToTTGPatterns(patterns, benefit);
  populateConcatOpToLLVMPatterns(typeConverter, patterns, benefit);
  populateScaledUpcastOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                       benefit);
}
} // namespace mlir::triton::AMD
