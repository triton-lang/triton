#include <triton/Dialect/TritonGPU/IR/Dialect.h>

namespace mlir::triton::gpu {

FailureOr<CTALayoutAttr> permuteCTALayout(CTALayoutAttr ctaLayout,
                                          ArrayRef<int> order);

}
