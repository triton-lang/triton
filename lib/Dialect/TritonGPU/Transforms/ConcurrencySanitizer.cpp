#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUCONCURRENCYSANITIZER
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class ConcurrencySanitizerPass
    : public impl::TritonGPUConcurrencySanitizerBase<ConcurrencySanitizerPass> {
public:
  void runOnOperation() override { ModuleOp module = getOperation(); }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
