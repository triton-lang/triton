#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createCombineOpsPass();

<<<<<<< HEAD
std::unique_ptr<Pass> createRewriteTensorPointerPass(int computeCapability = 80,
                                                     bool isROCM = false);
=======
std::unique_ptr<Pass> createReorderBroadcastPass();

std::unique_ptr<Pass>
createRewriteTensorPointerPass(int computeCapability = 80);
>>>>>>> 5df904233c11a65bd131ead7268f84cca7804275

} // namespace triton

#define GEN_PASS_REGISTRATION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

} // namespace mlir

#endif
