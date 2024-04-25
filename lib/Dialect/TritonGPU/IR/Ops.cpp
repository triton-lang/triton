
#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

///--- LocalLoadOp ---
void LocalLoadOp::build(OpBuilder &builder, OperationState &state, Type result,
                        Value src) {
  LocalLoadOp::build(builder, state, result, src, /*mask=*/{}, /*other=*/{});
}
} // namespace gpu
} // namespace triton
} // namespace mlir
