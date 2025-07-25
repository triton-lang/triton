#ifndef NVIDIA_NVWS_TRANSFORMS_UTILITY_H_
#define NVIDIA_NVWS_TRANSFORMS_UTILITY_H_

#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace nvws {

static const char *kArefTagAttrName = "aref_tag";

Operation *createAlloc(OpBuilder &builder, Location loc,
                       gpu::MemDescType memDescType, Value src);

ArefCreateOp createArefCreateOp(OpBuilder &builder,
                                const SmallVector<Type> &arefTypes,
                                const SmallVector<Value> &allocOps,
                                Location loc);
} // namespace nvws
} // namespace triton
} // namespace mlir

#endif // NVIDIA_NVWS_TRANSFORMS_UTILITY_H_
