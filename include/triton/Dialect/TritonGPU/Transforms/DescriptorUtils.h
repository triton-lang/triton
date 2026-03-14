#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_UTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_UTILS_H_

#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {

// Callback to apply a favorable shared encoding from the uses of the op
using FindEncodingFromUsersFn = llvm::function_ref<Attribute(Operation *)>;
// Provides a backend specific fallback encoding when no encoding is found
using GetFallbackSharedEncodingFn = llvm::function_ref<Attribute(
    RankedTensorType, CGAEncodingAttr, ArrayRef<int64_t>, unsigned)>;
// Callback to update the shared encoding for the shape of the descriptor
using UpdateEncodingForShapeFn = llvm::function_ref<SharedEncodingTrait(
    Operation *, SharedEncodingTrait, RankedTensorType)>;
// Callback Function to determine when to force default for an op
using IsForcedToDefaultFn = llvm::function_ref<bool(Operation *)>;

// Compute shared memory encodings for all descriptors on a per-function basis
void assignMemoryLayouts(ModuleOp &mod,
                         FindEncodingFromUsersFn findEncodingFromUsers,
                         GetFallbackSharedEncodingFn getFallbackSharedEncoding,
                         UpdateEncodingForShapeFn updateEncodingForShape,
                         IsForcedToDefaultFn isForcedToDefault);

CGAEncodingAttr updateCGALayoutForShape(CGAEncodingAttr cgaLayout,
                                        ArrayRef<int64_t> shape);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_UTILS_H_
