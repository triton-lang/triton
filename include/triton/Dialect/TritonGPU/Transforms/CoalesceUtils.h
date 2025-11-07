
#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_

#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::gpu {
void setCoalescedEncoding(MLIRContext *context,
                          ModuleAxisInfoAnalysis &axisInfoAnalysis,
                          Operation *op, int numWarps, int threadsPerWarp,
                          triton::gpu::CTALayoutAttr CTALayout,
                          SmallVector<int64_t> shapePerCTA,
                          llvm::MapVector<Operation *, Attribute> &layoutMap);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
