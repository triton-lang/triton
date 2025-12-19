
#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_

#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {
BlockedEncodingAttr buildCoalescedEncoding(
    MLIRContext *context, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    Operation *op, int numWarps, int threadsPerWarp,
    triton::gpu::CGAEncodingAttr cgaLayout, SmallVector<int64_t> shapePerCTA);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
