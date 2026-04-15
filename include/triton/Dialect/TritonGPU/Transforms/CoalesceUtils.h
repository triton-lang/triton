
#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_

#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::triton::gpu {
class CoalesceSliceCache {
public:
  ArrayRef<Operation *> getSlice(Operation *op);

private:
  llvm::DenseMap<Operation *, unsigned> opToSliceIndex;
  SmallVector<SmallVector<Operation *>> slices;
};

BlockedEncodingAttr buildCoalescedEncoding(
    ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op, int numWarps,
    int threadsPerWarp, triton::gpu::CGAEncodingAttr cgaLayout,
    SmallVector<int64_t> shapePerCTA, CoalesceSliceCache *sliceCache = nullptr);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
