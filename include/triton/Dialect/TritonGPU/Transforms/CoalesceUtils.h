
#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_

#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {
// Compute dependency information once per function, before changing its IR.
class CoalescedEncodingAnalysis {
public:
  explicit CoalescedEncodingAnalysis(ModuleAxisInfoAnalysis &axisInfoAnalysis)
      : axisInfoAnalysis(axisInfoAnalysis) {}

  BlockedEncodingAttr getEncoding(Operation *op, int numWarps,
                                  int threadsPerWarp, CGAEncodingAttr cgaLayout,
                                  ArrayRef<int64_t> shapePerCTA);

private:
  void computeSliceInfo(FunctionOpInterface func);
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
  DenseMap<Operation *, unsigned> elementsPerThread;
};
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
