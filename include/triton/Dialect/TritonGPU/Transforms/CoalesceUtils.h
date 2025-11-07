
#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_

namespace mlir {
namespace triton {
namespace gpu {
void setCoalescedEncoding(MLIRContext *context, ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                        int numWarps, int threadsPerWarp,
                        llvm::MapVector<Operation *, Attribute> &layoutMap);
} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_