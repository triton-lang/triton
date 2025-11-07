
#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_

namespace mlir::triton::gpu {
using CTALayoutProvider =
      llvm::function_ref<triton::gpu::CTALayoutAttr(RankedTensorType refType)>;

using ShapeProvider =
      llvm::function_ref<SmallVector<int64_t>(RankedTensorType refType, triton::gpu::CTALayoutAttr ctaLayout)>;

void setCoalescedEncoding(MLIRContext *context, 
    ModuleAxisInfoAnalysis &axisInfoAnalysis, 
    Operation *op,
    int numWarps, 
    int threadsPerWarp,
    CTALayoutProvider CTALayoutProvider,
    ShapeProvider shapeProvider,
    llvm::MapVector<Operation *, Attribute> &layoutMap);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_COALESCINGUTILS_H_