#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "third_party/amd/include/Utils/Utility.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

// In distributed layouts, tensors are divided into CTA tiles.
// A CTA tile represents the smallest contiguous portion of a tensor that is
// distributed across all threads and warps within a workgroup. The ExtractSlice
// operation extracts a portion of the tensor that is a multiple of CTA tiles.

namespace {

struct ExtractSliceOpConversion
    : public ConvertOpToLLVMPattern<amdgpu::ExtractSliceOp> {
  explicit ExtractSliceOpConversion(LLVMTypeConverter &typeConverter,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<amdgpu::ExtractSliceOp>(typeConverter, benefit) {
  }

  LogicalResult processLayout(amdgpu::ExtractSliceOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto srcTy = cast<RankedTensorType>(op.getSource().getType());
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcShape = srcTy.getShape();
    auto dstShape = dstTy.getShape();

    auto vals = unpackLLElements(loc, adaptor.getSource(), rewriter);
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(srcTy);
    auto srcCTAShape = LLVM::AMD::multiDimElementwise<int64_t, unsigned>(
        srcShape, shapePerCTATile, std::divides<unsigned>());
    auto dstCTAShape = LLVM::AMD::multiDimElementwise<int64_t, unsigned>(
        dstShape, shapePerCTATile, std::divides<unsigned>());

    auto numCTATiles = std::accumulate(dstCTAShape.begin(), dstCTAShape.end(),
                                       1, std::multiplies<>());
    auto offsets = op.getStaticOffsets();
    auto firstTileCoordinate =
        LLVM::AMD::multiDimElementwise<int64_t, unsigned>(
            offsets, shapePerCTATile, std::divides<unsigned>());

    Attribute srcEncoding = srcTy.getEncoding();
    Attribute dstEncoding = dstTy.getEncoding();
    auto linearLayoutSrc = triton::gpu::toLinearLayout(srcShape, srcEncoding);
    auto linearLayoutDst = triton::gpu::toLinearLayout(dstShape, dstEncoding);

    auto srcCTAOrder =
        LLVM::AMD::getCTATileOrder(srcTy.getContext(), linearLayoutSrc);
    auto dstCTAOrder =
        LLVM::AMD::getCTATileOrder(srcTy.getContext(), linearLayoutDst);

    unsigned elemsPerThreadPerCTA =
        triton::gpu::getTotalElemsPerThread(srcTy) /
        std::accumulate(srcCTAShape.begin(), srcCTAShape.end(), 1,
                        std::multiplies<>());

    // 1. Process CTA tiles in the destination tensor according to the
    // destination's linear layout order of CTA tiles.
    // 2. For each tile position in the destination tensor, compute its
    // corresponding position in the source tensor.
    // 3. Copy the values from the source tile to the destination slice.
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < numCTATiles; i++) {
      auto coordInDstTensor =
          mlir::LLVM::delinearize(i, dstCTAShape, dstCTAOrder);
      auto coordInSrcTensor =
          LLVM::AMD::multiDimElementwise<unsigned, unsigned>(
              coordInDstTensor, firstTileCoordinate, std::plus<unsigned>());
      auto linearIdxInSrcTensor =
          mlir::LLVM::linearize(coordInSrcTensor, srcCTAShape, srcCTAOrder);

      size_t startIdx = linearIdxInSrcTensor * elemsPerThreadPerCTA;
      llvm::append_range(resultVals, llvm::ArrayRef(vals).slice(
                                         startIdx, elemsPerThreadPerCTA));
    }
    Value ret = packLLElements(loc, this->getTypeConverter(), resultVals,
                               rewriter, dstTy);

    rewriter.replaceOp(op, ret);
    return success();
  }

  LogicalResult
  matchAndRewrite(amdgpu::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSource().getType();
    return processLayout(op, adaptor, rewriter);
  }
};
} // namespace

namespace mlir::triton::AMD {

void populateExtractSliceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit) {
  patterns.add<ExtractSliceOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
