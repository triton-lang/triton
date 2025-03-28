#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <numeric>

using namespace mlir;
using namespace mlir::triton;

// clang-format off
//===--------------------------------------------------------------------------------===//
//   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
//   # WO   #  W1 #                                     |                                #
//   #      #     #                                     |                                #
//   #  #   #  #  #                                     |                                #
//   # W2   # W3  #   ....                              |                                #
//   #      #     #                                     |  SkipElems                     #
//   #  #   #  #  #                                     |                                #
//   #                                                  |                                #
//   #                                        Slice     |                                #
//   #    .                                 /        \  |                                #
//   #    .                                /          \ |                                #
//   #    .                               /            \|                                #
//   #                                    #   #  #  #  #                                 #
//   #                                    #  W0  #  W1 #                                 #
//   #                                    #      #     #                                 #
//   #                                    #  #   #  #  #    tensorStride                 #
//   #                                    #  W2  #  W3 # --------------------------------#
//   #                                    #      #     #                                 #
//   #                                    #  #   #  #  #                                 #
//   #          tensorStride              #  W0  #  W1 #                                 #
//   # ---------------------------------- #      #     #                                 #
//   #                                    #  #   #  #  #                                 #
//   #                                    #  W2  #  W3 #                                 #
//   #                                    #      #     #                                 #
//   #                                    #  #   #  #  # ---> lastIdx                    #
//   #                                         .                                         #
//   #                                         .                                         #
//   #                                         .                                         #
//   #                                                                                   #
//   #                                                                                   #
//   #                                                                                   #
//   #                                                                                   #
//   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
//===--------------------------------------------------------------------------------===//
// clang-format on

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
    auto srcShape = srcTy.getShape();
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto vals = unpackLLElements(loc, adaptor.getSource(), rewriter);
    auto sizePerThread =
        triton::gpu::toLinearEncoding(srcTy).getSizePerThread();
    auto totalSizePerThread = product<unsigned>(sizePerThread);
    auto order = triton::gpu::getOrder(srcTy);

    // Calculate valid total number of workers in each dimension
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(srcTy);
    for (size_t i = 0; i < shapePerCTATile.size(); ++i) {
      shapePerCTATile[i] =
          std::min(static_cast<unsigned>(srcShape[i]), shapePerCTATile[i]);
    }

    // ranks of the source and the destination are euqal; checked in the
    // verifier
    const auto rank = srcTy.getRank();
    SmallVector<int64_t> sizes;
    for (auto i = 0; i < rank; ++i) {
      sizes.push_back(resultTy.getDimSize(i));
    }

    auto offsets = op.getStaticOffsets();

    // Calculate offsets and sizes in terms of CTA units.
    SmallVector<int64_t> CTAOffsets;
    SmallVector<int64_t> CTASizes;
    SmallVector<int64_t> CTAPerShape;
    for (size_t dim = 0; dim < rank; ++dim) {
      CTAOffsets.push_back(offsets[dim] / shapePerCTATile[dim]);
      CTASizes.push_back(sizes[dim] / shapePerCTATile[dim]);
      CTAPerShape.push_back(srcShape[dim] / shapePerCTATile[dim]);
    }
    SmallVector<int64_t> CTAStrides(CTAPerShape.size());
    std::exclusive_scan(CTAPerShape.rbegin(), CTAPerShape.rend(),
                        CTAStrides.begin(), 1, std::multiplies<>{});
    std::reverse(CTAStrides.begin(), CTAStrides.end());

    // The diagram above illustrates the graphical representation of the
    // skipElems, tensorStride, and lastIdx variables.
    auto tensorStride =
        (CTAPerShape[order[0]] - CTASizes[order[0]]) * totalSizePerThread;

    unsigned skipElems = 0;
    unsigned lastIdx = 0;
    for (size_t dim = 0; dim < rank; ++dim) {
      skipElems += CTAOffsets[dim] * CTAStrides[dim];
      lastIdx += (CTAOffsets[dim] + CTASizes[dim] - 1) * CTAStrides[dim];
    }
    skipElems *= totalSizePerThread;
    lastIdx = totalSizePerThread * (lastIdx + 1);

    assert(lastIdx <= vals.size());

    SmallVector<Value> resultVals;
    for (int i = skipElems; i < lastIdx; i += tensorStride) {
      for (int j = 0; j < totalSizePerThread * CTASizes[order[0]]; ++j, ++i) {
        assert(i < lastIdx);
        resultVals.push_back(vals[i]);
      }
    }
    Value ret = packLLElements(loc, this->getTypeConverter(), resultVals,
                               rewriter, resultTy);

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
