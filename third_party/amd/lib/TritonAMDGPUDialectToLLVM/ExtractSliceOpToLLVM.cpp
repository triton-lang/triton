#include "/triton/third_party/amd/lib/TritonAMDGPUDialectToLLVM/Utility.h"
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
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult processLayout(amdgpu::ExtractSliceOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto srcTy = cast<RankedTensorType>(op.getSource().getType());
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto vals = unpackLLElements(loc, adaptor.getSource(), rewriter);
    auto offsets = op.getStaticOffsets();

    auto linearLayoutSrc = triton::gpu::toLinearLayout(srcTy);
    auto outDimNames = llvm::to_vector(linearLayoutSrc.getOutDimNames());
    // Call transposeOuts, to ensure that order of input and output tensor
    // element coordinates are compatible on stage 7 in algorithm below.
    auto linearLayoutDst =
        triton::gpu::toLinearLayout(dstTy).transposeOuts(outDimNames);

    // Algorithm:
    // 1. for every src element
    // 2.   get src element coordinates
    // 3.   save mapping from element coords to src register idx
    // 4. for every dst register
    // 5.   get dst element coordinates relative to tile start
    // 6.   add coordinates of tile start relative to parent tensor
    // 7.   find source register number which holds dst value
    // 8.   copy from corresponding src register
    auto ctx = rewriter.getContext();
    int rank = srcTy.getRank();
    StringAttr kReg = StringAttr::get(ctx, "register");
    auto dstRegBases = linearLayoutDst.getBases().lookup(kReg);

    // Mapping from tensors element location to src register id
    // Steps 1), 2) and 3).
    auto srcElemToReg =
        mlir::LLVM::AMD::mapRegToCoordinates(linearLayoutSrc, ctx);

    // for every output register get element coords, copy corresponding src
    // register
    int dstRegNum = 1 << dstRegBases.size();
    SmallVector<Value> resultVals;

    // 4. for every dst register
    for (int regId = 0; regId < dstRegNum; ++regId) {
      // 5.   get dst element coordinates relative to tile start
      auto elemCoords =
          mlir::LLVM::AMD::getElemCoordsFromReg(linearLayoutDst, regId, ctx);
      // 6.   add coordinates of tile start relative to parent tensor
      for (int i = 0; i < rank; ++i)
        elemCoords[i].second += offsets[i];
      assert(srcElemToReg.contains(elemCoords));
      // 7.   find source register number which holds dst value
      auto srcRegId = srcElemToReg.lookup(elemCoords);
      // 8.   copy from corresponding src register
      resultVals.push_back(vals[srcRegId]);
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
