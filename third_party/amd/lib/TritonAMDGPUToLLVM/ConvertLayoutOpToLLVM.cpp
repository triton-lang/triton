#include "Analysis/AMDGPUAllocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::ConvertLayoutOp;
using ::triton::gpu::LinearEncodingAttr;

namespace {

// Match MFMA->Linear Layout conversion
static bool matchMFMAAndLinearLayoutCase(RankedTensorType srcTy,
                                         RankedTensorType dstTy) {
  auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcTy.getEncoding());
  auto linearLayout = dyn_cast<LinearEncodingAttr>(dstTy.getEncoding());
  if (!mfmaLayout || !linearLayout)
    return false;

  std::optional<LinearLayout> storeLL =
      mlir::triton::gpu::chooseMfmaLikeStoreLayout(srcTy);
  return linearLayout.getLinearLayout() ==
         storeLL.value_or(LinearLayout::empty());
};

class ConvertLayoutOpMFMAToLinearConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpMFMAToLinearConversion(LLVMTypeConverter &typeConverter,
                                        const TargetInfoBase &targetInfo,
                                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto dstType = cast<RankedTensorType>(op.getType());

    if (!matchMFMAAndLinearLayoutCase(srcType, dstType))
      return failure();

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (inVals.empty() || inVals.size() % 8 != 0)
      return failure();

    auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcType.getEncoding());
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();
    assert((mDim == 32 || mDim == 16) && mDim == nDim &&
           "Expected MFMA size 32 or 16");
    assert(triton::gpu::lookupThreadsPerWarp(rewriter) == 64 &&
           "Expected warp size 64 for MFMA");

    auto elemTy = srcType.getElementType();
    auto vecTy = vec_ty(elemTy, 2);

    SmallVector<Value> outVals;
    auto idx0 = b.i32_val(0);
    auto idx1 = b.i32_val(1);
    auto intrinsicName = mDim == 32 ? "llvm.amdgcn.permlane32.swap"
                                    : "llvm.amdgcn.permlane16.swap";
    // Convert MFMA layout to a MFMA-like linear layout where each thread
    // holds 8 consecutive elements
    for (size_t idx = 0; idx < inVals.size(); idx += 8) {
      SmallVector<Value, 4> inVecs;
      for (size_t vIdx = 0; vIdx < 4; vIdx++) {
        Value vec = b.undef(vecTy);
        vec = b.insert_element(vecTy, vec, inVals[idx + vIdx * 2 + 0], idx0);
        vec = b.insert_element(vecTy, vec, inVals[idx + vIdx * 2 + 1], idx1);
        inVecs.push_back(vec);
      }

      Value resVec0, resVec1, resVec2, resVec3;

      // Swap the row 2 and 3 of vec0 and the row 0 and 1 of vec2
      MLIRContext *ctx = rewriter.getContext();
      Type retType = struct_ty({i32_ty, i32_ty});
      Value falseVal = b.false_val();
      Value perm =
          LLVM::createLLVMIntrinsicCallOp(
              rewriter, loc, intrinsicName, retType,
              ValueRange{b.bitcast(inVecs[0], i32_ty),
                         b.bitcast(inVecs[2], i32_ty), falseVal, falseVal})
              ->getResult(0);
      resVec0 = b.bitcast(b.extract_val(i32_ty, perm, 0), vecTy);
      resVec2 = b.bitcast(b.extract_val(i32_ty, perm, 1), vecTy);

      // Swap the row 2 and 3 of vec1 and the row 0 and 1 of vec3
      perm = LLVM::createLLVMIntrinsicCallOp(
                 rewriter, loc, intrinsicName, retType,
                 ValueRange{b.bitcast(inVecs[1], i32_ty),
                            b.bitcast(inVecs[3], i32_ty), falseVal, falseVal})
                 ->getResult(0);
      resVec1 = b.bitcast(b.extract_val(i32_ty, perm, 0), vecTy);
      resVec3 = b.bitcast(b.extract_val(i32_ty, perm, 1), vecTy);

      for (Value res : {resVec0, resVec1, resVec2, resVec3})
        for (Value idx : {idx0, idx1})
          outVals.push_back(b.extract_element(elemTy, res, idx));
    }

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

class ConvertLayoutForcedPadding
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
public:
  ConvertLayoutForcedPadding(LLVMTypeConverter &typeConverter,
                             const TargetInfoBase &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasAttr(mlir::triton::AMD::AttrSharedMemPadded))
      return failure();
    auto srcType = op.getSrc().getType();
    auto dstType = op.getType();
    if (!cvtNeedsSharedMemory(srcType, dstType))
      return failure();

    auto result = transferWithinBlockPadding(op, adaptor.getSrc(), targetInfo,
                                             getTypeConverter(), rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};
} // namespace

void mlir::triton::AMD::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpMFMAToLinearConversion>(typeConverter, targetInfo,
                                                      benefit);
  patterns.add<ConvertLayoutForcedPadding>(typeConverter, targetInfo, benefit);
  // No need to convert when ForcedSwizzling as it's already the default
  // lowering
}
