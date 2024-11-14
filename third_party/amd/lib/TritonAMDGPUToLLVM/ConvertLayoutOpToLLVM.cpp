#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace SharedToDotOperandMFMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandMFMA

namespace SharedToDotOperandWMMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandWMMA

namespace {
struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      return lowerSharedToDotOperand(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  /// Lower ttg.local_load in dot operand layout if the operand parent layout is
  /// MFMA or WMMA.
  ///
  /// \returns value with packed loaded values or empty value if this local_load
  /// is not supproted.
  Value lowerSharedToDotOperandMMA(
      triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
      const LLVMTypeConverter *typeConverter,
      ConversionPatternRewriter &rewriter,
      const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto llvmElemTy = typeConverter->convertType(
        cast<MemDescType>(src.getType()).getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    Value res;
    auto dopOpParent = dotOperandLayout.getParent();
    if (!isOuter &&
        isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>(dopOpParent)) {
      auto sharedToDotConvert = isa<AMDMfmaEncodingAttr>(dopOpParent)
                                    ? SharedToDotOperandMFMA::convertLayout
                                    : SharedToDotOperandWMMA::convertLayout;
      res = sharedToDotConvert(dotOperandLayout.getOpIdx(), rewriter, loc, src,
                               dotOperandLayout, smemObj, typeConverter,
                               tid_val());
    } else {
      assert(false && "unsupported layout found");
    }
    return res;
  }

  // shared -> matrix_core_dot_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::LocalLoadOp op,
                          triton::gpu::LocalLoadOpAdaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto dstTensorTy = cast<RankedTensorType>(dst.getType());
    auto srcTensorTy = cast<MemDescType>(src.getType());
    auto dotOperandLayout =
        cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());
    auto sharedLayout = cast<SharedEncodingAttr>(srcTensorTy.getEncoding());

    bool isOuter{};
    int K{};
    if (dotOperandLayout.getOpIdx() == 0) // $a
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[1]];
    isOuter = K == 1;
    Value res = lowerSharedToDotOperandMMA(op, adaptor, typeConverter, rewriter,
                                           dotOperandLayout, isOuter);
    if (!res)
      return failure();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertLayoutOpMFMAToDotOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  explicit ConvertLayoutOpMFMAToDotOpConversion(
      LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp>(typeConverter,
                                                             benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto dstType = cast<RankedTensorType>(op.getType());

    if (!matchMFMAAndDotOperandShuffleCase(srcType, dstType))
      return failure();

    /*
    Using wave shuffle to convert layouts:
    1) Input MMA layout (32x32, fp8, 16 values):
     _____________________________________________________________
    |(t0  v0 v1 v2 v3) (t32 v0 v1 v2 v3) ... (t32 v12 v13 v14 v15)|
    | ...                                ...                      |
    |(t31 v0 v1 v2 v3) (t63 v0 v1 v2 v3) ... (t63 v12 v13 v14 v15)|
    |_____________________________________________________________|

    2) Output Dot operand layout (two 32x16 tiles, fp8, 8 values each):
     ____________________________________________________________   ___
    |(t0  v0 v1 v2 v3 v4 v5 v6 v7) (t32 v0 v1 v2 v3 v4 v5 v6 v7) | |
    | ...                           ...                          | |...
    |(t31 v0 v1 v2 v3 v4 v5 v6 v7) (t63 v0 v1 v2 v3 v4 v5 v6 v7) | |
    |____________________________________________________________| |___
    */

    auto loc = op.getLoc();

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (inVals.empty() || inVals.size() % 8 != 0)
      return failure();

    Value threadId = tid_val();
    Value warpSize = i32_val(64); // MFMA Warp Size
    Value laneId = urem(threadId, warpSize);
    Value laneOffset = i32_val(32);
    Value mask = icmp_slt(laneId, laneOffset);
    Value addr0 = select(mask, add(laneId, laneOffset), laneId);
    Value addr1 = select(mask, laneId, sub(laneId, laneOffset));

    SmallVector<Value> outVals;
    auto elemTy = int_ty(8);
    auto vecTy = vec_ty(elemTy, 4);
    for (size_t startIdx = 0; startIdx < inVals.size(); startIdx += 8) {
      Value vec0 = undef(vecTy);
      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        vec0 =
            insert_element(vecTy, vec0, inVals[startIdx + vIdx], i32_val(vIdx));
      }
      Value vec1 = undef(vecTy);
      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        vec1 = insert_element(vecTy, vec1, inVals[startIdx + vIdx + 4],
                              i32_val(vIdx));
      }

      Value shflVec0 =
          bitcast(targetInfo.shuffleIdx(rewriter, loc,
                                        bitcast(vec0, int_ty(32)), addr0),
                  vecTy);
      Value shflVec1 =
          bitcast(targetInfo.shuffleIdx(rewriter, loc,
                                        bitcast(vec1, int_ty(32)), addr1),
                  vecTy);

      Value firstVec = select(mask, vec0, shflVec1);
      Value secondVec = select(mask, shflVec0, vec1);

      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        outVals.push_back(extract_element(elemTy, firstVec, i32_val(vIdx)));
      }
      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        outVals.push_back(extract_element(elemTy, secondVec, i32_val(vIdx)));
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

namespace mlir::triton::AMD {
void populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<LocalLoadOpConversion>(typeConverter, benefit);
  patterns.add<ConvertLayoutOpMFMAToDotOpConversion>(typeConverter, targetInfo,
                                                     benefit);
}
} // namespace mlir::triton::AMD
