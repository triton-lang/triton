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
using ::mlir::triton::gpu::MemDescType;
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

    auto loc = op.getLoc();

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (inVals.empty() || inVals.size() % 8 != 0)
      return failure();

    auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcType.getEncoding());
    assert((mfmaLayout.getMDim() == 16 || mfmaLayout.getMDim() == 32) &&
           "Expected MFMA size 16 or 32");
    assert(triton::gpu::getWarpSize(mfmaLayout) == 64 &&
           "Expected warp size 64 for MFMA");

    auto elemTy = int_ty(8);
    auto vecTy = vec_ty(elemTy, 4);

    Value c16 = i32_val(16);
    Value c32 = i32_val(32);
    Value c48 = i32_val(48);
    Value c64 = i32_val(64);

    Value threadId = tid_val();
    Value laneId = urem(threadId, c64);

    Value mask0 = icmp_slt(laneId, c32);
    Value mask1 = icmp_slt(urem(laneId, c32), c16);

    Value addrShift16 = urem(add(laneId, c16), c64);
    Value addrShift32 = urem(add(laneId, c32), c64);
    Value addrShift48 = urem(add(laneId, c48), c64);

    SmallVector<Value> outVals;
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

      Value resVec0, resVec1;
      if (mfmaLayout.getMDim() == 32) {
        /*
        Using wave shuffle to convert layouts (32x32x16 case):
        1) Input MMA layout (32x32, fp8, 16 values):
         _____________________________________________________________
        |(t0  v0 v1 v2 v3) (t32 v0 v1 v2 v3) ... (t32 v12 v13 v14 v15)|
        | ...                                ...                      |
        |(t31 v0 v1 v2 v3) (t63 v0 v1 v2 v3) ... (t63 v12 v13 v14 v15)|
        |_____________________________________________________________|

        2) Output Dot operand layout (two 32x16 tiles, fp8, 8 values each):
         ____________________________________________________________  ___
        |(t0  v0 v1 v2 v3 v4 v5 v6 v7) (t32 v0 v1 v2 v3 v4 v5 v6 v7) ||
        | ...                           ...                          ||...
        |(t31 v0 v1 v2 v3 v4 v5 v6 v7) (t63 v0 v1 v2 v3 v4 v5 v6 v7) ||
        |____________________________________________________________||___
        */

        Value shflVec0 =
            bitcast(targetInfo.shuffleIdx(
                        rewriter, loc, bitcast(vec0, int_ty(32)), addrShift32),
                    vecTy);
        Value shflVec1 =
            bitcast(targetInfo.shuffleIdx(
                        rewriter, loc, bitcast(vec1, int_ty(32)), addrShift32),
                    vecTy);

        resVec0 = select(mask0, vec0, shflVec1);
        resVec1 = select(mask0, shflVec0, vec1);
      } else if (mfmaLayout.getMDim() == 16) {
        /*
        16x16x32 case:
        1) Input MMA layout (two 16x16, fp8, 4 values each):
         _________________________________________________________  ___________
        |(t0  v0 v1 v2 v3) (t16 v0 v1 v2 v3) ... (t48 v0 v1 v2 v3)||(t0  v4 ...
        | ...                                ...                  || ...
        |(t15 v0 v1 v2 v3) (t31 v0 v1 v2 v3) ... (t63 v0 v1 v2 v3)||(t15 v4 ...
        |_________________________________________________________||___________

        2) Output Dot operand layout (16x32 tile, fp8, 8 values):
         ________________________________________________________________
        |(t0  v0 v1 v2 v3 v4 v5 v6 v7) ... (t48 v0 v1 v2 v3 v4 v5 v6 v7) |
        | ...                          ...                               |
        |(t15 v0 v1 v2 v3 v4 v5 v6 v7) ... (t63 v0 v1 v2 v3 v4 v5 v6 v7) |
        |________________________________________________________________|
        */

        Value shflVec0_16 =
            bitcast(targetInfo.shuffleIdx(
                        rewriter, loc, bitcast(vec0, int_ty(32)), addrShift16),
                    vecTy);
        Value shflVec0_32 =
            bitcast(targetInfo.shuffleIdx(
                        rewriter, loc, bitcast(vec0, int_ty(32)), addrShift32),
                    vecTy);
        Value shflVec1_32 =
            bitcast(targetInfo.shuffleIdx(
                        rewriter, loc, bitcast(vec1, int_ty(32)), addrShift32),
                    vecTy);
        Value shflVec1_48 =
            bitcast(targetInfo.shuffleIdx(
                        rewriter, loc, bitcast(vec1, int_ty(32)), addrShift48),
                    vecTy);

        resVec0 = select(mask0, select(mask1, vec0, shflVec0_16),
                         select(mask1, shflVec1_32, shflVec1_48));
        resVec1 = select(mask0, select(mask1, shflVec0_16, shflVec0_32),
                         select(mask1, shflVec1_48, vec1));
      }

      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        outVals.push_back(extract_element(elemTy, resVec0, i32_val(vIdx)));
      }
      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        outVals.push_back(extract_element(elemTy, resVec1, i32_val(vIdx)));
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
