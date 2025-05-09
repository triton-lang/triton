#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MemDescType;
using ::triton::gpu::LinearEncodingAttr;

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
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (inVals.empty() || inVals.size() % 8 != 0)
      return failure();

    auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcType.getEncoding());
    assert((mfmaLayout.getMDim() == 16 || mfmaLayout.getMDim() == 32) &&
           "Expected MFMA size 16 or 32");
    assert(triton::gpu::lookupThreadsPerWarp(rewriter) == 64 &&
           "Expected warp size 64 for MFMA");

    auto elemTy = int_ty(8);
    auto vecTy = vec_ty(elemTy, 4);

    Value c16 = b.i32_val(16);
    Value c32 = b.i32_val(32);
    Value c48 = b.i32_val(48);
    Value c64 = b.i32_val(64);

    Value threadId = getThreadId(rewriter, loc);
    Value laneId = b.urem(threadId, c64);

    Value mask0 = b.icmp_slt(laneId, c32);
    Value mask1 = b.icmp_slt(b.urem(laneId, c32), c16);

    Value addrShift16 = b.urem(b.add(laneId, c16), c64);
    Value addrShift32 = b.urem(b.add(laneId, c32), c64);
    Value addrShift48 = b.urem(b.add(laneId, c48), c64);

    SmallVector<Value> outVals;
    for (size_t startIdx = 0; startIdx < inVals.size(); startIdx += 8) {
      Value vec0 = b.undef(vecTy);
      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        vec0 = b.insert_element(vecTy, vec0, inVals[startIdx + vIdx],
                                b.i32_val(vIdx));
      }
      Value vec1 = b.undef(vecTy);
      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        vec1 = b.insert_element(vecTy, vec1, inVals[startIdx + vIdx + 4],
                                b.i32_val(vIdx));
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

        Value shflVec0 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec0, int_ty(32)),
                                  addrShift32),
            vecTy);
        Value shflVec1 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec1, int_ty(32)),
                                  addrShift32),
            vecTy);

        resVec0 = b.select(mask0, vec0, shflVec1);
        resVec1 = b.select(mask0, shflVec0, vec1);
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

        Value shflVec0_16 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec0, int_ty(32)),
                                  addrShift16),
            vecTy);
        Value shflVec0_32 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec0, int_ty(32)),
                                  addrShift32),
            vecTy);
        Value shflVec1_32 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec1, int_ty(32)),
                                  addrShift32),
            vecTy);
        Value shflVec1_48 = b.bitcast(
            targetInfo.shuffleIdx(rewriter, loc, b.bitcast(vec1, int_ty(32)),
                                  addrShift48),
            vecTy);

        resVec0 = b.select(mask0, b.select(mask1, vec0, shflVec0_16),
                           b.select(mask1, shflVec1_32, shflVec1_48));
        resVec1 = b.select(mask0, b.select(mask1, shflVec0_16, shflVec0_32),
                           b.select(mask1, shflVec1_48, vec1));
      }

      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        outVals.push_back(b.extract_element(elemTy, resVec0, b.i32_val(vIdx)));
      }
      for (size_t vIdx = 0; vIdx < 4; ++vIdx) {
        outVals.push_back(b.extract_element(elemTy, resVec1, b.i32_val(vIdx)));
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

// Match MFMA->Linear Layout conversion
static bool matchMFMAAndLinearLayoutCase(RankedTensorType srcTy,
                                         RankedTensorType dstTy) {
  auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcTy.getEncoding());
  auto linearLayout = dyn_cast<LinearEncodingAttr>(dstTy.getEncoding());
  if (!mfmaLayout || !linearLayout)
    return false;

  std::optional<LinearLayout> srcLL =
      mlir::triton::gpu::chooseMfmaLikeStoreLayout(srcTy);
  if (!srcLL)
    return false;

  MLIRContext *ctx = linearLayout.getContext();
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kRegister = StringAttr::get(ctx, "register");
  auto srcBase = srcLL.value().getBases();
  auto srcReg = srcBase.lookup(kRegister);
  auto srcLane = srcBase.lookup(kLane);
  auto dstBases = linearLayout.getLinearLayout().getBases();
  auto dstReg = dstBases.lookup(kRegister);
  auto dstLane = dstBases.lookup(kLane);
  return dstReg == srcReg && dstLane == srcLane;
};

struct ConvertLayoutOpMFMAToLinearConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  explicit ConvertLayoutOpMFMAToLinearConversion(
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

    if (!matchMFMAAndLinearLayoutCase(srcType, dstType))
      return failure();

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (inVals.empty() || inVals.size() % 8 != 0)
      return failure();

    auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcType.getEncoding());
    assert(mfmaLayout.getMDim() == 32 && "Expected MFMA size 32");
    assert(triton::gpu::lookupThreadsPerWarp(rewriter) == 64 &&
           "Expected warp size 64 for MFMA");

    auto elemTy = srcType.getElementType();
    auto vecTy = vec_ty(elemTy, 2);

    SmallVector<Value> outVals;
    auto idx0 = b.i32_val(0);
    auto idx1 = b.i32_val(1);
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
              rewriter, loc, "llvm.amdgcn.permlane32.swap", retType,
              ValueRange{b.bitcast(inVecs[0], i32_ty),
                         b.bitcast(inVecs[2], i32_ty), falseVal, falseVal})
              ->getResult(0);
      resVec0 = b.bitcast(b.extract_val(i32_ty, perm, 0), vecTy);
      resVec2 = b.bitcast(b.extract_val(i32_ty, perm, 1), vecTy);

      // Swap the row 2 and 3 of vec1 and the row 0 and 1 of vec3
      perm = LLVM::createLLVMIntrinsicCallOp(
                 rewriter, loc, "llvm.amdgcn.permlane32.swap", retType,
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

protected:
  const TargetInfoBase &targetInfo;
};
} // namespace

void mlir::triton::AMD::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpMFMAToDotOpConversion>(typeConverter, targetInfo,
                                                     benefit);
  patterns.add<ConvertLayoutOpMFMAToLinearConversion>(typeConverter, targetInfo,
                                                      benefit);
}
