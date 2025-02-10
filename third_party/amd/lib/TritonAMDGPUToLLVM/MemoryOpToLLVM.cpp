#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MemDescType;

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
      const DotOperandEncodingAttr &dotOperandLayout) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto llvmElemTy = typeConverter->convertType(
        cast<MemDescType>(src.getType()).getElementType());

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    Value res;
    auto dopOpParent = dotOperandLayout.getParent();
    if (isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>(dopOpParent)) {
      auto sharedToDotConvert = isa<AMDMfmaEncodingAttr>(dopOpParent)
                                    ? SharedToDotOperandMFMA::convertLayout
                                    : SharedToDotOperandWMMA::convertLayout;
      res = sharedToDotConvert(dotOperandLayout.getOpIdx(), rewriter, loc, src,
                               dotOperandLayout, smemObj, typeConverter,
                               b.tid_val());
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
    Value dst = op.getResult();
    auto dstTensorTy = cast<RankedTensorType>(dst.getType());
    auto dotOperandLayout =
        cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());

    Value res = lowerSharedToDotOperandMMA(op, adaptor, typeConverter, rewriter,
                                           dotOperandLayout);
    if (!res)
      return failure();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct TransLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  TransLocalLoadOpConversion(const LLVMTypeConverter &converter,
                             const AMD::TargetInfo &targetInfo,
                             PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (canUseTransLoad(srcTy, dstTy)) {
      return lowerSharedToDotOperandTransLL(op, adaptor, getTypeConverter(),
                                            rewriter);
    }
    return failure();
  }

private:
  bool checkLayoutProperties(MemDescType srcTy, RankedTensorType dstTy) const {
    // Verify the layout properties required for using the ds_read_tr
    // instruction. This instruction is used to load non-k contiguous tensors
    // from shared memory into a dot layout with an MFMA layout parent.
    auto dotEnc = llvm::dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    if (!dotEnc) {
      return false;
    }

    auto mfmaEnc = llvm::dyn_cast<AMDMfmaEncodingAttr>(dotEnc.getParent());
    if (!mfmaEnc) {
      return false;
    }

    auto sharedEnc =
        dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding());
    if (!sharedEnc)
      return false;

    int rank = dstTy.getRank();
    const int kDim = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
    return kDim != sharedEnc.getOrder()[0];
  }

  bool checkPerformanceProperties(MemDescType srcTy,
                                  RankedTensorType dstTy) const {
    // The transposed load lowering logic assumes that double-rate MFMA (
    // mfma32x32x16 and mfma16x16x32) instructions are used whenever possible.
    // This code verifies whether double-rate MFMA instructions are being used
    // and falls back to the default path if they are not. (Note: The lowering
    // logic for double-rate MFMA is the same as for single-rate (mfma32x32x8
    // and mfma16x16x16) with kpack=2). This check should be removed once
    // double-rate MFMA support is fully implemented in the compiler, leaving
    // only an assertion. Currently, single-rate configurations with kpack=1 are
    // still in use, so in such cases, we revert to the default lowering logic
    // without LDS transpose read instructions.
    auto dotEnc = llvm::cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto mfmaEnc = llvm::cast<AMDMfmaEncodingAttr>(dotEnc.getParent());

    int rank = dstTy.getRank();
    int32_t kWidth = dotEnc.getKWidth();
    const int32_t mDim = mfmaEnc.getMDim();
    assert((mDim == 32 || mDim == 16) && "Invalid MFMA instruction dimension");

    // Single rate MFMA insts: mfma32x32x8, mfma16x16x16
    const int kSize16bSingleRateMfma32 = 8;
    const int kSize16bSingleRateMfma16 = 16;
    const int largeTileThreshold16b =
        (mDim == 32) ? kSize16bSingleRateMfma32 : kSize16bSingleRateMfma16;
    const auto shape = dstTy.getShape();
    const int kDim = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;

    const bool isLargeTile16b = shape[kDim] > largeTileThreshold16b;
    const int expectedKWidth16b = isLargeTile16b ? 8 : 4;

    return kWidth == expectedKWidth16b;
  }

  bool canUseTransLoad(MemDescType srcTy, RankedTensorType dstTy) const {
    auto bitwidth = typeConverter->convertType(dstTy.getElementType())
                        .getIntOrFloatBitWidth();

    // 1. Check GPU arch properties.
    if (!targetInfo.canUseLDSTransLoad(bitwidth)) {
      return false;
    }

    // 2. Check layout properties.
    if (!checkLayoutProperties(srcTy, dstTy)) {
      return false;
    }

    // 3. Check performance properties.
    if (!checkPerformanceProperties(srcTy, dstTy)) {
      return false;
    }

    // 4. Check current limitations.
    if (bitwidth != 16) {
      return false;
    }

    return true;
  }

  LogicalResult
  lowerSharedToDotOperandTransLL(triton::gpu::LocalLoadOp op,
                                 triton::gpu::LocalLoadOpAdaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dotEnc = cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto shape = dstTy.getShape();

    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto dsReadTransLayout = chooseDsReadB64Tr16Layout(dotEnc, shape, bitwidth);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    SmallVector<Value> outVals;
    bool valid = emitTransferBetweenRegistersAndShared(
        dsReadTransLayout, srcTy, llvmElemTy,
        /*maxVecElems=*/std::nullopt, smemObj, loc, rewriter, targetInfo,
        [&](VectorType vecTy, Value vecAddr) {
          auto dsReadOp =
              rewriter.create<ROCDL::ds_read_tr16_b64>(loc, vecTy, vecAddr);
          Value vecVal = dsReadOp.getResult();
          for (int v = 0; v < vecTy.getNumElements(); v++) {
            outVals.push_back(
                b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
          }
        });

    assert(valid && "Failed to emit LDS transpose load operations");
    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  PatternBenefit transBenefit = PatternBenefit(benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, benefit);
  patterns.add<TransLocalLoadOpConversion>(typeConverter, targetInfo,
                                           transBenefit);
}
