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

    if (auto dotLayout = dyn_cast<DotOperandEncodingAttr>(dstLayout)) {
      if (isa<AMDMfmaEncodingAttr>(dotLayout.getParent())) {
        if (canUseTransLoad(srcTy, dstTy)) {
          return lowerSharedToDotOperandTransLL(op, adaptor, getTypeConverter(),
                                                rewriter);
        }
      }
    }
    return failure();
  }

private:
  bool canUseTransLoad(MemDescType srcTy, RankedTensorType dstTy) const {
    auto bitwidth = typeConverter->convertType(dstTy.getElementType())
                        .getIntOrFloatBitWidth();

    auto dotEnc = llvm::cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto mfmaEnc = llvm::dyn_cast<AMDMfmaEncodingAttr>(dotEnc.getParent());
    // Currently, only transpose loading for 16-bit element types is supported.
    // Support for 8-bit types will be added in a future patch.
    if (!mfmaEnc || bitwidth != 16) {
      return false;
    }

    // The transposed load lowering logic assumes that double-rate MFMA
    // instructions are used whenever possible. This code verifies whether
    // double-rate MFMA instructions are being used and falls back to the
    // default path if they are not. (Note: The lowering logic for double-rate
    // MFMA is the same as for single-rate with kpack=2). This check should be
    // removed once double-rate MFMA support is fully implemented in the
    // compiler, leaving only an assertion. Currently, single-rate
    // configurations with kpack=1 are still in use, so in such cases, we revert
    // to the default lowering logic without LDS transposed read instructions.
    int rank = dstTy.getRank();
    int32_t kWidth = dotEnc.getKWidth();
    const int32_t mDim = mfmaEnc.getMDim();
    assert((mDim == 32 || mDim == 16) && "Invalid MFMA instruction dimension");

    const int largeTileThreshold = (mDim == 32) ? 8 : 16;
    const auto shape = dstTy.getShape();
    const int kDim = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;

    const bool isLargeTile = shape[kDim] > largeTileThreshold;
    const int expectedKWidth = isLargeTile ? 8 : 4;

    if (kWidth != expectedKWidth)
      return false;

    auto sharedEnc =
        dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding());
    if (!sharedEnc)
      return false;

    bool nonKContig = kDim != sharedEnc.getOrder()[0];
    return nonKContig && targetInfo.canUseLDSTransLoad(bitwidth);
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
    auto ldsTransLayout = chooseLDSTransLayout(dotEnc, shape, bitwidth);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    SmallVector<Value> outVals;
    SmallVector<Value> elemsI32;
    bool valid = emitTransferBetweenRegistersAndShared(
        ldsTransLayout, srcTy, llvmElemTy,
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
