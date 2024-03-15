#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

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
    if (dstLayout.isa<DotOperandEncodingAttr>() &&
        dstLayout.cast<DotOperandEncodingAttr>()
            .getParent()
            .isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>()) {
      return lowerSharedToDotOperand(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  // shared -> dot_operand if the result layout is mfma
  Value lowerSharedToDotOperandMMA(
      triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
      const LLVMTypeConverter *typeConverter,
      ConversionPatternRewriter &rewriter,
      const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto llvmElemTy = typeConverter->convertType(
        src.getType().cast<MemDescType>().getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    Value res;
    auto dopOpParent = dotOperandLayout.getParent();
    if (!isOuter &&
        dopOpParent.isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>()) {
      auto sharedToDotConvert = dopOpParent.isa<AMDMfmaEncodingAttr>()
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
    auto dstTensorTy = dst.getType().cast<RankedTensorType>();
    auto srcTensorTy = src.getType().cast<MemDescType>();
    auto dotOperandLayout =
        dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto sharedLayout = srcTensorTy.getEncoding().cast<SharedEncodingAttr>();

    bool isOuter{};
    int K{};
    if (dotOperandLayout.getOpIdx() == 0) // $a
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[1]];
    isOuter = K == 1;
    Value res = lowerSharedToDotOperandMMA(op, adaptor, typeConverter, rewriter,
                                           dotOperandLayout, isOuter);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::ConvertLayoutOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (srcLayout.isa<AMDMfmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMfmaToDotOperand(op, adaptor, rewriter);
    }
    return failure();
  }

private:
  LogicalResult
  lowerMfmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    if (isMfmaToDotShortcut(srcTy, dstTy)) {
      // get source values
      auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      unsigned elems = getTotalElemsPerThread(srcTy);
      Type elemTy =
          this->getTypeConverter()->convertType(srcTy.getElementType());
      // for the destination type, we need to pack values together
      // so they can be consumed by tensor core operations
      SmallVector<Value> vecVals;
      SmallVector<Type> types;
      auto elemSize = elemTy.getIntOrFloatBitWidth();
      // TODO: Support types other than float16 and
      // bf16 (represented as int16 in llvm ir).
      assert((type::isFloat(elemTy) || type::isInt(elemTy)) && elemSize == 16);
      // vecSize is an number of sequential elements stored by one thread
      // - For MFMA (nonKDim == 32) encoding it is 4
      // - For MFMA (nonKDim == 32) operand encoding it is
      // dotOperandEndocing::kWidth,
      //   which is 4 for fp16 and bfloat16 dtypes
      //
      // For mentioned types MFMA and MFMA operand layouts are the same
      const unsigned vecSize = 4;
      Type vecTy = vec_ty(elemTy, vecSize);
      types = SmallVector<Type>(elems / vecSize, vecTy);
      for (unsigned i = 0; i < elems; i += vecSize) {
        Value packed = rewriter.create<LLVM::UndefOp>(loc, vecTy);
        for (unsigned j = 0; j < vecSize; j++)
          packed = insert_element(vecTy, packed, vals[i + j], i32_val(j));
        vecVals.push_back(packed);
      }
      Value view =
          packLLElements(loc, getTypeConverter(), vecVals, rewriter, dstTy);
      rewriter.replaceOp(op, view);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace AMD {
void populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, benefit);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
} // namespace AMD
