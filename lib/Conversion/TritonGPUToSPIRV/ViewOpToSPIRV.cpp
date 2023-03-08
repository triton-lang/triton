#include "ViewOpToSPIRV.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::getElementsFromStruct;
using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::spirv::getStructFromElements;
using ::mlir::triton::gpu::getElemsPerThread;

struct SplatOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::SplatOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::SplatOp>::ConvertTritonGPUOpToSPIRVPattern;

  // Convert SplatOp or arith::ConstantOp with SplatElementsAttr to a
  // spirv::StructType value.
  //
  // @elemType: the element type in operand.
  // @resType: the return type of the Splat-like op.
  // @constVal: a spirv::ConstantOp or other scalar value.
  static Value convertSplatLikeOp(Type elemType, Type resType, Value constVal,
                                  TypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
    auto tensorTy = resType.cast<RankedTensorType>();
    if (tensorTy.getEncoding().isa<BlockedEncodingAttr>() ||
        tensorTy.getEncoding().isa<SliceEncodingAttr>()) {
      auto srcType = typeConverter->convertType(elemType);
      Value elemSrc;
      if (elemType == srcType) {
        elemSrc = constVal;
      } else {
        elemSrc = bitcast(constVal, srcType);
      }
      size_t elemsPerThread = getElemsPerThread(tensorTy);
      SmallVector<Value> elems(elemsPerThread, elemSrc);
      SmallVector<Type> elemTypes(elems.size(), srcType);
      auto structTy = spirv::StructType::get(elemTypes);

      return getStructFromElements(loc, elems, rewriter, structTy);
    } else if (auto dotLayout =
                   tensorTy.getEncoding()
                       .dyn_cast<triton::gpu::DotOperandEncodingAttr>()) {
      return convertSplatLikeOpWithDotOperandLayout(
          dotLayout, resType, elemType, constVal, typeConverter, rewriter, loc);
    } else if (auto mmaLayout =
                   tensorTy.getEncoding().dyn_cast<MmaEncodingAttr>()) {
      return convertSplatLikeOpWithMmaLayout(
          mmaLayout, resType, elemType, constVal, typeConverter, rewriter, loc);
    } else
      assert(false && "Unsupported layout found in ConvertSplatLikeOp");

    return {};
  }

  static Value convertSplatLikeOpWithDotOperandLayout(
      const triton::gpu::DotOperandEncodingAttr &layout, Type resType,
      Type elemType, Value constVal, TypeConverter *typeConverter,
      ConversionPatternRewriter &rewriter, Location loc) {
    assert(0);
  }

  static Value convertSplatLikeOpWithMmaLayout(
      const MmaEncodingAttr &layout, Type resType, Type elemType,
      Value constVal, TypeConverter *typeConverter,
      ConversionPatternRewriter &rewriter, Location loc) {
    assert(0);
  }

  LogicalResult matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto src = adaptor.getSrc();
    auto spirvStruct = convertSplatLikeOp(src.getType(), op.getType(), src,
                                       getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, {spirvStruct});
    return success();
  }
};

// This pattern helps to convert arith::ConstantOp(with SplatElementsAttr),
// the logic is the same as triton::SplatOp, so the underlying implementation
// is reused.
struct ArithConstantSplatOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<arith::ConstantOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      arith::ConstantOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    if (!value.dyn_cast<SplatElementsAttr>())
      return failure();

    auto loc = op->getLoc();

    auto values = op.getValue().dyn_cast<SplatElementsAttr>();
    auto elemType = values.getElementType();

    Attribute val;
    if (elemType.isBF16() || type::isFloat(elemType)) {
      val = values.getValues<FloatAttr>()[0];
    } else if (type::isInt(elemType)) {
      val = values.getValues<IntegerAttr>()[0];
    } else {
      llvm::errs() << "ArithConstantSplatOpSPIRVConversion get unsupported type: "
                   << value.getType() << "\n";
      return failure();
    }

    auto constOp = rewriter.create<spirv::ConstantOp>(loc, elemType, val);
    auto llStruct = SplatOpSPIRVConversion::convertSplatLikeOp(
        elemType, op.getType(), constOp, getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, llStruct);

    return success();
  }
};

template <typename SourceOp>
struct ViewLikeOpSPIRVConversion : public ConvertTritonGPUOpToSPIRVPattern<SourceOp> {
  using OpAdaptor = typename SourceOp::Adaptor;
  using ConvertTritonGPUOpToSPIRVPattern<SourceOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We cannot directly run `rewriter.replaceOp(op, adaptor.src())`
    // due to MLIR's restrictions
    Location loc = op->getLoc();
    auto resultTy = op.getType().template cast<RankedTensorType>();
    unsigned elems = getElemsPerThread(resultTy);
    Type elemTy =
        this->getTypeConverter()->convertType(resultTy.getElementType());
    SmallVector<Type> types(elems, elemTy);
    Type structTy = spirv::StructType::get(types);
    auto vals = getElementsFromStruct(loc, adaptor.getSrc(), rewriter);
    Value view = getStructFromElements(loc, vals, rewriter, structTy);
    rewriter.replaceOp(op, view);
    return success();
  }
};

void populateViewOpToSPIRVPatterns(mlir::SPIRVTypeConverter &typeConverter,
                                   mlir::MLIRContext *context,
                                   mlir::RewritePatternSet &patterns,
                                   int numWarps,
                                   mlir::AxisInfoAnalysis &axisInfoAnalysis,
                                   const mlir::Allocation *allocation,
                                   mlir::Value smem,
                                   mlir::PatternBenefit benefit) {
  patterns.add<ViewLikeOpSPIRVConversion<triton::ViewOp>>(typeConverter, context, benefit);
  patterns.add<SplatOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<ArithConstantSplatOpSPIRVConversion>(typeConverter, context, benefit);
}
