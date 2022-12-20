#include "ViewOpToLLVM.h"
#include "DotOpHelpers.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::DotOpMmaV1ConversionHelper;
using ::mlir::LLVM::DotOpMmaV2ConversionHelper;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStructFromElements;
using ::mlir::LLVM::getElementsFromStruct;
using ::mlir::triton::gpu::getElemsPerThread;

Value SplatOpConversion::convertSplatLikeOp(
    Type elemType, Type resType, Value constVal,
    TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, Location loc) {
  auto tensorTy = resType.cast<RankedTensorType>();
  if (tensorTy.getEncoding().isa<BlockedEncodingAttr>() ||
      tensorTy.getEncoding().isa<SliceEncodingAttr>()) {
    auto srcType = typeConverter->convertType(elemType);
    auto llSrc = bitcast(constVal, srcType);
    size_t elemsPerThread = getElemsPerThread(tensorTy);
    llvm::SmallVector<Value> elems(elemsPerThread, llSrc);
    llvm::SmallVector<Type> elemTypes(elems.size(), srcType);
    auto structTy =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elemTypes);

    return getStructFromElements(loc, elems, rewriter, structTy);
  } else if (auto mmaLayout =
                 tensorTy.getEncoding().dyn_cast<MmaEncodingAttr>()) {
    return convertSplatLikeOpWithMmaLayout(
        mmaLayout, resType, elemType, constVal, typeConverter, rewriter, loc);
  } else
    assert(false && "Unsupported layout found in ConvertSplatLikeOp");

  return {};
}

Value SplatOpConversion::convertSplatLikeOpWithMmaLayout(
    const MmaEncodingAttr &layout,
    Type resType, Type elemType, Value constVal,
    TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, Location loc) {
  auto tensorTy = resType.cast<RankedTensorType>();
  auto shape = tensorTy.getShape();
  if (layout.isAmpere()) {
    auto [repM, repN] = DotOpMmaV2ConversionHelper::getRepMN(tensorTy);
    size_t fcSize = 4 * repM * repN;

    auto structTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), SmallVector<Type>(fcSize, elemType));
    return getStructFromElements(loc, SmallVector<Value>(fcSize, constVal),
                                 rewriter, structTy);
  }
  if (layout.isVolta()) {
    DotOpMmaV1ConversionHelper helper(layout);
    int repM = helper.getRepM(shape[0]);
    int repN = helper.getRepN(shape[1]);
    // According to mma layout of v1, each thread process 8 elements.
    int elems = 8 * repM * repN;

    auto structTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), SmallVector<Type>(elems, elemType));
    return getStructFromElements(loc, SmallVector<Value>(elems, constVal),
                                 rewriter, structTy);
  }

  assert(false && "Unsupported mma layout found");
  return {};
}

LogicalResult SplatOpConversion::matchAndRewrite(
    triton::SplatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op->getLoc();
  auto src = adaptor.src();
  auto llStruct = convertSplatLikeOp(src.getType(), op.getType(), src,
                                     getTypeConverter(), rewriter, loc);
  rewriter.replaceOp(op, {llStruct});
  return success();
}

LogicalResult ArithConstantSplatOpConversion::matchAndRewrite(
    arith::ConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto value = op.getValue();
  if (!value.dyn_cast<SplatElementsAttr>())
    return failure();

  auto loc = op->getLoc();

  LLVM::ConstantOp arithConstantOp;
  auto values = op.getValue().dyn_cast<SplatElementsAttr>();
  auto elemType = values.getElementType();

  Attribute val;
  if (type::isInt(elemType)) {
    val = values.getValues<IntegerAttr>()[0];
  } else if (type::isFloat(elemType)) {
    val = values.getValues<FloatAttr>()[0];
  } else {
    llvm::errs() << "ArithConstantSplatOpConversion get unsupported type: "
                 << value.getType() << "\n";
    return failure();
  }

  auto constOp = rewriter.create<LLVM::ConstantOp>(loc, elemType, val);
  auto llStruct = SplatOpConversion::convertSplatLikeOp(
      elemType, op.getType(), constOp, getTypeConverter(), rewriter, loc);
  rewriter.replaceOp(op, llStruct);

  return success();
}

LogicalResult CatOpConversion::matchAndRewrite(
    CatOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  auto resultTy = op.getType().template cast<RankedTensorType>();
  unsigned elems = getElemsPerThread(resultTy);
  Type elemTy =
      this->getTypeConverter()->convertType(resultTy.getElementType());
  SmallVector<Type> types(elems, elemTy);
  // unpack input values
  auto lhsVals = getElementsFromStruct(loc, adaptor.lhs(), rewriter);
  auto rhsVals = getElementsFromStruct(loc, adaptor.rhs(), rewriter);
  // concatenate (and potentially reorder) values
  SmallVector<Value> retVals;
  for (Value v : lhsVals)
    retVals.push_back(v);
  for (Value v : rhsVals)
    retVals.push_back(v);
  // pack and replace
  Type structTy = LLVM::LLVMStructType::getLiteral(this->getContext(), types);
  Value ret = getStructFromElements(loc, retVals, rewriter, structTy);
  rewriter.replaceOp(op, ret);
  return success();
}

LogicalResult TransOpConversion::matchAndRewrite(
    triton::TransOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  auto srcSmemObj =
      getSharedMemoryObjectFromStruct(loc, adaptor.src(), rewriter);
  SmallVector<Value> dstStrides = {srcSmemObj.strides[1],
                                   srcSmemObj.strides[0]};
  SmallVector<Value> dstOffsets = {srcSmemObj.offsets[1],
                                   srcSmemObj.offsets[0]};
  auto dstSmemObj =
      SharedMemoryObject(srcSmemObj.base, dstStrides, dstOffsets);
  auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
  rewriter.replaceOp(op, retVal);
  return success();
}

void populateViewOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, int numWarps,
    AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    PatternBenefit benefit) {
  patterns.add<ViewLikeOpConversion<triton::ViewOp>>(typeConverter, benefit);
  patterns.add<ViewLikeOpConversion<triton::ExpandDimsOp>>(typeConverter,
                                                           benefit);
  patterns.add<SplatOpConversion>(typeConverter, benefit);
  patterns.add<ArithConstantSplatOpConversion>(typeConverter, benefit);
  patterns.add<CatOpConversion>(typeConverter, benefit);
  patterns.add<TransOpConversion>(typeConverter, benefit);
}

