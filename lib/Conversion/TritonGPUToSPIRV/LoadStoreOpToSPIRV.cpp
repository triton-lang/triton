#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "LoadStoreOpToSPIRV.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::getElementsFromStruct;
using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::spirv::getStructFromElements;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreSPIRVConversionBase {
  explicit LoadStoreSPIRVConversionBase(AxisInfoAnalysis &axisAnalysisPass)
      : axisAnalysisPass(axisAnalysisPass) {}

  // Get corresponding LLVM element values of \param value.
  static SmallVector<Value> getSPIRVElems(Value value, Value spirvValue,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc) {
    if (!value)
      return {};
    if (!spirvValue.getType().isa<spirv::StructType>())
      return {spirvValue};
    // Here, we assume that all inputs should have a blockedLayout
    auto valueVals = getElementsFromStruct(loc, spirvValue, rewriter);
    return valueVals;
  }

  unsigned getContiguity(Value ptr) const {
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  AxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::LoadOp>,
      public LoadStoreSPIRVConversionBase {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::LoadOp>::ConvertTritonGPUOpToSPIRVPattern;

  LoadOpSPIRVConversion(SPIRVTypeConverter &converter, MLIRContext *context,
                   AxisInfoAnalysis &axisAnalysisPass, PatternBenefit benefit)
      : ConvertTritonGPUOpToSPIRVPattern<triton::LoadOp>(converter, context, benefit),
        LoadStoreSPIRVConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    Value spirvPtr = adaptor.getPtr();
    Value spirvMask = adaptor.getMask();
    Value spirvOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getResult().getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getContiguity(ptr);
    unsigned numElems = getElemsPerThread(ptr.getType());
    if (spirvMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    // Get the SPIRV values for pointers
    auto ptrElems = getSPIRVElems(ptr, spirvPtr, rewriter, loc);
    assert(ptrElems.size() == numElems);

    // Get the SPIRV values for mask
    SmallVector<Value> maskElems;
    if (spirvMask) {
      maskElems = getSPIRVElems(mask, spirvMask, rewriter, loc);
      assert(maskElems.size() == numElems);
    }

    // Get the SPIRV values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && valueElemTy.isa<IntegerType>() &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat()) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    auto otherElems = getSPIRVElems(other, spirvOther, rewriter, loc);

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNbits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    assert(vec == 1 && "no vec support yet");
    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is GEP with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNbits);
      const size_t totalWidth = valueElemNbits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNbits;
      assert(wordNElems * nWords * numVecs == numElems);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.
      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);

      // scalar load
      // Create block structure for the masked load.
      auto *preheader = rewriter.getInsertionBlock();
      auto opPosition = rewriter.getInsertionPoint();
      auto *tailblock = rewriter.splitBlock(preheader, opPosition);
      tailblock->addArgument(valueElemTy, loc);
      auto *condblock = rewriter.createBlock(tailblock);

      // Test the mask
      rewriter.setInsertionPoint(preheader, preheader->end());

      // Prediction false to use the other value.
      auto other_ = rewriter.create<spirv::UndefOp>(loc, valueElemTy);
      rewriter.create<mlir::cf::CondBranchOp>(loc, pred, condblock, tailblock, ValueRange{other_});

      // Do the load
      rewriter.setInsertionPoint(condblock, condblock->end());
      Value ret = rewriter.create<spirv::LoadOp>(loc, ptrElems[vecStart]);
      rewriter.create<mlir::cf::BranchOp>(loc, tailblock, ValueRange{ret});

      rewriter.setInsertionPoint(tailblock, tailblock->begin());

      loadedVals.push_back(*tailblock->args_begin());
    } // end vec

    Type spirvResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct =
        getStructFromElements(loc, loadedVals, rewriter, spirvResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpSPIRVConversion
        : public ConvertTritonGPUOpToSPIRVPattern<triton::StoreOp>,
          public LoadStoreSPIRVConversionBase {
  using ConvertTritonGPUOpToSPIRVPattern<
          triton::StoreOp>::ConvertTritonGPUOpToSPIRVPattern;

  StoreOpSPIRVConversion(SPIRVTypeConverter &converter, MLIRContext *context,
                    AxisInfoAnalysis &axisAnalysisPass, PatternBenefit benefit)
          : ConvertTritonGPUOpToSPIRVPattern<triton::StoreOp>(converter, context, benefit),
            LoadStoreSPIRVConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value value = op.getValue();

    Value spirvPtr = adaptor.getPtr();
    Value spirvMask = adaptor.getMask();
    Value spirvValue = adaptor.getValue();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
            typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getContiguity(ptr);
    unsigned numElems = getElemsPerThread(ptr.getType());

    auto ptrElems = getSPIRVElems(ptr, spirvPtr, rewriter, loc);
    auto valueElems = getSPIRVElems(value, spirvValue, rewriter, loc);
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    SmallVector<Value> maskElems;
    if (spirvMask) {
      maskElems = getSPIRVElems(mask, spirvMask, rewriter, loc);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    const size_t dtsize =
            std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNbits = dtsize * 8;

    const int numVecs = numElems / vec;
    assert(vec == 1 && "not support vector store yet");
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is AddPtr with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNbits);
      const size_t totalWidth = valueElemNbits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNbits;
      assert(wordNElems * nWords * numVecs == numElems);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      Value maskVal = spirvMask ? maskElems[vecStart] : int_val(1, 1);

      // scalar store
      // Create block structure for the masked load.
      auto *preheader = rewriter.getInsertionBlock();
      auto opPosition = rewriter.getInsertionPoint();
      auto *tailblock = rewriter.splitBlock(preheader, opPosition);
      auto *condblock = rewriter.createBlock(tailblock);

      // Test the mask
      rewriter.setInsertionPoint(preheader, preheader->end());
      rewriter.create<mlir::cf::CondBranchOp>(loc, maskVal, condblock, tailblock);

      // Do the Store
      rewriter.setInsertionPoint(condblock, condblock->end());
      rewriter.create<spirv::StoreOp>(loc, ptrElems[vecStart], valueElems[vecStart]);
      rewriter.create<mlir::cf::BranchOp>(loc, tailblock);

      rewriter.setInsertionPoint(tailblock, tailblock->begin());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void populateLoadStoreOpToSPIRVPatterns(
        mlir::SPIRVTypeConverter &typeConverter, mlir::MLIRContext *context, RewritePatternSet &patterns,
        int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
        const Allocation *allocation, Value smem,
        ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
        PatternBenefit benefit) {
  patterns.add<LoadOpSPIRVConversion>(typeConverter, context, axisInfoAnalysis, benefit);
  patterns.add<StoreOpSPIRVConversion>(typeConverter, context, axisInfoAnalysis, benefit);
}
