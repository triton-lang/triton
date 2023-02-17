#include "DotOpToLLVM.h"
#include "DotOpHelpers.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::DotOpFMAConversionHelper;
using ::mlir::LLVM::DotOpMmaV1ConversionHelper;
using ::mlir::LLVM::getElementsFromStruct;
using ::mlir::LLVM::getStructFromElements;
using ::mlir::LLVM::MMA16816ConversionHelper;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;

struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::DotOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value A = op.a();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShape = A.getType().cast<RankedTensorType>().getShape();
    size_t reduceAxis = 1;
    unsigned K = AShape[reduceAxis];
    bool isOuter = K == 1;

    MmaEncodingAttr mmaLayout = D.getType()
                                    .cast<RankedTensorType>()
                                    .getEncoding()
                                    .dyn_cast<MmaEncodingAttr>();
    if (!isOuter && mmaLayout && supportMMA(op, mmaLayout.getVersionMajor())) {
      if (mmaLayout.isVolta())
        return convertMMA884(op, adaptor, rewriter);
      if (mmaLayout.isAmpere())
        return convertMMA16816(op, adaptor, rewriter);

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (D.getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .isa<BlockedEncodingAttr>())
      return convertFMADot(op, adaptor, rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }

private:
  // Convert to mma.m16n8k16
  LogicalResult convertMMA16816(triton::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto mmaLayout = op.getResult()
                         .getType()
                         .cast<RankedTensorType>()
                         .getEncoding()
                         .cast<MmaEncodingAttr>();

    Value A = op.a();
    Value B = op.b();
    Value C = op.c();

    MMA16816ConversionHelper mmaHelper(A.getType(), mmaLayout,
                                       getThreadId(rewriter, loc), rewriter,
                                       getTypeConverter(), loc);

    auto ATensorTy = A.getType().cast<RankedTensorType>();
    auto BTensorTy = B.getType().cast<RankedTensorType>();

    assert(ATensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
           BTensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
           "Both $a and %b should be DotOperand layout.");

    Value loadedA, loadedB, loadedC;
    loadedA = adaptor.a();
    loadedB = adaptor.b();
    loadedC = mmaHelper.loadC(op.c(), adaptor.c());

    return mmaHelper.convertDot(A, B, C, op.d(), loadedA, loadedB, loadedC, op,
                                adaptor);
  }
  /// Convert to mma.m8n8k4
  LogicalResult convertMMA884(triton::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto loc = op.getLoc();

    Value A = op.a();
    Value B = op.b();
    Value D = op.getResult();
    auto mmaLayout = D.getType()
                         .cast<RankedTensorType>()
                         .getEncoding()
                         .cast<MmaEncodingAttr>();
    auto ALayout = A.getType()
                       .cast<RankedTensorType>()
                       .getEncoding()
                       .cast<DotOperandEncodingAttr>();
    auto BLayout = B.getType()
                       .cast<RankedTensorType>()
                       .getEncoding()
                       .cast<DotOperandEncodingAttr>();

    auto ATensorTy = A.getType().cast<RankedTensorType>();
    auto BTensorTy = B.getType().cast<RankedTensorType>();
    auto DTensorTy = D.getType().cast<RankedTensorType>();
    auto AShape = ATensorTy.getShape();
    auto BShape = BTensorTy.getShape();

    bool isARow = ALayout.getIsMMAv1Row().cast<BoolAttr>().getValue();
    bool isBRow = BLayout.getIsMMAv1Row().cast<BoolAttr>().getValue();
    auto [isARow_, isBRow_, isAVec4_, isBVec4_, mmaId] =
        mmaLayout.decodeVoltaLayoutStates();
    assert(isARow == isARow_);
    assert(isBRow == isBRow_);

    DotOpMmaV1ConversionHelper helper(mmaLayout);

    unsigned numM = helper.getNumM(AShape[0], isARow, isAVec4_);
    unsigned numN = helper.getNumN(BShape[1], isBRow, isBVec4_);
    unsigned NK = AShape[1];

    auto has = helper.extractLoadedOperand(adaptor.a(), NK, rewriter);
    auto hbs = helper.extractLoadedOperand(adaptor.b(), NK, rewriter);

    // Initialize accumulators with external values, the acc holds the
    // accumulator value that is shared between the MMA instructions inside a
    // DotOp, we can call the order of the values the accumulator-internal
    // order.
    SmallVector<Value> acc = getElementsFromStruct(loc, adaptor.c(), rewriter);
    size_t resSize = acc.size();

    // The resVals holds the final result of the DotOp.
    // NOTE The current order of resVals is different from acc, we call it the
    // accumulator-external order. and
    SmallVector<Value> resVals(resSize);

    auto getIdx = [&](int m, int n) {
      std::vector<size_t> idx{{
          (m * 2 + 0) + (n * 4 + 0) * numM, // row0
          (m * 2 + 0) + (n * 4 + 1) * numM,
          (m * 2 + 1) + (n * 4 + 0) * numM, // row1
          (m * 2 + 1) + (n * 4 + 1) * numM,
          (m * 2 + 0) + (n * 4 + 2) * numM, // row2
          (m * 2 + 0) + (n * 4 + 3) * numM,
          (m * 2 + 1) + (n * 4 + 2) * numM, // row3
          (m * 2 + 1) + (n * 4 + 3) * numM,
      }};
      return idx;
    };

    auto callMMA = [&](unsigned m, unsigned n, unsigned k) {
      auto ha = has.at({m, k});
      auto hb = hbs.at({n, k});

      PTXBuilder builder;
      auto idx = getIdx(m, n);

      // note: using "=f" for float leads to cleaner PTX
      bool isIntMMA = DTensorTy.getElementType().isInteger(32);
      auto *resOprs = builder.newListOperand(8, isIntMMA ? "=r" : "=f");
      auto *AOprs = builder.newListOperand({
          {ha.first, "r"},
          {ha.second, "r"},
      });

      auto *BOprs = builder.newListOperand({
          {hb.first, "r"},
          {hb.second, "r"},
      });
      auto *COprs = builder.newListOperand();
      for (int i = 0; i < 8; ++i)
        COprs->listAppend(builder.newOperand(acc[idx[i]], std::to_string(i)));

      auto mma = builder.create("mma.sync.aligned.m8n8k4")
                     ->o(isARow ? "row" : "col")
                     .o(isBRow ? "row" : "col")
                     .o("f32.f16.f16.f32");

      mma(resOprs, AOprs, BOprs, COprs);

      Value res =
          builder.launch(rewriter, loc, helper.getMmaRetType(ATensorTy));

      for (auto i = 0; i < 8; i++) {
        Value elem = extract_val(f32_ty, res, i32_arr_attr(i));
        acc[idx[i]] = elem;
      }
    };

    for (unsigned k = 0; k < NK; k += 4)
      for (unsigned m = 0; m < numM / 2; ++m)
        for (unsigned n = 0; n < numN / 2; ++n) {
          callMMA(m, n, k);
        }

    // res holds the same layout of acc
    for (size_t i = 0; i < acc.size(); ++i) {
      resVals[i] = acc[i];
    }

    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(resSize, type::f32Ty(ctx)));
    Value res = getStructFromElements(loc, resVals, rewriter, structTy);
    rewriter.replaceOp(op, res);
    return success();
  }

  LogicalResult convertFMADot(triton::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto A = op.a();
    auto B = op.b();
    auto C = op.c();
    auto D = op.getResult();

    auto aTensorTy = A.getType().cast<RankedTensorType>();
    auto bTensorTy = B.getType().cast<RankedTensorType>();
    auto dTensorTy = D.getType().cast<RankedTensorType>();

    auto aShape = aTensorTy.getShape();
    auto bShape = bTensorTy.getShape();

    BlockedEncodingAttr dLayout =
        dTensorTy.getEncoding().cast<BlockedEncodingAttr>();
    auto order = dLayout.getOrder();
    auto cc = getElementsFromStruct(loc, adaptor.c(), rewriter);

    DotOpFMAConversionHelper helper(dLayout);
    Value llA = adaptor.a();
    Value llB = adaptor.b();

    auto sizePerThread = getSizePerThread(dLayout);
    auto shapePerCTA = getShapePerCTA(dLayout);

    int K = aShape[1];
    int M = aShape[0];
    int N = bShape[1];

    int mShapePerCTA =
        order[0] == 1 ? shapePerCTA[order[1]] : shapePerCTA[order[0]];
    int mSizePerThread =
        order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
    int nShapePerCTA =
        order[0] == 0 ? shapePerCTA[order[1]] : shapePerCTA[order[0]];
    int nSizePerThread =
        order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];

    auto has = helper.getValueTableFromStruct(llA, K, M, mShapePerCTA,
                                              mSizePerThread, rewriter, loc);
    auto hbs = helper.getValueTableFromStruct(llB, K, N, nShapePerCTA,
                                              nSizePerThread, rewriter, loc);

    SmallVector<Value> ret = cc;
    bool isCRow = order[0] == 1;

    for (unsigned k = 0; k < K; k++) {
      for (unsigned m = 0; m < M; m += mShapePerCTA)
        for (unsigned n = 0; n < N; n += nShapePerCTA)
          for (unsigned mm = 0; mm < mSizePerThread; ++mm)
            for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
              int mIdx = m / mShapePerCTA * mSizePerThread + mm;
              int nIdx = n / nShapePerCTA * nSizePerThread + nn;

              int z = isCRow ? mIdx * N / nShapePerCTA * mSizePerThread + nIdx
                             : nIdx * M / mShapePerCTA * nSizePerThread + mIdx;
              ret[z] = rewriter.create<LLVM::FMulAddOp>(
                  loc, has[{m + mm, k}], hbs[{n + nn, k}], ret[z]);
            }
    }

    auto res = getStructFromElements(
        loc, ret, rewriter,
        struct_ty(SmallVector<Type>(ret.size(), ret[0].getType())));
    rewriter.replaceOp(op, res);

    return success();
  }
};

void populateDotOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, int numWarps,
                                 AxisInfoAnalysis &axisInfoAnalysis,
                                 const Allocation *allocation, Value smem,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, allocation, smem, benefit);
}
