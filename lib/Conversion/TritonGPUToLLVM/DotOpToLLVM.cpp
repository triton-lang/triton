#include "DotOpToLLVM.h"
#include "DotOpHelpers.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::DotOpFMAConversionHelper;
using ::mlir::LLVM::MMA16816ConversionHelper;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;

// v1
using ValueTable = std::map<std::pair<int, int>, std::pair<Value, Value>>;

static Type getMmaRetType(TensorType operand) {
  auto *ctx = operand.getContext();
  Type fp32Ty = type::f32Ty(ctx);
  // f16*f16+f32->f32
  return struct_ty(SmallVector<Type>{8, fp32Ty});
}

static ValueTable
extractLoadedOperand(Value llStruct, int NK,
                     ConversionPatternRewriter &rewriter,
                     TritonGPUToLLVMTypeConverter *typeConverter, Type type) {
  ValueTable rcds;
  SmallVector<Value> elems = typeConverter->unpackLLElements(
      llStruct.getLoc(), llStruct, rewriter, type);

  int offset = 0;
  for (int i = 0; offset < elems.size(); ++i) {
    for (int k = 0; k < NK; k += 4) {
      rcds[{i, k}] = std::make_pair(elems[offset], elems[offset + 1]);
      offset += 2;
    }
  }

  return rcds;
}

// ---
// v2

typedef MMA16816ConversionHelper::ValueTable ValueTableV2;

Value loadC(Value tensor, Value llTensor) {
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  size_t fcSize = triton::gpu::getElemsPerThread(tensor.getType());

  assert(tensorTy.getEncoding().isa<MmaEncodingAttr>() &&
         "Currently, we only support $c with a mma layout.");
  // Load a normal C tensor with mma layout, that should be a
  // LLVM::struct with fcSize elements.
  auto structTy = llTensor.getType().cast<LLVM::LLVMStructType>();
  assert(structTy.getBody().size() == fcSize &&
         "DotOp's $c operand should pass the same number of values as $d in "
         "mma layout.");
  return llTensor;
}

ValueTableV2 getValuesFromDotOperandLayoutStruct(
    TritonGPUToLLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter, Value value, int n0, int n1,
    RankedTensorType type) {

  auto elems = typeConverter->unpackLLElements(loc, value, rewriter, type);
  int offset{};
  ValueTableV2 vals;
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; j++) {
      vals[{2 * i, 2 * j}] = elems[offset++];
      vals[{2 * i, 2 * j + 1}] = elems[offset++];
      vals[{2 * i + 1, 2 * j}] = elems[offset++];
      vals[{2 * i + 1, 2 * j + 1}] = elems[offset++];
    }
  }
  return vals;
}

enum class TensorCoreType : uint8_t {
  // floating-point tensor core instr
  FP32_FP16_FP16_FP32 = 0, // default
  FP32_BF16_BF16_FP32,
  FP32_TF32_TF32_FP32,
  // integer tensor core instr
  INT32_INT1_INT1_INT32, // Not implemented
  INT32_INT4_INT4_INT32, // Not implemented
  INT32_INT8_INT8_INT32, // Not implemented
  //
  NOT_APPLICABLE,
};

Type getMmaRetType(TensorCoreType mmaType, MLIRContext *ctx) {
  Type fp32Ty = type::f32Ty(ctx);
  Type i32Ty = type::i32Ty(ctx);
  Type fp32x4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp32Ty));
  Type i32x4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, i32Ty));
  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return fp32x4Ty;
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return i32x4Ty;
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }

  return Type{};
}

TensorCoreType getMmaType(triton::DotOp op) {
  Value A = op.getA();
  Value B = op.getB();
  auto aTy = A.getType().cast<RankedTensorType>();
  auto bTy = B.getType().cast<RankedTensorType>();
  // d = a*b + c
  auto dTy = op.getD().getType().cast<RankedTensorType>();

  if (dTy.getElementType().isF32()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP32_FP16_FP16_FP32;
    if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
      return TensorCoreType::FP32_BF16_BF16_FP32;
    if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
        op.getAllowTF32())
      return TensorCoreType::FP32_TF32_TF32_FP32;
  } else if (dTy.getElementType().isInteger(32)) {
    if (aTy.getElementType().isInteger(8) && bTy.getElementType().isInteger(8))
      return TensorCoreType::INT32_INT8_INT8_INT32;
  }

  return TensorCoreType::NOT_APPLICABLE;
}

inline static const std::map<TensorCoreType, std::string> mmaInstrPtx = {
    {TensorCoreType::FP32_FP16_FP16_FP32,
     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"},
    {TensorCoreType::FP32_BF16_BF16_FP32,
     "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"},
    {TensorCoreType::FP32_TF32_TF32_FP32,
     "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"},

    {TensorCoreType::INT32_INT1_INT1_INT32,
     "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc"},
    {TensorCoreType::INT32_INT4_INT4_INT32,
     "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32"},
    {TensorCoreType::INT32_INT8_INT8_INT32,
     "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"},
};

LogicalResult convertDot(TritonGPUToLLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Value a, Value b, Value c, Value d, Value loadedA,
                         Value loadedB, Value loadedC, DotOp op,
                         DotOpAdaptor adaptor) {
  auto aTensorTy = a.getType().cast<RankedTensorType>();
  auto bTensorTy = b.getType().cast<RankedTensorType>();
  auto dTensorTy = d.getType().cast<RankedTensorType>();

  SmallVector<int64_t> aShape(aTensorTy.getShape().begin(),
                              aTensorTy.getShape().end());
  auto dShape = dTensorTy.getShape();
  int bitwidth = aTensorTy.getElementType().getIntOrFloatBitWidth();
  auto repA =
      aTensorTy.getEncoding().cast<DotOperandEncodingAttr>().getMMAv2Rep(
          aTensorTy.getShape(), bitwidth);
  auto repB =
      bTensorTy.getEncoding().cast<DotOperandEncodingAttr>().getMMAv2Rep(
          bTensorTy.getShape(), bitwidth);

  assert(repA[1] == repB[0]);
  int repM = repA[0], repN = repB[1], repK = repA[1];

  // shape / shape_per_cta
  auto ha = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                loadedA, repM, repK, aTensorTy);
  auto hb = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                loadedB, std::max(repN / 2, 1),
                                                repK, bTensorTy);
  auto fc = typeConverter->unpackLLElements(loc, loadedC, rewriter, dTensorTy);

  auto mmaType = getMmaType(op);

  auto callMma = [&](unsigned m, unsigned n, unsigned k) {
    unsigned colsPerThread = repN * 2;
    PTXBuilder builder;
    auto &mma = *builder.create(mmaInstrPtx.at(mmaType));
    // using =r for float32 works but leads to less readable ptx.
    bool isIntMMA = dTensorTy.getElementType().isInteger(32);
    auto retArgs = builder.newListOperand(4, isIntMMA ? "=r" : "=f");
    auto aArgs = builder.newListOperand({
        {ha[{m, k}], "r"},
        {ha[{m + 1, k}], "r"},
        {ha[{m, k + 1}], "r"},
        {ha[{m + 1, k + 1}], "r"},
    });
    auto bArgs =
        builder.newListOperand({{hb[{n, k}], "r"}, {hb[{n, k + 1}], "r"}});
    auto cArgs = builder.newListOperand();
    for (int i = 0; i < 4; ++i) {
      cArgs->listAppend(builder.newOperand(fc[m * colsPerThread + 4 * n + i],
                                           std::to_string(i)));
      // reuse the output registers
    }

    mma(retArgs, aArgs, bArgs, cArgs);
    Value mmaOut =
        builder.launch(rewriter, loc, getMmaRetType(mmaType, op.getContext()));

    Type elemTy = mmaOut.getType().cast<LLVM::LLVMStructType>().getBody()[0];
    for (int i = 0; i < 4; ++i)
      fc[m * colsPerThread + 4 * n + i] = extract_val(elemTy, mmaOut, i);
  };

  for (int k = 0; k < repK; ++k)
    for (int m = 0; m < repM; ++m)
      for (int n = 0; n < repN; ++n)
        callMma(2 * m, n, 2 * k);

  Type resElemTy = dTensorTy.getElementType();

  for (auto &elem : fc) {
    elem = bitcast(elem, resElemTy);
  }

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      op.getContext(), SmallVector<Type>(fc.size(), resElemTy));
  Value res = typeConverter->packLLElements(loc, fc, rewriter, structTy);
  rewriter.replaceOp(op, res);

  return success();
}

// ---
// fma

// ---

struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::DotOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value A = op.getA();
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

    Value A = op.getA();
    Value B = op.getB();
    Value C = op.getC();

    auto ATensorTy = A.getType().cast<RankedTensorType>();
    auto BTensorTy = B.getType().cast<RankedTensorType>();

    assert(ATensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
           BTensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
           "Both $a and %b should be DotOperand layout.");

    Value loadedA, loadedB, loadedC;
    loadedA = adaptor.getA();
    loadedB = adaptor.getB();
    loadedC = loadC(op.getC(), adaptor.getC());

    MMA16816ConversionHelper mmaHelper(A.getType(), mmaLayout,
                                       getThreadId(rewriter, loc), rewriter,
                                       getTypeConverter(), loc);
    // return mmaHelper.convertDot(A, B, C, op.getD(), loadedA, loadedB,
    // loadedC,
    //                             op, adaptor);

    return convertDot(getTypeConverter(), rewriter, op.getLoc(), A, B, C,
                      op.getD(), loadedA, loadedB, loadedC, op, adaptor);
  }
  /// Convert to mma.m8n8k4

  LogicalResult convertMMA884(triton::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto loc = op.getLoc();

    Value A = op.getA();
    Value B = op.getB();
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

    bool isARow = ALayout.getMMAv1IsRow();
    bool isBRow = BLayout.getMMAv1IsRow();
    auto [isARow_, isBRow_, isAVec4_, isBVec4_, _] =
        mmaLayout.decodeVoltaLayoutStates();
    assert(isARow == isARow_);
    assert(isBRow == isBRow_);

    unsigned numM = ALayout.getMMAv1NumOuter(AShape);
    unsigned numN = BLayout.getMMAv1NumOuter(BShape);
    unsigned NK = AShape[1];

    auto has = extractLoadedOperand(adaptor.getA(), NK, rewriter,
                                    getTypeConverter(), ATensorTy);
    auto hbs = extractLoadedOperand(adaptor.getB(), NK, rewriter,
                                    getTypeConverter(), BTensorTy);

    // Initialize accumulators with external values, the acc holds the
    // accumulator value that is shared between the MMA instructions inside a
    // DotOp, we can call the order of the values the accumulator-internal
    // order.
    SmallVector<Value> acc = getTypeConverter()->unpackLLElements(
        loc, adaptor.getC(), rewriter, DTensorTy);
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

      Value res = builder.launch(rewriter, loc, getMmaRetType(ATensorTy));

      for (auto i = 0; i < 8; i++) {
        Value elem = extract_val(f32_ty, res, i);
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

    Value res =
        getTypeConverter()->packLLElements(loc, resVals, rewriter, DTensorTy);
    rewriter.replaceOp(op, res);
    return success();
  }

  LogicalResult convertFMADot(triton::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto A = op.getA();
    auto B = op.getB();
    auto C = op.getC();
    auto D = op.getResult();

    auto aTensorTy = A.getType().cast<RankedTensorType>();
    auto bTensorTy = B.getType().cast<RankedTensorType>();
    auto dTensorTy = D.getType().cast<RankedTensorType>();

    auto aShape = aTensorTy.getShape();
    auto bShape = bTensorTy.getShape();

    BlockedEncodingAttr dLayout =
        dTensorTy.getEncoding().cast<BlockedEncodingAttr>();
    auto order = dLayout.getOrder();
    auto cc = getTypeConverter()->unpackLLElements(loc, adaptor.getC(),
                                                   rewriter, dTensorTy);

    DotOpFMAConversionHelper helper(dLayout);
    Value llA = adaptor.getA();
    Value llB = adaptor.getB();

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
                                              mSizePerThread, rewriter, loc,
                                              getTypeConverter(), aTensorTy);
    auto hbs = helper.getValueTableFromStruct(llB, K, N, nShapePerCTA,
                                              nSizePerThread, rewriter, loc,
                                              getTypeConverter(), bTensorTy);

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

    auto res =
        getTypeConverter()->packLLElements(loc, ret, rewriter, dTensorTy);
    rewriter.replaceOp(op, res);

    return success();
  }
};

void populateDotOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, int numWarps,
                                 AxisInfoAnalysis &axisInfoAnalysis,
                                 const Allocation *allocation, Value smem,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, allocation, smem, benefit);
}
