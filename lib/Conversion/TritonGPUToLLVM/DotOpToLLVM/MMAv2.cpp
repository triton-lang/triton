#include "../DotOpToLLVM.h"
#include "../Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;

using ValueTableV2 = std::map<std::pair<unsigned, unsigned>, Value>;

Value loadC(Value tensor, Value llTensor,
            TritonGPUToLLVMTypeConverter *typeConverter, Location loc,
            ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = tensor.getContext();
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  size_t fcSize = triton::gpu::getTotalElemsPerThread(tensor.getType());

  assert(tensorTy.getEncoding().isa<MmaEncodingAttr>() &&
         "Currently, we only support $c with a mma layout.");
  // Load a normal C tensor with mma layout, that should be a
  // LLVM::struct with fcSize elements.
  auto structTy = llTensor.getType().cast<LLVM::LLVMStructType>();
  assert(structTy.getBody().size() == fcSize &&
         "DotOp's $c operand should pass the same number of values as $d in "
         "mma layout.");

  auto numMmaRets = tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
  assert(numMmaRets == 4 || numMmaRets == 2);
  if (numMmaRets == 4) {
    return llTensor;
  } else if (numMmaRets == 2) {
    auto cPack = SmallVector<Value>();
    auto cElemTy = tensorTy.getElementType();
    int numCPackedElem = 4 / numMmaRets;
    Type cPackTy = vec_ty(cElemTy, numCPackedElem);
    for (int i = 0; i < fcSize; i += numCPackedElem) {
      Value pack = rewriter.create<LLVM::UndefOp>(loc, cPackTy);
      for (int j = 0; j < numCPackedElem; ++j) {
        pack = insert_element(
            cPackTy, pack, extract_val(cElemTy, llTensor, i + j), i32_val(j));
      }
      cPack.push_back(pack);
    }

    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(cPack.size(), cPackTy));
    Value result =
        typeConverter->packLLElements(loc, cPack, rewriter, structTy);
    return result;
  }

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
  FP16_FP16_FP16_FP16,
  // integer tensor core instr
  INT32_INT1_INT1_INT32, // Not implemented
  INT32_INT4_INT4_INT32, // Not implemented
  INT32_INT8_INT8_INT32, // Not implemented
  //
  NOT_APPLICABLE,
};

Type getMmaRetType(TensorCoreType mmaType, MLIRContext *ctx) {
  Type fp32Ty = type::f32Ty(ctx);
  Type fp16Ty = type::f16Ty(ctx);
  Type i32Ty = type::i32Ty(ctx);
  Type fp32x4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp32Ty));
  Type i32x4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, i32Ty));
  Type fp16x2Pack2Ty = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(2, vec_ty(fp16Ty, 2)));
  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP16_FP16_FP16_FP16:
    return fp16x2Pack2Ty;
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
  } else if (dTy.getElementType().isF16()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP16_FP16_FP16_FP16;
  }

  return TensorCoreType::NOT_APPLICABLE;
}

inline static const std::map<TensorCoreType, std::string> mmaInstrPtxTuring = {
    {TensorCoreType::FP32_FP16_FP16_FP32,
     "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"},

    {TensorCoreType::INT32_INT8_INT8_INT32,
     "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32"},

    {TensorCoreType::FP16_FP16_FP16_FP16,
     "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16"},
};

inline static const std::map<TensorCoreType, std::string> mmaInstrPtxAmpere = {
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

    {TensorCoreType::FP16_FP16_FP16_FP16,
     "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"},
};

static void callMmaTuringInt8(PTXBuilder &builder, unsigned m, unsigned n,
                              unsigned k, mlir::triton::PTXInstr &mma,
                              unsigned numMmaRets, unsigned colsPerThread,
                              int numCPackedElem, ValueTableV2 &ha,
                              ValueTableV2 &hb, const SmallVector<Value> &fc) {
  auto retArgs1 = builder.newListOperand(numMmaRets / 2, "=r");
  auto retArgs2 = builder.newListOperand(numMmaRets / 2, "=r");
  auto cArgs1 = builder.newListOperand();
  for (int i = 0; i < numMmaRets / 2; ++i) {
    cArgs1->listAppend(
        builder.newOperand(fc[(m * colsPerThread + 4 * n) / numCPackedElem + i],
                           std::to_string(i)));
    // reuse the output registers
  }
  auto cArgs2 = builder.newListOperand();
  for (int i = numMmaRets / 2; i < numMmaRets; ++i) {
    cArgs2->listAppend(
        builder.newOperand(fc[(m * colsPerThread + 4 * n) / numCPackedElem + i],
                           std::to_string(i)));
    // reuse the output registers
  }
  auto aArgs1 = builder.newListOperand({
      {ha[{m, k}], "r"},
  });
  auto bArgs1 = builder.newListOperand({
      {hb[{n, k}], "r"},
  });
  auto aArgs2 = builder.newListOperand({
      {ha[{m, k + 1}], "r"},
  });
  auto bArgs2 = builder.newListOperand({{hb[{n, k + 1}], "r"}});
  auto aArgs3 = builder.newListOperand({
      {ha[{m + 1, k}], "r"},
  });
  auto bArgs3 = builder.newListOperand({
      {hb[{n, k}], "r"},
  });
  auto aArgs4 = builder.newListOperand({
      {ha[{m + 1, k + 1}], "r"},
  });
  auto bArgs4 = builder.newListOperand({{hb[{n, k + 1}], "r"}});
  mma(retArgs1, aArgs1, bArgs1, cArgs1);
  mma(retArgs1, aArgs2, bArgs2, cArgs1);
  mma(retArgs2, aArgs3, bArgs3, cArgs2);
  mma(retArgs2, aArgs4, bArgs4, cArgs2);
}

static void callMmaTuringFp16(PTXBuilder &builder, unsigned m, unsigned n,
                              unsigned k, mlir::triton::PTXInstr &mma,
                              unsigned numMmaRets, unsigned colsPerThread,
                              int numCPackedElem, ValueTableV2 &ha,
                              ValueTableV2 &hb, const SmallVector<Value> &fc,
                              bool isAccF16) {
  auto retArgs = builder.newListOperand(numMmaRets, isAccF16 ? "=r" : "=f");
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i) {
    cArgs->listAppend(
        builder.newOperand(fc[(m * colsPerThread + 4 * n) / numCPackedElem + i],
                           std::to_string(i)));
    // reuse the output registers
  }
  auto aArgs1 = builder.newListOperand({
      {ha[{m, k}], "r"},
      {ha[{m + 1, k}], "r"},
  });
  auto bArgs1 = builder.newListOperand({{hb[{n, k}], "r"}});
  auto aArgs2 = builder.newListOperand({
      {ha[{m, k + 1}], "r"},
      {ha[{m + 1, k + 1}], "r"},
  });
  auto bArgs2 = builder.newListOperand({{hb[{n, k + 1}], "r"}});
  mma(retArgs, aArgs1, bArgs1, cArgs);
  mma(retArgs, aArgs2, bArgs2, cArgs);
}

static void callMmaAmpere(PTXBuilder &builder, unsigned m, unsigned n,
                          unsigned k, mlir::triton::PTXInstr &mma,
                          unsigned numMmaRets, unsigned colsPerThread,
                          int numCPackedElem, ValueTableV2 &ha,
                          ValueTableV2 &hb, const SmallVector<Value> &fc,
                          bool isAccF16, bool isIntMMA) {
  auto retArgs =
      builder.newListOperand(numMmaRets, isIntMMA || isAccF16 ? "=r" : "=f");
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i) {
    cArgs->listAppend(
        builder.newOperand(fc[(m * colsPerThread + 4 * n) / numCPackedElem + i],
                           std::to_string(i)));
    // reuse the output registers
  }
  auto aArgs = builder.newListOperand({
      {ha[{m, k}], "r"},
      {ha[{m + 1, k}], "r"},
      {ha[{m, k + 1}], "r"},
      {ha[{m + 1, k + 1}], "r"},
  });
  auto bArgs =
      builder.newListOperand({{hb[{n, k}], "r"}, {hb[{n, k + 1}], "r"}});
  mma(retArgs, aArgs, bArgs, cArgs);
}

LogicalResult convertDot(TritonGPUToLLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Value a, Value b, Value c, Value d, Value loadedA,
                         Value loadedB, Value loadedC, DotOp op,
                         DotOpAdaptor adaptor, bool isTuring) {
  MLIRContext *ctx = c.getContext();
  auto aTensorTy = a.getType().cast<RankedTensorType>();
  auto bTensorTy = b.getType().cast<RankedTensorType>();
  auto dTensorTy = d.getType().cast<RankedTensorType>();

  auto aShapePerCTA = triton::gpu::getShapePerCTA(aTensorTy);
  auto bShapePerCTA = triton::gpu::getShapePerCTA(bTensorTy);
  auto dShapePerCTA = triton::gpu::getShapePerCTA(dTensorTy);

  int bitwidth = aTensorTy.getElementType().getIntOrFloatBitWidth();
  auto dotOpA = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  auto repA = dotOpA.getParent().cast<MmaEncodingAttr>().getMMAv2Rep(
      aShapePerCTA, bitwidth, dotOpA.getOpIdx());
  auto dotOpB = bTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  auto repB = dotOpB.getParent().cast<MmaEncodingAttr>().getMMAv2Rep(
      bShapePerCTA, bitwidth, dotOpB.getOpIdx());

  assert(repA[1] == repB[0]);
  int repM = repA[0], repN = repB[1], repK = repA[1];

  // shape / shape_per_cta
  auto ha = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                loadedA, repM, repK, aTensorTy);
  auto hb = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                loadedB, std::max(repN / 2, 1),
                                                repK, bTensorTy);
  auto fc = typeConverter->unpackLLElements(loc, loadedC, rewriter, dTensorTy);
  auto numMmaRets = dTensorTy.getElementType().getIntOrFloatBitWidth() / 8;
  int numCPackedElem = 4 / numMmaRets;

  auto mmaType = getMmaType(op);

  const auto &mmaInstructions =
      isTuring ? mmaInstrPtxTuring : mmaInstrPtxAmpere;

  auto callMma = [&](unsigned m, unsigned n, unsigned k) {
    unsigned colsPerThread = repN * 2;
    PTXBuilder builder;
    auto &mma = *builder.create(mmaInstructions.at(mmaType));
    // using =r for float32 works but leads to less readable ptx.
    bool isIntMMA = dTensorTy.getElementType().isInteger(32);
    bool isAccF16 = dTensorTy.getElementType().isF16();

    if (isTuring) {
      if (isIntMMA) // Turing int8
        callMmaTuringInt8(builder, m, n, k, mma, numMmaRets, colsPerThread,
                          numCPackedElem, ha, hb, fc);
      else // Turing fp16
        callMmaTuringFp16(builder, m, n, k, mma, numMmaRets, colsPerThread,
                          numCPackedElem, ha, hb, fc, isAccF16);
    } else { // Ampere
      callMmaAmpere(builder, m, n, k, mma, numMmaRets, colsPerThread,
                    numCPackedElem, ha, hb, fc, isAccF16, isIntMMA);
    }

    Value mmaOut =
        builder.launch(rewriter, loc, getMmaRetType(mmaType, op.getContext()));

    Type elemTy = mmaOut.getType().cast<LLVM::LLVMStructType>().getBody()[0];
    for (int i = 0; i < numMmaRets; ++i) {
      fc[(m * colsPerThread + 4 * n) / numCPackedElem + i] =
          extract_val(elemTy, mmaOut, i);
    }
  };

  for (int k = 0; k < repK; ++k)
    for (int m = 0; m < repM; ++m)
      for (int n = 0; n < repN; ++n)
        callMma(2 * m, n, 2 * k);

  Type resElemTy = dTensorTy.getElementType();

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(fc.size() * numCPackedElem, resElemTy));
  SmallVector<Value> results(fc.size() * numCPackedElem);
  for (int i = 0; i < fc.size(); ++i) {
    for (int j = 0; j < numCPackedElem; ++j) {
      results[i * numCPackedElem + j] =
          numCPackedElem > 1
              ? bitcast(extract_element(fc[i], i32_val(j)), resElemTy)
              : bitcast(fc[i], resElemTy);
    }
  }
  Value res = typeConverter->packLLElements(loc, results, rewriter, structTy);

  rewriter.replaceOp(op, res);

  return success();
}

LogicalResult convertMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                         TritonGPUToLLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring) {
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
  loadedC =
      loadC(op.getC(), adaptor.getC(), typeConverter, op.getLoc(), rewriter);

  return convertDot(typeConverter, rewriter, op.getLoc(), A, B, C, op.getD(),
                    loadedA, loadedB, loadedC, op, adaptor, isTuring);
}

// Convert to mma.m16n8k8
LogicalResult convertMMA1688(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                             TritonGPUToLLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter) {
  return convertMMA(op, adaptor, typeConverter, rewriter, true /*isTuring*/);
}

// Convert to mma.m16n8k16
LogicalResult convertMMA16816(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                              TritonGPUToLLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter) {
  return convertMMA(op, adaptor, typeConverter, rewriter, false /*isTuring*/);
}
