#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Intrinsics.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrderForDotOperand;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

namespace {

using ValueTableV2 = std::map<std::array<int, 3>, Value>;

SmallVector<Value> loadC(Value tensor, Value llTensor, Location loc,
                         ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  size_t fcSize = triton::gpu::getTotalElemsPerThread(tensor.getType());

  assert(isa<NvidiaMmaEncodingAttr>(tensorTy.getEncoding()) &&
         "Currently, we only support $c with a mma layout.");
  auto elems = unpackTensorElements(loc, llTensor, rewriter, tensorTy);
  assert(elems.size() == fcSize);

  auto numMmaRets = tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
  assert(numMmaRets == 8 || numMmaRets == 4 || numMmaRets == 2);
  if (numMmaRets == 8 || numMmaRets == 4) {
    return elems;
  } else if (numMmaRets == 2) {
    auto cPack = SmallVector<Value>();
    auto cElemTy = tensorTy.getElementType();
    int numCPackedElem = 4 / numMmaRets;
    Type cPackTy = vec_ty(cElemTy, numCPackedElem);
    for (int i = 0; i < fcSize; i += numCPackedElem) {
      Value pack = LLVM::UndefOp::create(rewriter, loc, cPackTy);
      for (int j = 0; j < numCPackedElem; ++j) {
        pack = b.insert_element(cPackTy, pack, elems[i + j], b.i32_val(j));
      }
      cPack.push_back(pack);
    }

    return cPack;
  }

  return elems;
}

// Per-thread MMA operand register counts along m, n, k. Registers hold 32 bits,
// or 64 bits for fp64. For m16n8k32 with i8 inputs, the counts are {2, 1, 2},
// with four i8 elements per register.
struct NumRegisters {
  int m;
  int n;
  int k;
};

// Base indices into the per-thread A/B tiles for one MMA.
// BaseOffset::m = NumRegisters.m * m where 0 <= m < repM.
// (Similarly for n and k.)
struct BaseOffset {
  int m;
  int n;
  int k;
};

ValueTableV2 getValuesFromDotOperandLayoutStruct(
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter, Value value, int batch, int repOuter,
    int repK, RankedTensorType type, const NumRegisters &numRegisters) {
  auto ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto elems = unpackTensorElements(loc, value, rewriter, type);
  auto eltTy = typeConverter->convertType(type.getElementType());
  auto bitwidth = eltTy.getIntOrFloatBitWidth();
  int numElemsPerVec = std::max(32 / bitwidth, 1u);

  auto dot = cast<DotOperandEncodingAttr>(type.getEncoding());
  int kWidth = dot.getKWidth();
  int outerRegs = dot.getOpIdx() == 0 ? numRegisters.m : numRegisters.n;

  auto reg = str_attr("register");
  auto srcLayout = triton::gpu::toLinearLayout(type);
  auto removeBroadcast = actionRemoveBroadcastedRegs(srcLayout);
  srcLayout = removeBroadcast.apply(srcLayout);
  elems = removeBroadcast.apply(elems);

  // Size the source groups using only the registers that remain.
  auto elemsPerThread = triton::gpu::LinearEncodingTrait::basesPerDim(
      srcLayout, reg, /*skipBroadcast=*/true);
  auto order =
      getOrderForDotOperand(dot.getOpIdx(), type.getRank(), /*kContig=*/true);
  int outerSize = std::min<int>(outerRegs, elemsPerThread[order[1]]);
  int kTileSize =
      std::min<int>(numRegisters.k, elemsPerThread[order[0]] / kWidth);

  int size = elems.size();
  // Reorder scalar entries in elems into MMA operand registers:
  //   element: scalar within a packed operand (32 bits, or 64 bits for fp64).
  //   k: packed operand position within a contiguous kWidth group.
  //   outer: operand along M for A, or N for B, within one MMA.
  //   kTiles: K operand from another kWidth group, for the same MMA.
  //   tile: group of MMAs, repeated along K, then M/N, then batch.
  //
  // Example: A, fp16, kWidth=4, shape=[16,64], one warp. Entries index the
  // unique elems for one lane. Each pair is (element=0, element=1), packed
  // into one 32-bit operand:
  //
  //                         kTiles=0                 kTiles=1
  //                      k=0       k=1            k=0       k=1
  //   tile=0  outer=0    (0,1)     (2,3)           (8,9)    (10,11)
  //           outer=1    (4,5)     (6,7)          (12,13)   (14,15)
  //   tile=1  outer=0   (16,17)   (18,19)         (24,25)   (26,27)
  //           outer=1   (20,21)   (22,23)         (28,29)   (30,31)
  //
  // Read one k column from both outer rows and both kTiles groups per MMA:
  //   tile=0, k=0 -> [(0,1), (4,5),  (8,9),  (12,13)]
  //   tile=0, k=1 -> [(2,3), (6,7), (10,11), (14,15)]
  //   tile=1 uses the same pattern, adding 16 to each scalar index.
  auto mapping =
      LinearLayout::identity1D(size, reg, reg)
          .reshapeIns(
              {{str_attr("element"), numElemsPerVec},
               {str_attr("k"), kWidth / numElemsPerVec},
               {str_attr("outer"), outerSize},
               {str_attr("kTiles"), kTileSize},
               {str_attr("tile"), size / (kWidth * outerSize * kTileSize)}})
          .transposeIns({str_attr("element"), str_attr("outer"),
                         str_attr("kTiles"), str_attr("k"), str_attr("tile")})
          .reshapeIns({{reg, size}});

  SmallVector<Value> mmaElems;
  for (int i = 0; i < size; ++i)
    mmaElems.push_back(elems[mapping.apply({{reg, i}})[0].second]);

  // Restore only the repeated slots required by the native MMA operands.
  dot = DotOperandEncodingAttr::get(ctx, dot.getOpIdx(), dot.getParent(),
                                    numElemsPerVec);
  elems =
      broadcastAs(mmaElems, triton::gpu::toLinearLayout(type.getShape(), dot));

  ValueTableV2 vals;
  int offset = 0;
  auto packVec = [&](std::array<int, 3> dstIdx) {
    Value vec = b.undef(vec_ty(eltTy, numElemsPerVec));
    for (int i = 0; i < numElemsPerVec; ++i)
      vec = b.insert_element(vec, b.bitcast(elems[offset + i], eltTy),
                             b.i32_val(i));
    vals[dstIdx] = bitwidth == 64 ? vec : b.bitcast(vec, i32_ty);
    offset += numElemsPerVec;
  };

  for (int batchIdx = 0; batchIdx < batch; ++batchIdx)
    for (int outer = 0; outer < repOuter; ++outer)
      for (int k = 0; k < repK; ++k)
        for (int vk = 0; vk < numRegisters.k; ++vk)
          for (int vo = 0; vo < outerRegs; ++vo)
            packVec(
                {batchIdx, outer * outerRegs + vo, k * numRegisters.k + vk});
  return vals;
}

enum class TensorCoreType : uint8_t {
  // floating-point tensor core instr
  FP32_FP16_FP16_FP32 = 0, // default
  FP32_BF16_BF16_FP32,
  FP32_TF32_TF32_FP32,
  FP16_FP16_FP16_FP16,
  // fp32 accumulator, fp8 operand
  FP32_FP8E5M2_FP8E5M2_FP32,
  FP32_FP8E5M2_FP8E4M3FN_FP32,
  FP32_FP8E4M3FN_FP8E5M2_FP32,
  FP32_FP8E4M3FN_FP8E4M3FN_FP32,
  // fp16 accumulator, fp8 operand
  FP16_FP8E5M2_FP8E5M2_FP16,
  FP16_FP8E5M2_FP8E4M3FN_FP16,
  FP16_FP8E4M3FN_FP8E5M2_FP16,
  FP16_FP8E4M3FN_FP8E4M3FN_FP16,
  // integer tensor core instr
  INT32_INT1_INT1_INT32, // Not implemented
  INT32_INT4_INT4_INT32, // Not implemented
  INT32_INT8_INT8_INT32, // Not implemented
  // double precision tensor core instr
  FP64_FP64_FP64_FP64,
  // scaled mxfp8 x mxfp8 matmul
  FP32_FP8E5M2_FP8E5M2_FP32_SCALE_VEC_1X,
  FP32_FP8E5M2_FP8E4M3FN_FP32_SCALE_VEC_1X,
  FP32_FP8E4M3FN_FP8E5M2_FP32_SCALE_VEC_1X,
  FP32_FP8E4M3FN_FP8E4M3FN_FP32_SCALE_VEC_1X,
  //
  FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X,
  FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X,
  //
  NOT_APPLICABLE,
};

static Type getMmaRetType(TensorCoreType mmaType, MLIRContext *ctx) {
  Type fp64Ty = type::f64Ty(ctx);
  Type fp32Ty = type::f32Ty(ctx);
  Type fp16Ty = type::f16Ty(ctx);
  Type i32Ty = type::i32Ty(ctx);
  Type fp64x2Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(2, fp64Ty));
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
  case TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32:
  case TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32:
  case TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32:
  case TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP16_FP8E5M2_FP8E5M2_FP16:
  case TensorCoreType::FP16_FP8E5M2_FP8E4M3FN_FP16:
  case TensorCoreType::FP16_FP8E4M3FN_FP8E5M2_FP16:
  case TensorCoreType::FP16_FP8E4M3FN_FP8E4M3FN_FP16:
    return fp16x2Pack2Ty;
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return i32x4Ty;
  case TensorCoreType::FP64_FP64_FP64_FP64:
    return fp64x2Ty;
  case TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X:
  case TensorCoreType::FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X:
    return fp32x4Ty;
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }

  return Type{};
}

static TensorCoreType getMmaTypeDotScaled(DotScaledOp op, RankedTensorType aTy,
                                          RankedTensorType bTy,
                                          RankedTensorType dTy) {
  if (dTy.getElementType().isF32()) {
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType())) {
      return TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32_SCALE_VEC_1X;
    }
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType())) {
      return TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32_SCALE_VEC_1X;
    }
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType())) {
      return TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32_SCALE_VEC_1X;
    }
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType())) {
      return TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32_SCALE_VEC_1X;
    }
    if (op.getBElemType() == ScaleDotElemType::E2M1 &&
        op.getAElemType() == ScaleDotElemType::E2M1) {
      if (isa<mlir::Float8E4M3FNType>(
              op.getBScale().getType().getElementType())) {
        return TensorCoreType::FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X;
      } else {
        return TensorCoreType::FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X;
      }
    }
  }
  return TensorCoreType::NOT_APPLICABLE;
}

static TensorCoreType getMmaTypeDot(DotOp op, RankedTensorType aTy,
                                    RankedTensorType bTy,
                                    RankedTensorType dTy) {
  if (dTy.getElementType().isF32()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP32_FP16_FP16_FP32;
    if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
      return TensorCoreType::FP32_BF16_BF16_FP32;
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType()))
      return TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32;
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType()))
      return TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32;
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType()))
      return TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32;
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType()))
      return TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32;
    if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
        op.getInputPrecision() == InputPrecision::TF32)
      return TensorCoreType::FP32_TF32_TF32_FP32;
  } else if (dTy.getElementType().isInteger(32)) {
    if (aTy.getElementType().isInteger(8) && bTy.getElementType().isInteger(8))
      return TensorCoreType::INT32_INT8_INT8_INT32;
  } else if (dTy.getElementType().isF16()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP16_FP16_FP16_FP16;
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType()))
      return TensorCoreType::FP16_FP8E5M2_FP8E5M2_FP16;
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType()))
      return TensorCoreType::FP16_FP8E5M2_FP8E4M3FN_FP16;
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType()))
      return TensorCoreType::FP16_FP8E4M3FN_FP8E5M2_FP16;
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType()))
      return TensorCoreType::FP16_FP8E4M3FN_FP8E4M3FN_FP16;
  } else if (dTy.getElementType().isF64()) {
    if (aTy.getElementType().isF64() && bTy.getElementType().isF64())
      return TensorCoreType::FP64_FP64_FP64_FP64;
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

static NVVM::MMATypes getMmaPtxType(Type type) {
  if (type.isF16())
    return NVVM::MMATypes::f16;
  if (type.isBF16())
    return NVVM::MMATypes::bf16;
  if (type.isF32())
    return NVVM::MMATypes::tf32;
  if (type.isF64())
    return NVVM::MMATypes::f64;
  if (type.isInteger(8))
    return NVVM::MMATypes::s8;
  if (isa<Float8E5M2Type>(type))
    return NVVM::MMATypes::e5m2;
  if (isa<Float8E4M3FNType>(type))
    return NVVM::MMATypes::e4m3;
  llvm_unreachable("unsupported MMA operand type");
}

static void callMmaTuringInt8(PTXBuilder &builder, int b,
                              const BaseOffset &base,
                              mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                              unsigned colsPerThread, int numCPackedElem,
                              ValueTableV2 &ha, ValueTableV2 &hb,
                              ArrayRef<Value> fc) {
  auto retArgs1 = builder.newListOperand(numMmaRets / 2, "=r");
  auto retArgs2 = builder.newListOperand(numMmaRets / 2, "=r");
  auto cArgs1 = builder.newListOperand();
  for (int i = 0; i < numMmaRets / 2; ++i) {
    cArgs1->listAppend(builder.newOperand(
        fc[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i],
        std::to_string(i)));
    // reuse the output registers
  }
  auto cArgs2 = builder.newListOperand();
  for (int i = numMmaRets / 2; i < numMmaRets; ++i) {
    cArgs2->listAppend(builder.newOperand(
        fc[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i],
        std::to_string(i)));
    // reuse the output registers
  }
  auto aArgs1 = builder.newListOperand({
      {ha[{b, base.m, base.k}], "r"},
  });
  auto bArgs1 = builder.newListOperand({
      {hb[{b, base.n, base.k}], "r"},
  });
  auto aArgs2 = builder.newListOperand({
      {ha[{b, base.m, base.k + 1}], "r"},
  });
  auto bArgs2 = builder.newListOperand({{hb[{b, base.n, base.k + 1}], "r"}});
  auto aArgs3 = builder.newListOperand({
      {ha[{b, base.m + 1, base.k}], "r"},
  });
  auto bArgs3 = builder.newListOperand({
      {hb[{b, base.n, base.k}], "r"},
  });
  auto aArgs4 = builder.newListOperand({
      {ha[{b, base.m + 1, base.k + 1}], "r"},
  });
  auto bArgs4 = builder.newListOperand({{hb[{b, base.n, base.k + 1}], "r"}});
  mma(retArgs1, aArgs1, bArgs1, cArgs1);
  mma(retArgs1, aArgs2, bArgs2, cArgs1);
  mma(retArgs2, aArgs3, bArgs3, cArgs2);
  mma(retArgs2, aArgs4, bArgs4, cArgs2);
}

static void callMmaTuringFp16(PTXBuilder &builder, int b,
                              const BaseOffset &base,
                              mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                              unsigned colsPerThread, int numCPackedElem,
                              ValueTableV2 &ha, ValueTableV2 &hb,
                              ArrayRef<Value> fc, bool isAccF16) {
  auto retArgs = builder.newListOperand(numMmaRets, isAccF16 ? "=r" : "=f");
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i) {
    cArgs->listAppend(builder.newOperand(
        fc[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i],
        std::to_string(i)));
    // reuse the output registers
  }
  auto aArgs1 = builder.newListOperand({
      {ha[{b, base.m, base.k}], "r"},
      {ha[{b, base.m + 1, base.k}], "r"},
  });
  auto bArgs1 = builder.newListOperand({{hb[{b, base.n, base.k}], "r"}});
  auto aArgs2 = builder.newListOperand({
      {ha[{b, base.m, base.k + 1}], "r"},
      {ha[{b, base.m + 1, base.k + 1}], "r"},
  });
  auto bArgs2 = builder.newListOperand({{hb[{b, base.n, base.k + 1}], "r"}});
  mma(retArgs, aArgs1, bArgs1, cArgs);
  mma(retArgs, aArgs2, bArgs2, cArgs);
}

static std::array<SmallVector<Value>, 3>
getMmaOperands(Location loc, ConversionPatternRewriter &rewriter, int batch,
               const BaseOffset &base, const NumRegisters &numRegisters,
               unsigned numMmaRets, unsigned colsPerThread,
               unsigned batchOffset, Type operandTy, ValueTableV2 &ha,
               ValueTableV2 &hb, ArrayRef<Value> accumulators) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  std::array<SmallVector<Value>, 3> operands;
  for (int k = 0; k < numRegisters.k; ++k)
    for (int m = 0; m < numRegisters.m; ++m)
      operands[0].push_back(
          b.bitcast(ha[{batch, base.m + m, base.k + k}], operandTy));
  for (int k = 0; k < numRegisters.k; ++k)
    operands[1].push_back(
        b.bitcast(hb[{batch, base.n, base.k + k}], operandTy));

  int numCPackedElem = operandTy.isF64() ? 1 : 4 / numMmaRets;
  int offset = (base.m * colsPerThread + numMmaRets * numCPackedElem * base.n) /
                   numCPackedElem +
               batchOffset * batch;
  llvm::append_range(operands[2], accumulators.slice(offset, numMmaRets));
  return operands;
}

static void declareConvergentMma(Operation *mma, llvm::Intrinsic::ID intrinsic,
                                 ConversionPatternRewriter &rewriter) {
  // The pinned LLVM MMA intrinsics omit convergence. Their typed NVVM ops
  // inherit it from this declaration when translated to LLVM IR.
  auto func = triton::gpu::appendOrGetExternFuncOp(
      rewriter, mma, llvm::Intrinsic::getName(intrinsic),
      triton::gpu::getFunctionType(mma->getResult(0).getType(),
                                   mma->getOperands()));
  func.setConvergent(true);
}

using EmitMmaCallback = std::function<Value(
    int b, int m, int n, int k, unsigned numMmaRets, unsigned colsPerThread,
    unsigned batchOffset, ValueTableV2 &ha, ValueTableV2 &hb,
    ArrayRef<Value> fc, Type resultTy, int repK)>;

LogicalResult convertMMAImpl(DotOpInterface op, Value llvmA, Value llvmB,
                             Value llvmC,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter,
                             TensorCoreType mmaType,
                             const NumRegisters &numRegisters,
                             const EmitMmaCallback &emitMma) {
  auto loc = op.getLoc();
  auto aType = cast<RankedTensorType>(op.getA().getType());
  auto bType = cast<RankedTensorType>(op.getB().getType());
  assert(mlir::isa<DotOperandEncodingAttr>(aType.getEncoding()) &&
         mlir::isa<DotOperandEncodingAttr>(bType.getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  Value cOperand = op->getOperand(2);
  auto fc = loadC(cOperand, llvmC, loc, rewriter);

  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto aTensorTy = cast<RankedTensorType>(op.getA().getType());
  auto bTensorTy = cast<RankedTensorType>(op.getB().getType());
  auto dTensorTy = cast<RankedTensorType>(op.getD().getType());

  auto aShapePerCTA = triton::gpu::getShapePerCTA(aTensorTy);
  auto bShapePerCTA = triton::gpu::getShapePerCTA(bTensorTy);
  auto dShapePerCTA = triton::gpu::getShapePerCTA(dTensorTy);

  int bitwidth = aTensorTy.getElementType().getIntOrFloatBitWidth();
  auto dotOpA = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  int kWidth = dotOpA.getKWidth();
  auto repA =
      cast<NvidiaMmaEncodingAttr>(dotOpA.getParent())
          .getRepForOperand(aShapePerCTA, bitwidth, kWidth, dotOpA.getOpIdx());
  auto dotOpB = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  auto repB =
      cast<NvidiaMmaEncodingAttr>(dotOpB.getParent())
          .getRepForOperand(bShapePerCTA, bitwidth, kWidth, dotOpB.getOpIdx());

  assert(repA[2] == repB[1]);
  assert(repA[0] == repB[0]);
  int repM = repA[1], repN = repB[2], repK = repA[2];
  int repBatch = repA[0];

  // We can reuse the same iteration order in
  // getValuesFromDotOperandLayoutStruct as both a and b are K-major
  assert(dotOpA.getRepOrder() == getOrderForDotOperand(dotOpA.getOpIdx(),
                                                       aShapePerCTA.size(),
                                                       /*kContig=*/true));
  auto ha = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                llvmA, repBatch, repM, repK,
                                                aTensorTy, numRegisters);

  assert(dotOpB.getRepOrder() == getOrderForDotOperand(dotOpB.getOpIdx(),
                                                       bShapePerCTA.size(),
                                                       /*kContig=*/true));
  auto hb = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                llvmB, repBatch, repN, repK,
                                                bTensorTy, numRegisters);

  int bitwidthRet = dTensorTy.getElementType().getIntOrFloatBitWidth();
  auto numMmaRets = bitwidthRet == 64 ? 2 : bitwidthRet / 8;
  int numCPackedElem = bitwidthRet == 64 ? 1 : 4 / numMmaRets;

  auto rank = dTensorTy.getRank();
  auto elemsPerThread = triton::gpu::getElemsPerThread(dTensorTy);
  auto batchOffset =
      elemsPerThread[rank - 2] * elemsPerThread[rank - 1] / numCPackedElem;
  Type resultTy = getMmaRetType(mmaType, op->getContext());
  auto callMma = [&](unsigned b, unsigned m, unsigned n, unsigned k) {
    unsigned colsPerThread = repN * 2;
    Value mmaOut = emitMma(b, static_cast<int>(m), static_cast<int>(n),
                           static_cast<int>(k), numMmaRets, colsPerThread,
                           batchOffset, ha, hb, fc, resultTy, repK);

    Type elemTy = cast<LLVM::LLVMStructType>(mmaOut.getType()).getBody()[0];
    for (int i = 0; i < numMmaRets; ++i) {
      fc[(numRegisters.m * static_cast<int>(m) * colsPerThread +
          numMmaRets * numCPackedElem * numRegisters.n * static_cast<int>(n)) /
             numCPackedElem +
         i + batchOffset * b] = tb.extract_val(elemTy, mmaOut, i);
    }
  };

  for (int b = 0; b < repBatch; ++b)
    for (int k = 0; k < repK; ++k)
      for (int m = 0; m < repM; ++m)
        for (int n = 0; n < repN; ++n) {
          callMma(b, m, n, k);
        }

  Type resElemTy = dTensorTy.getElementType();

  // replace with new packed result
  SmallVector<Value> results(fc.size() * numCPackedElem);
  for (int i = 0; i < fc.size(); ++i) {
    for (int j = 0; j < numCPackedElem; ++j) {
      results[i * numCPackedElem + j] =
          numCPackedElem > 1
              ? tb.bitcast(tb.extract_element(fc[i], tb.i32_val(j)), resElemTy)
              : tb.bitcast(fc[i], resElemTy);
    }
  }
  Value res =
      packTensorElements(loc, typeConverter, results, rewriter, dTensorTy);

  rewriter.replaceOp(op, res);

  return success();
}

LogicalResult convertMMAWithType(DotOpInterface op, Value llvmA, Value llvmB,
                                 Value llvmC,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter,
                                 TensorCoreType mmaType, bool isTuring) {
  auto aElemTy = cast<RankedTensorType>(op.getA().getType()).getElementType();
  auto bElemTy = cast<RankedTensorType>(op.getB().getType()).getElementType();
  std::array<NVVM::MMATypes, 2> ptxTypes = {getMmaPtxType(aElemTy),
                                            getMmaPtxType(bElemTy)};
  std::optional<NVVM::MMAIntOverflow> intOverflow;
  if (aElemTy.isInteger(8))
    intOverflow = NVVM::MMAIntOverflow::satfinite;
  std::string mmaInstruction;
  if (isTuring)
    mmaInstruction = mmaInstrPtxTuring.at(mmaType);
  if (auto dotI8 = dyn_cast<triton::instrument::DotI8Op>(op.getOperation())) {
    ptxTypes = {dotI8.getASigned() ? NVVM::MMATypes::s8 : NVVM::MMATypes::u8,
                dotI8.getBSigned() ? NVVM::MMATypes::s8 : NVVM::MMATypes::u8};
    intOverflow = NVVM::MMAIntOverflow::wrapped;
    if (isTuring) {
      mmaInstruction = "mma.sync.aligned.m8n8k16.row.col.s32.";
      mmaInstruction += dotI8.getASigned() ? "s8." : "u8.";
      mmaInstruction += dotI8.getBSigned() ? "s8.s32" : "u8.s32";
    }
  }
  std::array<int64_t, 3> shape = {aElemTy.isF64() ? 8 : 16, 8,
                                  256 / aElemTy.getIntOrFloatBitWidth()};
  NumRegisters numRegisters = (mmaType == TensorCoreType::FP64_FP64_FP64_FP64)
                                  ? NumRegisters{1, 1, 1}
                                  : NumRegisters{2, 1, 2};

  EmitMmaCallback emit = [&](int b, int m, int n, int k, unsigned numMmaRets,
                             unsigned colsPerThread, unsigned batchOffset,
                             ValueTableV2 &ha, ValueTableV2 &hb,
                             ArrayRef<Value> fc, Type resultTy, int /*repK*/) {
    auto dTy = cast<RankedTensorType>(op.getD().getType());
    bool isIntMMA = dTy.getElementType().isInteger(32);
    bool isAccF16 = dTy.getElementType().isF16();
    bool isFp64MMA = dTy.getElementType().isF64();
    const unsigned numCPackedElem = isFp64MMA ? 1u : 4u / numMmaRets;
    BaseOffset base{numRegisters.m * m, numRegisters.n * n, numRegisters.k * k};
    if (isTuring) {
      PTXBuilder builder;
      auto &mma = *builder.create(mmaInstruction);
      assert(b == 0 && "Turing only supports batch size 1");
      if (isIntMMA)
        callMmaTuringInt8(builder, b, base, mma, numMmaRets, colsPerThread,
                          numCPackedElem, ha, hb, fc);
      else
        callMmaTuringFp16(builder, b, base, mma, numMmaRets, colsPerThread,
                          numCPackedElem, ha, hb, fc, isAccF16);
      return builder.launch(rewriter, op.getLoc(), resultTy);
    }

    Type operandTy = rewriter.getI32Type();
    if (isFp64MMA)
      operandTy = rewriter.getF64Type();
    else if (mmaType == TensorCoreType::FP32_FP16_FP16_FP32 ||
             mmaType == TensorCoreType::FP16_FP16_FP16_FP16)
      operandTy = vec_ty(rewriter.getF16Type(), 2);
    auto [aOperands, bOperands, cOperands] =
        getMmaOperands(op.getLoc(), rewriter, b, base, numRegisters, numMmaRets,
                       colsPerThread, batchOffset, operandTy, ha, hb, fc);
    auto mma = NVVM::MmaOp::create(rewriter, op.getLoc(), resultTy, aOperands,
                                   bOperands, cOperands, shape, std::nullopt,
                                   intOverflow, ptxTypes, std::nullopt);
    auto intrinsic = NVVM::MmaOp::getIntrinsicID(
        shape[0], shape[1], shape[2], mma.getB1Op(),
        mma.getIntOverflowBehavior(), mma.getLayoutA(), mma.getLayoutB(),
        *mma.getMultiplicandAPtxType(), *mma.getMultiplicandBPtxType(),
        mma.accumPtxType(), mma.resultPtxType());
    declareConvergentMma(mma, intrinsic, rewriter);
    return mma.getResult();
  };

  return convertMMAImpl(op, llvmA, llvmB, llvmC, typeConverter, rewriter,
                        mmaType, numRegisters, emit);
}

} // namespace

LogicalResult convertMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                         const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring) {
  auto aTensorTy = op.getA().getType();
  auto bTensorTy = op.getB().getType();
  auto dTensorTy = op.getD().getType();

  TensorCoreType mmaType = getMmaTypeDot(op, aTensorTy, bTensorTy, dTensorTy);
  if (mmaType == TensorCoreType::NOT_APPLICABLE ||
      (isTuring && !mmaInstrPtxTuring.contains(mmaType)))
    return op.emitError(
        "unsupported MMA instruction for the given operand/result types");

  return convertMMAWithType(op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
                            typeConverter, rewriter, mmaType, isTuring);
}

LogicalResult convertMMA(triton::instrument::DotI8Op op,
                         triton::instrument::DotI8Op::Adaptor adaptor,
                         const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring) {
  return convertMMAWithType(op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
                            typeConverter, rewriter,
                            TensorCoreType::INT32_INT8_INT8_INT32, isTuring);
}

LogicalResult convertMMADotScaled(triton::DotScaledOp op,
                                  triton::DotScaledOp::Adaptor adaptor,
                                  const LLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter) {
  auto aTensorTy = cast<RankedTensorType>(op.getA().getType());
  auto bTensorTy = cast<RankedTensorType>(op.getB().getType());
  auto dTensorTy = cast<RankedTensorType>(op.getD().getType());

  TensorCoreType mmaType =
      getMmaTypeDotScaled(op, aTensorTy, bTensorTy, dTensorTy);
  if (mmaType == TensorCoreType::NOT_APPLICABLE)
    return op.emitError(
        "unsupported MMA instruction for the given operand/result types");

  SmallVector<Value> unpackedAScale = unpackTensorElements(
      op.getLoc(), adaptor.getAScale(), rewriter, op.getAScale().getType());
  SmallVector<Value> unpackedBScale = unpackTensorElements(
      op.getLoc(), adaptor.getBScale(), rewriter, op.getBScale().getType());

  bool isFp4 = op.getAElemType() == ScaleDotElemType::E2M1;
  std::array<NVVM::MMATypes, 2> ptxTypes = {
      isFp4 ? NVVM::MMATypes::e2m1 : getMmaPtxType(aTensorTy.getElementType()),
      isFp4 ? NVVM::MMATypes::e2m1 : getMmaPtxType(bTensorTy.getElementType())};
  int scaleVecMode = 1;
  auto scaleVecSize = NVVM::ScaleVecSize::X1;
  auto scaleFormat = NVVM::BlockScaleFormat::UE8M0;
  if (mmaType == TensorCoreType::FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X) {
    scaleVecMode = 2;
    scaleVecSize = NVVM::ScaleVecSize::X2;
  } else if (mmaType == TensorCoreType::FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X) {
    scaleVecMode = 4;
    scaleVecSize = NVVM::ScaleVecSize::X4;
    scaleFormat = NVVM::BlockScaleFormat::UE4M3;
  }
  auto kind = isFp4 ? NVVM::MMABlockScaleKind::MXF4NVF4
                    : NVVM::MMABlockScaleKind::MXF8F6F4;
  std::array<int64_t, 3> shape = {16, 8, isFp4 ? 64 : 32};
  NumRegisters numRegisters = {2, 1, 2};
  EmitMmaCallback emit = [&](int b, int m, int n, int k, unsigned numMmaRets,
                             unsigned colsPerThread, unsigned batchOffset,
                             ValueTableV2 &aTable, ValueTableV2 &bTable,
                             ArrayRef<Value> cValues, Type resultTy, int repK) {
    auto tb = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    auto i32 = IntegerType::get(op->getContext(), 32);

    auto packElements = [&](ArrayRef<Value> bytes, int loc,
                            int numBytes) -> Value {
      Value packed = tb.zext(i32, bytes[loc]);
      for (int i = 1; i < numBytes; ++i) {
        Value byte = tb.zext(i32, bytes[loc + i]);
        Value shifted = tb.shl(byte, tb.i32_val(i * 8));
        packed = tb.or_(packed, shifted);
      }
      return packed;
    };

    Value aScaleValue =
        packElements(unpackedAScale, m * repK * scaleVecMode + k * scaleVecMode,
                     scaleVecMode);
    Value bScaleValue =
        packElements(unpackedBScale, n * repK * scaleVecMode + k * scaleVecMode,
                     scaleVecMode);

    BaseOffset base{numRegisters.m * m, numRegisters.n * n, numRegisters.k * k};
    auto [aOperands, bOperands, cOperands] = getMmaOperands(
        op.getLoc(), rewriter, b, base, numRegisters, numMmaRets, colsPerThread,
        batchOffset, i32, aTable, bTable, cValues);
    Value zero = tb.int_val(16, 0);
    auto mma = NVVM::MmaBlockScaleOp::create(
        rewriter, op.getLoc(), resultTy, aOperands, bOperands, cOperands,
        aScaleValue, zero, zero, bScaleValue, zero, zero, shape, ptxTypes,
        scaleVecSize, scaleFormat, kind);
    auto intrinsic = NVVM::MmaBlockScaleOp::getIntrinsicID(
        shape[0], shape[1], shape[2], *mma.getMultiplicandAPtxType(),
        *mma.getMultiplicandBPtxType(), NVVM::MMATypes::f32,
        mma.getScaleVecSize(), mma.getBlockScaleFormat(), mma.getKind());
    declareConvergentMma(mma, intrinsic, rewriter);
    return mma.getResult();
  };

  return convertMMAImpl(op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
                        typeConverter, rewriter, mmaType, numRegisters, emit);
}
