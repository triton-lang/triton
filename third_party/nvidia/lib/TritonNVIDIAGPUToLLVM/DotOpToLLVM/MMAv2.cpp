#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SmallVector.h"

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
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto elems = unpackTensorElements(loc, value, rewriter, type);
  auto eltTy = typeConverter->convertType(type.getElementType());
  auto bitwidth = eltTy.getIntOrFloatBitWidth();
  int numElemsPerVec = std::max(32 / bitwidth, 1u);
  auto vecTy = vec_ty(eltTy, numElemsPerVec);

  auto dot = cast<DotOperandEncodingAttr>(type.getEncoding());
  int kWidth = dot.getKWidth();
  int outerRegs = dot.getOpIdx() == 0 ? numRegisters.m : numRegisters.n;

  auto reg = rewriter.getStringAttr("register");
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

  auto element = rewriter.getStringAttr("element");
  auto outer = rewriter.getStringAttr("outer");
  auto k = rewriter.getStringAttr("k");
  auto kTiles = rewriter.getStringAttr("kTiles");
  auto tile = rewriter.getStringAttr("tile");
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
          .reshapeIns({{element, numElemsPerVec},
                       {k, kWidth / numElemsPerVec},
                       {outer, outerSize},
                       {kTiles, kTileSize},
                       {tile, size / (kWidth * outerSize * kTileSize)}})
          .transposeIns({element, outer, kTiles, k, tile})
          .reshapeIns({{reg, size}});

  SmallVector<Value> mmaElems;
  for (int i = 0; i < size; ++i)
    mmaElems.push_back(elems[mapping.apply({{reg, i}})[0].second]);

  // Restore only the repeated slots required by the native MMA operands.
  auto mmaDot = DotOperandEncodingAttr::get(type.getContext(), dot.getOpIdx(),
                                            dot.getParent(), numElemsPerVec);
  elems = broadcastAs(mmaElems,
                      triton::gpu::toLinearLayout(type.getShape(), mmaDot));

  ValueTableV2 vals;
  int offset = 0;
  auto packVec = [&](std::array<int, 3> dstIdx) {
    Value vec = b.undef(vecTy);
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

    {TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32,
     "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"},
    {TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32,
     "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32"},
    {TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32,
     "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32"},
    {TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32,
     "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"},

    {TensorCoreType::FP16_FP8E5M2_FP8E5M2_FP16,
     "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16"},
    {TensorCoreType::FP16_FP8E5M2_FP8E4M3FN_FP16,
     "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e4m3.f16"},
    {TensorCoreType::FP16_FP8E4M3FN_FP8E5M2_FP16,
     "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16"},
    {TensorCoreType::FP16_FP8E4M3FN_FP8E4M3FN_FP16,
     "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16"},

    {TensorCoreType::FP64_FP64_FP64_FP64,
     "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"},
};

inline static const std::map<TensorCoreType, std::string> mmaInstrPtxScaled = {
    {TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32_SCALE_VEC_1X,
     "mma.sync.aligned.m16n8k32.row.col."
     "kind::mxf8f6f4.block_scale.scale_vec::"
     "1X.f32.e5m2.e5m2.f32.ue8m0"},
    {TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32_SCALE_VEC_1X,
     "mma.sync.aligned.m16n8k32.row.col."
     "kind::mxf8f6f4.block_scale.scale_vec::"
     "1X.f32.e5m2.e4m3.f32.ue8m0"},
    {TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32_SCALE_VEC_1X,
     "mma.sync.aligned.m16n8k32.row.col."
     "kind::mxf8f6f4.block_scale.scale_vec::"
     "1X.f32.e4m3.e5m2.f32.ue8m0"},
    {TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32_SCALE_VEC_1X,
     "mma.sync.aligned.m16n8k32.row.col."
     "kind::mxf8f6f4.block_scale.scale_vec::"
     "1X.f32.e4m3.e4m3.f32.ue8m0"},
    {TensorCoreType::FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X,
     "mma.sync.aligned.m16n8k64.row.col."
     "kind::mxf4nvf4.block_scale.scale_vec::"
     "2X.f32.e2m1.e2m1.f32.ue8m0"},
    {TensorCoreType::FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X,
     "mma.sync.aligned.m16n8k64.row.col."
     "kind::mxf4nvf4.block_scale.scale_vec::"
     "4X.f32.e2m1.e2m1.f32.ue4m3"},
};

static void callMmaTuringInt8(PTXBuilder &builder, int b,
                              const BaseOffset &base,
                              mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                              unsigned colsPerThread, int numCPackedElem,
                              ValueTableV2 &ha, ValueTableV2 &hb,
                              const SmallVector<Value> &fc) {
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
                              const SmallVector<Value> &fc, bool isAccF16) {
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

// Emit m8n8k4 fp64 MMA instructions.
// With numRegisters.m=1, numRegisters.k=1: emits a single m8n8k4.
// With numRegisters.m=2, numRegisters.k=2: emits 2*2=4 m8n8k4 grouped as
// m16n8k8
static void callMmaAmpereFp64(PTXBuilder &builder, int b,
                              const BaseOffset &base,
                              mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                              unsigned colsPerThread, int numCPackedElem,
                              unsigned batchOffset, ValueTableV2 &ha,
                              ValueTableV2 &hb, const SmallVector<Value> &fc,
                              int kRegs, int mRegs) {
  // Each m sub-tile gets numMmaRets/mRegs results (2 f64 values per m8n8k4).
  int retsPerM = numMmaRets / mRegs;

  // Build ret/c operand lists for each m sub-tile.
  SmallVector<PTXBuilder::Operand *> retArgsList, cArgsList;
  for (int vm = 0; vm < mRegs; ++vm) {
    auto *retArgs = builder.newListOperand(retsPerM, "=d");
    auto *cArgs = builder.newListOperand();
    for (int i = 0; i < retsPerM; ++i) {
      cArgs->listAppend(
          builder.newOperand(fc[((base.m + vm) * colsPerThread +
                                 numMmaRets * numCPackedElem * base.n) /
                                    numCPackedElem +
                                i + batchOffset * b],
                             std::to_string(i)));
    }
    retArgsList.push_back(retArgs);
    cArgsList.push_back(cArgs);
  }

  for (int vk = 0; vk < kRegs; ++vk) {
    auto bArgs = builder.newListOperand({{hb[{b, base.n, base.k + vk}], "d"}});
    for (int vm = 0; vm < mRegs; ++vm) {
      auto aArgs =
          builder.newListOperand({{ha[{b, base.m + vm, base.k + vk}], "d"}});
      mma(retArgsList[vm], aArgs, bArgs, cArgsList[vm]);
    }
  }
}

// Unified MMAV2 function for Ampere and HopperF64 architectures
static void callMmaV2(PTXBuilder &builder, int b, const BaseOffset &base,
                      mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                      unsigned colsPerThread, int numCPackedElem,
                      unsigned batchOffset, ValueTableV2 &ha, ValueTableV2 &hb,
                      const SmallVector<Value> &fc,
                      const std::string &constraintRet,
                      const std::string &constraintAB, int kRegs) {
  auto retArgs = builder.newListOperand(numMmaRets, constraintRet);
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i) {
    cArgs->listAppend(builder.newOperand(
        fc[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i +
           batchOffset * b],
        std::to_string(i)));
    // reuse the output registers
  }

  auto aArgs = builder.newListOperand();
  for (int vk = 0; vk < kRegs; ++vk) {
    aArgs->listAppend(
        builder.newOperand(ha[{b, base.m, base.k + vk}], constraintAB));
    aArgs->listAppend(
        builder.newOperand(ha[{b, base.m + 1, base.k + vk}], constraintAB));
  }

  auto bArgs = builder.newListOperand();
  for (int vk = 0; vk < kRegs; ++vk) {
    bArgs->listAppend(
        builder.newOperand(hb[{b, base.n, base.k + vk}], constraintAB));
  }

  mma(retArgs, aArgs, bArgs, cArgs);
}

static void callMmaScaled(PTXBuilder &builder, int b, const BaseOffset &base,
                          mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                          unsigned colsPerThread, ValueTableV2 &aTable,
                          ValueTableV2 &bTable,
                          const SmallVector<Value> &cValues, Value aScaleValue,
                          Value bScaleValue, int kRegs) {
  int numCPackedElem = 4 / static_cast<int>(numMmaRets);
  auto retArgs = builder.newListOperand(numMmaRets, "=f");
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i)
    cArgs->listAppend(builder.newOperand(
        cValues[(base.m * colsPerThread + 4 * base.n) / numCPackedElem + i],
        std::to_string(i)));

  auto aArgs = builder.newListOperand();
  for (int vk = 0; vk < kRegs; ++vk) {
    aArgs->listAppend(
        builder.newOperand(aTable[{b, base.m, base.k + vk}], "r"));
    aArgs->listAppend(
        builder.newOperand(aTable[{b, base.m + 1, base.k + vk}], "r"));
  }

  auto bArgs = builder.newListOperand();
  for (int vk = 0; vk < kRegs; ++vk)
    bArgs->listAppend(
        builder.newOperand(bTable[{b, base.n, base.k + vk}], "r"));

  SmallVector<PTXBuilder::Operand *> ops{retArgs, aArgs, bArgs, cArgs};

  auto appendScale = [&](Value scale, unsigned byteId, unsigned threadId) {
    ops.push_back(builder.newOperand(scale, "r"));
    auto sel = builder.newListOperand();
    sel->listAppend(builder.newConstantOperand(std::to_string(byteId)));
    sel->listAppend(builder.newConstantOperand(std::to_string(threadId)));
    ops.push_back(sel);
  };

  // Use only byteId=0 since each thread sign-extends a single i8 scale
  // into i32 instead of packing 4 bytes.
  appendScale(aScaleValue, 0, 0);
  appendScale(bScaleValue, 0, 0);

  mma(ops);
}

using EmitMmaCallback = std::function<void(
    PTXBuilder &builder, int b, int m, int n, int k,
    mlir::triton::PTXInstr &mma, unsigned numMmaRets, unsigned colsPerThread,
    unsigned batchOffset, ValueTableV2 &ha, ValueTableV2 &hb,
    const SmallVector<Value> &fc, RankedTensorType dTensorTy, int repK)>;

LogicalResult convertMMAImpl(
    DotOpInterface op, Value llvmA, Value llvmB, Value llvmC,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    TensorCoreType mmaType, const NumRegisters &numRegisters,
    const std::string &mmaInstruction, const EmitMmaCallback &emitMma) {
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
  auto callMma = [&](unsigned b, unsigned m, unsigned n, unsigned k) {
    PTXBuilder builder;
    auto &mma = *builder.create(mmaInstruction);
    // using =r for float32 works but leads to less readable ptx.
    unsigned colsPerThread = repN * 2;
    emitMma(builder, b, static_cast<int>(m), static_cast<int>(n),
            static_cast<int>(k), mma, numMmaRets, colsPerThread, batchOffset,
            ha, hb, fc, dTensorTy, repK);

    Value mmaOut =
        builder.launch(rewriter, loc, getMmaRetType(mmaType, op->getContext()));

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

LogicalResult convertMMAWithInstruction(
    DotOpInterface op, Value llvmA, Value llvmB, Value llvmC,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    TensorCoreType mmaType, const std::string &mmaInstruction, bool isTuring) {
  NumRegisters numRegisters = (mmaType == TensorCoreType::FP64_FP64_FP64_FP64)
                                  ? NumRegisters{1, 1, 1}
                                  : NumRegisters{2, 1, 2};

  EmitMmaCallback emit = [&](PTXBuilder &builder, int b, int m, int n, int k,
                             mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                             unsigned colsPerThread, unsigned batchOffset,
                             ValueTableV2 &ha, ValueTableV2 &hb,
                             const SmallVector<Value> &fc, RankedTensorType dTy,
                             int /*repK*/) {
    bool isIntMMA = dTy.getElementType().isInteger(32);
    bool isAccF16 = dTy.getElementType().isF16();
    bool isFp64MMA = dTy.getElementType().isF64();
    const unsigned numCPackedElem = isFp64MMA ? 1u : 4u / numMmaRets;
    BaseOffset base{numRegisters.m * m, numRegisters.n * n, numRegisters.k * k};
    if (isTuring) {
      assert(b == 0 && "Turing only supports batch size 1");
      if (isIntMMA)
        callMmaTuringInt8(builder, b, base, mma, numMmaRets, colsPerThread,
                          numCPackedElem, ha, hb, fc);
      else
        callMmaTuringFp16(builder, b, base, mma, numMmaRets, colsPerThread,
                          numCPackedElem, ha, hb, fc, isAccF16);
    } else {
      if (isFp64MMA) {
        callMmaAmpereFp64(builder, b, base, mma, numMmaRets, colsPerThread,
                          numCPackedElem, batchOffset, ha, hb, fc,
                          numRegisters.k, numRegisters.m);
      } else {
        callMmaV2(builder, b, base, mma, numMmaRets, colsPerThread,
                  numCPackedElem, batchOffset, ha, hb, fc,
                  isIntMMA || isAccF16 ? "=r" : "=f", "r", numRegisters.k);
      }
    }
  };

  return convertMMAImpl(op, llvmA, llvmB, llvmC, typeConverter, rewriter,
                        mmaType, numRegisters, mmaInstruction, emit);
}

} // namespace

LogicalResult convertMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                         const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring) {
  auto aTensorTy = op.getA().getType();
  auto bTensorTy = op.getB().getType();
  auto dTensorTy = op.getD().getType();

  TensorCoreType mmaType = getMmaTypeDot(op, aTensorTy, bTensorTy, dTensorTy);
  const auto &instrMap = isTuring ? mmaInstrPtxTuring : mmaInstrPtxAmpere;
  if (instrMap.find(mmaType) == instrMap.end())
    return op.emitError(
        "unsupported MMA instruction for the given operand/result types");

  return convertMMAWithInstruction(op, adaptor.getA(), adaptor.getB(),
                                   adaptor.getC(), typeConverter, rewriter,
                                   mmaType, instrMap.at(mmaType), isTuring);
}

LogicalResult convertMMA(triton::instrument::DotI8Op op,
                         triton::instrument::DotI8Op::Adaptor adaptor,
                         const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring) {
  std::string mmaInstruction = isTuring
                                   ? "mma.sync.aligned.m8n8k16.row.col.s32."
                                   : "mma.sync.aligned.m16n8k32.row.col.s32.";
  mmaInstruction += op.getASigned() ? "s8." : "u8.";
  mmaInstruction += op.getBSigned() ? "s8.s32" : "u8.s32";
  return convertMMAWithInstruction(op, adaptor.getA(), adaptor.getB(),
                                   adaptor.getC(), typeConverter, rewriter,
                                   TensorCoreType::INT32_INT8_INT8_INT32,
                                   mmaInstruction, isTuring);
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
  if (mmaInstrPtxScaled.find(mmaType) == mmaInstrPtxScaled.end())
    return op.emitError(
        "unsupported MMA instruction for the given operand/result types");

  SmallVector<Value> unpackedAScale = unpackTensorElements(
      op.getLoc(), adaptor.getAScale(), rewriter, op.getAScale().getType());
  SmallVector<Value> unpackedBScale = unpackTensorElements(
      op.getLoc(), adaptor.getBScale(), rewriter, op.getBScale().getType());

  NumRegisters numRegisters = {2, 1, 2};
  EmitMmaCallback emit = [&](PTXBuilder &builder, int b, int m, int n, int k,
                             mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                             unsigned colsPerThread, unsigned batchOffset,
                             ValueTableV2 &aTable, ValueTableV2 &bTable,
                             const SmallVector<Value> &cValues,
                             RankedTensorType dTy, int repK) {
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

    int scaleVecMode;
    if (mmaInstrPtxScaled.at(mmaType).find("1X") != std::string::npos) {
      scaleVecMode = 1;
    } else if (mmaType ==
               TensorCoreType::FP32_FP4E2M1_FP4E2M1_FP32_SCALE_VEC_2X) {
      scaleVecMode = 2;
    } else if (mmaType == TensorCoreType::FP32_NVFP4_NVFP4_FP32_SCALE_VEC_4X) {
      scaleVecMode = 4;
    } else {
      llvm_unreachable("Unsupported scale vector mode!");
    }
    Value aScaleValue =
        packElements(unpackedAScale, m * repK * scaleVecMode + k * scaleVecMode,
                     scaleVecMode);
    Value bScaleValue =
        packElements(unpackedBScale, n * repK * scaleVecMode + k * scaleVecMode,
                     scaleVecMode);

    BaseOffset base{numRegisters.m * m, numRegisters.n * n, numRegisters.k * k};
    callMmaScaled(builder, b, base, mma, numMmaRets, colsPerThread, aTable,
                  bTable, cValues, aScaleValue, bScaleValue, numRegisters.k);
  };

  return convertMMAImpl(op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
                        typeConverter, rewriter, mmaType, numRegisters,
                        mmaInstrPtxScaled.at(mmaType), emit);
}
