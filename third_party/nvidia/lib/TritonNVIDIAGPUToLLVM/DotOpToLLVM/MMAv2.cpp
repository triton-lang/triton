#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrderForDotOperand;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

using ValueTableV2 = std::map<std::array<int, 3>, Value>;

Value loadC(Value tensor, Value llTensor,
            const LLVMTypeConverter *typeConverter, Location loc,
            ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = tensor.getContext();
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  size_t fcSize = triton::gpu::getTotalElemsPerThread(tensor.getType());

  assert(isa<NvidiaMmaEncodingAttr>(tensorTy.getEncoding()) &&
         "Currently, we only support $c with a mma layout.");
  // Load a normal C tensor with mma layout, that should be a
  // LLVM::struct with fcSize elements.
  auto structTy = cast<LLVM::LLVMStructType>(llTensor.getType());
  assert(structTy.getBody().size() == fcSize &&
         "DotOp's $c operand should pass the same number of values as $d in "
         "mma layout.");

  auto numMmaRets = tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
  assert(numMmaRets == 8 || numMmaRets == 4 || numMmaRets == 2);
  if (numMmaRets == 8 || numMmaRets == 4) {
    return llTensor;
  } else if (numMmaRets == 2) {
    auto cPack = SmallVector<Value>();
    auto cElemTy = tensorTy.getElementType();
    int numCPackedElem = 4 / numMmaRets;
    Type cPackTy = vec_ty(cElemTy, numCPackedElem);
    for (int i = 0; i < fcSize; i += numCPackedElem) {
      Value pack = rewriter.create<LLVM::UndefOp>(loc, cPackTy);
      for (int j = 0; j < numCPackedElem; ++j) {
        pack = b.insert_element(cPackTy, pack,
                                b.extract_val(cElemTy, llTensor, i + j),
                                b.i32_val(j));
      }
      cPack.push_back(pack);
    }

    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(cPack.size(), cPackTy));
    Value result =
        packLLElements(loc, typeConverter, cPack, rewriter, structTy);
    return result;
  }

  return llTensor;
}

ValueTableV2 getValuesFromDotOperandLayoutStruct(
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter, Value value, int batch, int repOuter,
    int repK, RankedTensorType type) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto elems = unpackLLElements(loc, value, rewriter);
  auto eltTy = typeConverter->convertType(type.getElementType());
  int offset{};
  ValueTableV2 vals;
  auto bitwidth = eltTy.getIntOrFloatBitWidth();
  auto numElemsPerVec = std::max(32 / bitwidth, 1u);
  auto vecTy = vec_ty(eltTy, numElemsPerVec);

  auto packVec = [&](std::array<int, 3> dstIdx) {
    Value vec = b.undef(vecTy);
    for (auto i = 0; i < numElemsPerVec; ++i) {
      vec = b.insert_element(vec, b.bitcast(elems[offset + i], eltTy),
                             b.i32_val(i));
    }
    if (bitwidth == 64) {
      vals[dstIdx] = vec;
    } else {
      vals[dstIdx] = b.bitcast(vec, i32_ty);
    }
    offset += numElemsPerVec;
  };

  auto dot = cast<DotOperandEncodingAttr>(type.getEncoding());
  auto kWidth = dot.getKWidth();
  auto largeK = bitwidth * kWidth > std::max(32u, bitwidth);

  assert((bitwidth != 64 || largeK == false) &&
         "Currently fp64 don't support largeK MMA");

  if (largeK) {
    // For layouts with a large K dimension, the original register layout needs
    // to be divided into multiple MMAs, where each MMA has contiguous 32 bits
    // along the K dimension per thread.
    // Using kWidth = 8 and bitwidth = 2 as an example,
    // we split the MMA into 4 sub-MMAs, each with a stride 4 x 32-bit along the
    // K dimension.
    llvm::SmallVector<unsigned> si;
    auto kIters = kWidth / (std::max(32 / bitwidth, 1u));

    if (dot.getOpIdx() == 0) {
      // Original register layout:
      //
      //   [0, 1, 2, 3, 4, 5, 6, 7], [16, 17, 18, 19, 20, 21, 22, 23, 23]
      //   [8, 9, 10, 11, 12, 13, 14, 15], [24, 25, 26, 27, 28, 29, 30, 31]
      //
      // Each element in the layout is a single bf16.
      //
      // To derive four independent MMA operations, a stride of 4 is applied to
      // the original register layout:
      //
      //  1st MMA: [[0, 1], [8, 9], [16, 17], [24, 25]]
      //  2nd MMA: [[2, 3], [10, 11], [18, 19], [26, 27]]
      //  3rd MMA: [[4, 5], [12, 13], [20, 21], [28, 29]]
      //  4th MMA: [[6, 7], [14, 15], [22, 23], [30, 31]]
      if (kIters <= repK) {
        for (size_t kRep = 0; kRep < kWidth / numElemsPerVec; ++kRep)
          for (size_t tile = 0; tile < 4; ++tile)
            for (size_t e = 0; e < numElemsPerVec; ++e) {
              si.push_back(kRep * numElemsPerVec + tile * kWidth + e);
            }
      } else {
        // Suppose kWidth=4 and type=fp32, so numElemsPerVec=1.
        // Each tile of the dot operand layout has a size of 16x32.
        // However, if the triton tensor size is 16x16, elements along the k
        // dimension are duplicated. Within each tile, each register
        // contains 2x8 elements arranged as follows:
        //
        //       tile0/0           tile0/1
        //   |<--kWidth=4-->|   |<--kWidth-->|
        //   |<-mmaWidth=2->|
        //   [0,  1,  2,  3]    [0,  1,  2,  3]
        //   [4,  5,  6,  7]    [4,  5,  6,  7]
        //
        // tile0/1 replicates the elements in tile0/0 along the k dimension.
        // For a tensor size of 32x32, the next tile on the m dimension is as
        // follows:
        //
        //       tile1/0              tile1/1
        //   |<--kWidth-->|       |<--kWidth-->|
        //   [8,  9, 10, 11],     [8,  9, 10, 11]
        //   [12, 13, 14, 15],    [12, 13, 14, 15]
        //
        // Within a single tile, we can perform two MMAs, and the
        // resulting register layout for each MMA is as follows:
        //
        //   1st MMA: [0, 4, 1, 5]
        //   2nd MMA: [2, 6, 3, 7]
        //   3rd MMA: [8, 12, 9, 13]
        //   4th MMA: [10, 14, 11, 15]
        //
        // Additionally, we should reorder the elements by moving the duplicated
        // elements to the end.  In the example above, we convert the order from
        // tile0/0, tile0/1, tile1/0, tile1/1 to tile0/0, tile1/0, tile0/1,
        // tile1/1, so that only the first two tiles will be used in the
        // computation.
        size_t elemsPerTile = 2 * 2 * kWidth;
        size_t elemsPerMma = 2 * 2 * numElemsPerVec;
        size_t mmaWidth = kWidth / numElemsPerVec / 2;
        size_t repMma = elemsPerTile / (mmaWidth * elemsPerMma);
        for (size_t rep = 0; rep < repMma; ++rep)
          for (size_t tile = 0; tile < elems.size() / elemsPerTile; ++tile)
            for (size_t mmaKWidth = 0; mmaKWidth < mmaWidth; ++mmaKWidth)
              for (size_t kTile = 0; kTile < 2; ++kTile)
                for (size_t mTile = 0; mTile < 2; ++mTile)
                  for (size_t e = 0; e < numElemsPerVec; ++e) {
                    si.push_back(rep * mmaWidth * elemsPerMma +
                                 mmaKWidth * 2 * numElemsPerVec +
                                 tile * elemsPerTile + mTile * kWidth +
                                 kTile * numElemsPerVec + e);
                  }
      }
    } else {
      // Original register layout:
      //
      //   [0, 1, 2, 3, 4, 5, 6, 7]^T, [8, 9, 10, 11, 12, 13, 14, 15]^T
      //
      // A stride of 4 is applied to derive four independent MMA operations:
      //
      //  1st MMA: [[0, 1], [8, 9]]
      //  2nd MMA: [[2, 3], [10, 11]]
      //  3rd MMA: [[4, 5], [12, 13]]
      //  4th MMA: [[6, 7], [14, 15]]
      if (kIters <= repK) {
        for (size_t kRep = 0; kRep < kWidth / numElemsPerVec; ++kRep)
          for (size_t tile = 0; tile < 2; ++tile)
            for (size_t e = 0; e < numElemsPerVec; ++e) {
              si.push_back(kRep * numElemsPerVec + tile * kWidth + e);
            }
      } else {
        // Suppose kWidth=4 and type=fp32.
        // Original register layout:
        //
        //       tile0/0        tile0/1
        //   [0, 1, 2, 3]^T, [0, 1, 2, 3]^T
        //
        // Similar to the opIdx=0 situation, we should reorder the elements by
        // moving the duplicated elements to the end.
        size_t elemsPerTile = 2 * kWidth;
        size_t elemsPerMma = 2 * numElemsPerVec;
        size_t mmaWidth = kWidth / numElemsPerVec / 2;
        size_t repMma = elemsPerTile / (mmaWidth * elemsPerMma);
        for (size_t rep = 0; rep < repMma; ++rep)
          for (size_t tile = 0; tile < elems.size() / elemsPerTile; ++tile)
            for (size_t mmaKWidth = 0; mmaKWidth < mmaWidth; ++mmaKWidth)
              for (size_t kTile = 0; kTile < 2; ++kTile)
                for (size_t e = 0; e < numElemsPerVec; ++e) {
                  si.push_back(rep * mmaWidth * elemsPerMma +
                               mmaKWidth * 2 * numElemsPerVec +
                               tile * elemsPerTile + kTile * numElemsPerVec +
                               e);
                }
      }
    }

    auto step = si.size();
    SmallVector<Value> perm(step);
    for (auto i = 0; i < elems.size() / step; ++i) {
      for (auto j = 0; j < step; ++j) {
        perm[j] = elems[i * step + si[j]];
      }
      std::copy(perm.begin(), perm.end(), elems.begin() + i * step);
    }
  }

  auto numVecM = 2;
  auto numVecN = 1;
  auto numVecK = bitwidth == 64 ? 4 : 2;

  if (dot.getOpIdx() == 0) {
    for (auto b = 0; b < batch; ++b)
      for (auto m = 0; m < repOuter; ++m)
        for (auto k = 0; k < repK; ++k)
          for (auto vk = 0; vk < numVecK; ++vk)
            for (auto vm = 0; vm < numVecM; ++vm)
              packVec({b, m * numVecM + vm, k * numVecK + vk});
  } else {
    for (auto b = 0; b < batch; ++b)
      for (auto n = 0; n < repOuter; ++n)
        for (auto k = 0; k < repK; ++k)
          for (auto vk = 0; vk < numVecK; ++vk)
            for (auto vn = 0; vn < numVecN; ++vn)
              packVec({b, n * numVecN + vn, k * numVecK + vk});
  }
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
  NOT_APPLICABLE,
};

static Type getMmaRetType(TensorCoreType mmaType, MLIRContext *ctx) {
  Type fp64Ty = type::f64Ty(ctx);
  Type fp32Ty = type::f32Ty(ctx);
  Type fp16Ty = type::f16Ty(ctx);
  Type i32Ty = type::i32Ty(ctx);
  Type fp64x4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp64Ty));
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
    return fp64x4Ty;
  case TensorCoreType::FP32_FP8E5M2_FP8E5M2_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP8E5M2_FP8E4M3FN_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP8E4M3FN_FP8E5M2_FP32_SCALE_VEC_1X:
  case TensorCoreType::FP32_FP8E4M3FN_FP8E4M3FN_FP32_SCALE_VEC_1X:
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

inline static const std::map<TensorCoreType, std::string> mmaInstrPtxHopper = {
    {TensorCoreType::FP64_FP64_FP64_FP64,
     "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64"},
};

inline static const std::map<TensorCoreType, std::string> mmaInstrPtxScaled = {
    // 1X variants (default kind::mxf8f6f4). We may switch to mxfp4 dynamically.
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
};

static void callMmaTuringInt8(PTXBuilder &builder, int b, int m, int n, int k,
                              mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                              unsigned colsPerThread, int numCPackedElem,
                              ValueTableV2 &ha, ValueTableV2 &hb,
                              const SmallVector<Value> &fc) {
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
      {ha[{b, m, k}], "r"},
  });
  auto bArgs1 = builder.newListOperand({
      {hb[{b, n, k}], "r"},
  });
  auto aArgs2 = builder.newListOperand({
      {ha[{b, m, k + 1}], "r"},
  });
  auto bArgs2 = builder.newListOperand({{hb[{b, n, k + 1}], "r"}});
  auto aArgs3 = builder.newListOperand({
      {ha[{b, m + 1, k}], "r"},
  });
  auto bArgs3 = builder.newListOperand({
      {hb[{b, n, k}], "r"},
  });
  auto aArgs4 = builder.newListOperand({
      {ha[{b, m + 1, k + 1}], "r"},
  });
  auto bArgs4 = builder.newListOperand({{hb[{b, n, k + 1}], "r"}});
  mma(retArgs1, aArgs1, bArgs1, cArgs1);
  mma(retArgs1, aArgs2, bArgs2, cArgs1);
  mma(retArgs2, aArgs3, bArgs3, cArgs2);
  mma(retArgs2, aArgs4, bArgs4, cArgs2);
}

static void callMmaTuringFp16(PTXBuilder &builder, int b, int m, int n, int k,
                              mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                              unsigned colsPerThread, int numCPackedElem,
                              ValueTableV2 &ha, ValueTableV2 &hb,
                              const SmallVector<Value> &fc, bool isAccF16) {
  auto retArgs = builder.newListOperand(numMmaRets, isAccF16 ? "=r" : "=f");
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i) {
    cArgs->listAppend(
        builder.newOperand(fc[(m * colsPerThread + 4 * n) / numCPackedElem + i],
                           std::to_string(i)));
    // reuse the output registers
  }
  auto aArgs1 = builder.newListOperand({
      {ha[{b, m, k}], "r"},
      {ha[{b, m + 1, k}], "r"},
  });
  auto bArgs1 = builder.newListOperand({{hb[{b, n, k}], "r"}});
  auto aArgs2 = builder.newListOperand({
      {ha[{b, m, k + 1}], "r"},
      {ha[{b, m + 1, k + 1}], "r"},
  });
  auto bArgs2 = builder.newListOperand({{hb[{b, n, k + 1}], "r"}});
  mma(retArgs, aArgs1, bArgs1, cArgs);
  mma(retArgs, aArgs2, bArgs2, cArgs);
}

// Repeat m8n8k4 (2, 1, 4) times, as m16n8k16 on hopper.
static void callMmaAmpereFp64(PTXBuilder &builder, int b, int m, int n, int k,
                              mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                              unsigned colsPerThread, int numCPackedElem,
                              unsigned batchOffset, ValueTableV2 &ha,
                              ValueTableV2 &hb, const SmallVector<Value> &fc) {
  auto retArgs1 = builder.newListOperand(numMmaRets / 2, "=d");
  auto retArgs2 = builder.newListOperand(numMmaRets / 2, "=d");
  auto cArgs1 = builder.newListOperand();
  for (int i = 0; i < numMmaRets / 2; ++i) {
    cArgs1->listAppend(builder.newOperand(
        fc[(m * colsPerThread + 4 * n) / numCPackedElem + i + batchOffset * b],
        std::to_string(i)));
    // reuse the output registers
  }
  auto cArgs2 = builder.newListOperand();
  for (int i = numMmaRets / 2; i < numMmaRets; ++i) {
    cArgs2->listAppend(builder.newOperand(
        fc[(m * colsPerThread + 4 * n) / numCPackedElem + i + batchOffset * b],
        std::to_string(i)));
    // reuse the output registers
  }

  for (int vk = 0; vk < 4; ++vk) {
    auto aArgs1 = builder.newListOperand({
        {ha[{b, m, k + vk}], "d"},
    });
    auto bArgs = builder.newListOperand({{hb[{b, n, k + vk}], "d"}});
    auto aArgs2 = builder.newListOperand({
        {ha[{b, m + 1, k + vk}], "d"},
    });
    mma(retArgs1, aArgs1, bArgs, cArgs1);
    mma(retArgs2, aArgs2, bArgs, cArgs2);
  }
}

// Unified MMAV2 function for Ampere and HopperF64 architectures
static void callMmaV2(PTXBuilder &builder, int b, int m, int n, int k,
                      mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                      unsigned colsPerThread, int numCPackedElem,
                      unsigned batchOffset, ValueTableV2 &ha, ValueTableV2 &hb,
                      const SmallVector<Value> &fc,
                      const std::string &constraintRet,
                      const std::string &constraintAB, int numVecK) {
  auto retArgs = builder.newListOperand(numMmaRets, constraintRet);
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i) {
    cArgs->listAppend(builder.newOperand(
        fc[(m * colsPerThread + 4 * n) / numCPackedElem + i + batchOffset * b],
        std::to_string(i)));
    // reuse the output registers
  }

  auto aArgs = builder.newListOperand();
  for (int vk = 0; vk < numVecK; ++vk) {
    aArgs->listAppend(builder.newOperand(ha[{b, m, k + vk}], constraintAB));
    aArgs->listAppend(builder.newOperand(ha[{b, m + 1, k + vk}], constraintAB));
  }

  auto bArgs = builder.newListOperand();
  for (int vk = 0; vk < numVecK; ++vk) {
    bArgs->listAppend(builder.newOperand(hb[{b, n, k + vk}], constraintAB));
  }

  mma(retArgs, aArgs, bArgs, cArgs);
}

static void callMmaScaled(PTXBuilder &builder, int b, int m, int n, int k,
                          mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                          unsigned colsPerThread, ValueTableV2 &aTable,
                          ValueTableV2 &bTable,
                          const SmallVector<Value> &cValues, Value aScaleValue,
                          Value bScaleValue) {
  int numCPackedElem = 4 / static_cast<int>(numMmaRets);
  auto retArgs = builder.newListOperand(numMmaRets, "=f");
  auto cArgs = builder.newListOperand();
  for (int i = 0; i < numMmaRets; ++i)
    cArgs->listAppend(builder.newOperand(
        cValues[(m * colsPerThread + 4 * n) / numCPackedElem + i],
        std::to_string(i)));

  auto aArgs = builder.newListOperand();
  for (int vk = 0; vk < 2; ++vk) {
    aArgs->listAppend(builder.newOperand(aTable[{b, m, k + vk}], "r"));
    aArgs->listAppend(builder.newOperand(aTable[{b, m + 1, k + vk}], "r"));
  }

  auto bArgs = builder.newListOperand();
  for (int vk = 0; vk < 2; ++vk)
    bArgs->listAppend(builder.newOperand(bTable[{b, n, k + vk}], "r"));

  SmallVector<PTXBuilder::Operand *> ops{retArgs, aArgs, bArgs, cArgs};

  auto appendScale = [&](Value scale, unsigned byteId, unsigned threadId) {
    ops.push_back(builder.newOperand(scale, "r"));
    auto sel = builder.newListOperand();
    sel->listAppend(builder.newConstantOperand(std::to_string(byteId)));
    sel->listAppend(builder.newConstantOperand(std::to_string(threadId)));
    ops.push_back(sel);
  };

  unsigned aByte = (m / 2) & 0x3;
  unsigned bByte = n & 0x3;
  // byteId, threadId selection logic for the scale factor
  // depends on getSM120DotScaledScaleLayout
  appendScale(aScaleValue, aByte, /*threadId*/ 0);
  appendScale(bScaleValue, bByte, /*threadId*/ 0);

  mma(ops);
}

using EmitMmaCallback = std::function<void(
    PTXBuilder &builder, int b, int m, int n, int k,
    mlir::triton::PTXInstr &mma, unsigned numMmaRets, unsigned colsPerThread,
    unsigned batchOffset, ValueTableV2 &ha, ValueTableV2 &hb,
    const SmallVector<Value> &fc, RankedTensorType dTensorTy, int repK)>;

LogicalResult
convertMMAImpl(DotOpInterface op, Value llvmA, Value llvmB, Value llvmC,
               const LLVMTypeConverter *typeConverter,
               ConversionPatternRewriter &rewriter, TensorCoreType mmaType,
               const std::map<TensorCoreType, std::string> &mmaInstructions,
               const EmitMmaCallback &emitMma) {
  auto loc = op.getLoc();
  auto aType = cast<RankedTensorType>(op.getA().getType());
  auto bType = cast<RankedTensorType>(op.getB().getType());
  assert(mlir::isa<DotOperandEncodingAttr>(aType.getEncoding()) &&
         mlir::isa<DotOperandEncodingAttr>(bType.getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  Value cOperand = op->getOperand(2);
  Value loadedC = loadC(cOperand, llvmC, typeConverter, loc, rewriter);

  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = op->getContext();

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
  auto ha = getValuesFromDotOperandLayoutStruct(
      typeConverter, loc, rewriter, llvmA, repBatch, repM, repK, aTensorTy);

  assert(dotOpB.getRepOrder() == getOrderForDotOperand(dotOpB.getOpIdx(),
                                                       bShapePerCTA.size(),
                                                       /*kContig=*/true));
  auto hb = getValuesFromDotOperandLayoutStruct(
      typeConverter, loc, rewriter, llvmB, repBatch, repN, repK, bTensorTy);

  auto fc = unpackLLElements(loc, loadedC, rewriter);

  int bitwidthRet = dTensorTy.getElementType().getIntOrFloatBitWidth();
  auto numMmaRets = bitwidthRet == 64 ? 4 : bitwidthRet / 8;
  int numCPackedElem = 4 / numMmaRets;

  if (mmaInstructions.find(mmaType) == mmaInstructions.end()) {
    return emitError(loc, "Unsupported MMA instruction for the given mma type");
  }
  auto rank = dTensorTy.getRank();
  auto elemsPerThread = triton::gpu::getElemsPerThread(dTensorTy);
  auto batchOffset =
      elemsPerThread[rank - 2] * elemsPerThread[rank - 1] / numCPackedElem;
  auto callMma = [&](unsigned b, unsigned m, unsigned n, unsigned k) {
    PTXBuilder builder;
    auto &mma = *builder.create(mmaInstructions.at(mmaType));
    // using =r for float32 works but leads to less readable ptx.
    unsigned colsPerThread = repN * 2;
    emitMma(builder, b, m, n, k, mma, numMmaRets, colsPerThread, batchOffset,
            ha, hb, fc, dTensorTy, repK);

    Value mmaOut =
        builder.launch(rewriter, loc, getMmaRetType(mmaType, op->getContext()));

    Type elemTy = cast<LLVM::LLVMStructType>(mmaOut.getType()).getBody()[0];
    for (int i = 0; i < numMmaRets; ++i) {
      fc[(m * colsPerThread + 4 * n) / numCPackedElem + i + batchOffset * b] =
          tb.extract_val(elemTy, mmaOut, i);
    }
  };

  for (int b = 0; b < repBatch; ++b)
    for (int k = 0; k < repK; ++k)
      for (int m = 0; m < repM; ++m)
        for (int n = 0; n < repN; ++n) {
          auto numVecK = bitwidth == 64 ? 4 : 2;
          callMma(b, 2 * m, n, k * numVecK);
        }

  Type resElemTy = dTensorTy.getElementType();

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(fc.size() * numCPackedElem, resElemTy));
  SmallVector<Value> results(fc.size() * numCPackedElem);
  for (int i = 0; i < fc.size(); ++i) {
    for (int j = 0; j < numCPackedElem; ++j) {
      results[i * numCPackedElem + j] =
          numCPackedElem > 1
              ? tb.bitcast(tb.extract_element(fc[i], tb.i32_val(j)), resElemTy)
              : tb.bitcast(fc[i], resElemTy);
    }
  }
  Value res = packLLElements(loc, typeConverter, results, rewriter, structTy);

  rewriter.replaceOp(op, res);

  return success();
}

LogicalResult convertMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                         const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring,
                         bool isHopperF64) {
  auto aTensorTy = op.getA().getType();
  auto bTensorTy = op.getB().getType();
  auto dTensorTy = op.getD().getType();

  TensorCoreType mmaType = getMmaTypeDot(op, aTensorTy, bTensorTy, dTensorTy);

  const auto &instrMap =
      isTuring ? mmaInstrPtxTuring
               : (isHopperF64 ? mmaInstrPtxHopper : mmaInstrPtxAmpere);
  EmitMmaCallback emit = [&](PTXBuilder &builder, int b, int m, int n, int k,
                             mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                             unsigned colsPerThread, unsigned batchOffset,
                             ValueTableV2 &ha, ValueTableV2 &hb,
                             const SmallVector<Value> &fc, RankedTensorType dTy,
                             int /*repK*/) {
    const unsigned numCPackedElem = 4u / numMmaRets;
    bool isIntMMA = dTy.getElementType().isInteger(32);
    bool isAccF16 = dTy.getElementType().isF16();
    bool isFp64MMA = dTy.getElementType().isF64();

    if (isTuring) {
      assert(b == 0 && "Turing only supports batch size 1");
      if (isIntMMA)
        callMmaTuringInt8(builder, b, m, n, k, mma, numMmaRets, colsPerThread,
                          numCPackedElem, ha, hb, fc);
      else
        callMmaTuringFp16(builder, b, m, n, k, mma, numMmaRets, colsPerThread,
                          numCPackedElem, ha, hb, fc, isAccF16);
    } else {
      if (isFp64MMA) {
        if (!isHopperF64) {
          callMmaAmpereFp64(builder, b, m, n, k, mma, numMmaRets, colsPerThread,
                            numCPackedElem, batchOffset, ha, hb, fc);
        } else {
          callMmaV2(builder, b, m, n, k, mma, numMmaRets, colsPerThread,
                    numCPackedElem, batchOffset, ha, hb, fc, "=d", "d", 4);
        }
      } else {
        callMmaV2(builder, b, m, n, k, mma, numMmaRets, colsPerThread,
                  numCPackedElem, batchOffset, ha, hb, fc,
                  isIntMMA || isAccF16 ? "=r" : "=f", "r", 2);
      }
    }
  };

  return convertMMAImpl(op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
                        typeConverter, rewriter, mmaType, instrMap, emit);
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

  const auto &instrMap = mmaInstrPtxScaled;

  SmallVector<Value> unpackedAScale =
      unpackLLElements(op.getLoc(), adaptor.getAScale(), rewriter);
  SmallVector<Value> unpackedBScale =
      unpackLLElements(op.getLoc(), adaptor.getBScale(), rewriter);

  EmitMmaCallback emit = [&](PTXBuilder &builder, int b, int m, int n, int k,
                             mlir::triton::PTXInstr &mma, unsigned numMmaRets,
                             unsigned colsPerThread, unsigned batchOffset,
                             ValueTableV2 &aTable, ValueTableV2 &bTable,
                             const SmallVector<Value> &cValues,
                             RankedTensorType dTy, int repK) {
    const unsigned numCPackedElem = 4u / numMmaRets;

    // aScaleValue, bScaleValue selection logic for the scale factor
    // depends on the layout selection in
    // LinearLayoutConversions.cpp::getSM120DotScaledScaleLayout
    auto tb2 = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    auto i32 = IntegerType::get(op->getContext(), 32);
    auto toI32 = [&](Value v) -> Value {
      if (v.getType().isInteger(32))
        return v;
      return tb2.zext(i32, v);
    };
    auto pack4BytesToI32 = [&](ArrayRef<Value> bytes) -> Value {
      Value acc = tb2.i32_val(0);
      for (int i = 0; i < 4; ++i) {
        Value bv = (i < (int)bytes.size()) ? toI32(bytes[i]) : tb2.i32_val(0);
        acc = tb2.or_(acc, tb2.shl(bv, tb2.i32_val(8 * i)));
      }
      return acc;
    };
    auto pack4ByGroupedIndex = [&](ArrayRef<Value> bytes, int idx,
                                   int groupSize) -> Value {
      int blocks = bytes.size() / 4;
      int maxIdx = blocks * groupSize;
      if (idx < maxIdx) {
        int base = (idx / groupSize) * 4;
        return pack4BytesToI32(bytes.slice(base, 4));
      }
      return pack4BytesToI32(bytes);
    };

    int chooseK = k / 2;
    bool interleavedB = (repK > 1);
    SmallVector<Value> KsliceBuf;
    auto Kslice = [&](ArrayRef<Value> bytes,
                      bool interleaved) -> ArrayRef<Value> {
      int sz = bytes.size();
      if (repK == 1)
        return bytes;
      assert(sz % repK == 0);
      int chunk = (sz / repK);
      if (!interleaved) {
        int beg = chooseK * chunk;
        return bytes.slice(beg, chunk);
      } else {
        KsliceBuf.clear();
        KsliceBuf.reserve(chunk);
        const int elementsPerGroup = 2;
        for (int group = 0; group < chunk / elementsPerGroup; ++group) {
          for (int elem = 0; elem < elementsPerGroup; ++elem) {
            int idx = group * elementsPerGroup * repK +
                      chooseK * elementsPerGroup + elem;
            if (idx < sz) {
              KsliceBuf.push_back(bytes[idx]);
            }
          }
        }
        if (chunk % elementsPerGroup != 0) {
          for (int nn = (chunk / elementsPerGroup) * elementsPerGroup;
               nn < chunk; ++nn) {
            int idx = nn * repK + chooseK;
            if (idx < sz) {
              KsliceBuf.push_back(bytes[idx]);
            }
          }
        }
        return ArrayRef<Value>(KsliceBuf);
      }
    };
    Value aScaleValue =
        pack4ByGroupedIndex(Kslice(unpackedAScale, false), m, 8);
    Value bScaleValue =
        pack4ByGroupedIndex(Kslice(unpackedBScale, interleavedB), n, 4);

    callMmaScaled(builder, b, m, n, k, mma, numMmaRets, colsPerThread, aTable,
                  bTable, cValues, aScaleValue, bScaleValue);
  };

  return convertMMAImpl(op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
                        typeConverter, rewriter, mmaType, instrMap, emit);
}
