#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include <tuple>

namespace mlir {
namespace {

inline bool isF8F6F4(mlir::Type t) {
  return llvm::isa<Float8E4M3FNType, Float8E5M2Type, Float6E3M2FNType,
                   Float6E2M3FNType, Float4E2M1FNType>(t);
}

using MfmaKey =
    std::tuple<unsigned /*version*/, unsigned /*mDim*/, unsigned /*nDim*/,
               TypeID /*aElemType*/, TypeID /*bElemType*/>;

inline MfmaKey composeMfmaKeyFor(unsigned version, unsigned mDim, unsigned nDim,
                                 Type aElemType, Type bElemType, bool withScale,
                                 bool useTF32) {
  if (withScale) {
    assert(version == 4 && isF8F6F4(aElemType) && isF8F6F4(bElemType));
    // For MXFP types, we have the same instruction, which uses FP4 as the key.
    aElemType = bElemType = Float4E2M1FNType::get(aElemType.getContext());
  } else if (useTF32 && aElemType.isF32() && bElemType.isF32()) {
    // In Triton we use fp32 with TF32 input precision to mean TF32 types.
    // In the MFMA map we use the proper TF32 type. So fix it here.
    assert(version == 3);
    aElemType = bElemType = FloatTF32Type::get(aElemType.getContext());
  }
  return {version, mDim, nDim, aElemType.getTypeID(), bElemType.getTypeID()};
}

using MfmaMapValue =
    std::tuple<StringRef /*symbol*/, unsigned /*kDim*/, unsigned /*kBase*/>;
using MfmaMap = llvm::DenseMap<MfmaKey, SmallVector<MfmaMapValue, 2>>;

class MfmaDatabase {
public:
  static const MfmaDatabase &get(MLIRContext *context) {
    static MfmaDatabase db(context);
    return db;
  }

  const MfmaMap &getMap() const { return mfmaMap; }

private:
  MfmaMap mfmaMap;

  explicit MfmaDatabase(MLIRContext *context);
};

MfmaDatabase::MfmaDatabase(MLIRContext *context) {
#define TRITON_MFMA_v(v, m, n, aET, bET, symbol, k, kBase)                     \
  {                                                                            \
    /*key=*/{v, m, n, aET.getTypeID(), bET.getTypeID()}, /*value=*/{           \
      {ROCDL::symbol::getOperationName(), k, kBase},                           \
    }                                                                          \
  }
#define TRITON_MFMA_v4_2k(m, n, aET, bET, symbol1, k1, kBase1, symbol2, k2,    \
                          kBase2)                                              \
  {                                                                            \
    /*key=*/{4, m, n, aET.getTypeID(), bET.getTypeID()}, /*value=*/{           \
      {ROCDL::symbol1::getOperationName(), k1, kBase1},                        \
          {ROCDL::symbol2::getOperationName(), k2, kBase2},                    \
    }                                                                          \
  }

#define TRITON_MFMA_v1to2(m, n, k, aET, bET, kBase, symbol)                    \
  TRITON_MFMA_v(1, m, n, k, aET, bET, kBase, symbol),                          \
      TRITON_MFMA_v(2, m, n, k, aET, bET, kBase, symbol)

#define TRITON_MFMA_v2to3(m, n, k, aET, bET, kBase, symbol)                    \
  TRITON_MFMA_v(2, m, n, k, aET, bET, kBase, symbol),                          \
      TRITON_MFMA_v(3, m, n, k, aET, bET, kBase, symbol)

#define TRITON_MFMA_v3to4(m, n, k, aET, bET, kBase, symbol)                    \
  TRITON_MFMA_v(3, m, n, k, aET, bET, kBase, symbol),                          \
      TRITON_MFMA_v(4, m, n, k, aET, bET, kBase, symbol)

#define TRITON_MFMA_v2to4(m, n, k, aET, bET, kBase, symbol)                    \
  TRITON_MFMA_v(2, m, n, k, aET, bET, kBase, symbol),                          \
      TRITON_MFMA_v3to4(m, n, k, aET, bET, kBase, symbol)

#define TRITON_MFMA_v1to3(m, n, k, aET, bET, kBase, symbol)                    \
  TRITON_MFMA_v(1, m, n, k, aET, bET, kBase, symbol),                          \
      TRITON_MFMA_v2to3(m, n, k, aET, bET, kBase, symbol)

#define TRITON_MFMA_v1to4(m, n, k, aET, bET, kBase, symbol)                    \
  TRITON_MFMA_v(1, m, n, k, aET, bET, kBase, symbol),                          \
      TRITON_MFMA_v2to4(m, n, k, aET, bET, kBase, symbol)

  Builder b(context);
  auto f32T = b.getF32Type();
  auto tf32T = b.getTF32Type();
  auto f16T = b.getF16Type();
  auto bf16T = b.getBF16Type();
  auto i8T = b.getI8Type();
  auto amdFp8T = b.getType<Float8E4M3FNUZType>();
  auto amdBf8T = b.getType<Float8E5M2FNUZType>();
  auto ocpFp8T = b.getType<Float8E4M3FNType>();
  auto ocpBf8T = b.getType<Float8E5M2Type>();
  auto fp4T = b.getType<Float4E2M1FNType>();

  mfmaMap = {
      // f32 inputs
      // mfma_f32_32x32x2f32
      TRITON_MFMA_v1to4(32, 32, f32T, f32T, mfma_f32_32x32x2f32, 2, 1),
      // mfma_f32_16x16x4f32
      TRITON_MFMA_v1to4(16, 16, f32T, f32T, mfma_f32_16x16x4f32, 4, 1),
      // mfma_f32_4x4x1f32 / mfma_f32_4x4x1_16B_f32
      TRITON_MFMA_v1to4(4, 4, f32T, f32T, mfma_f32_4x4x1f32, 16, 1),
      TRITON_MFMA_v1to4(4, 64, f32T, f32T, mfma_f32_4x4x1f32, 1, 1),
      TRITON_MFMA_v1to4(64, 4, f32T, f32T, mfma_f32_4x4x1f32, 1, 1),

      // xf32
      // mfma.xf32.16x16x8xf32
      TRITON_MFMA_v(3, 16, 16, tf32T, tf32T, mfma_f32_16x16x8_xf32, 8, 2),
      // mfma.xf32.32x32x4.xf32
      TRITON_MFMA_v(3, 32, 32, tf32T, tf32T, mfma_f32_32x32x4_xf32, 4, 2),

      // f16 inputs
      // mfma_f32_32x32x16_f16 & mfma_f32_32x32x8f16
      TRITON_MFMA_v4_2k(32, 32, f16T, f16T, mfma_f32_32x32x16_f16, 16, 8,
                        mfma_f32_32x32x8f16, 8, 4),
      // mfma_f32_32x32x8f16
      TRITON_MFMA_v1to3(32, 32, f16T, f16T, mfma_f32_32x32x8f16, 8, 4),
      // mfma_f32_16x16x32_f16 & mfma_f32_16x16x16f16
      TRITON_MFMA_v4_2k(16, 16, f16T, f16T, mfma_f32_16x16x32_f16, 32, 8,
                        mfma_f32_16x16x16f16, 16, 4),
      // mfma_f32_16x16x16f16
      TRITON_MFMA_v1to3(16, 16, f16T, f16T, mfma_f32_16x16x16f16, 16, 4),
      // mfma_f32_4x4x4f16
      TRITON_MFMA_v1to4(4, 4, f16T, f16T, mfma_f32_4x4x4f16, 64, 4),
      TRITON_MFMA_v1to4(4, 64, f16T, f16T, mfma_f32_4x4x4f16, 4, 4),
      TRITON_MFMA_v1to4(64, 4, f16T, f16T, mfma_f32_4x4x4f16, 4, 4),

      // bf16 inputs
      // mfma_f32_32x32x16_bf16 & mfma_f32_32x32x8_bf16_1K
      TRITON_MFMA_v4_2k(32, 32, bf16T, bf16T, mfma_f32_32x32x16_bf16, 16, 8,
                        mfma_f32_32x32x8bf16_1k, 8, 4),
      // mfma_f32_32x32x8_bf16_1K
      TRITON_MFMA_v2to3(32, 32, bf16T, bf16T, mfma_f32_32x32x8bf16_1k, 8, 4),
      // mfma_f32_16x16x32_bf16 & mfma_f32_16x16x16_bf16_1K
      TRITON_MFMA_v4_2k(16, 16, bf16T, bf16T, mfma_f32_16x16x32_bf16, 32, 8,
                        mfma_f32_16x16x16bf16_1k, 16, 4),
      // mfma_f32_16x16x16_bf16_1K
      TRITON_MFMA_v2to3(16, 16, bf16T, bf16T, mfma_f32_16x16x16bf16_1k, 16, 4),
      // mfma_f32_32x32x4_bf16
      TRITON_MFMA_v(1, 32, 32, bf16T, bf16T, mfma_f32_32x32x4bf16, 4, 2),
      // mfma_f32_16x16x8_bf16
      TRITON_MFMA_v(1, 16, 16, bf16T, bf16T, mfma_f32_16x16x8bf16, 8, 2),
      // mfma_f32_4x4x4_bf16_1K
      TRITON_MFMA_v2to4(4, 4, bf16T, bf16T, mfma_f32_4x4x4bf16_1k, 64, 4),
      TRITON_MFMA_v2to4(4, 64, bf16T, bf16T, mfma_f32_4x4x4bf16_1k, 4, 4),
      TRITON_MFMA_v2to4(64, 4, bf16T, bf16T, mfma_f32_4x4x4bf16_1k, 4, 4),
      // mfma_f32_4x4x2_bf16
      TRITON_MFMA_v(1, 4, 4, bf16T, bf16T, mfma_f32_4x4x2bf16, 32, 2),
      TRITON_MFMA_v(1, 4, 64, bf16T, bf16T, mfma_f32_4x4x2bf16, 2, 2),
      TRITON_MFMA_v(1, 64, 4, bf16T, bf16T, mfma_f32_4x4x2bf16, 2, 2),

      // fp8/bf8 inputs
      // mfma_f32_32x32x16_FP8_FP8
      TRITON_MFMA_v(4, 32, 32, ocpFp8T, ocpFp8T, mfma_f32_32x32x16_fp8_fp8, 16,
                    8),
      TRITON_MFMA_v(3, 32, 32, amdFp8T, amdFp8T, mfma_f32_32x32x16_fp8_fp8, 16,
                    8),
      // mfma_f32_32x32x16_FP8_BF8
      TRITON_MFMA_v(4, 32, 32, ocpFp8T, ocpBf8T, mfma_f32_32x32x16_fp8_bf8, 16,
                    8),
      TRITON_MFMA_v(3, 32, 32, amdFp8T, amdBf8T, mfma_f32_32x32x16_fp8_bf8, 16,
                    8),
      // mfma_f32_32x32x16_BF8_FP8
      TRITON_MFMA_v(4, 32, 32, ocpBf8T, ocpFp8T, mfma_f32_32x32x16_bf8_fp8, 16,
                    8),
      TRITON_MFMA_v(3, 32, 32, amdBf8T, amdFp8T, mfma_f32_32x32x16_bf8_fp8, 16,
                    8),
      // mfma_f32_32x32x16_BF8_BF8
      TRITON_MFMA_v(4, 32, 32, ocpBf8T, ocpBf8T, mfma_f32_32x32x16_bf8_bf8, 16,
                    8),
      TRITON_MFMA_v(3, 32, 32, amdBf8T, amdBf8T, mfma_f32_32x32x16_bf8_bf8, 16,
                    8),
      // mfma_f32_16x16x32_FP8_FP8
      TRITON_MFMA_v(4, 16, 16, ocpFp8T, ocpFp8T, mfma_f32_16x16x32_fp8_fp8, 32,
                    8),
      TRITON_MFMA_v(3, 16, 16, amdFp8T, amdFp8T, mfma_f32_16x16x32_fp8_fp8, 32,
                    8),
      // mfma_f32_16x16x32_FP8_BF8
      TRITON_MFMA_v(4, 16, 16, ocpFp8T, ocpBf8T, mfma_f32_16x16x32_fp8_bf8, 32,
                    8),
      TRITON_MFMA_v(3, 16, 16, amdFp8T, amdBf8T, mfma_f32_16x16x32_fp8_bf8, 32,
                    8),
      // mfma_f32_16x16x32_BF8_FP8
      TRITON_MFMA_v(3, 16, 16, amdBf8T, amdFp8T, mfma_f32_16x16x32_bf8_fp8, 32,
                    8),
      TRITON_MFMA_v(4, 16, 16, ocpBf8T, ocpFp8T, mfma_f32_16x16x32_bf8_fp8, 32,
                    8),
      // mfma_f32_16x16x32_BF8_BF8
      TRITON_MFMA_v(4, 16, 16, ocpBf8T, ocpBf8T, mfma_f32_16x16x32_bf8_bf8, 32,
                    8),
      TRITON_MFMA_v(3, 16, 16, amdBf8T, amdBf8T, mfma_f32_16x16x32_bf8_bf8, 32,
                    8),

      // int8 inputs
      // mfma_i32_32x32x16i8
      TRITON_MFMA_v3to4(32, 32, i8T, i8T, mfma_i32_32x32x16_i8, 16, 8),
      // mfma_i32_32x32x8i8
      TRITON_MFMA_v1to2(32, 32, i8T, i8T, mfma_i32_32x32x8i8, 8, 4),
      // mfma_i32_16x16x32i8
      TRITON_MFMA_v3to4(16, 16, i8T, i8T, mfma_i32_16x16x32_i8, 32, 8),
      // mfma_i32_16x16x16i8
      TRITON_MFMA_v1to2(16, 16, i8T, i8T, mfma_i32_16x16x16i8, 16, 4),
      // mfma_i32_4x4x4i8
      TRITON_MFMA_v1to4(4, 4, i8T, i8T, mfma_i32_4x4x4i8, 64, 4),
      TRITON_MFMA_v1to4(4, 64, i8T, i8T, mfma_i32_4x4x4i8, 4, 4),
      TRITON_MFMA_v1to4(64, 4, i8T, i8T, mfma_i32_4x4x4i8, 4, 4),

      // Scaled mfma f8f6f4
      // mfma_scale_F32_16x16x128_F8F6F4
      TRITON_MFMA_v(4, 16, 16, fp4T, fp4T, mfma_scale_f32_16x16x128_f8f6f4, 128,
                    32),
      // mfma_scale_F32_32x32x64_F8F6F4
      TRITON_MFMA_v(4, 32, 32, fp4T, fp4T, mfma_scale_f32_32x32x64_f8f6f4, 64,
                    32),
  };
}

} // namespace

FailureOr<MfmaIntrinsic>
MfmaIntrinsic::selectFor(int version, unsigned mDim, unsigned nDim,
                         unsigned inputKDim, Type aElemType, Type bElemType,
                         bool withScale, bool useTF32) {
  const auto &mfmaDatabase = MfmaDatabase::get(aElemType.getContext());
  const auto &mfmaMap = mfmaDatabase.getMap();
  MfmaKey key = composeMfmaKeyFor(version, mDim, nDim, aElemType, bElemType,
                                  withScale, useTF32);

  auto it = mfmaMap.find(key);
  if (it == mfmaMap.end())
    return failure();
  llvm::outs() << "found key\n";

  const SmallVector<MfmaMapValue, 2> &values = it->second;
  if (values.size() == 1) {
    auto [symbol, k, kBase] = values.front();
    return MfmaIntrinsic(symbol, mDim, nDim, k, kBase, aElemType, bElemType);
  }

  // We have more than one instrinsics. Prefer larger K ones.
  for (const auto &value : values) {
    auto [symbol, k, kBase] = values.front();
    if (inputKDim >= k)
      return MfmaIntrinsic(symbol, mDim, nDim, k, kBase, aElemType, bElemType);
  }
  return failure();
}
} // namespace mlir
