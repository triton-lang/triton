#include "TritonAMDGPUTransforms/WmmaGroup.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include <tuple>

namespace mlir {
namespace {

//===----------------------------------------------------------------------===//
// Wmma intrinsic query key
//===----------------------------------------------------------------------===//

// The tuple used as key to query WMMA intrinsic map.
using WmmaKey = std::tuple<unsigned /*version*/, unsigned /*mDim*/,
                           unsigned /*nDim*/, TypeID /*aElemType*/,
                           TypeID /*bElemType*/, unsigned /*operatorBitWidth*/,
                           TypeID /*dElemType*/, unsigned /*resultBitWidth*/>;

//===----------------------------------------------------------------------===//
// WMMA intrinsic map
//===----------------------------------------------------------------------===//

using WmmaMapValue =
    std::tuple<StringRef /*symbol*/, unsigned /*kDim*/, unsigned /*kBase*/>;
using WmmaMap = llvm::DenseMap<WmmaKey, SmallVector<WmmaMapValue, 2>>;

class WmmaDatabase {
public:
  static const WmmaMap &get(MLIRContext *context) {
    static WmmaDatabase db(context);
    return db.wmmaMap;
  }

private:
  explicit WmmaDatabase(MLIRContext *context);

  WmmaMap wmmaMap;
};

WmmaDatabase::WmmaDatabase(MLIRContext *context) {
// Macro for defining WMMA intrinsics at a specific gfx version.
#define TRITON_WMMA_v(v, m, n, aET, bET, opW, dET, resW, symbol, k, kBase)     \
  {                                                                            \
    /*key=*/                                                                   \
    {v, m, n, aET.getTypeID(), bET.getTypeID(), opW, dET.getTypeID(), resW},   \
    /*value=*/{                                                                \
      {symbol, k, kBase},                                                      \
    }                                                                          \
  }

// For certain architectures, we can have two intrinsics with the same M/N but
// different K. Order matters here: case1 will be preferred to case2.
#define TRITON_WMMA_v_2case(v, m, n, aET, bET, opW, dET, resW, symbol1, k1,    \
                            kBase1, symbol2, k2, kBase2)                       \
  {                                                                            \
    /*key=*/                                                                   \
    {v, m, n, aET.getTypeID(), bET.getTypeID(), opW, dET.getTypeID(), resW},   \
    /*value=*/{                                                                \
      {symbol1, k1, kBase1}, {symbol2, k2, kBase2},                            \
    }                                                                          \
  }

#define TRITON_WMMA_v2_2case(m, n, aET, bET, opW, dET, resW, symbol1, k1,      \
                             kBase1, symbol2, k2, kBase2)                      \
  TRITON_WMMA_v_2case(2, m, n, aET, bET, opW, dET, resW, symbol1, k1, kBase1,  \
                      symbol2, k2, kBase2)

  Builder b(context);
  auto f32T = b.getF32Type();
  auto f16T = b.getF16Type();
  auto bf16T = b.getBF16Type();
  auto i32T = b.getIntegerType(32);
  auto i8T = b.getIntegerType(8);
  auto i4T = b.getIntegerType(4);

  auto ocpFp8T = b.getType<Float8E4M3FNType>();
  auto ocpBf8T = b.getType<Float8E5M2Type>();

  wmmaMap = {
      // wmma_f16_16x16x16_f16
      TRITON_WMMA_v(1, 16, 16, f16T, f16T, 16, f16T, 16,
                    "llvm.amdgcn.wmma.f16.16x16x16.f16", 16, 16),
      TRITON_WMMA_v(2, 16, 16, f16T, f16T, 16, f16T, 16,
                    "llvm.amdgcn.wmma.f16.16x16x16.f16", 16, 8),

      // wmma_f32_16x16x16_bf16
      TRITON_WMMA_v(1, 16, 16, bf16T, bf16T, 16, f32T, 32,
                    "llvm.amdgcn.wmma.f32.16x16x16.bf16", 16, 16),
      TRITON_WMMA_v(2, 16, 16, bf16T, bf16T, 16, f32T, 32,
                    "llvm.amdgcn.wmma.f32.16x16x16.bf16", 16, 8),

      // wmma_f32_16x16x16_f16
      TRITON_WMMA_v(1, 16, 16, f16T, f16T, 16, f32T, 32,
                    "llvm.amdgcn.wmma.f32.16x16x16.f16", 16, 16),
      TRITON_WMMA_v(2, 16, 16, f16T, f16T, 16, f32T, 32,
                    "llvm.amdgcn.wmma.f32.16x16x16.f16", 16, 8),

      // wmma_bf16_16x16x16_bf16
      TRITON_WMMA_v(1, 16, 16, bf16T, bf16T, 16, bf16T, 16,
                    "llvm.amdgcn.wmma.bf16.16x16x16.bf16", 16, 16),
      TRITON_WMMA_v(2, 16, 16, bf16T, bf16T, 16, bf16T, 16,
                    "llvm.amdgcn.wmma.bf16.16x16x16.bf16", 16, 8),

      // wmma_i32_16x16x16_iu4
      TRITON_WMMA_v(1, 16, 16, i4T, i4T, 4, i32T, 32,
                    "llvm.amdgcn.wmma.i32.16x16x16.iu4", 16, 16),

      // wmma_i32_16x16x32_iu4 && wmma_i32_16x16x16_iu4
      TRITON_WMMA_v2_2case(16, 16, i4T, i4T, 4, i32T, 32,
                           "llvm.amdgcn.wmma.i32.16x16x32.iu4", 32, 16,
                           "llvm.amdgcn.wmma.i32.16x16x16.iu4", 16, 8),

      // wmma_i32_16x16x16_iu8
      TRITON_WMMA_v(1, 16, 16, i8T, i8T, 8, i32T, 32,
                    "llvm.amdgcn.wmma.i32.16x16x16.iu8", 16, 16),
      TRITON_WMMA_v(2, 16, 16, i8T, i8T, 8, i32T, 32,
                    "llvm.amdgcn.wmma.i32.16x16x16.iu8", 16, 8),

      // wmma_f32_16x16x16_fp8_fp8
      TRITON_WMMA_v(2, 16, 16, ocpFp8T, ocpFp8T, 8, f32T, 32,
                    "llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8", 16, 8),

      // wmma_f32_16x16x16_fp8_bf8
      TRITON_WMMA_v(2, 16, 16, ocpFp8T, ocpBf8T, 8, f32T, 32,
                    "llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8", 16, 8),

      // wmma_f32_16x16x16_bf8_fp8
      TRITON_WMMA_v(2, 16, 16, ocpBf8T, ocpFp8T, 8, f32T, 32,
                    "llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8", 16, 8),

      // wmma_f32_16x16x16_bf8_bf8
      TRITON_WMMA_v(2, 16, 16, ocpBf8T, ocpBf8T, 8, f32T, 32,
                    "llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8", 16, 8),

  };
}

} // namespace

//===----------------------------------------------------------------------===//
// Wmma intrinsic selection
//===----------------------------------------------------------------------===//

FailureOr<WmmaIntrinsic>
WmmaIntrinsic::selectFor(int version, unsigned mDim, unsigned nDim,
                         unsigned inputKDim, Type aElemType, Type bElemType,
                         Type dElemType) {

  const WmmaMap &wmmaMap = WmmaDatabase::get(aElemType.getContext());
  WmmaKey key = {version,
                 mDim,
                 nDim,
                 aElemType.getTypeID(),
                 bElemType.getTypeID(),
                 aElemType.getIntOrFloatBitWidth(),
                 dElemType.getTypeID(),
                 dElemType.getIntOrFloatBitWidth()};

  auto it = wmmaMap.find(key);
  if (it == wmmaMap.end())
    return failure();

  const SmallVector<WmmaMapValue, 2> &values = it->second;

  // If We have more than one instrinsics, prefer those with a larger K.
  for (const auto [symbol, k, kBase] : llvm::drop_end(values)) {
    if (inputKDim >= k)
      return WmmaIntrinsic(symbol, mDim, nDim, k, kBase, aElemType, bElemType,
                           dElemType);
  }

  // We always have one choice--the only / smallest-K intrinsic.
  auto [symbol, k, kBase] = values.back();

  return WmmaIntrinsic(symbol, mDim, nDim, k, kBase, aElemType, bElemType,
                       dElemType);
}
} // namespace mlir
