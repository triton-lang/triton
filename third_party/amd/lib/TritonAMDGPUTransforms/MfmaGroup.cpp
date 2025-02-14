#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include <array>
#include <tuple>

#define TRITON_AMD_MAX_MFMA_VERSION 4

namespace mlir {

namespace {
using MfmaKey = std::tuple<unsigned /*mDim*/, unsigned /*nDim*/,
                           TypeID /*aElemType*/, TypeID /*bElemType*/>;

inline MfmaKey composeMfmaKeyFor(unsigned mDim, unsigned nDim, Type aElemType,
                                 Type bElemType) {
  return {mDim, nDim, aElemType.getTypeID(), bElemType.getTypeID()};
}

class MfmaDataBase {
public:
  static MfmaDataBase &get(MLIRContext *context) {
    static MfmaDataBase db(context);
    return db;
  }

private:
  using PerVersionDataBase =
      llvm::DenseMap<MfmaKey, SmallVector<MfmaIntrinsic, 2>>;

  explicit MfmaDataBase(MLIRContext *context);

  std::array<PerVersionDataBase, TRITON_AMD_MAX_MFMA_VERSION + 1> db;
};

MfmaDataBase::MfmaDataBase(MLIRContext *context) {
#define TRITON_AMD_ADD_MFMA_v(v, m, n, k, aET, bET, kBase, symbol)             \
  db[v][composeMfmaKeyFor(m, n, aET, bET)].emplace_back(                       \
      ROCDL::symbol::getOperationName(), m, n, k, kBase, aET, bET);

#define TRITON_AMD_ADD_MFMA_v1to2(m, n, k, aET, bET, kBase, symbol)            \
  TRITON_AMD_ADD_MFMA_v(1, m, n, k, aET, bET, kBase, symbol);                  \
  TRITON_AMD_ADD_MFMA_v(2, m, n, k, aET, bET, kBase, symbol);

#define TRITON_AMD_ADD_MFMA_v3to4(m, n, k, aET, bET, kBase, symbol)            \
  TRITON_AMD_ADD_MFMA_v(3, m, n, k, aET, bET, kBase, symbol);                  \
  TRITON_AMD_ADD_MFMA_v(4, m, n, k, aET, bET, kBase, symbol);

#define TRITON_AMD_ADD_MFMA_v2to4(m, n, k, aET, bET, kBase, symbol)            \
  TRITON_AMD_ADD_MFMA_v(2, m, n, k, aET, bET, kBase, symbol);                  \
  TRITON_AMD_ADD_MFMA_v3to4(m, n, k, aET, bET, kBase, symbol);

#define TRITON_AMD_ADD_MFMA_v1to4(m, n, k, aET, bET, kBase, symbol)            \
  TRITON_AMD_ADD_MFMA_v(1, m, n, k, aET, bET, kBase, symbol);                  \
  TRITON_AMD_ADD_MFMA_v2to4(m, n, k, aET, bET, kBase, symbol);

  Builder b(context);
  auto f32T = b.getF32Type();
  auto f16T = b.getF16Type();
  auto bf16T = b.getBF16Type();
  auto i8T = b.getI8Type();
  auto amdFp8T = b.getType<Float8E4M3FNUZType>();
  auto amdBf8T = b.getType<Float8E5M2FNUZType>();

  // f32 inputs
  // mfma_f32_32x32x2f32
  TRITON_AMD_ADD_MFMA_v1to4(32, 32, 2, f32T, f32T, 1, mfma_f32_32x32x2f32);
  // mfma_f32_16x16x4f32
  TRITON_AMD_ADD_MFMA_v1to4(16, 16, 4, f32T, f32T, 1, mfma_f32_16x16x4f32);
  // mfma_f32_4x4x1f32 / mfma_f32_4x4x1_16B_f32
  TRITON_AMD_ADD_MFMA_v1to4(4, 4, 16, f32T, f32T, 1, mfma_f32_4x4x1f32);
  TRITON_AMD_ADD_MFMA_v1to4(4, 64, 1, f32T, f32T, 1, mfma_f32_4x4x1f32);
  TRITON_AMD_ADD_MFMA_v1to4(64, 4, 1, f32T, f32T, 1, mfma_f32_4x4x1f32);

  // f16 inputs
  // mfma_f32_32x32x16_f16
  TRITON_AMD_ADD_MFMA_v(4, 32, 32, 16, f16T, f16T, 8, mfma_f32_32x32x16_f16);
  // mfma_f32_32x32x8f16
  TRITON_AMD_ADD_MFMA_v1to4(32, 32, 8, f16T, f16T, 4, mfma_f32_32x32x8f16);
  // mfma_f32_16x16x32_f16
  TRITON_AMD_ADD_MFMA_v(4, 16, 16, 32, f16T, f16T, 8, mfma_f32_16x16x32_f16);
  // mfma_f32_16x16x16f16
  TRITON_AMD_ADD_MFMA_v1to4(16, 16, 16, f16T, f16T, 4, mfma_f32_16x16x16f16);
  // mfma_f32_4x4x4f16
  TRITON_AMD_ADD_MFMA_v1to4(4, 4, 64, f16T, f16T, 4, mfma_f32_4x4x4f16);
  TRITON_AMD_ADD_MFMA_v1to4(4, 64, 4, f16T, f16T, 4, mfma_f32_4x4x4f16);
  TRITON_AMD_ADD_MFMA_v1to4(64, 4, 4, f16T, f16T, 4, mfma_f32_4x4x4f16);

  // bf16 inputs
  // mfma_f32_32x32x16_bf16
  TRITON_AMD_ADD_MFMA_v(4, 32, 32, 16, bf16T, bf16T, 8, mfma_f32_32x32x16_bf16);
  // mfma_f32_32x32x8_bf16_1K
  TRITON_AMD_ADD_MFMA_v2to4(32, 32, 8, bf16T, bf16T, 4,
                            mfma_f32_32x32x8bf16_1k);
  // mfma_f32_16x16x32_bf16
  TRITON_AMD_ADD_MFMA_v(4, 16, 16, 32, bf16T, bf16T, 8, mfma_f32_16x16x32_bf16);
  // mfma_f32_16x16x16_bf16_1K
  TRITON_AMD_ADD_MFMA_v2to4(16, 16, 16, bf16T, bf16T, 4,
                            mfma_f32_16x16x16bf16_1k);
  // mfma_f32_32x32x4_bf16
  TRITON_AMD_ADD_MFMA_v(1, 32, 32, 4, bf16T, bf16T, 2, mfma_f32_32x32x4bf16);
  // mfma_f32_16x16x8_bf16
  TRITON_AMD_ADD_MFMA_v(1, 16, 16, 8, bf16T, bf16T, 2, mfma_f32_16x16x8bf16);
  // mfma_f32_4x4x4_bf16_1K
  TRITON_AMD_ADD_MFMA_v2to4(4, 4, 64, bf16T, bf16T, 4, mfma_f32_4x4x4bf16_1k);
  TRITON_AMD_ADD_MFMA_v2to4(4, 64, 4, bf16T, bf16T, 4, mfma_f32_4x4x4bf16_1k);
  TRITON_AMD_ADD_MFMA_v2to4(64, 4, 4, bf16T, bf16T, 4, mfma_f32_4x4x4bf16_1k);
  // mfma_f32_4x4x2_bf16
  TRITON_AMD_ADD_MFMA_v(1, 4, 4, 32, bf16T, bf16T, 2, mfma_f32_4x4x2bf16);
  TRITON_AMD_ADD_MFMA_v(1, 4, 64, 2, bf16T, bf16T, 2, mfma_f32_4x4x2bf16);
  TRITON_AMD_ADD_MFMA_v(1, 64, 4, 2, bf16T, bf16T, 2, mfma_f32_4x4x2bf16);

  // fp8/bf8 inputs
  // mfma_f32_32x32x16_FP8_FP8
  TRITON_AMD_ADD_MFMA_v3to4(32, 32, 16, amdFp8T, amdFp8T, 8,
                            mfma_f32_32x32x16_fp8_fp8);
  // mfma_f32_32x32x16_FP8_BF8
  TRITON_AMD_ADD_MFMA_v3to4(32, 32, 16, amdFp8T, amdBf8T, 8,
                            mfma_f32_32x32x16_fp8_bf8);
  // mfma_f32_32x32x16_BF8_FP8
  TRITON_AMD_ADD_MFMA_v3to4(32, 32, 16, amdBf8T, amdFp8T, 8,
                            mfma_f32_32x32x16_bf8_fp8);
  // mfma_f32_32x32x16_BF8_BF8
  TRITON_AMD_ADD_MFMA_v3to4(32, 32, 16, amdBf8T, amdBf8T, 8,
                            mfma_f32_32x32x16_bf8_bf8);
  // mfma_f32_16x16x32_FP8_FP8
  TRITON_AMD_ADD_MFMA_v3to4(16, 16, 32, amdFp8T, amdFp8T, 8,
                            mfma_f32_16x16x32_fp8_fp8);
  // mfma_f32_16x16x32_FP8_BF8
  TRITON_AMD_ADD_MFMA_v3to4(16, 16, 32, amdFp8T, amdBf8T, 8,
                            mfma_f32_16x16x32_fp8_bf8);
  // mfma_f32_16x16x32_BF8_FP8
  TRITON_AMD_ADD_MFMA_v3to4(16, 16, 32, amdBf8T, amdFp8T, 8,
                            mfma_f32_16x16x32_bf8_fp8);
  // mfma_f32_16x16x32_BF8_BF8
  TRITON_AMD_ADD_MFMA_v3to4(16, 16, 32, amdBf8T, amdBf8T, 8,
                            mfma_f32_16x16x32_bf8_bf8);

  // int8 inputs
  // mfma_i32_32x32x16i8
  TRITON_AMD_ADD_MFMA_v3to4(32, 32, 16, i8T, i8T, 8, mfma_i32_32x32x16_i8);
  // mfma_i32_32x32x8i8
  TRITON_AMD_ADD_MFMA_v1to2(32, 32, 8, i8T, i8T, 4, mfma_i32_32x32x8i8);
  // mfma_i32_16x16x32i8
  TRITON_AMD_ADD_MFMA_v3to4(16, 16, 32, i8T, i8T, 8, mfma_i32_16x16x32_i8);
  // mfma_i32_16x16x16i8
  TRITON_AMD_ADD_MFMA_v1to2(16, 16, 16, i8T, i8T, 4, mfma_i32_16x16x16i8);
  // mfma_i32_4x4x4i8
  TRITON_AMD_ADD_MFMA_v1to4(4, 4, 64, i8T, i8T, 4, mfma_i32_4x4x4i8);
  TRITON_AMD_ADD_MFMA_v1to4(4, 64, 4, i8T, i8T, 4, mfma_i32_4x4x4i8);
  TRITON_AMD_ADD_MFMA_v1to4(64, 4, 4, i8T, i8T, 4, mfma_i32_4x4x4i8);

  // Scaled mfma f8f6f4
  // mfma_scale_F32_16x16x128_F8F6F4
}

} // namespace

static bool isF8F6F4(mlir::Type t) {
  return llvm::isa<Float8E4M3FNType, Float8E5M2Type, Float6E3M2FNType,
                   Float6E2M3FNType, Float4E2M1FNType>(t);
}

static MfmaTypeId chooseAppropriateMfmaId(mlir::Type dataTypeA,
                                          mlir::Type dataTypeB,
                                          bool allowXF32) {
  if (dataTypeA.isF32() && dataTypeB.isF32()) {
    if (allowXF32)
      return MfmaTypeId::Xf32TyId;
    else
      return MfmaTypeId::Fp32TyId;
  }
  if (dataTypeA.isF16() && dataTypeB.isF16()) {
    return MfmaTypeId::Fp16TyId;
  }
  if (dataTypeA.isBF16() && dataTypeB.isBF16()) {
    return MfmaTypeId::Bf16TyId;
  }
  if (dataTypeA.isInteger(8) && dataTypeB.isInteger(8)) {
    return MfmaTypeId::I8TyId;
  }
  if (llvm::isa<Float8E4M3FNUZType>(dataTypeA) &&
      llvm::isa<Float8E4M3FNUZType>(dataTypeB)) {
    return MfmaTypeId::Fp8Fp8TyId;
  }
  if (llvm::isa<Float8E4M3FNUZType>(dataTypeA) &&
      llvm::isa<Float8E5M2FNUZType>(dataTypeB)) {
    return MfmaTypeId::Fp8Bf8TyId;
  }
  if (llvm::isa<Float8E5M2FNUZType>(dataTypeA) &&
      llvm::isa<Float8E4M3FNUZType>(dataTypeB)) {
    return MfmaTypeId::Bf8Fp8TyId;
  }
  if (llvm::isa<Float8E5M2FNUZType>(dataTypeA) &&
      llvm::isa<Float8E5M2FNUZType>(dataTypeB)) {
    return MfmaTypeId::Bf8Bf8TyId;
  }
  if (llvm::isa<Float8E5M2Type>(dataTypeA) &&
      llvm::isa<Float8E5M2Type>(dataTypeB)) {
    return MfmaTypeId::Fp16TyId;
  }
  if (isF8F6F4(dataTypeA) && isF8F6F4(dataTypeB)) {
    return MfmaTypeId::F8F6F4TyId;
  }
  llvm_unreachable("Unsupported input argument type.");
}

using MfmaInsnGroupMap = llvm::DenseMap<MfmaInsnGroupSelectKey, MfmaInsnAttr,
                                        MfmaInsnGroupSelectKeyInfo>;

auto getMfmaInsnGroupAttrMap = []() -> const MfmaInsnGroupMap & {
  static MfmaInsnGroupMap MfmaInsnMap{
      // xf32
      // mfma.xf32.16x16x8xf32
      {{16, 16, 0, MfmaTypeId::Xf32TyId, 3},
       {16, 16, 8, 2, ROCDL::mfma_f32_16x16x8_xf32::getOperationName()}},
      // mfma.xf32.32x32x4.xf32
      {{32, 32, 0, MfmaTypeId::Xf32TyId, 3},
       {32, 32, 4, 2, ROCDL::mfma_f32_32x32x4_xf32::getOperationName()}},

      // scaled mfma f8f6f4
      // mfma_scale_F32_16x16x128_F8F6F4
      {{16, 16, 0, MfmaTypeId::F8F6F4TyId, 4},
       {16, 16, 128, 32,
        ROCDL::mfma_scale_f32_16x16x128_f8f6f4::getOperationName()}},
      // mfma_scale_F32_32x32x64_F8F6F4
      {{32, 32, 0, MfmaTypeId::F8F6F4TyId, 4},
       {32, 32, 64, 32,
        ROCDL::mfma_scale_f32_32x32x64_f8f6f4::getOperationName()}},
      // mfma_f32_16x16x32_BF8_BF8
      {{16, 16, 0, MfmaTypeId::Bf8Bf8TyId, 3},
       {16, 16, 32, 8, ROCDL::mfma_f32_16x16x32_bf8_bf8::getOperationName()}},
      {{16, 16, 0, MfmaTypeId::Bf8Bf8TyId, 4},
       {16, 16, 32, 8, ROCDL::mfma_f32_16x16x32_bf8_bf8::getOperationName()}},
  };
  return MfmaInsnMap;
};

std::pair<mlir::Type, mlir::Type> TypesFromMfmaId(mlir::MLIRContext *ctx,
                                                  MfmaTypeId id) {
  auto f8e5m2 = Float8E5M2Type::get(ctx);
  auto f8e4m3fnuz = Float8E4M3FNUZType::get(ctx);
  auto f8e5m2fnuz = Float8E5M2FNUZType::get(ctx);
  auto f16 = Float16Type::get(ctx);
  auto bf16 = BFloat16Type::get(ctx);
  auto f32 = Float32Type::get(ctx);
  auto i8 = IntegerType::get(ctx, 8, IntegerType::Signed);
  switch (id) {
  case MfmaTypeId::Xf32TyId:
  case MfmaTypeId::Fp32TyId:
    return {f32, f32};
  case MfmaTypeId::Fp16TyId:
    return {f16, f16};
  case MfmaTypeId::Bf16TyId:
    return {bf16, bf16};
  case MfmaTypeId::I8TyId:
    return {i8, i8};
  case MfmaTypeId::Fp8Fp8TyId:
    return {f8e4m3fnuz, f8e4m3fnuz};
  case MfmaTypeId::Fp8Bf8TyId:
    return {f8e4m3fnuz, f8e5m2fnuz};
  case MfmaTypeId::Bf8Fp8TyId:
    return {f8e5m2fnuz, f8e4m3fnuz};
  case MfmaTypeId::Bf8Bf8TyId:
    return {f8e5m2fnuz, f8e5m2fnuz};
  default:
    llvm_unreachable("unsupported MfmaTypeId!");
  }
}

FailureOr<MfmaInsn> MfmaInsn::selectMfma(unsigned mDim, unsigned nDim,
                                         unsigned kDim, Type elementTypeA,
                                         Type elementTypeB, int mfmaVersion,
                                         bool allowXF32) {
  auto mfmaInsnAttrMap = getMfmaInsnGroupAttrMap();
  MfmaTypeId mfmaId =
      chooseAppropriateMfmaId(elementTypeA, elementTypeB, allowXF32);
  unsigned kDimKey = 0;
  // gfx950: select double-rate mfma ops for large kDim
  if (mfmaVersion == 4 &&
      (mfmaId == MfmaTypeId::Fp16TyId || mfmaId == MfmaTypeId::Bf16TyId)) {
    if (mDim == 16 && nDim == 16 && kDim >= 32)
      kDimKey = 32;
    else if (mDim == 32 && nDim == 32 && kDim >= 16)
      kDimKey = 16;
  }
  MfmaInsnGroupSelectKey key = {mDim, nDim, kDimKey, mfmaId, mfmaVersion};

  auto it = mfmaInsnAttrMap.find(key);
  if (it == mfmaInsnAttrMap.end())
    return failure();
  if (mfmaId == MfmaTypeId::F8F6F4TyId) {
    return MfmaInsn(elementTypeA, elementTypeB, it->second);
  } else {
    auto [instrElementTypeA, instrElementTypeB] =
        TypesFromMfmaId(elementTypeA.getContext(), mfmaId);
    return MfmaInsn(instrElementTypeA, instrElementTypeB, it->second);
  }
}

MfmaInsn::MfmaInsn(Type elementTypeA, Type elementTypeB,
                   const MfmaInsnAttr &attr)
    : elementTypeA(elementTypeA), elementTypeB(elementTypeB), attr(attr) {}

unsigned MfmaInsn::getKDim() { return attr.k; }
unsigned MfmaInsn::getMDim() { return attr.m; }
unsigned MfmaInsn::getNDim() { return attr.n; }
StringRef MfmaInsn::getInsnName() { return attr.insn; }
unsigned MfmaInsn::getKBase() { return attr.kBase; }
Type MfmaInsn::getElementTypeA() { return elementTypeA; }
Type MfmaInsn::getElementTypeB() { return elementTypeB; }
} // namespace mlir
