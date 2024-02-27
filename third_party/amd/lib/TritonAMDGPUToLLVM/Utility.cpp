#include "Utility.h"
#include "TritonGPUToLLVMBase.h"
#include "TypeConverter.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"

namespace mlir {

namespace LLVM {
using namespace mlir::triton;

namespace AMD {
static Value commonShflSync(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, Value i, int strideInt,
                            NVVM::ShflKind mode, Value clamp) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  // On AMD, the ds_swizzle_b32 and ds_permute_b32 instructions work on
  // 32bit/dwords so we need promote to 32 here.
  auto valType = val.getType();
  if (!valType.isInteger(32) && bits <= 32) {
    if (!valType.isIntOrIndex())
      val = bitcast(val, int_ty(bits));
    if (bits < 32)
      val = sext(i32_ty, val);

    val = commonShflSync(loc, rewriter, val, i, strideInt, mode, clamp);

    if (bits < 32)
      val = trunc(int_ty(bits), val);
    if (!valType.isIntOrIndex())
      val = bitcast(val, valType);
    return val;
  }

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = commonShflSync(loc, rewriter, val0, i, strideInt, mode, clamp);
    val1 = commonShflSync(loc, rewriter, val1, i, strideInt, mode, clamp);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Value threadId =
      rewriter.create<::mlir::gpu::ThreadIdOp>(loc, ::mlir::gpu::Dimension::x);
  threadId = rewriter.create<arith::IndexCastOp>(loc, i32_ty, threadId);
  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = i32_val(iWarpSize);
  Value laneId = urem(threadId, warpSize);
  auto bpermute = [&](Value lane) {
    // Multiple lineId by 4. (More on permute instruction semantics:
    // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf#page=180
    Value byteOffset = i32_val(2);
    Value permuteAddr = shl(lane, byteOffset);
    return rewriter.create<ROCDL::DsBpermuteOp>(loc, valType, permuteAddr, val);
  };

  switch (mode) {
  case NVVM::ShflKind::bfly:
    if (strideInt > 16) {
      Value threadId =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  loc, TypeRange{i32_ty},
                  ValueRange{rewriter.create<::mlir::gpu::ThreadIdOp>(
                      loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x)})
              .getResult(0);
      Value stride = i32_val(32);
      Value lineId = xor_(threadId, stride);
      return bpermute(lineId);
    } else {
      // This map facilates the butterfly shuffle pattern for a stride less
      // than 16. The pattern stride is the key of the map.
      DenseMap<short, unsigned int> masks{
          {16, 0x401F}, {8, 0x201F}, {4, 0x101F}, {2, 0x081F}, {1, 0x041F}};
      Value offset = i32_val(masks[strideInt]);
      return rewriter.create<ROCDL::DsSwizzleOp>(loc, valType, val, offset);
    }
    break;
  case NVVM::ShflKind::up: {
    Value mask = icmp_slt(laneId, i);
    Value delta = sub(laneId, i);
    Value index = select(mask, laneId, delta);
    return bpermute(index);
  }
  case NVVM::ShflKind::idx:
    return bpermute(i);
  default:
    assert(false && "Unsupported ShflKind");
    break;
  }
  return Value();
}

Value shflSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
               int i) {
  return commonShflSync(loc, rewriter, val, i32_val(i), i, NVVM::ShflKind::bfly,
                        i32_val(0x1f));
}

Value shflUpSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i) {
  return commonShflSync(loc, rewriter, val, i32_val(i), i, NVVM::ShflKind::up,
                        i32_val(0x0));
}

Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  int i) {
  return shflIdxSync(loc, rewriter, val, i32_val(i));
}

Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  Value i) {
  return commonShflSync(loc, rewriter, val, i, 0, NVVM::ShflKind::idx,
                        i32_val(0x1f));
}

Value storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                  Value val, Value pred) {
  auto ty = val.getType();
  auto val_ty = vec_ty(ty, 1);
  Value vec_val = undef(val_ty);
  vec_val = insert_element(val_ty, vec_val, val, i32_val(0));

  auto vec_ty = vec_ty(i1_ty, 1);
  Value vec_pred = undef(vec_ty);
  vec_pred = insert_element(vec_ty, vec_pred, pred, i32_val(0));

  rewriter.create<LLVM::MaskedStoreOp>(loc, vec_val, ptr, vec_pred, 4);
  return val;
}

template<class T>
T getVal(const std::string& val) {
  if (val == "max") {
    return std::numeric_limits<T>::max();
  } else if (val == "min") {
    return std::numeric_limits<T>::min();
  } else if (val == "zero") {
    return T(0);
  } else if (val == "one") {
    return T(1);
  }
  return T(110);
}

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Type elemTy, Value pred, const std::string& val) {
  auto loaded = rewriter.create<scf::IfOp>(loc, pred,
    [&](OpBuilder& builder, Location loc) {
      auto loadVal = load(elemTy, ptr);
      builder.create<scf::YieldOp>(loc, ValueRange(loadVal));
    },
    [&](OpBuilder& builder, Location loc) {
      Value initVal;
      if (elemTy.isF16()) {
        float fVal = getVal<float>(val);
        initVal = f16_val(fVal);
      } else if (elemTy.isInteger(32)) {
        int ival = getVal<int>(val);
        initVal = i32_val(ival);
      } else if (elemTy.isInteger(64)) {
        int64_t i64Val = getVal<int64_t>(val);
        initVal = int_val(64, i64Val);
      } else if (elemTy.isF64()) {
        double dVal = getVal<double>(val);
        initVal = f64_val(dVal);
      } else {
        float fVal = getVal<float>(val);
        initVal = f32_val(fVal);
      }
      builder.create<mlir::scf::YieldOp>(loc, ValueRange({initVal}));
    });
  return loaded->getResult(0);
}


} // namespace AMD

} // namespace LLVM
} // namespace mlir
