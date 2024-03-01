#include "TargetInfo.h"
#include "amd/include/TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
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
bool TargetInfo::supportMaximumMinimum() const { return false; }
Value TargetInfo::callBallotOp(ConversionPatternRewriter &rewriter,
                               Location loc, Type type, Value cmp) const {
  auto stringAttr = rewriter.getStringAttr("llvm.amdgcn.ballot");
  SmallVector<Value> operands = {cmp};
  Value asmResult =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, type, stringAttr, operands)
          ->getResult(0);
  return asmResult;
}

Value TargetInfo::storeShared(ConversionPatternRewriter &rewriter, Location loc,
                              Value ptr, Value val, Value pred) const {
  store(val, ptr);
  return val;
}

Value TargetInfo::loadShared(ConversionPatternRewriter &rewriter, Location loc,
                             Value ptr, Type elemTy, Value pred) const {
  return load(elemTy, ptr);
}

Value TargetInfo::shflSync(Location loc, ConversionPatternRewriter &rewriter,
                           Value val, int i) const {
  return commonShflSync(loc, rewriter, val, i32_val(i), i, NVVM::ShflKind::bfly,
                        i32_val(0x1f));
}

Value TargetInfo::shflUpSync(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return commonShflSync(loc, rewriter, val, i32_val(i), i, NVVM::ShflKind::up,
                        i32_val(0x0));
}

Value TargetInfo::shflIdxSync(Location loc, ConversionPatternRewriter &rewriter,
                              Value val, int i) const {
  return shflIdxSync(loc, rewriter, val, i32_val(i));
}

Value TargetInfo::shflIdxSync(Location loc, ConversionPatternRewriter &rewriter,
                              Value val, Value i) const {
  return commonShflSync(loc, rewriter, val, i, 0, NVVM::ShflKind::idx,
                        i32_val(0x1f));
}

bool TargetInfo::warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce) const {
  return false;
}
unsigned TargetInfo::getShuffleIndex(triton::ReduceOp op, unsigned N,
                                     unsigned numLaneToReduce) const {
  unsigned shuffleIdx = N;
  auto srcTys = op.getInputTypes();
  auto inputTy = srcTys[0].cast<RankedTensorType>();
  auto inMfma =
      inputTy.getEncoding().dyn_cast<triton::gpu::AMDMfmaEncodingAttr>();
  if (inMfma && inMfma.getIsTransposed()) {
    assert(numLaneToReduce == 2 || numLaneToReduce == 4);
    // for mfma 32x32 adjacent threads in y dimension in transposed MFMA
    // layout are 32 apart: [[0 0 0 0 32 32 32 32 ...] [1 1 1 1 33 33 33 33
    // ...] ...]. for mfma 16x16 adjacent threads in y dimension in
    // transposed MFMA layout are 16 apart: [[0 0 0 0 16 16 16 16 32 32 32
    // 32 ...] [1 1 1 1 33 33 33 33 ...] ...].
    const int warpSize = 64;
    shuffleIdx = warpSize / N / 2;
  }
  return shuffleIdx;
}
} // namespace AMD