#include "Utility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"

namespace {
enum class ShflKind : uint32_t {
  bfly = 0,
  up = 1,
  down = 2,
  idx = 3,
};
}

namespace mlir::LLVM::AMD {
static Value shuffleCommon(Location loc, ConversionPatternRewriter &rewriter,
                           Value val, Value i, int strideInt, ShflKind mode,
                           Value clamp) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  // On AMD, the ds_swizzle_b32 and ds_permute_b32 instructions work on
  // 32bit/dwords so we need promote to 32 here.
  auto valType = val.getType();
  if (!valType.isInteger(32) && bits <= 32) {
    if (!valType.isIntOrIndex())
      val = bitcast(val, int_ty(bits));
    if (bits < 32)
      val = sext(i32_ty, val);

    val = shuffleCommon(loc, rewriter, val, i, strideInt, mode, clamp);

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
    val0 = shuffleCommon(loc, rewriter, val0, i, strideInt, mode, clamp);
    val1 = shuffleCommon(loc, rewriter, val1, i, strideInt, mode, clamp);
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
  case ShflKind::bfly:
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
  case ShflKind::up: {
    Value mask = icmp_slt(laneId, i);
    Value delta = sub(laneId, i);
    Value index = select(mask, laneId, delta);
    return bpermute(index);
  }
  case ShflKind::idx:
    return bpermute(i);
  default:
    assert(false && "Unsupported ShflKind");
    break;
  }
  return Value();
}

Value shuffleXor(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), i, ShflKind::bfly,
                       i32_val(0x1f));
}

Value shuffleUp(Location loc, ConversionPatternRewriter &rewriter, Value val,
                int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), i, ShflKind::up,
                       i32_val(0x0));
}

Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i) {
  return shuffleIdx(loc, rewriter, val, i32_val(i));
}

Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 Value i) {
  return shuffleCommon(loc, rewriter, val, i, 0, ShflKind::idx, i32_val(0x1f));
}

Value llGetPid(Location loc, ConversionPatternRewriter &rewriter,
               ModuleOp moduleOp, int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
  Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[axis]);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

Value llLoad(ConversionPatternRewriter &rewriter, Location loc,
             const TypeConverter *converter, Value ptr, Type elemTy, Value pred,
             unsigned vecStart, SmallVector<Value> otherElems) {
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *afterLoad =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  afterLoad->addArgument({elemTy}, {loc});
  Block *trueBlock = rewriter.createBlock(afterLoad);
  Block *falseBlock =
      rewriter.splitBlock(trueBlock, rewriter.getInsertionPoint());
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, falseBlock);
  rewriter.setInsertionPointToStart(trueBlock);
  auto loadOp = rewriter.create<LLVM::LoadOp>(loc, elemTy, ptr);
  rewriter.create<LLVM::BrOp>(loc, loadOp->getResult(0), afterLoad);
  rewriter.setInsertionPointToStart(falseBlock);
  auto valueElemTy = getElementTypeOrSelf(elemTy);
  mlir::Attribute zero = rewriter.getZeroAttr(valueElemTy);
  Value zeroVal;
  if (auto shapedTy = elemTy.dyn_cast<mlir::ShapedType>()) {
    auto denseValue = DenseElementsAttr::get(shapedTy, zero);
    zeroVal = rewriter.create<LLVM::ConstantOp>(loc, elemTy, denseValue);
  } else {
    zeroVal = rewriter.create<LLVM::ConstantOp>(loc, elemTy, zero);
  }
  Value falseVal = zeroVal;
  // If we need to mask the loaded value with other elements
  if (otherElems.size() != 0) {
    auto vecTy = dyn_cast<VectorType>(elemTy);
    assert(vecTy && "Expected vector type");
    assert(ptr.getType().cast<LLVMPointerType>().getAddressSpace() == 0 &&
           "Expected to only mask global memory loads");
    auto vec = vecTy.getNumElements();
    Value v = undef(elemTy);
    for (size_t s = 0; s < vec; ++s) {
      Value otherElem = otherElems[vecStart + s];
      Value indexVal = createIndexConstant(rewriter, loc, converter, s);
      v = insert_element(elemTy, v, otherElem, indexVal);
    }
    falseVal = v;
  }
  rewriter.create<LLVM::BrOp>(loc, falseVal, afterLoad);
  rewriter.setInsertionPointToStart(afterLoad);
  Value loadVal = afterLoad->getArgument(0);
  return loadVal;
}

Value llStore(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
              Value val, Value pred) {
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *afterStore =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Block *trueBlock = rewriter.createBlock(afterStore);
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, afterStore);
  rewriter.setInsertionPointToStart(trueBlock);
  auto storeOp = rewriter.create<LLVM::StoreOp>(loc, val, ptr);
  rewriter.create<LLVM::BrOp>(loc, afterStore);
  rewriter.setInsertionPointToStart(afterStore);
  return val;
}

} // namespace mlir::LLVM::AMD
