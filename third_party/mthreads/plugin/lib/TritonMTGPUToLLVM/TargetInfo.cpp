#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

using mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;

namespace mlir::triton::MUSA {

bool TargetInfo::supportMaximumMinimum() const {
  // TODO(lingfeng.qiu): Mtcc currently does not support llvm.minimum and
  // llvm.maximum.
  return false;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  // TODO(lingfeng.qiu): Figure out whether MTGPU support CTA clusters.
  // On AMD hardware we don't have CTA clusters like NVIDIA. So this will always
  // be zero. Whoever calling into this should make sure the whole program does
  // not try to utilize CTA clusters.
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

Value TargetInfo::ballot(ConversionPatternRewriter &rewriter, Location loc,
                         Type type, Value cmp) const {
  auto int32Ty = rewriter.getI32Type();
  return rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
                                           rewriter.getI32IntegerAttr(0));
}

void TargetInfo::storeShared(ConversionPatternRewriter &rewriter, Location loc,
                             Value ptr, Value val, Value pred) const {
  mlir::LLVM::MUSA::llStore(rewriter, loc, ptr, val, pred);
}

Value TargetInfo::loadShared(ConversionPatternRewriter &rewriter, Location loc,
                             const TypeConverter *converter, Value ptr,
                             Type elemTy, Value pred) const {
  Value falseVal = rewriter.create<arith::ConstantOp>(
      loc, elemTy, rewriter.getZeroAttr(elemTy));
  return mlir::LLVM::MUSA::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal);
}

Value TargetInfo::shuffleXor(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, int i) const {
  return mlir::LLVM::MUSA::MTGPU_shuffleXor(loc, rewriter, val, i, 128);
}

Value TargetInfo::shuffleUp(ConversionPatternRewriter &rewriter, Location loc,
                            Value val, int i) const {
  return mlir::LLVM::MUSA::MTGPU_shuffleUp(loc, rewriter, val, i, 128);
}

Value TargetInfo::shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, int i) const {
  return mlir::LLVM::MUSA::MTGPU_shuffleIdx(loc, rewriter, val, i, 128);
}

Value TargetInfo::shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, Value i) const {
  return mlir::LLVM::MUSA::MTGPU_shuffleIdx(loc, rewriter, val, i, 128);
}

Value TargetInfo::programId(ConversionPatternRewriter &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::MUSA::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce) const {
  return false;
}

bool TargetInfo::processReplicaUsingStMatrix(
    ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
    SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
    ArrayRef<unsigned> outOrd, unsigned accumNumReplicates,
    int swizzleByteWidth) const {
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__mt_umulhi" : "__mt_umul64hi";
  return funcName;
}

void TargetInfo::printf(ConversionPatternRewriter &rewriter,
                        Value formatStrStart, int /*formatStrByteCount*/,
                        ValueRange args) const {
  return;
}

void TargetInfo::assertFail(ConversionPatternRewriter &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  return;
}

} // namespace mlir::triton::MUSA
