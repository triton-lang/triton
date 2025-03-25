//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
// clang-format off
#include "xpu/lib/Conversion/TritonXPUToLLVM/TargetInfo.h"  // TargetInfo

#include "triton/Analysis/UtilityXPU.h"
#include "xpu/lib/Conversion/TritonXPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// clang-format on

using namespace mlir;

namespace mlir {
namespace triton {
namespace xpu {

bool TargetInfo::supportMaximumMinimum() const {
  llvm_unreachable("not impl");
  return false;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::ballot(ConversionPatternRewriter &rewriter, Location loc,
                         Type type, Value cmp) const {
  llvm_unreachable("not impl");
  return Value();
}

void TargetInfo::storeShared(ConversionPatternRewriter &rewriter, Location loc,
                             Value ptr, Value val, Value pred) const {
  llvm_unreachable("not impl");
}

Value TargetInfo::loadShared(ConversionPatternRewriter &rewriter, Location loc,
                             const TypeConverter *converter, Value ptr,
                             Type elemTy, Value pred) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleXor(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleUp(ConversionPatternRewriter &rewriter, Location loc,
                            Value val, int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, int i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
                             Value val, Value i) const {
  llvm_unreachable("not impl");
  return Value();
}

Value TargetInfo::programId(ConversionPatternRewriter &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::XPU::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce) const {
  llvm_unreachable("not impl");
  return false;
}

bool TargetInfo::processReplicaUsingStMatrix(
    ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
    SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
    ArrayRef<unsigned> outOrd, unsigned accumNumReplicates,
    int swizzlingByteWidth) const {
  llvm_unreachable("not impl");
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "_ZN3xpu6umulhiEjj" : "Unsupported";
  return funcName;
}

void TargetInfo::printf(ConversionPatternRewriter &rewriter,
                        Value formatStrStart, int /*formatStrByteCount*/,
                        ValueRange args) const {
  llvm_unreachable("not impl");
}

void TargetInfo::assertFail(ConversionPatternRewriter &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  llvm_unreachable("not impl");
}

uint32_t TargetInfo::getXPUArch() const { return this->xpu_arch; }
uint32_t TargetInfo::getXPUBufferSize() const { return this->buffer_size; }

} // namespace xpu
} // namespace triton
} // namespace mlir
