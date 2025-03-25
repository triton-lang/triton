//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_CONVERSION_TRITONXPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONXPU_TO_LLVM_UTILITY_H

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

#define addrspace_cast(...)                                                    \
  rewriter.create<LLVM::AddrSpaceCastOp>(loc, __VA_ARGS__)
#define allocate(...) rewriter.create<LLVM::AllocaOp>(loc, __VA_ARGS__)
#define idx_val(...)                                                           \
  LLVM::createIndexConstant(rewriter, loc, this->getTypeConverter(),           \
                            __VA_ARGS__)
#define sdiv(...) rewriter.create<LLVM::SDivOp>(loc, __VA_ARGS__)
#define srem(...) rewriter.create<LLVM::SRemOp>(loc, __VA_ARGS__)
#define load_sm(...) rewriter.create<LLVM::LoadOp>(loc, __VA_ARGS__)
#define store_sm(...) rewriter.create<LLVM::StoreOp>(loc, __VA_ARGS__)
#define xpu_barrier() rewriter.create<mlir::LLVM::XPU::BarrierOp>(loc)
#define i16_val(...)                                                           \
  LLVM::createLLVMIntegerConstant(rewriter, loc, 16, __VA_ARGS__)

namespace mlir::triton {
inline size_t align(size_t elemNum, Type elemTy, size_t target) {
  size_t elemBit = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                       ? 64u
                       : elemTy.getIntOrFloatBitWidth();
  size_t elemBytes = (elemBit / 8u) ? (elemBit / 8u) : 1;
  size_t aligned = (elemNum * elemBytes + target - 1) / target * target;
  return aligned / elemBytes;
}
} // namespace mlir::triton

namespace mlir::LLVM::XPU {

Value llGetPid(Location loc, ConversionPatternRewriter &rewriter,
               ModuleOp moduleOp, int axis);

Value createDeviceCall(StringRef funcName, ConversionPatternRewriter &rewriter,
                       Operation *op, Type &elemTy, ValueRange &operands,
                       Location &loc);

void createDeviceCall(StringRef funcName, ConversionPatternRewriter &rewriter,
                      Operation *op, ValueRange &operands, Location &loc);

SmallVector<SmallVector<unsigned>>
emitOffsetForClusterLayout(const triton::xpu::ClusterLayoutAttr &clusterLayout,
                           RankedTensorType type);

inline Value getGridDim(RewriterBase &rewriter, Location loc) {
  Value gridDim =
      rewriter.create<::mlir::gpu::GridDimOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, gridDim);
}

inline Value getBlockDim(RewriterBase &rewriter, Location loc) {
  Value blockDim =
      rewriter.create<::mlir::gpu::BlockDimOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockDim);
}

inline Value getBlockId(RewriterBase &rewriter, Location loc) {
  Value blockId =
      rewriter.create<::mlir::gpu::BlockIdOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

} // namespace mlir::LLVM::XPU

#endif // TRITON_CONVERSION_TRITONXPU_TO_LLVM_UTILITY_H
