//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITON_CONVERSION_TRITONXPU_TO_LLVM_PATTERNS_TRITON_XPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONXPU_TO_LLVM_PATTERNS_TRITON_XPU_OP_TO_LLVM_H

// clang-format off
// Dialect
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

#include "triton/Analysis/Membar.h"                 // ModuleMembarAnalysis
#include "triton/Analysis/AxisInfo.h"               // ModuleAxisInfoAnalysis
#include "triton/Analysis/Allocation.h"             // ModuleAllocation


#include "triton/Analysis/Utility.h"
#include "xpu/lib/Conversion/TritonXPUToLLVM/Utility.h"

#include "xpu/lib/Conversion/TritonXPUToLLVM/TargetInfo.h"  // TargetInfo
#include "triton/Conversion/TritonXPUToLLVM/TypeConverter.h" // TritonXPUToLLVMTypeConverter

#include "llvm/Support/ErrorHandling.h"
// clang-format on

namespace mlir {
namespace triton {
namespace xpu {

//===----------------------------------------------------------------------===//
// triton::xpu::LoadOp, triton::xpu::StoreOp, triton::xpu::AllocaOp,
// triton::xpu::GM2LMOp, triton::xpu::LM2GMOp
//===----------------------------------------------------------------------===//
void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);

//===----------------------------------------------------------------------===//
// mlir::gpu::ThreadIdOp -> mlir::LLVM::XPU::CoreIdOp
// mlir::gpu::BlockIdOp -> mlir::LLVM::XPU::LoadParamOp[0]
// mlir::gpu::GridDimOp -> mlir::LLVM::XPU::LoadParamOp[1]
// mlir::gpu::BlockDimOp -> mlir::LLVM::XPU::LoadParamOp[2]
//
// Collect a set of patterns to convert from the mlir::gpu dialect to
// mlir::LLVM::XPU.
//===----------------------------------------------------------------------===//
void populateGPUToXPUConversionPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const TargetInfo &targetInfo,
                                        PatternBenefit benefit);

//===----------------------------------------------------------------------===//
// triton::xpu::MakeRangeOp
//===----------------------------------------------------------------------===//
void populateMakeRangeOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                      const TargetInfoBase &targetInfo,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit);

//===----------------------------------------------------------------------===//
// triton::xpu::ExtractOp
//===----------------------------------------------------------------------===//
void populateTTXPUUtilityOpToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit);

//===----------------------------------------------------------------------===//
// triton::xpu::ExtractOp
//===----------------------------------------------------------------------===//
void populateTTXPUVectorizedOpToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit);

//===----------------------------------------------------------------------===//
// arith::ExtFOp -> LLVM::FPExtOp         arith::TruncFOp -> LLVM::FPTruncOp
// arith::SIToFPOp -> LLVM::SIToFPOp      arith::FPToSIOp -> LLVM::FPToSIOp
// triton::PreciseSqrtOp -> LLVM::SqrtOp
// arith::AddFOp -> LLVM::FAddOp          arith::SubFOp, LLVM::FSubOp
// arith::MulFOp -> LLVM::FMulOp          arith::DivFOp, LLVM::FDivOp
// triton::PreciseDivFOp -> LLVM::FDivOp
//===----------------------------------------------------------------------===//
void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfo &targetInfo,
    PatternBenefit benefit);

//===----------------------------------------------------------------------===//
// triton::xpu::ConvertLayoutOp -> <>
//===----------------------------------------------------------------------===//
void populateConvertLayoutOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfo &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

//===----------------------------------------------------------------------===//
// triton::ExpandDimsOp -> <>
//===----------------------------------------------------------------------===//
void populateViewOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);

//===----------------------------------------------------------------------===//
// triton::xpu::ReduceOp -> calculation logic
//===----------------------------------------------------------------------===//
void populateReduceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfo &targetInfo,
                                    PatternBenefit benefit);

//===----------------------------------------------------------------------===//
// triton::FuncOp -> LLVM::FuncOp
//===----------------------------------------------------------------------===//
void populateFuncOpConversionPattern(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

//===----------------------------------------------------------------------===//
// triton::GetNumProgramsOp -> LLVM::LoadParamOp
//===----------------------------------------------------------------------===//
void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONXPU_TO_LLVM_PATTERNS_TRITON_XPU_OP_TO_LLVM_H
