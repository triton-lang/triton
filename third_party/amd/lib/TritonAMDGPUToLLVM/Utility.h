#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_UTILITY_H

#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
namespace mlir::LLVM::AMD {

const char Predicated_Load[] = "__predicated_load";
const char Predicated_Store[] = "__predicated_store";

Value shuffleXor(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i);
Value shuffleUp(Location loc, ConversionPatternRewriter &rewriter, Value val,
                int i);
Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i);
Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 Value i);

Value llGetPid(Location loc, ConversionPatternRewriter &rewriter,
               ModuleOp moduleOp, int axis);

// Loads from shared or global memory with predication.
// `otherElems` is used to mask out the elements that are not loaded
Value llLoad(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
             Type elemTy, Value pred, Value falseVal);

// Stores to shared or global memory with predication.
void llStore(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
             Value val, Value pred);
} // namespace mlir::LLVM::AMD

#endif
