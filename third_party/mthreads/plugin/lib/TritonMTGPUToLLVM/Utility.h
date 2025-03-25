#ifndef TRITON_CONVERSION_TRITONMTGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONMTGPU_TO_LLVM_UTILITY_H

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace LLVM {

namespace MUSA {
const char Predicated_Load[] = "__predicated_load";
const char Predicated_Store[] = "__predicated_store";

// Value getSRegValue(OpBuilder &b, Location loc, const std::string &sRegStr);
Value MTGPU_shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                       unsigned width);
Value MTGPU_shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                      unsigned width);
Value MTGPU_shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                       unsigned width);
Value MTGPU_shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                       unsigned width);

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis);

// Loads from shared or global memory with predication.
// `otherElems` is used to mask out the elements that are not loaded
Value llLoad(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
             Type elemTy, Value pred, Value falseVal);

// Stores to shared or global memory with predication.
void llStore(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
             Value val, Value pred);
} // namespace MUSA
} // namespace LLVM

} // namespace mlir

#endif
