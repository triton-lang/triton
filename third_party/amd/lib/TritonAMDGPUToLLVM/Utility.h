#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_UTILITY_H

#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"

#include "../lib/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"

namespace mlir {
namespace LLVM {
using namespace mlir::triton;
namespace AMD {

Value shflSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
               int i);
Value shflUpSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i);
Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  int i);
Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  Value i);
} // namespace AMD

} // namespace LLVM
} // namespace mlir

#endif
