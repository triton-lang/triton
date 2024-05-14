#ifndef TRITON_CONVERSION_TRITONCPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONCPU_TO_LLVM_UTILITY_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::triton;

// TODO: Do better refactoring.
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "ttcpu_to_llvm"

#endif
