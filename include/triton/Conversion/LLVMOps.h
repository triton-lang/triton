#ifndef TRITON_CONVERSION_LLVM_OPS_H
#define TRITON_CONVERSION_LLVM_OPS_H

// Older MLIR packages expose core operations and intrinsics in one header.
#if __has_include("mlir/Dialect/LLVMIR/LLVMOps.h") &&                          \
                  __has_include("mlir/Dialect/LLVMIR/LLVMIntrinsicOps.h")
#include "mlir/Dialect/LLVMIR/LLVMOps.h"
#else
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#endif

#endif // TRITON_CONVERSION_LLVM_OPS_H
