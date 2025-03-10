#ifndef TRITON_CONVERSION_PROTONGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_PROTONGPU_TO_LLVM_UTILITY_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace proton {
namespace gpu {

static Value getGlobalScratchPtr(Location loc, RewriterBase &rewriter,
                                 FunctionOpInterface funcOp,
                                 Value allocOffset = {}) {
  auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() - 1);
  ModuleOp mod = funcOp.getOperation()->getParentOfType<ModuleOp>();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  Value bufferPtr = b.gep(mlir::LLVM::LLVMPointerType::get(ctx, 1), i8_ty,
                          gmemBase, allocOffset);
  return bufferPtr;
}
} // namespace gpu
} // namespace proton
} // namespace triton
} // namespace mlir
#endif
