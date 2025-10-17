#ifndef NVG_CONVERSION_PASSES_H
#define NVG_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/NVGToLLVM/NVGToLLVMPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "nvidia/include/NVGToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "nvidia/include/NVGToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
