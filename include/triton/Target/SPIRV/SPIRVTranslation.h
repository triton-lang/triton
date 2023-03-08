#ifndef TRITON_SPIRVTRANSLATION_H
#define TRITON_SPIRVTRANSLATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"
#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace triton {

LogicalResult assembleSPIRV(std::string spirvCode, raw_ostream &output);

LogicalResult disassembleSPIRV(uint32_t* binary_ptr, size_t binary_size, raw_ostream &output);

// Translate TritonGPU dialect to SPIRV, return null if failed.
std::string
translateTritonGPUToSPIRVIR(mlir::ModuleOp module,
                        int computeCapability);

} // namespace triton
} // namespace mlir

#endif
