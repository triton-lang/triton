#ifndef TRITON_TARGET_PTXTRANSLATION_H
#define TRITON_TARGET_PTXTRANSLATION_H

#include "triton/driver/dispatch.h"

#include <string>

namespace mlir {

class ModuleOp;

} // namespace mlir

namespace triton {



// Translate TritonGPU IR to PTX code.
std::string translateTritonGPUToPTX(mlir::ModuleOp module, int cc, int version);

} // namespace triton

#endif
