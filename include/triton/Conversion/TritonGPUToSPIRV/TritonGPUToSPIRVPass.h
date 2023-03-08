//
// Created by guangyey on 12/28/22.
//

#ifndef TRITON_TRITONGPUTOSPIRVPASS_H
#define TRITON_TRITONGPUTOSPIRVPASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton {


std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToSPIRVPass(int computeCapability = 80);

} // namespace triton

} // namespace mlir

#endif //TRITON_TRITONGPUTOSPIRVPASS_H
