#ifndef TRITONTOTRITONCPU_CONVERSION_PASSES_H
#define TRITONTOTRITONCPU_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace cpu {

#define GEN_PASS_DECL
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertElementwiseOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertMemoryOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertPtrOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertDotOp();
std::unique_ptr<OperationPass<ModuleOp>> createConvertControlFlowOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertHistogramOp();
std::unique_ptr<OperationPass<ModuleOp>> createConvertReductionOp();

void tritonToTritonCPUPipelineBuilder(OpPassManager &pm);
void registerTritonToTritonCPUPipeline();

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"

} // namespace cpu
} // namespace triton

} // namespace mlir

#endif
