#ifndef TritonCPUTransforms_CONVERSION_PASSES_H
#define TritonCPUTransforms_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace cpu {

enum class Ukernels {
  OneDNN,
  XSMM,
};

#define GEN_PASS_DECL
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertUnsupportedOps();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertUnsupportedOps(bool promoteBf16ToFp32,
                            bool convertMixedPrecisionMatmul,
                            bool promoteLibMathToFp32);
std::unique_ptr<OperationPass<ModuleOp>> createDecomposeFpConversions();
std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeFpConversions(bool decomposeBf16Conversions,
                             bool decomposeFp8Conversions);
std::unique_ptr<OperationPass<ModuleOp>> createOptimizeMasks();

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotProduct();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotProduct(bool useHorizontalSum);

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToAMX();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotToAMX(bool convertInt8, bool convertFp16, bool convertBf16);
std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToFMA();
std::unique_ptr<OperationPass<ModuleOp>> createConvertDotGeneric();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertVectorContractToArmNeonBFMMLA();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotToNanokernel(std::string cpuFeatures = "");

std::unique_ptr<OperationPass<ModuleOp>>
createUnrollAndReorderElementwiseOps(std::string cpuFeatures = "");

std::unique_ptr<OperationPass<ModuleOp>> createCanonicalize();

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotOpToUkernelOps(
    Ukernels ukernels = mlir::triton::cpu::Ukernels::OneDNN);

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"

} // namespace cpu
} // namespace triton

} // namespace mlir

#endif
