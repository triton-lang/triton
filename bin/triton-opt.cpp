#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
} // namespace test
} // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::registerTritonGPUPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerConvertTritonGPUToLLVMPass();

  // TODO: register Triton & TritonGPU passes
  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
                  mlir::triton::gpu::TritonGPUDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::gpu::GPUDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton (GPU) optimizer driver\n", registry));
}
