#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"


int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect,
                  mlir::triton::gpu::TritonGPUDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Triton (GPU) optimizer driver\n", registry)
  );
}
