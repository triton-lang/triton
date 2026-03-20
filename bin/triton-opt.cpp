#include "./RegisterTritonDialects.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

int main(int argc, char **argv) {

  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton (GPU) optimizer driver\n", registry));
}
