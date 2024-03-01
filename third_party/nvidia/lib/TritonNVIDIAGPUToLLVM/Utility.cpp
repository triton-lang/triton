#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
// #include "mlir/Dialect/LLVMIR/NVVMDialect.h"
// #include "triton/Dialect/NVGPU/IR/Dialect.h"
namespace mlir {

namespace LLVM {

namespace NVIDIA {
using namespace mlir::triton;

Value getSRegValue(OpBuilder &b, Location loc, const std::string &sRegStr) {
  PTXBuilder builder;
  auto &mov = builder.create("mov")->o("u32");
  auto *destOpr = builder.newOperand("=r");
  auto *sRegOpr = builder.newConstantOperand(sRegStr);
  mov(destOpr, sRegOpr);
  Value val = builder.launch(b, loc, b.getIntegerType(32), false);
  return val;
}
} // namespace NVIDIA
} // namespace LLVM
} // namespace mlir
