#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::proton::gpu::NVIDIA {

Value TargetInfo::clock(ConversionPatternRewriter &rewriter, Location loc,
                        bool isClock64) const {
  int width = isClock64 ? 64 : 32;
  std::string dtype = isClock64 ? "u64" : "u32";
  std::string reg = isClock64 ? "%clock64" : "%clock";

  PTXBuilder builder;
  auto &mov = builder.create("mov")->o(dtype);
  auto *destOpr = builder.newOperand("=r");
  auto *sRegOpr = builder.newConstantOperand(reg);
  mov(destOpr, sRegOpr);
  Value val =
      builder.launch(rewriter, loc, rewriter.getIntegerType(width), true);

  return val;
}

} // namespace mlir::triton::proton::gpu::NVIDIA
