#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h" // TODO(fywkevin): move Utility.h to include/
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

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  return LLVM::NVIDIA::getSRegValue(rewriter, loc, "smid");
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (mlir::isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else if (mlir::isa<proton::gpu::StackMemorySpaceAttr>(addressSpace)) {
    spaceId = 1;
  } else if (mlir::isa<proton::gpu::HeapMemorySpaceAttr>(addressSpace)) {
    spaceId = 1;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace, "
                             "StackMemorySpace, and HeapMemorySpace for now");
  }
  return spaceId;
}

} // namespace mlir::triton::proton::gpu::NVIDIA
