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

  auto getClockReg = [&](const std::string &clkName) {
    PTXBuilder builder;
    auto &movLow = builder.create("mov")->o("u32");
    auto *destLowOpr = builder.newOperand("=r");
    auto *sRegLowOpr = builder.newConstantOperand(clkName);
    movLow(destLowOpr, sRegLowOpr);
    Value clkLow32 =
        builder.launch(rewriter, loc, rewriter.getIntegerType(32), true);
    return clkLow32;
  };

  Value clkLow32 = getClockReg("%clock");

  if (!isClock64)
    return clkLow32;

  Value clkHigh32 = getClockReg("%clock_hi");

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value clkLow64 = b.zext(i64_ty, clkLow32);
  Value clkHigh64 = b.zext(i64_ty, clkHigh32);
  Value clock64 = b.or_(b.shl(clkHigh64, b.i64_val(32)), clkLow64);
  return clock64;
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  return LLVM::NVIDIA::getSRegValue(rewriter, loc, "smid");
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (mlir::isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else if (mlir::isa<proton::gpu::GlobalMemorySpaceAttr>(addressSpace)) {
    spaceId = 1;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace, "
                             "and GlobalMemorySpace for now");
  }
  return spaceId;
}

int TargetInfo::getIndexPtrAddrSpace() const {
  // Internal buffer index is private to each thread, we use generic address
  // space for NV GPUs. See detail discussion:
  // https://llvm.org/docs/NVPTXUsage.html#address-spaces
  // The reason we don't use address space 5 is due to the downstream compiler
  // generates incorrect `cvta` instruction for %SP/%SPL register that causes
  // IMA when we perform thread-private memory access like `ld.local`.
  return 0;
}

} // namespace mlir::triton::proton::gpu::NVIDIA
