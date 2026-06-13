#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h" // TODO(fywkevin): move Utility.h to include/
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::proton::gpu::NVIDIA {

Value TargetInfo::globalTime(ConversionPatternRewriter &rewriter,
                             Location loc) const {
  return NVVM::GlobalTimerOp::create(rewriter, loc, i64_ty);
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  return NVVM::SmIdOp::create(rewriter, loc, i32_ty);
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
