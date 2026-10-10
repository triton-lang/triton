#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h" // TODO(fywkevin): move Utility.h to include/
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/NVPTXAddrSpace.h"

namespace mlir::triton::proton::gpu::NVIDIA {

Value TargetInfo::clock(ConversionPatternRewriter &rewriter, Location loc,
                        bool isClock64) const {
  // Keep counter reads ordered with memory accesses.
  Value clock = LLVM::createLLVMIntrinsicCallOp(
                    rewriter, loc, "llvm.readcyclecounter", i64_ty, {})
                    .getResult(0);
  if (!isClock64)
    clock = LLVM::TruncOp::create(rewriter, loc, i32_ty, clock);
  return clock;
}

Value TargetInfo::globalTime(ConversionPatternRewriter &rewriter,
                             Location loc) const {
  // globaltimer is a 64-bit global clock counter in nanoseconds.
  // Reference:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer
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

unsigned TargetInfo::getPtrAddressSpace(triton::PtrAddrSpace space) const {
  switch (space) {
  case triton::PtrAddrSpace::Global:
  // Global memory read-only marking is not carried via a different LLVM
  // address space. We channel it through other mechanisms.
  case triton::PtrAddrSpace::Constant:
    return llvm::NVPTXAS::ADDRESS_SPACE_GLOBAL;
  case triton::PtrAddrSpace::Descriptor:
    return llvm::NVPTXAS::ADDRESS_SPACE_GENERIC;
  }
  llvm_unreachable("unknown PtrAddrSpace");
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
