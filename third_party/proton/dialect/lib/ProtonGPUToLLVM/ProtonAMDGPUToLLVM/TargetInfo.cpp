#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "third_party/amd/include/TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::proton::gpu::AMD {

Value TargetInfo::clock(ConversionPatternRewriter &rewriter, Location loc,
                        bool isClock64) const {
  //TODO(crobeck): fix this to actually support the 64-bit clock value
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  StringRef clock64IntrinsicName = "llvm.amdgcn.s.memtime";
  Value clock64Val =  LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, clock64IntrinsicName, i64_ty, {}).getResult(0);
  return rewriter.create<LLVM::TruncOp>(loc, i32_ty, clock64Val);
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto res = builder.newOperand("=s");
  auto hwreg = builder.newConstantOperand("hwreg(HW_REG_HW_ID, 8, 4)");
  gethwid(res, hwreg);
  builder.create<>("s_waitcnt lgkmcnt(0)")->operator()();
  return builder.launch(rewriter, loc, i32_ty, false);
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (mlir::isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else if (mlir::isa<proton::gpu::StackMemorySpaceAttr>(addressSpace)) {
    spaceId = 1;
  } else if (mlir::isa<proton::gpu::GlobalMemorySpaceAttr>(addressSpace)) {
    spaceId = 1;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace, "
                             "StackMemorySpace, and GlobalMemorySpace for now");
  }
  return spaceId;
}

} // namespace mlir::triton::proton::gpu::AMD
