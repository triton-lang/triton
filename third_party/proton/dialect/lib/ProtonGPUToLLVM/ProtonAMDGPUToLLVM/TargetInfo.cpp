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
  // NV has both a 32 bit and 64 bit clock intrinsic. On AMD we only have
  // s_memtime which is 64 bit. However truncating the 64 bit version
  // in cases of requesting 32 bit should be fine, since in 64 bits, 
  // after 0x0000.0000.ffff.ffff comes 0x0000.0001.0000.0000, and 
  // truncating that to 32 bits gives zero, effectively wrapping from 
  // 0xffff.ffff to 0x0000.0000. 
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  StringRef clock64IntrinsicName = "llvm.amdgcn.s.memtime";
  Value clockVal =  LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, clock64IntrinsicName, i64_ty, {}).getResult(0);
  if(!isClock64)
	  clockVal = rewriter.create<LLVM::TruncOp>(loc, i32_ty, clockVal);
  return clockVal;
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto cu_id = builder.newOperand("=s");
  auto hwreg = builder.newConstantOperand("hwreg(HW_REG_HW_ID, 8, 4)");
  gethwid(cu_id, hwreg);

//  	auto &getxccid = *builder.create("s_getreg_b32");
//  	auto xcc_res = builder.newOperand("=s");
//	auto xcc_reg = builder.newConstantOperand("hwreg(HW_REG_XCC_ID)");
//  	gethwid(xcc_res, xcc_reg);
  
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
