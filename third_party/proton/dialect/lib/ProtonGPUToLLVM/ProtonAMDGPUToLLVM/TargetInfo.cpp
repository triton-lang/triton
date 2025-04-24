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
#define HW_ID_CU_ID_SIZE    4
#define HW_ID_CU_ID_OFFSET  8

#define HW_ID_SE_ID_SIZE    3
#define HW_ID_SE_ID_OFFSET  13

#define HW_REG_XCC_ID_SIZE    3
#define HW_REG_XCC_ID_OFFSET  0

//gfx942
#define CU_PER_XCD 40
#define CU_PER_SE 10

//gfx950
//#define CU_PER_XCD 32
//#define CU_PER_SE 10

static Value getXCCID(ConversionPatternRewriter &rewriter,
                              Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto xcc_id = builder.newOperand("=s");
  auto xcc_reg = builder.newConstantOperand("hwreg(HW_REG_XCC_ID, HW_REG_XCC_ID_OFFSET, HW_REG_XCC_ID_SIZE)");
  gethwid(xcc_id, xcc_reg);
  return builder.launch(rewriter, loc, i32_ty, false);
}

static Value getCUID(ConversionPatternRewriter &rewriter,
                              Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto cu_id = builder.newOperand("=s");
  auto hwreg = builder.newConstantOperand("hwreg(HW_REG_HW_ID, HW_ID_CU_ID_OFFSET, HW_ID_CU_ID_SIZE)");
  gethwid(cu_id, hwreg);
  return builder.launch(rewriter, loc, i32_ty, false);
}

static Value getSEID(ConversionPatternRewriter &rewriter,
                              Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto se_id = builder.newOperand("=s");
  auto hwreg = builder.newConstantOperand("hwreg(HW_REG_HW_ID, HW_ID_SE_ID_OFFSET, HW_ID_SE_ID_SIZE)");
  gethwid(se_id, hwreg);
  return builder.launch(rewriter, loc, i32_ty, false);
}


Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  GCNBuilder builder; 
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto &gethwid = *builder.create("s_getreg_b32");

  Value xcc_id;
  llvm::AMDGPU::GPUKind GPUKind = llvm::AMDGPU::parseArchAMDGCN(this->arch);
  switch (GPUKind) {
  case llvm::AMDGPU::GK_GFX90A:
	   xcc_id = b.i32_val(0);
  case llvm::AMDGPU::GK_GFX942:
  case llvm::AMDGPU::GK_GFX950:
	  xcc_id = getXCCID(rewriter, loc);
  default:
     llvm::report_fatal_error("unsupported arch");
  }
  Value cu_id = getCUID(rewriter, loc); //local CU ID
  Value se_id = getSEID(rewriter, loc); 
  builder.create<>("s_waitcnt lgkmcnt(0)")->operator()();

  //global_cu_id = xcc_id * CU_PER_XCD + se_id * CU_PER_SE + cu_id 
  if(GPUKind==llvm::AMDGPU::GK_GFX942)
	  cu_id = b.add(b.add(b.mul(xcc_id, b.i32_val(CU_PER_XCD)), b.mul(se_id, b.i32_val(CU_PER_SE))), cu_id);

  return cu_id;
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
