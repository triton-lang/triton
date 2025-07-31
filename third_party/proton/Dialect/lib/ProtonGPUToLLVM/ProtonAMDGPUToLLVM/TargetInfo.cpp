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
  Value clockVal = LLVM::createLLVMIntrinsicCallOp(
                       rewriter, loc, clock64IntrinsicName, i64_ty, {})
                       .getResult(0);
  if (!isClock64)
    clockVal = rewriter.create<LLVM::TruncOp>(loc, i32_ty, clockVal);

  return clockVal;
}

// TODO(crobeck): move these into a util file
static Value getXCCID(ConversionPatternRewriter &rewriter, Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto xcc_id = builder.newOperand("=s");
  // 0=HW_REG_XCC_ID_OFFSET, 3=HW_REG_XCC_ID_SIZE
  auto xcc_reg = builder.newConstantOperand("hwreg(HW_REG_XCC_ID, 0, 3)");
  gethwid(xcc_id, xcc_reg);
  return builder.launch(rewriter, loc, i32_ty, false);
}

static Value getCUID(ConversionPatternRewriter &rewriter, Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto cu_id = builder.newOperand("=s");
  // 8=HW_ID_CU_ID_OFFSET, 4=HW_ID_CU_ID_SIZE
  auto hwreg = builder.newConstantOperand("hwreg(HW_REG_HW_ID, 8, 4)");
  gethwid(cu_id, hwreg);
  return builder.launch(rewriter, loc, i32_ty, false);
}

static Value getSEID(ConversionPatternRewriter &rewriter, Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto se_id = builder.newOperand("=s");
  // 13=HW_ID_SE_ID_OFFSET, 3=HW_ID_SE_ID_SIZE
  auto hwreg = builder.newConstantOperand("hwreg(HW_REG_HW_ID, 13, 3)");
  gethwid(se_id, hwreg);
  return builder.launch(rewriter, loc, i32_ty, false);
}

static uint32_t getCU_PER_XCD(llvm::AMDGPU::GPUKind GPUKind) {
  switch (GPUKind) {
  case llvm::AMDGPU::GK_GFX942:
    return 40;
  case llvm::AMDGPU::GK_GFX950:
    return 32;
  default:
    llvm_unreachable("unsupported arch");
  }
}

static uint32_t getCU_PER_SE(llvm::AMDGPU::GPUKind GPUKind) {
  switch (GPUKind) {
  case llvm::AMDGPU::GK_GFX942:
    return 10;
  case llvm::AMDGPU::GK_GFX950:
    return 10;
  default:
    llvm_unreachable("unsupported arch");
  }
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  GCNBuilder builder;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto &gethwid = *builder.create("s_getreg_b32");

  Value xcc_id = b.i32_val(0);
  llvm::AMDGPU::GPUKind GPUKind = llvm::AMDGPU::parseArchAMDGCN(this->arch);
  // For now only support gfx90a, gfx942, and gfx950
  switch (GPUKind) {
  case llvm::AMDGPU::GK_GFX90A:
    break;
  case llvm::AMDGPU::GK_GFX942:
  case llvm::AMDGPU::GK_GFX950:
    xcc_id = getXCCID(rewriter, loc);
    break;
  default:
    llvm::report_fatal_error("unsupported arch");
  }
  // on gfx90a the local cu_id == global cu_id
  Value cu_id = getCUID(rewriter, loc); // local CU ID
  Value se_id = getSEID(rewriter, loc);
  builder.create<>("s_waitcnt lgkmcnt(0)")->operator()();

  // For XCC based architectures to get a unique CU id for a wave:
  // global_cu_id = xcc_id * CU_PER_XCD + se_id * CU_PER_SE + cu_id (local)
  if (GPUKind == llvm::AMDGPU::GK_GFX942 ||
      GPUKind == llvm::AMDGPU::GK_GFX950) {
    uint32_t CU_PER_XCD = getCU_PER_XCD(GPUKind);
    uint32_t CU_PER_SE = getCU_PER_SE(GPUKind);
    cu_id = b.add(b.add(b.mul(xcc_id, b.i32_val(CU_PER_XCD)),
                        b.mul(se_id, b.i32_val(CU_PER_SE))),
                  cu_id);
  }

  return cu_id;
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
  // Internal buffer index is private to each thread, we use thread local
  // address space for AMD GPUs. See detail discussion:
  // https://llvm.org/docs/AMDGPUUsage.html#address-spaces
  return 5;
}

} // namespace mlir::triton::proton::gpu::AMD
