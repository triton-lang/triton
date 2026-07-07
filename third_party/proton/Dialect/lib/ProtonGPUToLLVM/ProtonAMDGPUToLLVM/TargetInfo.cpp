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
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  llvm::AMDGPU::GPUKind GPUKind = llvm::AMDGPU::parseArchAMDGCN(this->arch);

  // gfx12 (incl. gfx1250) removed s_memtime and its llvm.amdgcn.s.memtime
  // intrinsic. On MI400 the shader-cycles counter was widened from 20 to
  // 64 bits and *moved out of the HW_REGISTERS[] array that backs
  // s_getreg_b32*; per the MI400 Shader Programming Guide §1.1.1.1:
  //   "SHADER_CYCLES is now 64 bits (was 20), and can only be read with
  //    S_GET_SHADER_CYCLES, not S_GETREG."
  // §3.4.13 "Time" confirms: the method to read the cycle counter is the
  // dedicated SOP1 instruction S_GET_SHADER_CYCLES_U64, which returns the
  // full 64-bit counter in a scalar-register pair. Trying to read
  // hwreg(HW_REG_SHADER_CYCLES_LO/HI) on gfx1250 assembles (LLVM's
  // subtarget table still accepts the legacy ids 29/30) but reads 0 at
  // runtime because the register is no longer wired to that path.
  //
  // Emit the new instruction via inline asm. Result is i64; truncate to
  // i32 for the isClock64=false case.
  if (GPUKind == llvm::AMDGPU::GK_GFX1250) {
    GCNBuilder builder;
    auto &getCycles = *builder.create("s_get_shader_cycles_u64");
    auto out = builder.newOperand("=s");
    getCycles(out);
    Value cy64 = builder.launch(rewriter, loc, i64_ty, /*hasSideEffect=*/true);
    if (isClock64)
      return cy64;
    return LLVM::TruncOp::create(rewriter, loc, i32_ty, cy64);
  }

  // NV has both a 32 bit and 64 bit clock intrinsic. On AMD we only have
  // s_memtime which is 64 bit. However truncating the 64 bit version
  // in cases of requesting 32 bit should be fine, since in 64 bits,
  // after 0x0000.0000.ffff.ffff comes 0x0000.0001.0000.0000, and
  // truncating that to 32 bits gives zero, effectively wrapping from
  // 0xffff.ffff to 0x0000.0000.
  StringRef clock64IntrinsicName = "llvm.amdgcn.s.memtime";
  Value clockVal = LLVM::createLLVMIntrinsicCallOp(
                       rewriter, loc, clock64IntrinsicName, i64_ty, {})
                       .getResult(0);
  if (!isClock64)
    clockVal = LLVM::TruncOp::create(rewriter, loc, i32_ty, clockVal);

  return clockVal;
}

Value TargetInfo::globalTime(ConversionPatternRewriter &rewriter,
                             Location loc) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  llvm::AMDGPU::GPUKind GPUKind = llvm::AMDGPU::parseArchAMDGCN(this->arch);

  Value globalTimeVal;
  // gfx12 (incl. gfx1250) removed the s_memrealtime instruction and its
  // matching llvm.amdgcn.s.memrealtime intrinsic; LLVM's AMDGPU ISel
  // aborts with "Cannot select: intrinsic %llvm.amdgcn.s.memrealtime" when
  // compiling for gfx1250. Use the gfx11/12 replacement sequence instead:
  //   s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)
  //   s_wait_kmcnt 0
  // Mirrors triton's own language/hip/utils.py::memrealtime() gfx12 path.
  if (GPUKind == llvm::AMDGPU::GK_GFX1250) {
    GCNBuilder builder;
    auto &sendmsg = *builder.create("s_sendmsg_rtn_b64");
    auto out = builder.newOperand("=s");
    auto msg = builder.newConstantOperand("sendmsg(MSG_RTN_GET_REALTIME)");
    sendmsg(out, msg);
    builder.create<>("s_wait_kmcnt 0")->operator()();
    globalTimeVal =
        builder.launch(rewriter, loc, i64_ty, /*hasSideEffect=*/true);
  } else {
    StringRef globalTimeIntrinsicName = "llvm.amdgcn.s.memrealtime";
    globalTimeVal = LLVM::createLLVMIntrinsicCallOp(
                        rewriter, loc, globalTimeIntrinsicName, i64_ty, {})
                        .getResult(0);
  }
  // The clock-generator runs at 100 MHz ==> 10 ns per clock.
  // Reference: Section 3.4.11 in the RDNA4 ISA manual
  // https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf
  return b.mul(globalTimeVal, b.i64_val(10));
}

// https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/include/hip/amd_detail/amd_device_functions.h#L898
// XCC_ID Register bit structure for gfx940-942, gfx950, gfx1250.
// XCC_ID      3:0     XCC the wave is assigned to.
// Use numeric hwreg id (HW_REG_XCC_ID == 20) rather than the symbolic name,
// because LLVM's AMDGPU AsmParser does not recognize the `HW_REG_XCC_ID`
// symbol on every target (notably gfx1250) and will abort when parsing.
static Value getXCCID(ConversionPatternRewriter &rewriter, Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto xcc_id = builder.newOperand("=s");
  auto xcc_reg = builder.newConstantOperand("hwreg(20, 0, 4)");
  gethwid(xcc_id, xcc_reg);
  return builder.launch(rewriter, loc, i32_ty, false);
}

// HW_ID Register bit structure for GCN and CDNA (reg 4).
// CU_ID       11:8    Compute Unit the wave is assigned to.
// Same numeric-id rationale as getXCCID above; gfx1250 LLVM does not accept
// the symbolic `HW_REG_HW_ID` form in inline asm and aborts during make_amdgcn.
static Value getCUID(ConversionPatternRewriter &rewriter, Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto cu_id = builder.newOperand("=s");
  auto hwreg = builder.newConstantOperand("hwreg(4, 8, 4)");
  gethwid(cu_id, hwreg);
  return builder.launch(rewriter, loc, i32_ty, false);
}
// SE_ID       15:13   Shader Engine the wave is assigned to for gfx940-942,
// gfx950, gfx1250.
static Value getSEID(ConversionPatternRewriter &rewriter, Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto se_id = builder.newOperand("=s");
  auto hwreg = builder.newConstantOperand("hwreg(4, 13, 3)");
  gethwid(se_id, hwreg);
  return builder.launch(rewriter, loc, i32_ty, false);
}

// gfx942 has 8 XCDs, each XCD contains 40 CUs per XCD but only 38/40 are active
// (total of 304 CUs). gfx950 has 8 XCDs, each XCD contains 36 CUs per XCD but
// only 32/36 active CUs (total 256 CUs). gfx1250 has 8 XCDs with 32 active CUs
// per XCD (total 256 CUs, all active) — verified via rocminfo: 256 CUs, 16
// SEs (2 per XCD), 2 SAs/SE, 4 SIMDs/CU, Wave32, LDS 320 KB.
static uint32_t getCU_PER_XCD(llvm::AMDGPU::GPUKind GPUKind) {
  switch (GPUKind) {
  case llvm::AMDGPU::GK_GFX942:
    return 38;
  case llvm::AMDGPU::GK_GFX950:
    return 32;
  case llvm::AMDGPU::GK_GFX1250:
    return 32;
  default:
    llvm_unreachable("unsupported arch for proton.");
  }
}

static uint32_t getCU_PER_SE(llvm::AMDGPU::GPUKind GPUKind) {
  switch (GPUKind) {
  case llvm::AMDGPU::GK_GFX942:
    return 10;
  case llvm::AMDGPU::GK_GFX950:
    return 10;
  // gfx1250: 256 active CUs / 16 SEs = 16 CUs per SE.
  case llvm::AMDGPU::GK_GFX1250:
    return 16;
  default:
    llvm_unreachable("unsupported arch");
  }
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  GCNBuilder builder;
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Value xcc_id = b.i32_val(0);
  llvm::AMDGPU::GPUKind GPUKind = llvm::AMDGPU::parseArchAMDGCN(this->arch);
  // Multi-XCD CDNA-family targets: gfx942, gfx950, gfx1250.
  switch (GPUKind) {
  case llvm::AMDGPU::GK_GFX942:
  case llvm::AMDGPU::GK_GFX950:
  case llvm::AMDGPU::GK_GFX1250:
    xcc_id = getXCCID(rewriter, loc);
    break;
  default:
    llvm::report_fatal_error("unsupported arch");
  }

  Value cu_id = getCUID(rewriter, loc); // local CU ID
  Value se_id = getSEID(rewriter, loc);
  builder.create<>("s_waitcnt lgkmcnt(0)")->operator()();

  // For XCC based architectures to get a unique CU id for a wave:
  // global_cu_id = xcc_id * CU_PER_XCD + se_id * CU_PER_SE + cu_id (local)
  if (GPUKind == llvm::AMDGPU::GK_GFX942 ||
      GPUKind == llvm::AMDGPU::GK_GFX950 ||
      GPUKind == llvm::AMDGPU::GK_GFX1250) {
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
