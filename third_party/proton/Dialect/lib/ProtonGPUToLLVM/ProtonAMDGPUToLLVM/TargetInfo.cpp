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
  // llvm.readcyclecounter lowers to the correct per-target counter:
  //   - gfx942/gfx950: s_memtime
  //   - gfx1250: s_get_shader_cycles_u64. gfx1250 removed s_memtime and
  //     moved the (now 64-bit) shader-cycles counter out of the s_getreg
  //     table; hwreg(HW_REG_SHADER_CYCLES_*) still assembles but reads 0 at
  //     runtime.
  //
  // Result is i64. Truncating to i32 for the isClock64=false case is fine:
  // after 0x0000.0000.ffff.ffff comes 0x0000.0001.0000.0000, so truncation
  // effectively wraps from 0xffff.ffff to 0x0000.0000.
  Value clockVal = LLVM::createLLVMIntrinsicCallOp(
                       rewriter, loc, "llvm.readcyclecounter", i64_ty, {})
                       .getResult(0);
  if (!isClock64)
    clockVal = LLVM::TruncOp::create(rewriter, loc, i32_ty, clockVal);

  return clockVal;
}

Value TargetInfo::globalTime(ConversionPatternRewriter &rewriter,
                             Location loc) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  llvm::AMDGPU::GPUKind GPUKind = getTritonTargetInfo().getGPUKind();

  Value globalTimeVal;
  if (GPUKind == llvm::AMDGPU::GK_GFX1250) {
    Value msg = b.i32_val(/*MSG_RTN_GET_REALTIME=*/131);
    globalTimeVal =
        LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, "llvm.amdgcn.s.sendmsg.rtn.i64", i64_ty, {msg})
            .getResult(0);
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

// The helpers below read the CDNA (gfx940-942, gfx950) wave-placement fields
// using the symbolic HW_REG_* names, which LLVM's AMDGPU AsmParser accepts on
// those targets and which document exactly which register is read.
//
// IMPORTANT: these registers do NOT exist under the same ids on gfx1250.
// The equivalent numeric ids decode to unrelated registers
// (hwreg(20,...) -> WAVE_SCRATCH_BASE_LO, hwreg(4,...) -> WAVE_STATE_PRIV), so
// they would read garbage. gfx1250 wave placement is read separately in
// processorId(); see below.
//
// https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/include/hip/amd_detail/amd_device_functions.h#L898
// XCC_ID Register bit structure for gfx940-942, gfx950.
// XCC_ID      3:0     XCC the wave is assigned to.
static Value getXCCID(ConversionPatternRewriter &rewriter, Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto xcc_id = builder.newOperand("=s");
  auto xcc_reg = builder.newConstantOperand("hwreg(HW_REG_XCC_ID, 0, 4)");
  gethwid(xcc_id, xcc_reg);
  return builder.launch(rewriter, loc, i32_ty, false);
}

// HW_ID Register bit structure for GCN and CDNA (reg 4).
// CU_ID       11:8    Compute Unit the wave is assigned to.
static Value getCUID(ConversionPatternRewriter &rewriter, Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto cu_id = builder.newOperand("=s");
  auto hwreg = builder.newConstantOperand("hwreg(HW_REG_HW_ID, 8, 4)");
  gethwid(cu_id, hwreg);
  return builder.launch(rewriter, loc, i32_ty, false);
}
// SE_ID       15:13   Shader Engine the wave is assigned to for gfx940-942,
// gfx950.
static Value getSEID(ConversionPatternRewriter &rewriter, Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto se_id = builder.newOperand("=s");
  auto hwreg = builder.newConstantOperand("hwreg(HW_REG_HW_ID, 13, 3)");
  gethwid(se_id, hwreg);
  return builder.launch(rewriter, loc, i32_ty, false);
}

// gfx1250 WAVE_HW_ID1 (hwreg id 23) holds the wave's WGP/SA/SIMD placement.
// Read the complete register once so all fields come from the same snapshot.
static Value getWaveHwId1(ConversionPatternRewriter &rewriter, Location loc) {
  GCNBuilder builder;
  auto &gethwid = *builder.create("s_getreg_b32");
  auto dst = builder.newOperand("=s");
  auto reg = builder.newConstantOperand("hwreg(HW_REG_WAVE_HW_ID1)");
  gethwid(dst, reg);
  return builder.launch(rewriter, loc, i32_ty, false);
}

// gfx1250 uses a separate packing in processorId().
static uint32_t getCU_PER_XCD(llvm::AMDGPU::GPUKind GPUKind) {
  switch (GPUKind) {
  case llvm::AMDGPU::GK_GFX942:
    return 38;
  case llvm::AMDGPU::GK_GFX950:
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
  default:
    llvm_unreachable("unsupported arch");
  }
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  llvm::AMDGPU::GPUKind GPUKind = getTritonTargetInfo().getGPUKind();

  // gfx1250 remapped the hwreg id space, so the CDNA reads
  // used below (hwreg 20 / hwreg 4) no longer name XCC_ID / HW_ID and every
  // wave would read the same constant, collapsing all traces onto one core.
  // Read the gfx1250 sources instead: SE_ID and XCC/AID_ID come from
  // s_sendmsg_rtn(MSG_RTN_GET_SE_AID_ID) (bits [3:0] and [19:16]); WGP_ID and
  // SA_ID come from WAVE_HW_ID1 (hwreg id 23). Together these four fields
  // uniquely identify a WGP. Pack the full documented field widths to avoid
  // relying on a particular number of XCCs or SEs.
  //
  // Granularity is per-WGP, not per-CU: the CU-within-WGP bit
  // (HW_ID1.SIMD_ID[0]) is intentionally omitted; add it if sub-WGP resolution
  // is ever needed.
  if (GPUKind == llvm::AMDGPU::GK_GFX1250) {
    Value msg = b.i32_val(/*MSG_RTN_GET_SE_AID_ID=*/135);
    Value seAid =
        LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, "llvm.amdgcn.s.sendmsg.rtn.i32", i32_ty, {msg})
            .getResult(0);
    Value se = b.and_(seAid, b.i32_val(0xF)); // SE_ID [3:0]
    Value xcc = b.and_(b.lshr(seAid, b.i32_val(16)),
                       b.i32_val(0xF)); // XCC/AID_ID [19:16]

    Value hwId1 = getWaveHwId1(rewriter, loc);
    Value wgp = b.and_(b.lshr(hwId1, b.i32_val(10)),
                       b.i32_val(0xF)); // WGP_ID [13:10]
    Value sa = b.and_(b.lshr(hwId1, b.i32_val(16)),
                      b.i32_val(0x1)); // SA_ID [16]

    // Collision-free packing:
    // xcc[12:9] | se[8:5] | sa[4] | wgp[3:0].
    Value id =
        b.or_(b.or_(b.or_(b.shl(xcc, b.i32_val(9)), b.shl(se, b.i32_val(5))),
                    b.shl(sa, b.i32_val(4))),
              wgp);
    return id;
  }

  GCNBuilder builder;
  Value xcc_id = b.i32_val(0);
  // Multi-XCD CDNA-family targets: gfx942, gfx950.
  switch (GPUKind) {
  case llvm::AMDGPU::GK_GFX942:
  case llvm::AMDGPU::GK_GFX950:
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
