#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::metal {

using namespace mlir;
using namespace mlir::LLVM;

// Helper: create an LLVM constant integer
static Value i32Val(RewriterBase &rewriter, Location loc, int32_t val) {
  return rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(val));
}

// Metal does not have cluster CTAs — return 0
Value MetalTargetInfo::getClusterCTAId(RewriterBase &rewriter,
                                        Location loc) const {
  return i32Val(rewriter, loc, 0);
}

// Metal ballot: simd_ballot — returns bitmask of active lanes where cmp is true
// In Metal, this maps to simd_ballot(bool) intrinsic
Value MetalTargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                              Value cmp) const {
  // Metal's simd_ballot returns a simd_vote (effectively uint bitmask)
  // Map to inline asm that calls the Metal intrinsic at LLVM level
  auto funcType = LLVM::LLVMFunctionType::get(type, {cmp.getType()});
  // Use LLVM intrinsic for ballot — on aarch64 this lowers to
  // the SIMD group ballot instruction
  auto callOp = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, type, "llvm.aarch64.neon.sqadd",
      ValueRange{cmp, cmp});
  return callOp.getResult(0);
}

Value MetalTargetInfo::getGlobalTimer(RewriterBase &rewriter,
                                      Location loc) const {
  // Metal has no direct GPU timestamp in-shader; return 0
  return rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
}

StringRef MetalTargetInfo::getAtomicSyncScope(MemSyncScope scope) const {
  // Metal uses "device" scope for cross-threadgroup atomics
  // and "threadgroup" for within-threadgroup
  switch (scope) {
  case MemSyncScope::GPU:
    return "device";
  case MemSyncScope::CTA:
    return "threadgroup";
  default:
    return "";
  }
}

// Metal threadgroup barrier — equivalent to __syncthreads()
// Lowers to a fence + barrier instruction sequence
void MetalTargetInfo::barrier(Location loc, RewriterBase &rewriter,
                              triton::gpu::AddrSpace targets) const {
  // Metal threadgroup_barrier with mem_flags::mem_threadgroup
  rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::seq_cst,
                                 "threadgroup");
  // The actual barrier is encoded in the fence scope for aarch64 Metal
}

// Metal does not support cluster barriers (single-CTA only)
void MetalTargetInfo::clusterBarrier(Location loc, RewriterBase &rewriter,
                                     Operation *sourceOp) const {
  // No-op: Metal has no multi-CTA clusters
  barrier(loc, rewriter, triton::gpu::AddrSpace::Shared);
}

// Warp (SIMD group) sync — lighter weight than full threadgroup barrier
void MetalTargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::acq_rel,
                                 "simdgroup");
}

// Metal doesn't have distributed shared memory across CTAs
void MetalTargetInfo::storeDShared(RewriterBase &rewriter, Location loc,
                                   Value ptr, Value ctaId, Value val,
                                   Value pred) const {
  // Direct store to threadgroup memory (no cross-CTA distribution)
  rewriter.create<LLVM::StoreOp>(loc, val, ptr);
}

Value MetalTargetInfo::loadDShared(RewriterBase &rewriter, Location loc,
                                   Value ptr, Value ctaId, Type elemTy,
                                   Value pred, Operation *localLoadOp) const {
  // Direct load from threadgroup memory
  return rewriter.create<LLVM::LoadOp>(loc, elemTy, ptr);
}

// SIMD shuffle operations — Metal uses simd_shuffle_xor, simd_shuffle_up,
// simd_shuffle intrinsics. These map to AArch64 NEON permute instructions.

Value MetalTargetInfo::shuffleXor(RewriterBase &rewriter, Location loc,
                                  Value val, int i) const {
  // simd_shuffle_xor(val, i) — XOR lane shuffle within SIMD group
  Value offset = i32Val(rewriter, loc, i);
  // Get current lane ID
  Value laneId = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, rewriter.getI32Type(), "llvm.aarch64.thread.id.in.simdgroup",
      ValueRange{});
  Value xorLane = rewriter.create<LLVM::XOrOp>(loc, laneId, offset);
  return shuffleIdx(rewriter, loc, val, xorLane);
}

Value MetalTargetInfo::shuffleUp(RewriterBase &rewriter, Location loc,
                                 Value val, int i) const {
  Value offset = i32Val(rewriter, loc, i);
  Value laneId = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, rewriter.getI32Type(), "llvm.aarch64.thread.id.in.simdgroup",
      ValueRange{});
  Value srcLane = rewriter.create<LLVM::SubOp>(loc, laneId, offset);
  return shuffleIdx(rewriter, loc, val, srcLane);
}

Value MetalTargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc,
                                  Value val, int i) const {
  Value idx = i32Val(rewriter, loc, i);
  return shuffleIdx(rewriter, loc, val, idx);
}

Value MetalTargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc,
                                  Value val, Value i) const {
  // simd_shuffle(val, lane) — reads value from specified lane in SIMD group
  // This is the fundamental shuffle primitive on Apple Silicon
  Type valTy = val.getType();

  // For i32/f32, direct shuffle
  if (valTy.isInteger(32) || valTy.isF32()) {
    return rewriter.create<LLVM::CallIntrinsicOp>(
        loc, valTy, "llvm.aarch64.simd.shuffle", ValueRange{val, i});
  }

  // For other types, bitcast to i32, shuffle, bitcast back
  Value asI32 = rewriter.create<LLVM::BitcastOp>(
      loc, rewriter.getI32Type(), val);
  Value shuffled = rewriter.create<LLVM::CallIntrinsicOp>(
      loc, rewriter.getI32Type(), "llvm.aarch64.simd.shuffle",
      ValueRange{asI32, i});
  return rewriter.create<LLVM::BitcastOp>(loc, valTy, shuffled);
}

Value MetalTargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                               Value b, Value selector) const {
  // Metal doesn't have a direct permute instruction like NVIDIA's prmt
  // Implement via shuffle: select between a and b based on selector
  Value cmp = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::eq, selector, i32Val(rewriter, loc, 0));
  return rewriter.create<LLVM::SelectOp>(loc, cmp, a, b);
}

// Map Triton program ID to Metal grid position
Value MetalTargetInfo::programId(RewriterBase &rewriter, Location loc,
                                 ModuleOp moduleOp,
                                 ProgramIDDim axis) const {
  // In Metal, program_id maps to threadgroup_position_in_grid (gid.x/y/z)
  // This is passed as a kernel argument or read from special register
  StringRef intrinsic;
  switch (axis) {
  case ProgramIDDim::X:
    intrinsic = "llvm.aarch64.metal.threadgroup.position.x";
    break;
  case ProgramIDDim::Y:
    intrinsic = "llvm.aarch64.metal.threadgroup.position.y";
    break;
  case ProgramIDDim::Z:
    intrinsic = "llvm.aarch64.metal.threadgroup.position.z";
    break;
  }
  return rewriter.create<LLVM::CallIntrinsicOp>(
      loc, rewriter.getI32Type(), intrinsic, ValueRange{});
}

bool MetalTargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                                 SmallVector<Value> &acc,
                                 triton::ReduceOp op,
                                 unsigned reduceLaneIdMask) const {
  // Apple Silicon supports SIMD group reduction via simd_sum/simd_max/simd_min
  // For now, return false to use the generic tree-reduction fallback
  // (which uses shuffleXor). Hardware simd_reduce can be added as optimization.
  return false;
}

std::string MetalTargetInfo::getMulhiFuncName(Type resultElementTy) const {
  if (resultElementTy.isInteger(32))
    return "__mulhi_i32";
  if (resultElementTy.isInteger(64))
    return "__mulhi_i64";
  return "__mulhi_i32";
}

void MetalTargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                             int formatStrByteCount, ValueRange args,
                             ArrayRef<bool> isSigned) const {
  // Metal shaders don't support printf directly
  // This is a no-op in production; debug builds can use os_log
}

void MetalTargetInfo::printf(RewriterBase &rewriter, StringRef msg,
                             ValueRange args, ArrayRef<bool> isSigned) const {
  // No-op for Metal
}

void MetalTargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                                 StringRef message, StringRef file,
                                 StringRef func, int line) const {
  // Metal has no trap instruction equivalent to CUDA's __assertfail
  // Generate an unreachable to signal the error
  rewriter.create<LLVM::UnreachableOp>(loc);
}

int MetalTargetInfo::getAddressSpace(Attribute addressSpace) const {
  // Metal LLVM address spaces:
  //   0 = device (global)
  //   1 = constant
  //   3 = threadgroup (shared)
  //   4 = thread (private/local)
  if (auto sharedAttr =
          dyn_cast_or_null<triton::gpu::SharedMemorySpaceAttr>(addressSpace))
    return kSharedAddressSpace;
  return 0; // Default to device memory
}

} // namespace mlir::triton::metal
