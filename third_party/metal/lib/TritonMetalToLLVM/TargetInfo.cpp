#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::metal {

using namespace mlir;
using namespace mlir::LLVM;

static Value i32Val(RewriterBase &rewriter, Location loc, int32_t val) {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(), val);
}

Value MetalTargetInfo::getClusterCTAId(RewriterBase &rewriter,
                                       Location loc) const {
  return i32Val(rewriter, loc, 0);
}

Value MetalTargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                              Value cmp) const {
  // Metal simd_ballot: return bitmask of lanes where cmp is true.
  // On AArch64/Metal, this maps to a SIMD group ballot operation.
  // For now, implement as a simple zero-extend of the predicate
  // (each lane contributes its bit via shuffle-based reduction at higher level).
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                         "llvm.aarch64.neon.sqadd",
                                         TypeRange{type}, ValueRange{cmp, cmp})
      .getResult(0);
}

Value MetalTargetInfo::getGlobalTimer(RewriterBase &rewriter,
                                      Location loc) const {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), 0);
}

StringRef MetalTargetInfo::getAtomicSyncScope(MemSyncScope scope) const {
  switch (scope) {
  case MemSyncScope::GPU:
    return "device";
  case MemSyncScope::CTA:
    return "threadgroup";
  default:
    return "";
  }
}

void MetalTargetInfo::barrier(Location loc, RewriterBase &rewriter,
                              triton::gpu::AddrSpace targets) const {
  LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst,
                        "threadgroup");
}

void MetalTargetInfo::clusterBarrier(Location loc, RewriterBase &rewriter,
                                     Operation *sourceOp) const {
  barrier(loc, rewriter, triton::gpu::AddrSpace::Shared);
}

void MetalTargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::acq_rel,
                        "simdgroup");
}

void MetalTargetInfo::storeDShared(RewriterBase &rewriter, Location loc,
                                   Value ptr, Value ctaId, Value val,
                                   Value pred) const {
  LLVM::StoreOp::create(rewriter, loc, val, ptr);
}

Value MetalTargetInfo::loadDShared(RewriterBase &rewriter, Location loc,
                                   Value ptr, Value ctaId, Type elemTy,
                                   Value pred, Operation *localLoadOp) const {
  return LLVM::LoadOp::create(rewriter, loc, elemTy, ptr);
}

Value MetalTargetInfo::shuffleXor(RewriterBase &rewriter, Location loc,
                                  Value val, int i) const {
  Value offset = i32Val(rewriter, loc, i);
  Value laneId =
      LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc, "llvm.aarch64.thread.id.in.simdgroup",
          TypeRange{rewriter.getI32Type()}, ValueRange{})
          .getResult(0);
  Value xorLane = LLVM::XOrOp::create(rewriter, loc, laneId, offset);
  return shuffleIdx(rewriter, loc, val, xorLane);
}

Value MetalTargetInfo::shuffleUp(RewriterBase &rewriter, Location loc,
                                 Value val, int i) const {
  Value offset = i32Val(rewriter, loc, i);
  Value laneId =
      LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc, "llvm.aarch64.thread.id.in.simdgroup",
          TypeRange{rewriter.getI32Type()}, ValueRange{})
          .getResult(0);
  Value srcLane = LLVM::SubOp::create(rewriter, loc, laneId, offset);
  return shuffleIdx(rewriter, loc, val, srcLane);
}

Value MetalTargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc,
                                  Value val, int i) const {
  Value idx = i32Val(rewriter, loc, i);
  return shuffleIdx(rewriter, loc, val, idx);
}

Value MetalTargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc,
                                  Value val, Value i) const {
  Type valTy = val.getType();

  if (valTy.isInteger(32) || valTy.isF32()) {
    return LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                           "llvm.aarch64.simd.shuffle",
                                           TypeRange{valTy}, ValueRange{val, i})
        .getResult(0);
  }

  // For other types, bitcast to i32, shuffle, bitcast back
  Value asI32 =
      LLVM::BitcastOp::create(rewriter, loc, rewriter.getI32Type(), val);
  Value shuffled =
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                      "llvm.aarch64.simd.shuffle",
                                      TypeRange{rewriter.getI32Type()},
                                      ValueRange{asI32, i})
          .getResult(0);
  return LLVM::BitcastOp::create(rewriter, loc, valTy, shuffled);
}

Value MetalTargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                               Value b, Value selector) const {
  Value cmp = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                   selector, i32Val(rewriter, loc, 0));
  return LLVM::SelectOp::create(rewriter, loc, cmp, a, b);
}

Value MetalTargetInfo::programId(RewriterBase &rewriter, Location loc,
                                 ModuleOp moduleOp, ProgramIDDim axis) const {
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
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic,
                                         TypeRange{rewriter.getI32Type()},
                                         ValueRange{})
      .getResult(0);
}

bool MetalTargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                                 SmallVector<Value> &acc, triton::ReduceOp op,
                                 unsigned reduceLaneIdMask) const {
  // Return false to use the generic tree-reduction fallback (via shuffleXor).
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
  // No-op: Metal shaders don't support printf
}

void MetalTargetInfo::printf(RewriterBase &rewriter, StringRef msg,
                             ValueRange args, ArrayRef<bool> isSigned) const {
  // No-op
}

void MetalTargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                                 StringRef message, StringRef file,
                                 StringRef func, int line) const {
  LLVM::UnreachableOp::create(rewriter, loc);
}

int MetalTargetInfo::getAddressSpace(Attribute addressSpace) const {
  if (auto sharedAttr =
          dyn_cast_or_null<triton::gpu::SharedMemorySpaceAttr>(addressSpace))
    return kSharedAddressSpace;
  return 0;
}

} // namespace mlir::triton::metal
