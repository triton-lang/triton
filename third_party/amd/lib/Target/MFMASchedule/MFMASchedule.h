//===- MFMASchedule.h - MFMA <-> memory interleave for gfx950 -------------===//
//
// The opt-in MFMA scheduler, an LLVM-IR pass that interleaves the MFMAs of a
// matrix-core hot loop with the memory operations independent of them and
// pins the order with llvm.amdgcn.sched.barrier. See MFMASchedule.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMASCHEDULE_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMASCHEDULE_H_

namespace llvm {
class Function;
} // namespace llvm

namespace mlir::triton::AMD {

// Schedule every block of F. Returns true iff a span was scheduled.
bool runMFMASchedulePass(llvm::Function &F);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMASCHEDULE_H_
