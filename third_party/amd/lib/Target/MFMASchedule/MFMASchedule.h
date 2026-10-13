//===- MFMASchedule.h - MFMA <-> memory interleave for gfx950 -------------===//
//
// The opt-in MFMA scheduler, an LLVM-IR pass that interleaves the MFMAs of a
// matrix-core hot loop with the memory operations independent of them and
// pins the order with llvm.amdgcn.sched.barrier. See MFMASchedule.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_AMD_TARGET_MFMASCHEDULE_H
#define TRITON_AMD_TARGET_MFMASCHEDULE_H

namespace llvm {
class Function;
} // namespace llvm

namespace mlir::triton::AMD {

// Schedule every block of F. Returns true iff a span was scheduled.
bool runMFMASchedulePass(llvm::Function &F);

} // namespace mlir::triton::AMD

#endif // TRITON_AMD_TARGET_MFMASCHEDULE_H
