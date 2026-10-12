//===- MFMASchedule.h - MFMA <-> memory interleave for CDNA3/4 ------------===//
//
// The opt-in MFMA scheduler, an LLVM-IR pass that interleaves the MFMAs of a
// matrix-core hot loop with the memory operations independent of them and
// pins the order with llvm.amdgcn.sched.barrier. See MFMASchedule.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMASCHEDULE_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMASCHEDULE_H_

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Function;
} // namespace llvm

namespace mlir::triton::AMD {

// Schedule every block of F for the target Arch ("gfx942" or "gfx950").
// Returns true iff a span was scheduled.
bool runMFMASchedulePass(llvm::Function &F, llvm::StringRef Arch);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMASCHEDULE_H_
