//===- MFMAVALUSchedule.h - MFMA <-> VALU co-execution for gfx950 ---------===//
//
// The LLVM-IR pass behind warp_pipeline_stage(schedule_hint="mfma-valu"):
// it declares, with llvm.amdgcn.sched.group.barrier, how the MFMAs of a
// marked stage interleave with the stage's VALU and transcendental work.
// See MFMAVALUSchedule.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMAVALUSCHEDULE_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMAVALUSCHEDULE_H_

namespace llvm {
class Function;
} // namespace llvm

namespace mlir::triton::AMD {

// Declare the co-execution schedule of every stage of F that
// ConvertWarpPipeline marked for it, and remove the markers. Returns true iff
// a marker was found.
bool runMFMAVALUSchedulePass(llvm::Function &F);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMAVALUSCHEDULE_H_
