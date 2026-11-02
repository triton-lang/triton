//===- MFMAScheduleUtils.h - Helpers shared by the MFMA schedulers --------===//
//
// Instruction classification and the MFMA cycle table used by both LLVM-IR
// scheduling passes of this library: the MFMA <-> memory interleave
// (MFMASchedule.cpp) and the MFMA <-> VALU co-execution declaration
// (MFMAVALUSchedule.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMASCHEDULEUTILS_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMASCHEDULEUTILS_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

namespace mlir::triton::AMD::MFMAScheduleUtils {

// How an instruction takes part in a schedule: an MFMA, a global-memory
// access (GR), an LDS read (LR), an LDS write (LW), or nothing the passes
// move.
enum class SchedKind { MFMA, GR, LR, LW, Other };

constexpr unsigned kLDSAddressSpace = 3;
constexpr unsigned kGlobalAddressSpace = 1;

inline bool isMFMAorWMMA(const llvm::Instruction &I) {
  const auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
  if (!CI || CI->isInlineAsm())
    return false;
  const llvm::Function *Callee = CI->getCalledFunction();
  if (!Callee || !Callee->isIntrinsic())
    return false;
  llvm::StringRef Name = Callee->getName();
  return Name.contains("mfma") || Name.contains("wmma");
}

inline bool isSchedBarrier(const llvm::Instruction &I) {
  if (const auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
    if (const llvm::Function *F = CI->getCalledFunction())
      return F->getIntrinsicID() == llvm::Intrinsic::amdgcn_sched_barrier;
  return false;
}

inline bool hasSchedBarrier(const llvm::Function &F) {
  for (const llvm::BasicBlock &BB : F)
    for (const llvm::Instruction &I : BB)
      if (isSchedBarrier(I))
        return true;
  return false;
}

inline SchedKind classifySchedInst(const llvm::Instruction &I) {
  if (isMFMAorWMMA(I))
    return SchedKind::MFMA;

  if (const auto *CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
    if (const llvm::Function *F = CI->getCalledFunction()) {
      if (F->isIntrinsic()) {
        llvm::StringRef Name = F->getName();
        // Buffer loads (to registers or to LDS), buffer stores, and the
        // global.load.lds family.
        if (Name.contains("buffer.load") ||
            Name.contains("raw.ptr.buffer.store") ||
            Name.contains("global.load") || Name.contains("global.store"))
          return SchedKind::GR;
        if (Name.contains("ds.read") || Name.contains("ds.load"))
          return SchedKind::LR;
      }
    }
  }

  // Plain loads and stores: LDS by address space 3, global by address space 1
  // (what a kernel gets when buffer ops do not apply to its pointers).
  if (const auto *LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
    if (LI->getPointerAddressSpace() == kLDSAddressSpace)
      return SchedKind::LR;
    if (LI->getPointerAddressSpace() == kGlobalAddressSpace)
      return SchedKind::GR;
  }
  if (const auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I)) {
    if (SI->getPointerAddressSpace() == kLDSAddressSpace)
      return SchedKind::LW;
    if (SI->getPointerAddressSpace() == kGlobalAddressSpace)
      return SchedKind::GR;
  }
  return SchedKind::Other;
}

inline bool isMemoryAnchor(SchedKind K) {
  return K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW;
}

// Issue-to-issue cost of an MFMA: its pass count times 4 cycles
// (SISchedule.td: 16x16xK shapes are 4 passes, 32x32xK shapes 8, whatever the
// element type). Returns 0 for a shape the table does not know; the callers
// then leave the span to LLVM's schedulers.
inline unsigned getMFMACycles(const llvm::Instruction &I) {
  if (!isMFMAorWMMA(I))
    return 0;
  const auto *CI = llvm::cast<llvm::CallInst>(&I);
  const llvm::Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return 0;
  llvm::StringRef Name = Callee->getName();

  // Scaled f8f6f4 MFMAs: the cost depends on the operand formats encoded in
  // cbsz (arg 3) and blgp (arg 4); a value above 1 is a 4- or 6-bit format,
  // which the unit runs at the 4-bit rate.
  auto scaledCycles = [&](unsigned Narrow, unsigned Wide) {
    if (auto *Cbsz = llvm::dyn_cast<llvm::ConstantInt>(CI->getArgOperand(3)))
      if (auto *Blgp = llvm::dyn_cast<llvm::ConstantInt>(CI->getArgOperand(4)))
        return (Cbsz->getZExtValue() > 1 && Blgp->getZExtValue() > 1) ? Narrow
                                                                      : Wide;
    return Wide;
  };
  if (Name.contains("mfma.scale.f32.16x16x128.f8f6f4"))
    return scaledCycles(16, 32);
  if (Name.contains("mfma.scale.f32.32x32x64.f8f6f4"))
    return scaledCycles(32, 64);

  static constexpr struct {
    llvm::StringRef Name;
    unsigned Cycles;
  } kFixedCycles[] = {
      {"mfma.f32.16x16x32.f16", 16},  {"mfma.f32.16x16x32.bf16", 16},
      {"mfma.i32.16x16x64.i8", 16},   {"mfma.f32.32x32x16.f16", 32},
      {"mfma.f32.32x32x16.bf16", 32}, {"mfma.i32.32x32x32.i8", 32},
      {"mfma.f32.16x16x32.fp8", 16},  {"mfma.f32.16x16x32.bf8", 16},
      {"mfma.f32.32x32x16.fp8", 32},  {"mfma.f32.32x32x16.bf8", 32},
      {"mfma.f32.16x16x16f16", 16},   {"mfma.f32.16x16x16bf16.1k", 16},
      {"mfma.f32.32x32x8f16", 32},    {"mfma.f32.32x32x8bf16.1k", 32},
  };
  for (const auto &Entry : kFixedCycles)
    if (Name.contains(Entry.Name))
      return Entry.Cycles;
  return 0;
}

} // namespace mlir::triton::AMD::MFMAScheduleUtils

#endif // TRITON_THIRD_PARTY_AMD_LIB_TARGET_MFMASCHEDULE_MFMASCHEDULEUTILS_H_
