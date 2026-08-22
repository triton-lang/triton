//===-- AMDGPUInstrUtils.h - AMDGPU instruction classification utilities --===//
//
// Minimal helpers for classifying AMDGPU instructions (barrier / waitcnt / VMEM
// / LDS) used when filtering scheduling-DAG edges. Only the classifiers the DAG
// builder needs are defined here.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_MIR_DAG_AMDGPUINSTRUTILS_H
#define TRITON_MIR_DAG_AMDGPUINSTRUTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

namespace llvm {
namespace mir_dag {

// TODO(tyb0807): these classifiers match on the opcode mnemonic string
// (e.g. `starts_with("DS_WRITE")`). This is deliberately used instead of the
// precise, TSFlags-based predicates in the AMDGPU target (SIInstrInfo::isDS /
// isVMEM / isFLAT, AMDGPU::decodeVmcnt/decodeLgkmcnt, etc.) because those live
// in `llvm/lib/Target/AMDGPU/` and are NOT part of the installed public headers
// -- a Triton user linking a prebuilt/wheel LLVM cannot include them. Only the
// generic `TargetInstrInfo::getName()` (public) is available here, so we match
// on names.
//
// String matching is brittle: it silently breaks if an opcode is renamed. Remove
// this file and switch to the target predicates once a public AMDGPU
// instruction-classification API is exposed upstream (i.e. reachable from
// `include/llvm/`).

/// Get the instruction name (opcode mnemonic) for a MachineInstr.
inline StringRef getInstrName(const MachineInstr &MI) {
  const MachineFunction *MF = MI.getParent()->getParent();
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
  return TII->getName(MI.getOpcode());
}

/// Check if instruction is S_BARRIER (MI300/MI350).
inline bool isBarrier(const MachineInstr &MI) {
  return getInstrName(MI) == "S_BARRIER";
}

/// Check if instruction is S_WAITCNT (not VSCNT variants).
inline bool isWaitcnt(const MachineInstr &MI) {
  StringRef Name = getInstrName(MI);
  return Name.starts_with("S_WAITCNT") && !Name.contains("VSCNT");
}

/// Check if instruction is a VMEM operation (BUFFER_*, GLOBAL_*, FLAT_*).
inline bool isVMEMOp(const MachineInstr &MI) {
  StringRef Name = getInstrName(MI);
  return Name.starts_with("BUFFER_LOAD") || Name.starts_with("BUFFER_STORE") ||
         Name.starts_with("BUFFER_ATOMIC") ||
         Name.starts_with("GLOBAL_LOAD") || Name.starts_with("GLOBAL_STORE") ||
         Name.starts_with("GLOBAL_ATOMIC") ||
         Name.starts_with("FLAT_LOAD") || Name.starts_with("FLAT_STORE") ||
         Name.starts_with("FLAT_ATOMIC");
}

/// Check if instruction is an LDS operation (DS_READ, DS_WRITE, DS atomics).
inline bool isLDSOp(const MachineInstr &MI) {
  StringRef Name = getInstrName(MI);
  return Name.starts_with("DS_READ") || Name.starts_with("DS_WRITE") ||
         Name.starts_with("DS_ADD") || Name.starts_with("DS_SUB") ||
         Name.starts_with("DS_MIN") || Name.starts_with("DS_MAX") ||
         Name.starts_with("DS_AND") || Name.starts_with("DS_OR") ||
         Name.starts_with("DS_XOR") || Name.starts_with("DS_INC") ||
         Name.starts_with("DS_DEC") || Name.starts_with("DS_CMPST") ||
         Name.starts_with("DS_WRAP");
}

} // namespace mir_dag
} // namespace llvm

#endif // TRITON_MIR_DAG_AMDGPUINSTRUTILS_H
