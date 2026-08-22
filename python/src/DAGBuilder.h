//===-- DAGBuilder.h - Build scheduling DAG from MachineFunction ----------===//
//
// Builds a scheduling DAG from a MachineBasicBlock and extracts dependency
// edges, using LLVM's ScheduleDAGInstrs. Used to dump the scheduling DAG
// alongside the MIR.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_MIR_DAG_DAGBUILDER_H
#define TRITON_MIR_DAG_DAGBUILDER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"

#include <memory>

namespace llvm {
class AAResults;
class AssumptionCache;
class BasicAAResult;
class LiveIntervals;
class MachineDominatorTree;
class MachineLoopInfo;
class TargetLibraryInfo;
class TargetLibraryInfoImpl;
class raw_ostream;
} // namespace llvm

namespace llvm {
namespace mir_dag {

/// Represents a dependency edge between two instructions.
struct DAGEdge {
  MachineInstr *Src;  /// Source instruction (must execute first).
  MachineInstr *Dst;  /// Destination instruction (depends on Src).

  /// Edge types we emit.
  ///
  /// Why a separate enum instead of reusing LLVM's SDep::Kind / SDep::OrderKind:
  /// this enum is emitted into the (bb, position)-keyed DAG text, which is what
  /// lets a MIR instruction be mapped back to its DAG SUnit robustly (by
  /// position, not by register name or LLVM-internal enum values). Keeping our
  /// own stable set of type names decouples that mapping from LLVM internals,
  /// which vary across versions.
  ///
  /// It is a deliberately FLATTENED, FILTERED projection: SDep splits the
  /// classification across two enums (Kind + the OrderKind sub-tag) and includes
  /// hint edges we intentionally drop. We collapse that to the minimal set of
  /// real ordering constraints the scheduler must respect:
  ///   SDep::Data/Anti/Output      -> Data/Anti/Output
  ///   SDep::Order + Barrier       -> Barrier
  ///   SDep::Order + MustAlias/Mem -> Memory
  ///   SDep::Order + Artificial/Cluster (and anything else) -> Other (dropped)
  enum Kind {
    Data,      /// RAW: Src defines reg, Dst uses it.
    Anti,      /// WAR: Src uses reg, Dst defines it.
    Output,    /// WAW: Both Src and Dst define the same reg.
    Memory,    /// Memory ordering: potential aliasing.
    Barrier,   /// Barrier synchronization.
    Other      /// Artificial, Cluster, etc. (not a real ordering constraint).
  };
  Kind Type;

  /// For Data edges: the register involved.
  Register Reg;

  /// Latency (informational).
  unsigned Latency = 0;
};

/// DAGBuilder - Builds scheduling DAG and extracts edges.
/// Uses LLVM's ScheduleDAGInstrs with MachineDominatorTree and MachineLoopInfo
/// for accurate dependency analysis including loop-carried dependencies.
class DAGBuilder {
public:
  /// \p LIS, if non-null, enables subregister (lane-mask) precise WAW/WAR
  /// dependencies matching LLVM's machine scheduler. It must be computed for
  /// \p MF and outlive this builder. If null, dependencies are whole-register
  /// (a safe over-approximation).
  explicit DAGBuilder(MachineFunction &MF, LiveIntervals *LIS = nullptr);
  ~DAGBuilder();

  /// Build DAG for a single basic block, return all edges.
  /// Filters out Artificial/Cluster edges (returns them as Kind::Other).
  SmallVector<DAGEdge, 64> buildDAG(MachineBasicBlock &MBB);

private:
  MachineFunction &MF;
  LiveIntervals *LIS;  // not owned
  std::unique_ptr<MachineDominatorTree> MDT;
  std::unique_ptr<MachineLoopInfo> MLI;

  /// Alias analysis infrastructure.
  std::unique_ptr<TargetLibraryInfoImpl> TLIImpl;
  std::unique_ptr<TargetLibraryInfo> TLI;
  std::unique_ptr<AssumptionCache> AC;
  std::unique_ptr<AAResults> AA;
  std::unique_ptr<BasicAAResult> BasicAA;

  /// Set up alias analysis for the function.
  void setupAliasAnalysis();
};

/// Helper to convert edge kind to string for debugging.
const char *edgeKindToString(DAGEdge::Kind K);

/// Emit one MachineFunction's scheduling DAG as a (bb, position)-keyed edge list
/// to \p os. \p LIS (if non-null) enables subregister-precise dependencies.
void emitSchedulingDAGForMF(raw_ostream &os, MachineFunction &MF,
                            LiveIntervals *LIS = nullptr);

/// Parse `mirText` into a fresh MachineFunction (for the given target) and return
/// its scheduling DAG as a (bb, position)-keyed edge list (see
/// emitSchedulingDAGForMF). Identity is (bb, pos), not register names, so the
/// parser's virtual-register renumbering does not matter: the parsed MF
/// enumerates instructions in mirText's order, so (bb, pos) maps 1:1 onto the
/// dumped MIR body lines. Returns "" on parse failure.
///
/// TODO(tyb0807): re-parsing the just-emitted MIR is a pragmatic choice. The
/// alternative -- building the DAG from the SAME MachineFunction the codegen
/// pipeline produced -- avoids the extra parse but requires either replicating
/// addPassesToGenerateCode and omitting its trailing FreeMachineFunctionPass, or
/// inserting a MachineFunctionPass at the machine-scheduler anchor via
/// TargetPassConfig::insertPass. Both are more (LLVM-version-fragile) glue;
/// revisit if the re-parse ever proves too costly or semantically divergent.
/// Re-parse is safe here because identity is positional, not register-based.
std::string buildSchedulingDAGText(StringRef mirText, StringRef triple,
                                   StringRef cpu);

} // namespace mir_dag
} // namespace llvm

#endif // TRITON_MIR_DAG_DAGBUILDER_H
