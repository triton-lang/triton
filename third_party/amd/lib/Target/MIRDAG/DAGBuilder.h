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
#include <string>

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
  MachineInstr *Src; /// Source instruction (must execute first).
  MachineInstr *Dst; /// Destination instruction (depends on Src).

  /// Edge types we emit.
  ///
  /// Why a separate enum instead of reusing LLVM's SDep::Kind /
  /// SDep::OrderKind: this enum is emitted into the (bb, position)-keyed DAG
  /// text, which is what lets a MIR instruction be mapped back to its DAG SUnit
  /// robustly (by position, not by register name or LLVM-internal enum values).
  /// Keeping our own stable set of type names decouples that mapping from LLVM
  /// internals, which vary across versions.
  ///
  /// It is a deliberately FLATTENED, FILTERED projection: SDep splits the
  /// classification across two enums (Kind + the OrderKind sub-tag) and
  /// includes hint edges we intentionally drop. We collapse that to the minimal
  /// set of real ordering constraints the scheduler must respect:
  ///   SDep::Data/Anti/Output      -> Data/Anti/Output
  ///   SDep::Order + Barrier       -> Barrier
  ///   SDep::Order + MustAlias/Mem -> Memory
  ///   SDep::Order + Artificial/Cluster (and anything else) -> Other (dropped)
  enum Kind {
    Data,    /// RAW: Src defines reg, Dst uses it.
    Anti,    /// WAR: Src uses reg, Dst defines it.
    Output,  /// WAW: Both Src and Dst define the same reg.
    Memory,  /// Memory ordering: potential aliasing.
    Barrier, /// Barrier synchronization.
    Other    /// Artificial, Cluster, etc. (not a real ordering constraint).
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

  /// Build the DAG for one scheduling region -- the half-open instruction
  /// range [\p Begin, \p End) within \p MBB -- and return all edges.
  /// Filters out Artificial/Cluster edges (returns them as Kind::Other).
  ///
  /// Use getSchedulingRegions to carve a block into the ranges LLVM's own
  /// scheduler would use; passing a range that spans a scheduling boundary
  /// yields a DAG missing the constraints that boundary stands for.
  SmallVector<DAGEdge, 64> buildDAG(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator Begin,
                                    MachineBasicBlock::iterator End);

private:
  MachineFunction &MF;
  LiveIntervals *LIS; // not owned
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

/// A scheduling region: the half-open instruction range LLVM's machine
/// scheduler treats as one reorderable unit.
struct SchedulingRegion {
  MachineBasicBlock::iterator Begin;
  MachineBasicBlock::iterator End;
};

/// Carve \p MBB into the same scheduling regions MachineScheduler would use,
/// in program order.
///
/// A region is a maximal run of instructions between scheduling boundaries;
/// the boundaries themselves belong to no region and are not reorderable. This
/// mirrors getSchedRegions in MachineScheduler.cpp -- note that a region is
/// generally a PART of a basic block, not the whole of it: on AMDGPU a
/// boundary is a terminator or label, an INLINEASM_BR, a call, a fake use, a
/// SCHED_BARRIER with mask 0, or any IDX0 write. Triton itself emits mask-0
/// sched barriers (see BlockPingpong and ConvertWarpPipeline) precisely to
/// stop the backend reordering across them, so multi-region blocks are the
/// common case rather than a corner case.
SmallVector<SchedulingRegion, 8> getSchedulingRegions(MachineBasicBlock &MBB);

/// Emit one MachineFunction's scheduling DAG as a (bb, position)-keyed edge
/// list to \p os, one section per scheduling region. \p LIS (if non-null)
/// enables subregister-precise dependencies.
void emitSchedulingDAGForMF(raw_ostream &os, MachineFunction &MF,
                            LiveIntervals *LIS = nullptr);

/// Arm DAG emission from the live codegen pipeline: the next
/// addPassesToEmitFile run for an AMDGCN target will emit its scheduling DAG
/// into \p Sink, built on the MachineFunction the pipeline itself produced.
///
/// The caller must run that pipeline with
/// `-stop-after=<livePipelineDAGPassName()>` instead of
/// `-stop-before=machine-scheduler`: the emitting pass takes the machine
/// scheduler's slot, so it runs at the same point with the same LiveIntervals
/// and preserves all analyses. Building the DAG does perturb the MF --
/// buildSchedGraph clears `undef` on subregister defs, expecting the scheduler
/// to re-add it -- so the pass restores those flags itself, leaving the dumped
/// MIR unchanged. Must be paired with disarmLivePipelineDAGEmission().
///
/// Emitting from the live MF means the `region <bb-number>` key is the same
/// block number the MIR printer puts in the `bb.N` labels, which re-parsing the
/// dumped text cannot guarantee (see DAGBuilder.cpp).
///
/// Not thread-safe: arming is process-global (see DAGBuilder.cpp).
void armLivePipelineDAGEmission(std::string *Sink);
void disarmLivePipelineDAGEmission();

/// Command-line name of the DAG-emitting pass, for `-stop-after`.
const char *livePipelineDAGPassName();

} // namespace mir_dag
} // namespace llvm

#endif // TRITON_MIR_DAG_DAGBUILDER_H
