//===-- DAGBuilder.cpp - Build scheduling DAG from MachineFunction --------===//
//
// Implementation of DAG building.
// Uses LLVM's ScheduleDAGInstrs to build the dependency graph, reusing the
// same DAG construction logic as the machine scheduler.
//
//===----------------------------------------------------------------------===//

#include "DAGBuilder.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/RegisterTargetPassConfigCallback.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::mir_dag;

namespace {

// Barrier edges are kept as-is. LLVM inserts the AMDGPU wait-count
// instructions (S_WAITCNT on gfx9/10, the split S_WAIT_* counters on gfx11+)
// in GCNPassConfig::addPreEmitPass, which runs after register allocation --
// long after the machine-scheduler point this MIR is dumped at. There are
// therefore no wait-count instructions in the DAG to reason about, and no
// counter semantics to filter barrier edges by. See
// test_no_waitcnt_before_machine_scheduler, which guards that ordering.

/// Check if a memory edge should be kept using proper alias analysis.
/// Uses MachineInstr::mayAlias with AAResults for precise aliasing.
bool shouldKeepMemoryEdge(const MachineInstr *Src, const MachineInstr *Dst,
                          AAResults *AA) {
  // Use LLVM's mayAlias with proper alias analysis.
  // Returns true if the instructions may alias (keep the edge).
  // Returns false if they definitely don't alias (filter the edge).
  return Src->mayAlias(AA, *Dst, /*UseTBAA=*/true);
}

/// A minimal ScheduleDAGInstrs subclass that builds the DAG for one region.
/// Reuses LLVM's dependency analysis from ScheduleDAGInstrs::buildSchedGraph.
class RegionScheduleDAG : public ScheduleDAGInstrs {
  AAResults *AA;
  LiveIntervals *LIS;

public:
  RegionScheduleDAG(MachineFunction &MF, const MachineLoopInfo *MLI,
                    AAResults *AA, LiveIntervals *LIS)
      : ScheduleDAGInstrs(MF, MLI, /*RemoveKillFlags=*/false), AA(AA),
        LIS(LIS) {}

  /// Build the DAG for a region and extract edges.
  SmallVector<DAGEdge, 64>
  buildAndExtractEdges(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator Begin,
                       MachineBasicBlock::iterator End) {
    SmallVector<DAGEdge, 64> Edges;

    if (Begin == End)
      return Edges;

    // Set up the region - must call startBlock before enterRegion.
    startBlock(&MBB);
    enterRegion(&MBB, Begin, End, std::distance(Begin, End));

    // buildSchedGraph MUTATES the MachineFunction when lane-mask tracking is
    // on:
    //
    //   // Clear undef flag, we'll re-add it later once we know which
    //   // subregister Def is first.
    //   MO.setIsUndef(false);      // ScheduleDAGInstrs::addVRegDefDeps
    //
    // The flag is re-added by the scheduler as it places instructions. We only
    // build the graph -- we never schedule -- so nothing would restore it, and
    // a subregister def that loses `undef` makes the MIR printed alongside the
    // DAG unparseable ("Use not jointly dominated by defs"). That is harmless
    // when the MF is a throwaway re-parse, but not when we run on the live
    // codegen pipeline's MF (see armLivePipelineDAGEmission), so snapshot the
    // flags and put them back ourselves.
    //
    // (Kill flags are safe: we pass RemoveKillFlags=false, and fixupKills only
    // runs from the post-RA scheduler.)
    SmallVector<std::pair<MachineInstr *, unsigned>, 16> UndefOperands;
    for (MachineBasicBlock::iterator I = Begin; I != End; ++I)
      for (unsigned OpIdx = 0, E = I->getNumOperands(); OpIdx != E; ++OpIdx) {
        const MachineOperand &MO = I->getOperand(OpIdx);
        if (MO.isReg() && MO.isUndef())
          UndefOperands.emplace_back(&*I, OpIdx);
      }

    // Call schedule() which calls buildSchedGraph internally.
    // This follows the same pattern as MachineScheduler.
    schedule();

    for (const auto &[MI, OpIdx] : UndefOperands)
      MI->getOperand(OpIdx).setIsUndef(true);

    // Extract edges from SUnits.
    for (const SUnit &SU : SUnits) {
      MachineInstr *SrcMI = SU.getInstr();
      if (!SrcMI)
        continue;

      for (const SDep &Succ : SU.Succs) {
        SUnit *SuccSU = Succ.getSUnit();
        if (!SuccSU)
          continue;

        MachineInstr *DstMI = SuccSU->getInstr();
        if (!DstMI)
          continue;

        DAGEdge Edge;
        Edge.Src = SrcMI;
        Edge.Dst = DstMI;
        Edge.Latency = Succ.getLatency();
        Edge.Reg = Register();

        // Classify edge type based on SDep kind and flags.
        switch (Succ.getKind()) {
        case SDep::Data:
          Edge.Type = DAGEdge::Data;
          Edge.Reg = Succ.getReg();
          break;

        case SDep::Anti:
          Edge.Type = DAGEdge::Anti;
          Edge.Reg = Succ.getReg();
          break;

        case SDep::Output:
          Edge.Type = DAGEdge::Output;
          Edge.Reg = Succ.getReg();
          break;

        case SDep::Order:
          // Order dependencies: classify by flags.
          if (Succ.isBarrier()) {
            Edge.Type = DAGEdge::Barrier;
          } else if (Succ.isMustAlias()) {
            // Must-alias edges are real dependencies - keep them.
            Edge.Type = DAGEdge::Memory;
          } else if (Succ.isNormalMemory()) {
            // May-alias edges: filter if we can prove non-aliasing using AA.
            if (!shouldKeepMemoryEdge(SrcMI, DstMI, AA))
              continue;
            Edge.Type = DAGEdge::Memory;
          } else if (Succ.isArtificial() || Succ.isCluster()) {
            Edge.Type = DAGEdge::Other;
          } else {
            Edge.Type = DAGEdge::Other;
          }
          break;

        default:
          Edge.Type = DAGEdge::Other;
          break;
        }

        Edges.push_back(Edge);
      }
    }

    // Clean up following MachineScheduler pattern.
    exitRegion();
    finishBlock();

    return Edges;
  }

  // Override schedule() to just build the graph without reordering. Pass
  // AAResults for precise memory aliasing, and LiveIntervals with
  // TrackLaneMasks=true so subregister (per-lane) WAW/WAR dependencies match
  // LLVM's machine scheduler instead of over-approximating whole-register defs.
  void schedule() override {
    // buildSchedGraph(AA, RPTracker, PDiffs, LIS, TrackLaneMasks). Lane-mask
    // tracking requires a real LiveIntervals; enable it only when we have one.
    buildSchedGraph(AA, /*RPTracker=*/nullptr, /*PDiffs=*/nullptr, LIS,
                    /*TrackLaneMasks=*/LIS != nullptr);
  }
};

} // anonymous namespace

DAGBuilder::DAGBuilder(MachineFunction &MF, LiveIntervals *LIS)
    : MF(MF), LIS(LIS) {
  // Compute dominator tree and loop info for accurate DAG building.
  MDT = std::make_unique<MachineDominatorTree>(MF);
  MLI = std::make_unique<MachineLoopInfo>(*MDT);

  // Set up alias analysis for memory dependency filtering.
  setupAliasAnalysis();
}

DAGBuilder::~DAGBuilder() = default;

void DAGBuilder::setupAliasAnalysis() {
  // Get the IR function from MachineFunction.
  const Function *F =
      MF.getFunction().getParent() ? &MF.getFunction() : nullptr;
  if (!F)
    return;

  Function &Func = const_cast<Function &>(*F);
  const Module *M = Func.getParent();
  if (!M)
    return;

  // Create TargetLibraryInfo from the module's target triple.
  TLIImpl =
      std::make_unique<TargetLibraryInfoImpl>(Triple(M->getTargetTriple()));
  TLI = std::make_unique<TargetLibraryInfo>(*TLIImpl, &Func);

  // Create AAResults with TLI.
  AA = std::make_unique<AAResults>(*TLI);

  // Create AssumptionCache for BasicAA.
  AC = std::make_unique<AssumptionCache>(Func);

  // Create BasicAA and add to AAResults.
  // Note: BasicAA needs DataLayout, Function, TLI, AssumptionCache.
  // We create a minimal setup without DominatorTree (optional for BasicAA).
  const DataLayout &DL = M->getDataLayout();

  // Create BasicAAResult and add to AAResults.
  // Store in member unique_ptr to ensure proper cleanup.
  BasicAA = std::make_unique<BasicAAResult>(DL, Func, *TLI, *AC);
  AA->addAAResult(*BasicAA);
}

SmallVector<SchedulingRegion, 8>
llvm::mir_dag::getSchedulingRegions(MachineBasicBlock &MBB) {
  const MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  // Same predicate as isSchedBoundary in MachineScheduler.cpp. Boundary
  // instructions belong to no region: the scheduler never moves them, and
  // nothing may be moved across them.
  auto isBoundary = [&](const MachineInstr &MI) {
    return MI.isCall() || MI.isFakeUse() ||
           TII->isSchedulingBoundary(MI, &MBB, MF);
  };

  SmallVector<SchedulingRegion, 8> Regions;
  auto Begin = MBB.begin();
  for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
    if (!isBoundary(*I))
      continue;
    if (Begin != I)
      Regions.push_back({Begin, I});
    Begin = std::next(I);
  }
  if (Begin != MBB.end())
    Regions.push_back({Begin, MBB.end()});
  return Regions;
}

SmallVector<DAGEdge, 64> DAGBuilder::buildDAG(MachineBasicBlock &MBB,
                                              MachineBasicBlock::iterator Begin,
                                              MachineBasicBlock::iterator End) {
  if (Begin == End)
    return {};

  RegionScheduleDAG DAG(MF, MLI.get(), AA.get(), LIS);
  return DAG.buildAndExtractEdges(MBB, Begin, End);
}

const char *llvm::mir_dag::edgeKindToString(DAGEdge::Kind K) {
  switch (K) {
  case DAGEdge::Data:
    return "Data";
  case DAGEdge::Anti:
    return "Anti";
  case DAGEdge::Output:
    return "Output";
  case DAGEdge::Memory:
    return "Memory";
  case DAGEdge::Barrier:
    return "Barrier";
  case DAGEdge::Other:
    return "Other";
  }
  return "Unknown";
}

// Emit one MachineFunction's scheduling DAG as a (bb, position)-keyed edge
// list. One section per SCHEDULING REGION, in program order:
//
//   region <bb-number> <region-index-within-bb>
//   node <pos> <def-reg|-> <opcode>
//   ...
//   edge <src-pos> <dst-pos> <Type> <latency> [<reg>]
//   ...
//
// `pos` is the instruction's index within the FULL basic block (boundaries
// included), so it maps directly onto the MIR body lines the scheduler
// reorders. Edge endpoints use the same index space. A region's `node` lines
// therefore carry a contiguous but not necessarily zero-based run of positions,
// and consecutive regions of one block leave a gap where the boundary sits.
//
// Only instructions INSIDE a region get a node. An instruction with no node --
// a terminator, a call, a mask-0 SCHED_BARRIER, an IDX0 write -- is a
// scheduling boundary: LLVM never moves it and never moves anything across it,
// so a consumer must pin it in place. Emitting nodes for them (as this used to)
// while building edges only for the region between them would advertise them as
// freely reorderable, which is exactly backwards.
//
// Only Data/Anti/Output/Memory/Barrier edges are emitted; Artificial/Cluster
// (DAGEdge::Other) are dropped. The def-reg column is a cross-check aid only --
// identity is (bb, pos).
void llvm::mir_dag::emitSchedulingDAGForMF(raw_ostream &os, MachineFunction &MF,
                                           LiveIntervals *LIS) {
  DenseMap<const MachineInstr *, int> posInBB;
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  for (MachineBasicBlock &MBB : MF) {
    int pos = 0;
    for (MachineInstr &MI : MBB)
      posInBB[&MI] = pos++;
  }

  DAGBuilder builder(MF, LIS);
  for (MachineBasicBlock &MBB : MF) {
    int regionIdx = 0;
    for (const SchedulingRegion &R : getSchedulingRegions(MBB)) {
      os << "region " << MBB.getNumber() << " " << regionIdx++ << "\n";
      for (auto I = R.Begin; I != R.End; ++I) {
        MachineInstr &MI = *I;
        StringRef opcode = TII->getName(MI.getOpcode());
        std::string defReg = "-";
        for (const MachineOperand &MO : MI.operands()) {
          if (MO.isReg() && MO.isDef() && !MO.isDead() && MO.getReg()) {
            std::string s;
            raw_string_ostream rs(s);
            rs << printReg(MO.getReg(), TRI);
            defReg = rs.str();
            break;
          }
        }
        os << "node " << posInBB[&MI] << " " << defReg << " " << opcode << "\n";
      }
      SmallPtrSet<const MachineInstr *, 32> inRegion;
      for (auto I = R.Begin; I != R.End; ++I)
        inRegion.insert(&*I);

      SmallVector<DAGEdge, 64> edges = builder.buildDAG(MBB, R.Begin, R.End);
      for (const DAGEdge &E : edges) {
        if (E.Type == DAGEdge::Other)
          continue;
        // ScheduleDAGInstrs gives ExitSU the instruction just past the region,
        // so edges to it name an instruction we did not emit a node for. Drop
        // them: that instruction is a scheduling boundary, and "everything in
        // the region precedes it" is already implied by it being one.
        if (!inRegion.contains(E.Src) || !inRegion.contains(E.Dst))
          continue;
        auto itS = posInBB.find(E.Src);
        auto itD = posInBB.find(E.Dst);
        if (itS == posInBB.end() || itD == posInBB.end())
          continue;
        os << "edge " << itS->second << " " << itD->second << " "
           << edgeKindToString(E.Type) << " " << E.Latency;
        if (E.Reg)
          os << " " << printReg(E.Reg, TRI);
        os << "\n";
      }
    }
  }
}

namespace {

// A MachineFunctionPass that emits each function's scheduling DAG. It requires
// LiveIntervalsWrapperPass so buildSchedGraph can use lane-mask (subregister)
// precise dependencies, matching LLVM's machine scheduler. The emitted text is
// appended to *Sink (set before the pass runs; a registered legacy pass must be
// default-constructible, so the output buffer is threaded through a static).
std::string *DAGTextSink = nullptr;

class EmitSchedulingDAGPass : public MachineFunctionPass {
public:
  static char ID;
  EmitSchedulingDAGPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Emit scheduling DAG (mir_dag)";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (DAGTextSink) {
      LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
      raw_string_ostream os(*DAGTextSink);
      llvm::mir_dag::emitSchedulingDAGForMF(os, MF, &LIS);
    }
    return false;
  }
};
char EmitSchedulingDAGPass::ID = 0;

} // namespace

namespace llvm {
void initializeEmitSchedulingDAGPassPass(PassRegistry &);
}
INITIALIZE_PASS_BEGIN(EmitSchedulingDAGPass, "mir-dag-emit-scheduling-dag",
                      "Emit scheduling DAG (mir_dag)", false, true)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(EmitSchedulingDAGPass, "mir-dag-emit-scheduling-dag",
                    "Emit scheduling DAG (mir_dag)", false, true)

//===----------------------------------------------------------------------===//
// Live-pipeline DAG emission
//===----------------------------------------------------------------------===//
//
// Build the DAG on the MachineFunction the codegen pipeline actually produced,
// instead of serializing it to MIR text and re-parsing it.
//
// Besides avoiding the round-trip and reusing the pipeline's LiveIntervals,
// this makes the `region <bb-number>` key unambiguous. On a re-parsed MF the
// MIR parser assigns block numbers in the order it reads blocks, so the number
// is the block's LAYOUT POSITION; the `bb.N` labels in the dumped text are the
// original numbers. The two agree only when layout order happens to match
// numbering, and a function laid out `bb.9 bb.12 bb.10 bb.11` silently hands
// one block's constraints to another. Emitting from the live MF keys on the
// same MachineBasicBlock the MIR printer labels, so the two cannot diverge.
//
// The obvious approach -- insertPass(&MachineSchedulerID, ...) -- does NOT work
// here. TargetPassConfig::addPass(Pass*) only appends a pass's inserted passes
// when that pass is itself added:
//
//   if (StopBefore == PassID) Stopped = true;
//   if (Started && !Stopped) { PM->add(P); /* add InsertedPasses for P */ }
//   else delete P;
//
// The MIR dump runs with -stop-before=machine-scheduler, so MachineScheduler
// sets Stopped and takes the `delete P` branch; anything inserted at that
// anchor is silently dropped. Anchoring on the preceding pass
// (RenameIndependentSubregs) is no better: our callback runs before
// GCNPassConfig::addOptimizedRegAlloc, so our entry would precede AMDGPU's own
// insertPass(&RenameIndependentSubregsID, &GCNRewritePartialRegUsesID) -- and
// GCNRewritePartialRegUses rewrites instructions, so the DAG would describe a
// different MF than the one dumped.
//
// Instead we take MachineScheduler's slot via substitutePass and stop after
// ourselves. addPass(AnalysisID) resolves the substitution before calling
// addPass(Pass*), so the pass ID tested against StopBefore/StopAfter is OURS:
// the pass is added and runs at exactly the point MachineScheduler would have,
// with the same LiveIntervals, and the pipeline stops immediately after. The
// pass preserves all analyses, and it restores the operand flags
// buildSchedGraph perturbs (see buildAndExtractEdges), so the MIR printed by
// addPassesToEmitFile is byte-identical to what -stop-before=machine-scheduler
// produced.

static std::string *LivePipelineSink = nullptr;

// Register the TargetPassConfig callback LAZILY, on first arm().
//
// It must NOT be a file-scope global: llvm::TargetPassConfigCallbacks (in
// RegisterTargetPassConfigCallback.cpp) is itself a file-scope SmallVector with
// a dynamic initializer, so a global here races it. When our global wins,
// push_back() runs on an unconstructed vector and the vector's own constructor
// then clears the registration -- the callback silently never fires. A
// function-local static is constructed on first call, long after all static
// initialization has completed.
static void ensureLivePipelineCallbackRegistered() {
  // Invoked for EVERY codegen pipeline created in this process (any target),
  // so it must be inert unless armed.
  static RegisterTargetPassConfigCallback Reg(
      [](TargetMachine &TM, legacy::PassManagerBase &, TargetPassConfig *PC) {
        if (!LivePipelineSink || !PC)
          return;
        if (!TM.getTargetTriple().isAMDGCN())
          return;
        PC->substitutePass(&MachineSchedulerID, &EmitSchedulingDAGPass::ID);
      });
  (void)Reg;
}

const char *llvm::mir_dag::livePipelineDAGPassName() {
  return "mir-dag-emit-scheduling-dag";
}

void llvm::mir_dag::armLivePipelineDAGEmission(std::string *Sink) {
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeEmitSchedulingDAGPassPass(Registry);
  initializeLiveIntervalsWrapperPassPass(Registry);
  ensureLivePipelineCallbackRegistered();
  LivePipelineSink = Sink;
  DAGTextSink = Sink;
}

void llvm::mir_dag::disarmLivePipelineDAGEmission() {
  LivePipelineSink = nullptr;
  DAGTextSink = nullptr;
}
