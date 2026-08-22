//===-- DAGBuilder.cpp - Build scheduling DAG from MachineFunction --------===//
//
// Implementation of DAG building.
// Uses LLVM's ScheduleDAGInstrs to build the dependency graph, reusing the
// same DAG construction logic as the machine scheduler.
//
//===----------------------------------------------------------------------===//

#include "DAGBuilder.h"
#include "AMDGPUInstrUtils.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::mir_dag;

namespace {

/// S_WAITCNT immediate encoding for GFX9/GFX10 (AMDGPU). This layout is public
/// (documented in AMD's GCN/CDNA ISA reference guides), but it is version-
/// specific:
///   bits [3:0]  = VM_CNT  (VMEM operations like BUFFER_LOAD)
///   bits [6:4]  = EXP_CNT (export operations)
///   bits [11:8] = LGKM_CNT (LDS/GDS/SMEM operations like DS_READ/DS_WRITE)
///
/// When a counter is at its max value, it means "don't wait" for that type.
/// GFX9: vmcnt max=15, expcnt max=7, lgkmcnt max=15.
///
/// This hardcodes the gfx9/10 layout, which is the packed S_WAITCNT used by
/// gfx9/gfx10 (MI300/MI350). gfx11+ (including gfx1250/MI450) does NOT use packed
/// S_WAITCNT -- it emits split counters (S_WAIT_LOADCNT/STORECNT/DSCNT/
/// TENSORCNT). So on gfx1250 isWaitcnt() is always false and this decode never
/// runs; the barrier-edge filter simply keeps all such edges there. It is only
/// exercised (and correct) on gfx9/10.
///
/// TODO(tyb0807): to also refine split-counter waits on gfx11+, use LLVM's
/// version-aware decoders (AMDGPU::decodeLoadcnt/decodeDscnt/... (IsaVersion,
/// Imm) in lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.h). They are not in the
/// installed public headers; adopt them once a public waitcnt-decode API is
/// exposed upstream.
struct WaitCounts {
  unsigned VMCnt;   // 0-15, 15 = don't wait for VMEM.
  unsigned ExpCnt;  // 0-7, 7 = don't wait for exports.
  unsigned LGKMCnt; // 0-15, 15 = don't wait for LDS/GDS/SMEM.

  static constexpr unsigned VMCntMax = 15;
  static constexpr unsigned ExpCntMax = 7;
  static constexpr unsigned LGKMCntMax = 15;

  bool waitsForVMEM() const { return VMCnt < VMCntMax; }
  bool waitsForLDS() const { return LGKMCnt < LGKMCntMax; }
  bool waitsForExport() const { return ExpCnt < ExpCntMax; }

  /// Two S_WAITCNTs are independent when the sets of hardware counters they
  /// each wait on do not overlap. Such waitcnts commute (each only stalls on
  /// its own disjoint counter), so there is no ordering dependency between them.
  bool disjointFrom(const WaitCounts &Other) const {
    return !(waitsForVMEM() && Other.waitsForVMEM()) &&
           !(waitsForLDS() && Other.waitsForLDS()) &&
           !(waitsForExport() && Other.waitsForExport());
  }
};

/// Decode S_WAITCNT immediate value into individual wait counts.
WaitCounts decodeWaitCnt(unsigned Imm) {
  WaitCounts WC;
  WC.VMCnt = Imm & 0xF;
  WC.ExpCnt = (Imm >> 4) & 0x7;
  WC.LGKMCnt = (Imm >> 8) & 0xF;
  return WC;
}


/// Get S_WAITCNT immediate operand value.
/// Returns the immediate value if found, or ~0u if not.
unsigned getWaitCntImm(const MachineInstr *MI) {
  for (const MachineOperand &MO : MI->operands()) {
    if (MO.isImm())
      return static_cast<unsigned>(MO.getImm());
  }
  return ~0u;
}

/// Check if an edge should be filtered based on S_WAITCNT semantics.
/// Returns true if the edge should be KEPT, false if it should be FILTERED.
bool shouldKeepBarrierEdge(const MachineInstr *Src, const MachineInstr *Dst) {
  // Only filter barrier edges to S_WAITCNT.
  if (!isWaitcnt(*Dst))
    return true;

  unsigned Imm = getWaitCntImm(Dst);
  if (Imm == ~0u)
    return true; // Can't decode, keep edge to be safe.

  WaitCounts WC = decodeWaitCnt(Imm);

  // S_WAITCNT -> S_WAITCNT: the scheduler emits an order edge between adjacent
  // waitcnts, but two waitcnts that wait on disjoint counter sets (e.g.
  // vmcnt-only vs lgkmcnt-only) are independent and commute. Filter that
  // spurious edge so a pass that legitimately relocates one waitcnt (e.g.
  // insert-cond-barrier hoisting a vmcnt throttle into a uniform guard block)
  // is not flagged as reordering a real dependency.
  if (isWaitcnt(*Src)) {
    unsigned SrcImm = getWaitCntImm(Src);
    if (SrcImm == ~0u)
      return true; // Can't decode source, keep edge to be safe.
    if (decodeWaitCnt(SrcImm).disjointFrom(WC))
      return false;
    return true;
  }

  // If source is VMEM and S_WAITCNT doesn't wait for VMEM, filter the edge.
  if (isVMEMOp(*Src) && !WC.waitsForVMEM())
    return false;

  // If source is LDS and S_WAITCNT doesn't wait for LDS, filter the edge.
  if (isLDSOp(*Src) && !WC.waitsForLDS())
    return false;

  return true;
}

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
  SmallVector<DAGEdge, 64> buildAndExtractEdges(MachineBasicBlock &MBB,
                                                MachineBasicBlock::iterator Begin,
                                                MachineBasicBlock::iterator End) {
    SmallVector<DAGEdge, 64> Edges;

    if (Begin == End)
      return Edges;

    // Set up the region - must call startBlock before enterRegion.
    startBlock(&MBB);
    enterRegion(&MBB, Begin, End, std::distance(Begin, End));

    // Call schedule() which calls buildSchedGraph internally.
    // This follows the same pattern as MachineScheduler.
    schedule();

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
            // Filter spurious barrier edges based on S_WAITCNT semantics.
            // E.g., BUFFER_LOAD -> S_WAITCNT with vmcnt=15 is not a real dep.
            if (!shouldKeepBarrierEdge(SrcMI, DstMI))
              continue;
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
  const Function *F = MF.getFunction().getParent()
                          ? &MF.getFunction()
                          : nullptr;
  if (!F)
    return;

  Function &Func = const_cast<Function &>(*F);
  const Module *M = Func.getParent();
  if (!M)
    return;

  // Create TargetLibraryInfo from the module's target triple.
  TLIImpl = std::make_unique<TargetLibraryInfoImpl>(Triple(M->getTargetTriple()));
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

SmallVector<DAGEdge, 64> DAGBuilder::buildDAG(MachineBasicBlock &MBB) {
  if (MBB.empty())
    return {};

  // Find end of schedulable region (before terminators).
  // ScheduleDAGInstrs doesn't handle terminators by default.
  auto RegionEnd = MBB.end();
  for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
    if (I->isTerminator()) {
      RegionEnd = I;
      break;
    }
  }

  // If no non-terminator instructions, return empty.
  if (RegionEnd == MBB.begin())
    return {};

  RegionScheduleDAG DAG(MF, MLI.get(), AA.get(), LIS);
  return DAG.buildAndExtractEdges(MBB, MBB.begin(), RegionEnd);
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

// Emit one MachineFunction's scheduling DAG as a (bb, position)-keyed edge list.
// One section per MachineBasicBlock:
//
//   region <bb-number>
//   node <pos> <def-reg|-> <opcode>
//   ...
//   edge <src-pos> <dst-pos> <Type> <latency> [<reg>]
//   ...
//
// `pos` is the instruction's index within the FULL basic block (terminators
// included), so it maps directly onto the MIR body lines the scheduler reorders.
// Edge endpoints use the same index space. Only Data/Anti/Output/Memory/Barrier
// edges are emitted; Artificial/Cluster (DAGEdge::Other) are dropped. The def-reg
// column is a cross-check aid only -- identity is (bb, pos).
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
    os << "region " << MBB.getNumber() << "\n";
    for (MachineInstr &MI : MBB) {
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
    SmallVector<DAGEdge, 64> edges = builder.buildDAG(MBB);
    for (const DAGEdge &E : edges) {
      if (E.Type == DAGEdge::Other)
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

// Parse `mirText` into a fresh MachineFunction and emit its scheduling DAG. A
// small legacy PassManager computes LiveIntervals (via LiveIntervalsWrapperPass)
// so the DAG has subregister-precise dependencies. Because identity is (bb,
// pos), the parser's vreg renumbering is irrelevant -- the parsed MF preserves
// mirText's instruction order.
//
// TODO(tyb0807): run EmitSchedulingDAGPass inside the original codegen
// PassManager (at the machine-scheduler anchor) so the DAG is built on the live
// MachineFunction directly, instead of serializing to MIR text and re-parsing
// it here. That avoids the re-parse round-trip and reuses the LiveIntervals
// already computed by the pipeline.
std::string llvm::mir_dag::buildSchedulingDAGText(StringRef mirText,
                                                  StringRef triple,
                                                  StringRef cpu) {
  using namespace llvm;

  llvm::Triple TT(triple.str());
  std::string error;
  const Target *target = TargetRegistry::lookupTarget(TT, error);
  if (!target)
    return "";

  TargetOptions options;
  std::unique_ptr<TargetMachine> TM(target->createTargetMachine(
      TT, cpu, /*Features=*/"", options, std::nullopt));
  if (!TM)
    return "";

  LLVMContext ctx;
  std::unique_ptr<MemoryBuffer> buffer = MemoryBuffer::getMemBuffer(mirText);
  std::unique_ptr<MIRParser> parser = createMIRParser(std::move(buffer), ctx);
  if (!parser)
    return "";

  std::unique_ptr<Module> M = parser->parseIRModule();
  if (!M)
    return "";
  M->setTargetTriple(TT);
  M->setDataLayout(TM->createDataLayout());

  // Register the passes we will run (and their dependencies) so the legacy
  // PassManager can resolve them.
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeEmitSchedulingDAGPassPass(Registry);
  initializeLiveIntervalsWrapperPassPass(Registry);
  initializeSlotIndexesWrapperPassPass(Registry);
  initializeMachineDominatorTreeWrapperPassPass(Registry);
  initializeMachineLoopInfoWrapperPassPass(Registry);

  std::string out;
  DAGTextSink = &out;

  {
    legacy::PassManager PM;
    // Construct the MMI wrapper explicitly with the TargetMachine and add it
    // first, so the PassManager does not default-construct one (which would have
    // a null TargetMachine and crash).
    auto *MMIWP = new MachineModuleInfoWrapperPass(TM.get());
    PM.add(MMIWP);
    PM.add(new EmitSchedulingDAGPass());
    if (parser->parseMachineFunctions(*M, MMIWP->getMMI())) {
      DAGTextSink = nullptr;
      return "";
    }
    PM.run(*M);
  }

  DAGTextSink = nullptr;
  return out;
}
