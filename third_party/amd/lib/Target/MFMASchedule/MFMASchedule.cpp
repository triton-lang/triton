//===- MFMASchedule.cpp - MFMA <-> memory interleave for gfx950 -----------===//
//
// An opt-in LLVM-IR pass that interleaves the MFMAs of a matrix-core hot loop
// with the memory operations that are independent of them, so the matrix unit
// stays busy while loads and LDS accesses issue. It runs on the optimized IR,
// before instruction selection, when the caller passes
// schedule_hint="mfma-schedule" (see HIPBackend.make_llir).
//
// The pass works per basic block. A block is cut into spans: a span is a
// run of MFMAs that ends at the first MFMA after a memory op, so the loads
// feeding a span's MFMAs sit in an earlier span and reordering inside a
// span is dependency-safe. Within a span the pass hoists MFMA-input prep,
// sinks MFMA-result extracts, spaces the MFMAs around the span's memory
// anchors with a throughput model (how many MFMA cycles each memory op's issue
// port occupancy needs), and emits an llvm.amdgcn.sched.barrier in front of
// each anchor so LLVM's machine schedulers keep the interleave. A block that
// already carries sched.barriers (an explicit hint, or a block-pingpong
// cluster boundary) is scheduled range by range between them.
//
// MFMA accumulators pinned to a register class by Gluon's cd_regclass (an
// empty tied inline asm on C and D) travel with their MFMA and are fenced, so
// the pins survive register allocation.
//
// Every block is scheduled transactionally: snapshot, schedule, verify, and
// roll the block back if the result is invalid, so a bad schedule never
// reaches codegen. The pass returns true iff a span was scheduled.
//
//===----------------------------------------------------------------------===//

#include "MFMASchedule.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tritonamdgpu-mfma-schedule"

namespace {

using namespace llvm;

// How an instruction takes part in the schedule: an MFMA, a global-memory
// access (GR), an LDS read (LR), an LDS write (LW), or nothing the pass moves.
enum class SchedKind { MFMA, GR, LR, LW, Other };

constexpr unsigned kLDSAddressSpace = 3;
constexpr unsigned kGlobalAddressSpace = 1;

// A memory op the span's MFMAs are spaced around.
struct AnchorInst {
  Instruction *I = nullptr;
  SchedKind Kind = SchedKind::Other;
};

// A span found by analyzeSpan: its first MFMA and how many MFMAs it holds.
struct SpanInfo {
  Instruction *SpanStart = nullptr;
  unsigned TotalMFMA = 0;
};

using SpanList = SmallVector<SpanInfo, 8>;

// The instruction range [Begin, End) of one span; End == nullptr means the
// end of the block.
struct Span {
  BasicBlock *BB = nullptr;
  Instruction *Begin = nullptr;
  Instruction *End = nullptr;
};

// What collectSpan found in a span.
struct SpanContents {
  // MFMA-input prep (shuffles, insertelements) to hoist to the span start.
  SmallVector<Instruction *, 16> Hoist;
  // MFMA-result extracts to sink past the span's last anchor.
  SmallVector<Instruction *, 16> Sink;
  // The last memory anchor, the sink point.
  Instruction *LastAnchor = nullptr;
  // The memory ops (GR/LR/LW) in program order.
  SmallVector<AnchorInst, 32> Anchors;
  // The span's MFMAs in program order.
  SmallVector<Instruction *, 32> MFMAInsts;
};

bool isMFMAorWMMA(const Instruction &I) {
  const auto *CI = dyn_cast<CallInst>(&I);
  if (!CI || CI->isInlineAsm())
    return false;
  const Function *Callee = CI->getCalledFunction();
  if (!Callee || !Callee->isIntrinsic())
    return false;
  StringRef Name = Callee->getName();
  return Name.contains("mfma") || Name.contains("wmma");
}

bool isSchedBarrier(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      return F->getIntrinsicID() == Intrinsic::amdgcn_sched_barrier;
  return false;
}

bool hasSchedBarrier(const Function &F) {
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB)
      if (isSchedBarrier(I))
        return true;
  return false;
}

bool isHoistTransparentInst(const Instruction &I) {
  return isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I);
}

bool isSinkTransparentInst(const Instruction &I) {
  return isa<ExtractElementInst>(I);
}

// A register-class pin: an empty inline asm whose output is tied to its input
// ("=a,0" / "=v,0"), which Triton's `cd_regclass` MFMA lowering puts around
// each MFMA tile's accumulator. It emits no instruction; it only forces the
// value into AGPRs or VGPRs at that point, so it has to stay right next to its
// MFMA: the pin on C directly before the MFMA, the pin on D directly after.
bool isRegClassPin(const Instruction &I) {
  const auto *CI = dyn_cast<CallInst>(&I);
  if (!CI || CI->arg_size() != 1 ||
      CI->getType() != CI->getArgOperand(0)->getType())
    return false;
  const auto *IA = dyn_cast<InlineAsm>(CI->getCalledOperand());
  if (!IA || !IA->getAsmString().empty())
    return false;
  StringRef Constraints = IA->getConstraintString();
  return Constraints == "=a,0" || Constraints == "=v,0";
}

// The pin that produces MFMA's accumulator operand C, if that is its only use.
Instruction *getCPin(Instruction *MFMA) {
  auto *CI = cast<CallInst>(MFMA);
  if (CI->arg_size() < 3 || CI->getArgOperand(2)->getType() != CI->getType())
    return nullptr;
  auto *Pin = dyn_cast<Instruction>(CI->getArgOperand(2));
  if (!Pin || !isRegClassPin(*Pin) || !Pin->hasOneUse() ||
      Pin->getParent() != MFMA->getParent())
    return nullptr;
  return Pin;
}

// The pin on MFMA's result D, if it is the result's only user.
Instruction *getDPin(Instruction *MFMA) {
  if (!MFMA->hasOneUse())
    return nullptr;
  auto *Pin = dyn_cast<Instruction>(*MFMA->user_begin());
  if (!Pin || !isRegClassPin(*Pin) || Pin->getParent() != MFMA->getParent())
    return nullptr;
  return Pin;
}

SchedKind classifySchedInst(const Instruction &I) {
  if (isMFMAorWMMA(I))
    return SchedKind::MFMA;

  if (const auto *CI = dyn_cast<CallInst>(&I)) {
    if (const Function *F = CI->getCalledFunction()) {
      if (F->isIntrinsic()) {
        StringRef Name = F->getName();
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
  if (const auto *LI = dyn_cast<LoadInst>(&I)) {
    if (LI->getPointerAddressSpace() == kLDSAddressSpace)
      return SchedKind::LR;
    if (LI->getPointerAddressSpace() == kGlobalAddressSpace)
      return SchedKind::GR;
  }
  if (const auto *SI = dyn_cast<StoreInst>(&I)) {
    if (SI->getPointerAddressSpace() == kLDSAddressSpace)
      return SchedKind::LW;
    if (SI->getPointerAddressSpace() == kGlobalAddressSpace)
      return SchedKind::GR;
  }
  return SchedKind::Other;
}

bool isMemoryAnchor(SchedKind K) {
  return K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW;
}

iterator_range<BasicBlock::iterator> instructionsInSpan(const Span &R) {
  auto ItBegin = R.Begin ? R.Begin->getIterator() : R.BB->begin();
  auto ItEnd = R.End ? R.End->getIterator() : R.BB->end();
  return make_range(ItBegin, ItEnd);
}

// Issue-to-issue cost of an MFMA on gfx950: its pass count times 4 cycles
// (SISchedule.td: 16x16xK shapes are 4 passes, 32x32xK shapes 8, whatever the
// element type). Returns 0 for a shape the table does not know; the span is
// then left to LLVM's schedulers.
unsigned getMFMACycles(const Instruction &I) {
  if (!isMFMAorWMMA(I))
    return 0;
  const auto *CI = cast<CallInst>(&I);
  const Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return 0;
  StringRef Name = Callee->getName();

  // Scaled f8f6f4 MFMAs: the cost depends on the operand formats encoded in
  // cbsz (arg 3) and blgp (arg 4); a value above 1 is a 4- or 6-bit format,
  // which the unit runs at the 4-bit rate.
  auto scaledCycles = [&](unsigned Narrow, unsigned Wide) {
    if (auto *Cbsz = dyn_cast<ConstantInt>(CI->getArgOperand(3)))
      if (auto *Blgp = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
        return (Cbsz->getZExtValue() > 1 && Blgp->getZExtValue() > 1) ? Narrow
                                                                      : Wide;
    return Wide;
  };
  if (Name.contains("mfma.scale.f32.16x16x128.f8f6f4"))
    return scaledCycles(16, 32);
  if (Name.contains("mfma.scale.f32.32x32x64.f8f6f4"))
    return scaledCycles(32, 64);

  static constexpr struct {
    StringRef Name;
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

// Width in bits of the value moved by an LDS access.
unsigned getLDSAccessBits(const Instruction *I) {
  if (const auto *LI = dyn_cast<LoadInst>(I))
    return LI->getType()->getPrimitiveSizeInBits();
  if (const auto *SI = dyn_cast<StoreInst>(I))
    return SI->getValueOperand()->getType()->getPrimitiveSizeInBits();
  if (const auto *CI = dyn_cast<CallInst>(I))
    return CI->getType()->getPrimitiveSizeInBits();
  return 0;
}

// Cycles of MFMA cover an LDS access asks for: the LDS port moves one byte
// per cycle in steady state, so the cost is proportional to the access width.
unsigned getLDSCoverCycles(const Instruction *I, unsigned MFMACycles) {
  unsigned Bits = getLDSAccessBits(I);
  return Bits ? (Bits / 8) : MFMACycles;
}

// MFMAs to place at this LDS access. Reads and writes share the one LDS issue
// port, so a running cycle balance is carried across the span's accesses:
// each access adds its cover cycles and takes floor(balance / MFMACycles)
// MFMAs, keeping the remainder for the next one.
unsigned takeMFMAsForLDS(const Instruction *I, unsigned MFMACycles,
                         unsigned &AccumCycles) {
  AccumCycles += getLDSCoverCycles(I, MFMACycles);
  unsigned N = AccumCycles / MFMACycles;
  AccumCycles -= N * MFMACycles;
  return N;
}

class MFMAScheduler {
public:
  // Schedule the spans of the range [Begin, End) of BB; End == nullptr means
  // the end of the block. A range with no schedulable span is left untouched.
  // Returns true if any span was scheduled.
  bool runRange(BasicBlock &BB, Instruction *Begin, Instruction *End) {
    SpanList Spans;
    analyzeSpan(Begin, End ? End->getIterator() : BB.end(), Spans);
    return scheduleSpans(BB, Spans, End);
  }

  bool runBlock(BasicBlock &BB) { return runRange(BB, &BB.front(), nullptr); }

  // Roll a block back to a pre-scheduling snapshot: erase the sched.barriers
  // the scheduler inserted and restore the recorded instruction order.
  static void restoreBlock(BasicBlock &BB,
                           const SmallVectorImpl<Instruction *> &Snapshot) {
    SmallPtrSet<const Instruction *, 32> Orig(Snapshot.begin(), Snapshot.end());
    SmallVector<Instruction *, 8> Inserted;
    for (Instruction &I : BB)
      if (!Orig.count(&I))
        Inserted.push_back(&I);
    for (Instruction *I : Inserted) {
      if (!I->use_empty())
        I->replaceAllUsesWith(PoisonValue::get(I->getType()));
      I->eraseFromParent();
    }
    for (size_t i = 1; i < Snapshot.size(); ++i)
      Snapshot[i]->moveAfter(Snapshot[i - 1]);
  }

private:
  // Cut the range [Begin, End) into spans in one program-order pass. A new
  // span opens at every MFMA that follows a memory op seen since the current
  // span's MFMAs began: that op feeds this MFMA, so it belongs to the next
  // span. By construction an MFMA's input loads land in an earlier span, so
  // reordering inside a span is dependency-safe.
  static void analyzeSpan(Instruction *Begin, BasicBlock::iterator End,
                          SpanList &Spans) {
    bool SeenMemoryOps = false;
    for (auto It = Begin->getIterator(); It != End; ++It) {
      Instruction &I = *It;
      SchedKind SK = classifySchedInst(I);
      if (isMemoryAnchor(SK))
        SeenMemoryOps = true;
      if (SK != SchedKind::MFMA)
        continue;
      if (Spans.empty() || SeenMemoryOps) {
        // Memory ops seen before a span's first MFMA are that span's own
        // setup, not a boundary.
        SeenMemoryOps = false;
        Spans.push_back({&I, 0});
      }
      Spans.back().TotalMFMA++;
    }
    LLVM_DEBUG({
      for (unsigned i = 0; i < Spans.size(); ++i)
        dbgs() << "  span " << i << ": " << Spans[i].TotalMFMA << " mfma\n";
    });
  }

  // Does I reach an MFMA through prep instructions and pins only?
  static bool feedsMFMA(Instruction *I) {
    SmallVector<Value *, 8> Worklist{I};
    SmallPtrSet<Value *, 16> Visited;
    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;
      for (User *U : V->users()) {
        if (auto *UI = dyn_cast<Instruction>(U)) {
          if (isMFMAorWMMA(*UI))
            return true;
          if (isHoistTransparentInst(*UI) || isRegClassPin(*UI))
            Worklist.push_back(UI);
        }
      }
    }
    return false;
  }

  // Does I come from an MFMA through extracts and pins only?
  static bool definedByMFMA(Instruction *I) {
    SmallVector<Value *, 8> Worklist{I};
    SmallPtrSet<Value *, 16> Visited;
    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;
      if (auto *DefI = dyn_cast<Instruction>(V)) {
        if (isMFMAorWMMA(*DefI))
          return true;
        if (isSinkTransparentInst(*DefI) || isRegClassPin(*DefI))
          for (Value *Op : DefI->operands())
            Worklist.push_back(Op);
      }
    }
    return false;
  }

  static SpanContents collectSpan(const Span &R) {
    SpanContents Res;

    // Hoisting moves a prep to the span start, which is only safe if every
    // operand already dominates that position: operands defined before the
    // span do, and so does a prep hoisted ahead of it. An operand defined
    // inside the span that is not hoisted (an anchor, or a rejected prep)
    // would end up after its use, so such a prep stays put.
    SmallPtrSet<const Instruction *, 32> SpanInsts;
    for (Instruction &I : instructionsInSpan(R))
      SpanInsts.insert(&I);
    SmallPtrSet<const Instruction *, 16> Hoisted;

    for (Instruction &I : instructionsInSpan(R)) {
      SchedKind K = classifySchedInst(I);
      if (isMemoryAnchor(K)) {
        Res.LastAnchor = &I;
        Res.Anchors.push_back({&I, K});
        continue;
      }
      if (K == SchedKind::MFMA) {
        Res.MFMAInsts.push_back(&I);
        continue;
      }
      if (isHoistTransparentInst(I)) {
        bool SafeToHoist = llvm::all_of(I.operands(), [&](Value *Op) {
          auto *OpI = dyn_cast<Instruction>(Op);
          return !OpI || OpI == R.Begin || !SpanInsts.count(OpI) ||
                 Hoisted.count(OpI);
        });
        if (SafeToHoist && feedsMFMA(&I)) {
          Res.Hoist.push_back(&I);
          Hoisted.insert(&I);
        }
        continue;
      }
      if (isSinkTransparentInst(I) && definedByMFMA(&I))
        Res.Sink.push_back(&I);
    }
    return Res;
  }

  // Hoist the MFMA-input prep to the span start and sink the MFMA-result
  // extracts past the last anchor, so the MFMA run in between can be reordered
  // freely. A span with no anchor has no sink point; its extracts stay, and
  // scheduleSpans skips such a span anyway.
  static SpanContents preprocessSpan(const Span &R) {
    SpanContents Res = collectSpan(R);
    Instruction *HoistPos = R.Begin;
    if (Instruction *DPin = getDPin(R.Begin))
      HoistPos = DPin; // keep the span start's D pin directly after it
    for (Instruction *I : llvm::reverse(Res.Hoist))
      I->moveAfter(HoistPos);
    if (Res.LastAnchor)
      for (Instruction *I : llvm::reverse(Res.Sink))
        I->moveAfter(Res.LastAnchor);
    return Res;
  }

  // Move one MFMA right after InsertPt, together with its register-class pins:
  // the C pin stays directly before it and the D pin directly after it.
  static void moveMFMAAfter(Instruction *MFMA, Instruction *InsertPt) {
    Instruction *CPin = getCPin(MFMA);
    Instruction *DPin = getDPin(MFMA);
    MFMA->moveAfter(InsertPt);
    if (CPin)
      CPin->moveBefore(MFMA->getIterator());
    if (DPin)
      DPin->moveAfter(MFMA);
  }

  // Move up to Count MFMAs, taken from the back of MFMAInsts[0, MFMAIdx), to
  // right after InsertPt. Each moved MFMA lands directly after InsertPt and
  // pushes the previously moved ones further away, so program order is kept.
  static void moveMFMAsAfter(SmallVectorImpl<Instruction *> &MFMAInsts,
                             unsigned &MFMAIdx, unsigned Count,
                             Instruction *InsertPt) {
    for (unsigned j = 0; j < Count && MFMAIdx > 0; ++j)
      moveMFMAAfter(MFMAInsts[--MFMAIdx], InsertPt);
  }

  // Space the span's MFMAs around its memory anchors. Every count is "how
  // many MFMAs of compute cover this memory op's issue-port occupancy":
  //   - a global load occupies its path for about 64 cycles, so it gets
  //     ceil(64 / mfma_cycles) MFMAs (1 if an LDS read follows directly);
  //   - LDS reads and writes draw MFMAs at the width-proportional rate of
  //     takeMFMAsForLDS;
  //   - 2 MFMAs drain the tail, and any surplus compute is split evenly
  //     between the span's head and tail (an odd MFMA favors the head).
  static void scheduleMFMAWithSpacing(SmallVectorImpl<AnchorInst> &Anchors,
                                      SmallVectorImpl<Instruction *> &MFMAInsts,
                                      unsigned MFMACycles) {
    unsigned MFMAPerGR = llvm::divideCeil(64, MFMACycles);
    unsigned NumGR = 0, NumGRBeforeLR = 0, TotalLDSCycles = 0;
    for (size_t j = 0; j < Anchors.size(); ++j) {
      if (Anchors[j].Kind == SchedKind::GR) {
        NumGR++;
        if (j + 1 < Anchors.size() && Anchors[j + 1].Kind == SchedKind::LR)
          NumGRBeforeLR++;
      } else {
        TotalLDSCycles += getLDSCoverCycles(Anchors[j].I, MFMACycles);
      }
    }
    unsigned Total = MFMAInsts.size();
    unsigned Needed = MFMAPerGR * (NumGR - NumGRBeforeLR) + NumGRBeforeLR +
                      TotalLDSCycles / MFMACycles + 2;
    unsigned Leftover = Total > Needed ? Total - Needed : 0;
    unsigned TailLeftover = Leftover / 2;
    LLVM_DEBUG(dbgs() << "  budget: total=" << Total << " needed=" << Needed
                      << " leftover=" << Leftover << "\n");

    // Walk the anchors backwards, taking MFMAs from the back of the run; what
    // is left unmoved at the front is the head's share of the surplus.
    unsigned MFMAIdx = Total;
    moveMFMAsAfter(MFMAInsts, MFMAIdx, 2 + TailLeftover, Anchors.back().I);
    unsigned LDSAccum = 0;
    for (size_t idx = Anchors.size(); idx-- > 0 && MFMAIdx > 0;) {
      unsigned Count;
      if (Anchors[idx].Kind == SchedKind::GR) {
        bool FollowedByLR =
            idx + 1 < Anchors.size() && Anchors[idx + 1].Kind == SchedKind::LR;
        Count = FollowedByLR ? 1 : MFMAPerGR;
      } else {
        Count = takeMFMAsForLDS(Anchors[idx].I, MFMACycles, LDSAccum);
      }
      moveMFMAsAfter(MFMAInsts, MFMAIdx, Count, Anchors[idx].I);
    }
  }

  // Insert llvm.amdgcn.sched.barrier(0) immediately after AfterI, so the pre-
  // and post-RA machine schedulers cannot move instructions across it.
  static void insertSchedBarrier(Instruction *AfterI) {
    Instruction *Next = AfterI->getNextNode();
    if (!Next)
      return;
    IRBuilder<> Builder(Next);
    Builder.CreateIntrinsic(Intrinsic::amdgcn_sched_barrier,
                            {Builder.getInt32(0)});
  }

  // Schedule the spans of one range; RangeEnd bounds the last span (nullptr
  // for the end of the block).
  static bool scheduleSpans(BasicBlock &BB, const SpanList &Spans,
                            Instruction *RangeEnd) {
    unsigned Scheduled = 0;
    for (unsigned i = 0; i < Spans.size(); ++i) {
      Span R;
      R.BB = &BB;
      R.Begin = Spans[i].SpanStart;
      R.End = i + 1 < Spans.size() ? Spans[i + 1].SpanStart : RangeEnd;

      // Before touching anything: skip a span whose MFMA shape the cycle
      // table does not know, or that has no memory anchor to interleave with.
      // Such spans are left to LLVM's schedulers.
      unsigned MFMACycles = getMFMACycles(*R.Begin);
      bool HasAnchor = llvm::any_of(instructionsInSpan(R), [](auto &I) {
        return isMemoryAnchor(classifySchedInst(I));
      });
      if (MFMACycles == 0 || !HasAnchor)
        continue;

      SpanContents Res = preprocessSpan(R);
      scheduleMFMAWithSpacing(Res.Anchors, Res.MFMAInsts, MFMACycles);

      // Pin the interleave. A fence in front of each anchor makes every window
      // between fences `anchor, mfma...`, with the memory op leading; a fence
      // after each D pin keeps the machine scheduler from reordering pinned
      // MFMAs between anchors, which would leave the pins behind as
      // v_accvgpr copies. Machine scheduling stays enabled everywhere else.
      for (const AnchorInst &A : Res.Anchors)
        if (Instruction *Prev = A.I->getPrevNode())
          insertSchedBarrier(Prev);
      for (Instruction *MFMA : Res.MFMAInsts)
        if (Instruction *DPin = getDPin(MFMA))
          insertSchedBarrier(DPin);
      ++Scheduled;
    }
    return Scheduled > 0;
  }
};

} // namespace

namespace mlir::triton::AMD {

bool runMFMASchedulePass(llvm::Function &F) {
  if (F.isDeclaration())
    return false;

  // A function that already carries sched.barriers is cut at them; the pass
  // never moves anything across one.
  const bool Spanned = hasSchedBarrier(F);
  MFMAScheduler Scheduler;
  bool Changed = false;
  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 64> Snapshot;
    for (Instruction &I : BB)
      Snapshot.push_back(&I);

    bool BlockChanged = false;
    if (!Spanned) {
      BlockChanged = Scheduler.runBlock(BB);
    } else {
      Instruction *RangeBegin = &BB.front();
      for (Instruction &I : BB) {
        if (!isSchedBarrier(I))
          continue;
        if (RangeBegin != &I)
          BlockChanged |= Scheduler.runRange(BB, RangeBegin, &I);
        RangeBegin = I.getNextNode();
      }
      if (RangeBegin)
        BlockChanged |= Scheduler.runRange(BB, RangeBegin, nullptr);
    }
    if (!BlockChanged)
      continue;
    if (verifyFunction(F, /*OS=*/nullptr)) {
      LLVM_DEBUG(dbgs() << "  invalid schedule in " << BB.getName()
                        << ", rolling the block back\n");
      MFMAScheduler::restoreBlock(BB, Snapshot);
      continue;
    }
    Changed = true;
  }
  return Changed;
}

} // namespace mlir::triton::AMD
