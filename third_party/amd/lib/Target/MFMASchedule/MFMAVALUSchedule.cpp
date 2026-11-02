//===- MFMAVALUSchedule.cpp - MFMA <-> VALU co-execution for gfx950 -------===//
//
// An LLVM-IR pass that declares, with llvm.amdgcn.sched.group.barrier, how the
// MFMAs of a warp-pipeline stage interleave with the stage's VALU and
// transcendental instructions, so the vector work issues in the shadow of the
// matrix instructions (the softmax of an attention kernel next to its dot).
// It runs on the optimized IR, before instruction selection, on the stages
// ConvertWarpPipeline marked for it (warp_pipeline_stage(schedule_hint=
// "mfma-valu")). A marker is an llvm.amdgcn.sched.group.barrier with mask 0
// and size 0 at the head of the stage; the stage extends to the next
// llvm.amdgcn.sched.barrier. The pass erases the marker.
//
// The pass never moves or rewrites an instruction. A declaration only adds
// edges to the machine scheduler's DAG (AMDGPUIGroupLP), which refuses an edge
// that would form a cycle, so it is binding where legal and ignored where not:
// the generated code is correct whatever the pass emits, and what the pass has
// to get right is the quality of the declaration. Four steps per stage:
//
//  1. Inventory. Every instruction is an MFMA, a VALU or a TRANS instruction
//     with its issue cycles, or something that issues nothing. The counts must
//     match what the machine scheduler will see: a packed f32 op is one
//     instruction (but two slots of window, since SIPreEmitPeephole unpacks it
//     in an MFMA shadow), the inner max/min of a chain folds into v_maximum3,
//     fneg and fabs become source modifiers. The ConvertWarpPipeline check
//     guarantees that no instruction of the stage depends on the stage's own
//     MFMAs, so every one of them can be placed in any window.
//  2. Depth. For every instruction, the number of transcendentals on the
//     longest in-stage dependency path ending at it. Instructions are grouped
//     into blocks by (depth, class): the VALU that depends on nothing
//     transcendental, then the exp2s, then the VALU that consumes them, and so
//     on. That is a topological order of class-homogeneous blocks, so a
//     declaration in block order is always satisfiable, while the program
//     order (sub, exp, sub, exp, ...) would ask for one group per alternation.
//  3. Pack. Each block is cut into windows of one MFMA's co-execution
//     capacity, balanced so that no window overflows. When the stage's vector
//     work fits the capacity, the surplus MFMAs go just before the last block,
//     which gives the transcendental results time to land before the VALU
//     that consumes them. When it does not fit, what cannot be hidden is
//     declared without an MFMA in front of it, and the packed ops are the ones
//     left exposed: covering one spends two slots of window to hide a single
//     issue.
//  4. Emit. One sched.group.barrier sequence at the end of the stage, in block
//     order, [MFMA 1][VALU n] or [MFMA 1][TRANS n] per window. IGroupLP fills
//     the groups bottom-up from the barrier, so the declaration has to sit
//     after every instruction of the stage.
//
//===----------------------------------------------------------------------===//

#include "MFMAVALUSchedule.h"
#include "MFMAScheduleUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <climits>
#include <functional>

#define DEBUG_TYPE "tritonamdgpu-mfma-valu-schedule"

namespace {

using namespace llvm;
using namespace mlir::triton::AMD::MFMAScheduleUtils;

// IGroupLP scheduling-group masks (AMDGPUIGroupLP.cpp, SchedGroupMask). TRANS
// is a class of its own: the VALU mask does not match transcendentals.
constexpr uint32_t kMaskVALU = 0x002;
constexpr uint32_t kMaskMFMA = 0x008;
constexpr uint32_t kMaskTRANS = 0x400;

// The co-execution model, validated on gfx950. An MFMA occupies the matrix
// unit for its issue cycles (getMFMACycles); the first kMFMAIssueCycles of
// them overlap nothing, the rest are a window in which vector instructions
// issue: 24 cycles behind a 32-cycle MFMA, 8 behind a 16-cycle one. A plain
// VALU instruction takes one 4-cycle slot of the window, a packed f32 op two
// (SIPreEmitPeephole unpacks it in the shadow), a transcendental 8 cycles, a
// permlane 20. The backend's own shadow budget (SIPreEmitPeephole:
// ReleaseAtCycle - 1 units at one unit per instruction) agrees with these
// window sizes.
constexpr int kMFMAIssueCycles = 8;
constexpr int kSlotCycles = 4;
constexpr int kTransCycles = 8;
constexpr int kPermlaneCycles = 20;
constexpr int kMinWindowCycles = 4;

// A stage needs at least this many MFMAs for a declaration to be worth it.
constexpr unsigned kMinMFMAs = 2;

enum class OpClass { VALU, TRANS };

// One vector instruction of a stage, as the declaration prices it.
struct SchedOp {
  Instruction *I = nullptr;
  OpClass Class = OpClass::VALU;
  int Cycles = 0;
  // Transcendentals on the longest in-stage dependency path ending here,
  // this instruction included.
  int Depth = 0;
  // A packed f32 fmul/fadd/fsub/fma: two slots when covered by an MFMA, one
  // issue when exposed.
  bool Packed = false;
};

// Everything the declaration of one stage is computed from.
struct StageInventory {
  SmallVector<Instruction *, 32> MFMAs;
  SmallVector<SchedOp, 64> Ops; // program order
  unsigned MinMFMACycles = 0;

  // Co-execution cycles behind one MFMA, sized for the shortest MFMA of the
  // stage so that no window overfills.
  int windowCycles() const {
    return std::max<int>(kMinWindowCycles,
                         (int)MinMFMACycles - kMFMAIssueCycles);
  }
  int totalCycles() const {
    int Total = 0;
    for (const SchedOp &Op : Ops)
      Total += Op.Cycles;
    return Total;
  }
};

StringRef intrinsicName(const Instruction &I) {
  if (const auto *CI = dyn_cast<CallInst>(&I))
    if (const Function *F = CI->getCalledFunction())
      if (F->isIntrinsic())
        return F->getName();
  return {};
}

// The marker ConvertWarpPipeline leaves at the head of a hinted stage.
bool isStageMarker(const Instruction &I) {
  const auto *CI = dyn_cast<CallInst>(&I);
  if (!CI)
    return false;
  const Function *F = CI->getCalledFunction();
  if (!F || F->getIntrinsicID() != Intrinsic::amdgcn_sched_group_barrier)
    return false;
  const auto *Mask = dyn_cast<ConstantInt>(CI->getArgOperand(0));
  const auto *Size = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  return Mask && Size && Mask->isZero() && Size->isZero();
}

// v_exp, v_log, v_rcp, ...: issued to the TRANS unit, which IGroupLP's VALU
// mask does not match.
bool isTransOp(const Instruction &I) {
  StringRef N = intrinsicName(I);
  return N.contains("exp2") || N.contains("exp.") || N.contains("log2") ||
         N.contains("sin") || N.contains("cos") || N.contains("sqrt") ||
         N.contains("rcp");
}

// fmaximum/fminimum/maxnum/minnum: the ops isel folds pairwise into
// v_maximum3 / v_minimum3.
bool isMaxMin(const Instruction &I) {
  StringRef N = intrinsicName(I);
  return N.contains("maximum") || N.contains("minimum") ||
         N.contains("maxnum") || N.contains("minnum");
}

bool isPermlane(const Instruction &I) {
  return intrinsicName(I).contains("permlane");
}

// A packed f32 fmul/fadd/fsub/fma: one v_pk_* instruction.
bool isPackedF32Op(const Instruction &I) {
  auto *VT = dyn_cast<FixedVectorType>(I.getType());
  if (!VT || VT->getNumElements() < 2 || !VT->getElementType()->isFloatTy())
    return false;
  if (isa<BinaryOperator>(I)) {
    unsigned Op = I.getOpcode();
    return Op == Instruction::FMul || Op == Instruction::FAdd ||
           Op == Instruction::FSub;
  }
  if (const auto *II = dyn_cast<IntrinsicInst>(&I)) {
    Intrinsic::ID ID = II->getIntrinsicID();
    return ID == Intrinsic::fmuladd || ID == Intrinsic::fma;
  }
  return false;
}

// Marks the max/min instructions the backend absorbs into a v_maximum3 /
// v_minimum3, which therefore issue nothing. Isel folds a chain pairwise and
// bottom-up, each fold taking one single-use max/min operand; walking the
// stage in program order and letting each max/min absorb at most one
// not-yet-committed single-use max/min operand reproduces that count (checked
// against the assembly of an attention softmax: 33 LLVM max/min, 17 issued).
void computeMax3Folds(Instruction *Begin, BasicBlock::iterator End,
                      const SmallPtrSetImpl<const Instruction *> &InStage,
                      SmallPtrSetImpl<const Instruction *> &Folded) {
  SmallPtrSet<const Instruction *, 32> Committed;
  for (auto It = Begin->getIterator(); It != End; ++It) {
    if (!isMaxMin(*It))
      continue;
    for (Value *Op : It->operands()) {
      auto *In = dyn_cast<Instruction>(Op);
      if (In && isMaxMin(*In) && InStage.count(In) && In->hasOneUse() &&
          !Folded.count(In) && !Committed.count(In)) {
        Folded.insert(In);
        Committed.insert(&*It);
        break;
      }
    }
  }
}

// Issue cycles of a vector instruction, and its class; 0 for an instruction
// that issues nothing, or nothing this pass models (it then floats, and
// IGroupLP places it where its dependencies allow).
int opCycles(const Instruction &I, OpClass &Class, bool &Packed) {
  Class = OpClass::VALU;
  Packed = false;
  if (isMFMAorWMMA(I))
    return 0;
  if (isTransOp(I)) {
    Class = OpClass::TRANS;
    return kTransCycles;
  }
  if (isPermlane(I))
    return kPermlaneCycles;
  const bool Vector = I.getType()->isVectorTy();
  if (!intrinsicName(I).empty()) {
    // max/min (the inner ops of a chain are folded away by computeMax3Folds)
    // and fma. fabs is a source modifier, like fneg below.
    StringRef N = intrinsicName(I);
    if (isMaxMin(I) || N.contains("fmuladd") || N.contains("fma.")) {
      Packed = isPackedF32Op(I);
      return Vector ? 2 * kSlotCycles : kSlotCycles;
    }
    return 0;
  }
  // fneg is never an instruction on AMDGPU: it folds into its consumer as a
  // source modifier. Counting it would declare an instruction that does not
  // exist.
  if (I.getOpcode() == Instruction::FNeg)
    return 0;
  // A packed convert is one v_cvt_pk instruction, one slot.
  if (isa<FPTruncInst>(I) || isa<FPExtInst>(I))
    return kSlotCycles;
  if (I.getType()->isFPOrFPVectorTy() &&
      (isa<BinaryOperator>(I) || isa<UnaryOperator>(I) || isa<SelectInst>(I))) {
    Packed = isPackedF32Op(I);
    return Vector ? 2 * kSlotCycles : kSlotCycles;
  }
  return 0;
}

// Depth of every priced instruction: the transcendentals on the longest
// in-stage dependency path ending at it, itself included. Loop-carried values
// and values from other stages are roots.
void computeDepths(StageInventory &Inv,
                   const SmallPtrSetImpl<const Instruction *> &InStage) {
  DenseMap<const Instruction *, int> Memo;
  std::function<int(const Instruction *)> depthOf =
      [&](const Instruction *I) -> int {
    auto Found = Memo.find(I);
    if (Found != Memo.end())
      return Found->second;
    int D = 0;
    if (!isa<PHINode>(I))
      for (const Value *V : I->operands())
        if (const auto *J = dyn_cast<Instruction>(V); J && InStage.count(J))
          D = std::max(D, depthOf(J));
    D += isTransOp(*I) ? 1 : 0;
    Memo[I] = D;
    return D;
  };
  for (SchedOp &Op : Inv.Ops)
    Op.Depth = depthOf(Op.I);
}

// Returns false when the stage is not one this pass declares: too few MFMAs,
// no vector work, an MFMA shape the cycle table does not know, or a memory
// access (the ConvertWarpPipeline check rejects such stages, so one here came
// from an O3 motion and the stage is left alone).
bool takeInventory(Instruction *Begin, Instruction *End, StageInventory &Inv) {
  BasicBlock *BB = Begin->getParent();
  auto ItEnd = End ? End->getIterator() : BB->end();
  SmallPtrSet<const Instruction *, 32> InStage;
  for (auto It = Begin->getIterator(); It != ItEnd; ++It)
    InStage.insert(&*It);
  SmallPtrSet<const Instruction *, 32> Folded;
  computeMax3Folds(Begin, ItEnd, InStage, Folded);

  for (auto It = Begin->getIterator(); It != ItEnd; ++It) {
    Instruction &I = *It;
    if (isMFMAorWMMA(I)) {
      unsigned C = getMFMACycles(I);
      if (C == 0) {
        LLVM_DEBUG(dbgs() << "  unmodelled MFMA, stage left alone: " << I
                          << "\n");
        return false;
      }
      Inv.MinMFMACycles =
          Inv.MinMFMACycles ? std::min(Inv.MinMFMACycles, C) : C;
      Inv.MFMAs.push_back(&I);
      continue;
    }
    if (isMemoryAnchor(classifySchedInst(I))) {
      LLVM_DEBUG(dbgs() << "  memory access in a co-exec stage, stage left "
                           "alone: "
                        << I << "\n");
      return false;
    }
    if (Folded.count(&I))
      continue;
    SchedOp Op;
    Op.I = &I;
    Op.Cycles = opCycles(I, Op.Class, Op.Packed);
    if (Op.Cycles > 0)
      Inv.Ops.push_back(Op);
  }
  if (Inv.MFMAs.size() < kMinMFMAs || Inv.Ops.empty())
    return false;
  computeDepths(Inv, InStage);
  return true;
}

// A class-homogeneous block of the stage's vector work, in declaration order.
struct Block {
  int Depth = 0;
  OpClass Class = OpClass::VALU;
  SmallVector<int, 32> Cycles; // per instruction, program order
};

// Blocks ordered by (depth, class), transcendentals before the VALU of the
// same depth (that VALU consumes them). Depth 0 holds VALU only.
SmallVector<Block, 4> buildBlocks(const StageInventory &Inv) {
  SmallVector<Block, 4> Blocks;
  for (const SchedOp &Op : Inv.Ops) {
    Block *B = nullptr;
    for (Block &Cand : Blocks)
      if (Cand.Depth == Op.Depth && Cand.Class == Op.Class)
        B = &Cand;
    if (!B) {
      Blocks.push_back({Op.Depth, Op.Class, {}});
      B = &Blocks.back();
    }
    B->Cycles.push_back(Op.Cycles);
  }
  llvm::stable_sort(Blocks, [](const Block &A, const Block &B) {
    if (A.Depth != B.Depth)
      return A.Depth < B.Depth;
    return A.Class == OpClass::TRANS && B.Class == OpClass::VALU;
  });
  return Blocks;
}

// Cut one block into groups of at most `Budget` cycles. Balanced, not
// first-fit: the block takes the minimum number of groups its cycle total
// needs, g = ceil(total / budget), and each group aims at the average of what
// is left over the groups still to come. That self-corrects after a wide op (a
// permlane forces a short group; the following groups take a little more) and
// lands on exactly g groups instead of fragmenting, while first-fit leaves the
// block's tail in a stub that wastes a window. Returns the instruction count
// of each group.
SmallVector<int, 32> packGroups(ArrayRef<int> Cycles, int Budget) {
  SmallVector<int, 32> Sizes;
  if (Cycles.empty())
    return Sizes;
  int Total = 0;
  for (int C : Cycles)
    Total += C;
  int G = std::max(1, (Total + Budget - 1) / Budget);
  int RemCyc = Total, RemG = G, Cur = 0, N = 0;
  for (int C : Cycles) {
    int Target = RemG > 0 ? (RemCyc + RemG - 1) / RemG : Budget;
    Target = std::min(Target, Budget);
    if (N > 0 && (Cur + C > Budget || (Cur >= Target && RemG > 1))) {
      Sizes.push_back(N);
      RemCyc -= Cur;
      if (RemG > 1)
        --RemG;
      Cur = 0;
      N = 0;
    }
    Cur += C;
    ++N;
  }
  if (N)
    Sizes.push_back(N);
  return Sizes;
}

class Declaration {
public:
  Declaration(Instruction *InsertPt, int SyncID)
      : B(InsertPt), SyncID(SyncID) {}

  void group(uint32_t Mask, int Size) {
    B.CreateIntrinsic(Intrinsic::amdgcn_sched_group_barrier,
                      {B.getInt32(Mask), B.getInt32(Size), B.getInt32(SyncID)});
  }
  void window(uint32_t Mask, int Size) {
    group(kMaskMFMA, 1);
    group(Mask, Size);
  }
  void bareMFMAs(int N) {
    for (int K = 0; K < N; ++K)
      group(kMaskMFMA, 1);
  }

private:
  IRBuilder<> B;
  int SyncID;
};

uint32_t maskOf(OpClass C) {
  return C == OpClass::TRANS ? kMaskTRANS : kMaskVALU;
}

// The stage's vector work fits the MFMA capacity: pack every block into
// windows, give each window an MFMA, and put the surplus MFMAs just before
// the last block. An unfilled window is free, but having it first delays
// every co-issued instruction behind it, while in front of the last block it
// gives the transcendental results time to land before their consumers.
void declareFitting(const StageInventory &Inv, ArrayRef<Block> Blocks,
                    int Window, Instruction *InsertPt, int SyncID) {
  const int M = Inv.MFMAs.size();
  SmallVector<SmallVector<int, 32>, 4> Groups;
  int Total = 0;
  for (const Block &Blk : Blocks) {
    Groups.push_back(packGroups(Blk.Cycles, Window));
    Total += Groups.back().size();
  }
  // More groups than MFMAs (fragmentation around a wide op): merge the
  // cheapest adjacent pair until they fit, so the excess lands in one or two
  // over-full windows instead of widening every window.
  while (Total > M) {
    size_t BI = 0, GI = 0;
    int Best = INT_MAX;
    for (size_t Bi = 0; Bi < Groups.size(); ++Bi)
      for (size_t Gi = 0; Gi + 1 < Groups[Bi].size(); ++Gi) {
        int Cost = Groups[Bi][Gi] + Groups[Bi][Gi + 1];
        if (Cost < Best) {
          Best = Cost;
          BI = Bi;
          GI = Gi;
        }
      }
    if (Best == INT_MAX)
      break;
    Groups[BI][GI] += Groups[BI][GI + 1];
    Groups[BI].erase(Groups[BI].begin() + GI + 1);
    --Total;
  }
  const int Bare = std::max(0, M - Total);

  Declaration D(InsertPt, SyncID);
  for (size_t Bi = 0; Bi < Blocks.size(); ++Bi) {
    if (Bi + 1 == Blocks.size())
      D.bareMFMAs(Bare);
    for (int N : Groups[Bi])
      D.window(maskOf(Blocks[Bi].Class), N);
  }

  LLVM_DEBUG({
    dbgs() << "  [mfma-valu] sync=" << SyncID << " M=" << M
           << " window=" << Window << "cyc total=" << Inv.totalCycles()
           << "cyc fits; blocks:";
    for (size_t Bi = 0; Bi < Blocks.size(); ++Bi) {
      dbgs() << " " << (Blocks[Bi].Class == OpClass::TRANS ? "T" : "V")
             << Blocks[Bi].Depth << "{";
      for (int N : Groups[Bi])
        dbgs() << N << " ";
      dbgs() << "}";
    }
    dbgs() << " bare=" << Bare << "\n";
  });
}

// The stage's vector work exceeds the MFMA capacity: no arrangement hides all
// of it, and the questions are which instructions get a window and where the
// rest sit. An instruction that cannot be packed (exp2, v_maximum3, v_cvt_pk,
// permlane) hides one issue per slot it takes, a packed op only half of one,
// so the unpackable ones get first claim on the windows and the packable ones
// are covered from the end of the stage backwards with whatever capacity is
// left; the exposed remainder then stays contiguous and where it already sits.
// The declaration keeps program order here: an exposed op is declared in a
// group with no MFMA in front of it. Returns false when the stage fits after
// all (the caller then packs it).
bool declareOverCapacity(const StageInventory &Inv, int Window,
                         Instruction *InsertPt, int SyncID) {
  const int M = Inv.MFMAs.size();
  const int Cap = Window * M;
  int Total = 0, Fixed = 0;
  for (const SchedOp &Op : Inv.Ops) {
    Total += Op.Cycles;
    if (!Op.Packed)
      Fixed += Op.Cycles;
  }
  if (Total <= Cap)
    return false;

  // A window may hold several groups of different classes: [MFMA 1][VALU 1]
  // [TRANS 1] asks for one MFMA with a sub and an exp2 behind it.
  struct Chunk {
    OpClass Class;
    int Cycles, N;
  };
  struct Slot {
    bool HasMFMA; // false: exposed, declared with no MFMA in front
    int Cycles;
    SmallVector<Chunk, 4> Chunks;
  };
  SmallVector<bool, 64> Covered(Inv.Ops.size(), false);

  // Decide coverage for a packable-op budget, then lay the stage out in
  // slots. Returns the number of windows the layout needs.
  auto decideAndLayout = [&](int Avail, SmallVectorImpl<Slot> &Out) {
    int Bought = 0;
    for (size_t K = 0; K < Inv.Ops.size(); ++K)
      Covered[K] = !Inv.Ops[K].Packed;
    for (size_t K = Inv.Ops.size(); K-- > 0;) {
      const SchedOp &Op = Inv.Ops[K];
      if (Op.Packed && Bought + Op.Cycles <= Avail) {
        Covered[K] = true;
        Bought += Op.Cycles;
      }
    }
    Out.clear();
    int Windows = 0;
    for (size_t K = 0; K < Inv.Ops.size(); ++K) {
      const SchedOp &Op = Inv.Ops[K];
      const bool WantMFMA = Covered[K];
      Slot *S = Out.empty() ? nullptr : &Out.back();
      const bool Fits = S && S->HasMFMA == WantMFMA &&
                        (!WantMFMA || S->Cycles + Op.Cycles <= Window);
      if (!Fits) {
        Out.push_back({WantMFMA, 0, {}});
        S = &Out.back();
        if (WantMFMA)
          ++Windows;
      }
      S->Cycles += Op.Cycles;
      if (!S->Chunks.empty() && S->Chunks.back().Class == Op.Class) {
        S->Chunks.back().Cycles += Op.Cycles;
        S->Chunks.back().N += 1;
      } else {
        S->Chunks.push_back({Op.Class, Op.Cycles, 1});
      }
    }
    return Windows;
  };

  // A window freed by tighter packing is capacity the packable ops can still
  // use, so feed it back into the budget a few times.
  int Avail = Cap - Fixed;
  SmallVector<Slot, 40> Slots;
  int Windows = decideAndLayout(Avail, Slots);
  for (int Iter = 0; Iter < 4 && Windows < M; ++Iter) {
    int Grown = Avail + (M - Windows) * Window;
    SmallVector<Slot, 40> Trial;
    int TW = decideAndLayout(Grown, Trial);
    if (TW > M)
      break;
    Avail = Grown;
    Slots = std::move(Trial);
    if (TW == Windows)
      break;
    Windows = TW;
  }
  Windows = decideAndLayout(Avail, Slots);

  // More windows than MFMAs: merge the cheapest adjacent pair of covered
  // slots, so the excess lands in one deliberately over-full window instead
  // of costing the last groups their windows.
  while (Windows > M) {
    size_t BI = Slots.size();
    int Best = INT_MAX;
    for (size_t K = 0; K + 1 < Slots.size(); ++K)
      if (Slots[K].HasMFMA && Slots[K + 1].HasMFMA) {
        int Cost = Slots[K].Cycles + Slots[K + 1].Cycles;
        if (Cost < Best) {
          Best = Cost;
          BI = K;
        }
      }
    if (BI == Slots.size())
      break;
    Slots[BI].Cycles += Slots[BI + 1].Cycles;
    for (const Chunk &C : Slots[BI + 1].Chunks)
      if (!Slots[BI].Chunks.empty() &&
          Slots[BI].Chunks.back().Class == C.Class) {
        Slots[BI].Chunks.back().Cycles += C.Cycles;
        Slots[BI].Chunks.back().N += C.N;
      } else {
        Slots[BI].Chunks.push_back(C);
      }
    Slots.erase(Slots.begin() + BI + 1);
    --Windows;
  }

  const int Bare = std::max(0, M - Windows);
  Declaration D(InsertPt, SyncID);
  for (size_t K = 0; K < Slots.size(); ++K) {
    const Slot &S = Slots[K];
    if (K + 1 == Slots.size())
      D.bareMFMAs(Bare);
    if (S.HasMFMA)
      D.group(kMaskMFMA, 1);
    for (const Chunk &C : S.Chunks)
      D.group(maskOf(C.Class), C.N);
  }

  LLVM_DEBUG({
    dbgs() << "  [mfma-valu] sync=" << SyncID << " M=" << M
           << " window=" << Window << "cyc total=" << Total << "cyc cap=" << Cap
           << "cyc over capacity; slots:";
    for (const Slot &S : Slots) {
      dbgs() << " " << (S.HasMFMA ? "[" : "*[");
      bool First = true;
      for (const Chunk &C : S.Chunks) {
        dbgs() << (First ? "" : "+") << (C.Class == OpClass::TRANS ? "T" : "V")
               << C.N;
        First = false;
      }
      dbgs() << "]" << S.Cycles;
    }
    dbgs() << " (* = exposed) bare=" << Bare << "\n";
  });
  return true;
}

// Declare the co-execution schedule of the stage [Begin, End); End == nullptr
// means the end of the block. Returns true iff a declaration was emitted.
bool declareStage(Instruction *Begin, Instruction *End, int SyncID) {
  StageInventory Inv;
  if (!takeInventory(Begin, End, Inv))
    return false;
  const int Window = Inv.windowCycles();
  const int M = Inv.MFMAs.size();
  Instruction *InsertPt = End ? End : Begin->getParent()->getTerminator();
  if (!InsertPt)
    return false;
  if (Inv.totalCycles() > Window * M &&
      declareOverCapacity(Inv, Window, InsertPt, SyncID))
    return true;
  declareFitting(Inv, buildBlocks(Inv), Window, InsertPt, SyncID);
  return true;
}

} // namespace

namespace mlir::triton::AMD {

bool runMFMAVALUSchedulePass(llvm::Function &F) {
  if (F.isDeclaration())
    return false;
  SmallVector<Instruction *, 8> Markers;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (isStageMarker(I))
        Markers.push_back(&I);
  if (Markers.empty())
    return false;

  int SyncID = 1;
  for (Instruction *Marker : Markers) {
    Instruction *Begin = Marker->getNextNode();
    Instruction *End = nullptr;
    for (Instruction *I = Begin; I; I = I->getNextNode())
      if (isSchedBarrier(*I)) {
        End = I;
        break;
      }
    Marker->eraseFromParent();
    if (!Begin || Begin == End)
      continue;
    LLVM_DEBUG(dbgs() << "[mfma-valu] stage " << SyncID << " in "
                      << Begin->getParent()->getName() << "\n");
    declareStage(Begin, End, SyncID);
    ++SyncID;
  }
  return true;
}

} // namespace mlir::triton::AMD
