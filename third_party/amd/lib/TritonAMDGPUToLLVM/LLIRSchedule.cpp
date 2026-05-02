#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tritonamdgpu-llir-schedule"

namespace {

using namespace llvm;

// Classifications used by the pass

enum class InstClass {
  MFMA,
  WMMA,
  schedBarrier,
  sBarrier,
  sWaitCnt,
  schedGroupBarrier,
  bufferLoadLDS,
  bufferStore,
  OtherIntrinsic,
  Load,
  Store,
  FAdd,
  FMul,
  FMA,
  Branch,
  Other
};

enum class MFMAInputSource { FullyPrefetched, SameRegionLoad, Unknown };

enum class SchedKind { MFMA, GR, LR, LW, CVT, Other };

// Structures used for region analysis/scheduling

struct MFMAStats {
  unsigned Total = 0;
  unsigned SameIter = 0;
  unsigned Prefetched = 0;
  unsigned Mixed = 0;
};

struct AnchorInst {
  Instruction *I = nullptr;
  SchedKind Kind = SchedKind::Other;
};

struct MFMARegionInfo {
  Instruction *Barrier = nullptr; // sched.barrier starting this region
  unsigned TotalMFMA = 0;
  unsigned FullyPrefetchedMFMA = 0;

  bool hasOnlyPrefetchedMFMA() const {
    return (TotalMFMA != 0) && (TotalMFMA == FullyPrefetchedMFMA);
  }
};

using MFMARegionList = SmallVector<MFMARegionInfo, 8>;
using BBMFMAAnalysisMap = DenseMap<const BasicBlock *, MFMARegionList>;
using InstRegionMap = DenseMap<const Instruction *, unsigned>;

struct BBRegion {
  BasicBlock *BB = nullptr;
  Instruction *Begin = nullptr; // First instruction in region (inclusive)
  Instruction *End =
      nullptr; // First instruction of next region or nullptr (exclusive)
};

struct MFMARegionCollectResult {
  SmallVector<Instruction *, 16> Hoist;
  SmallVector<Instruction *, 16> Sink;
  Instruction *LastAnchor = nullptr;
  SmallVector<AnchorInst, 32> Anchors;
  SmallVector<Instruction *, 32> MFMAInsts;
};

// Utilities grouped for clarity

struct Utils {
  static Loop *findMainLoop(LoopInfo &LI) {
    Loop *mainLoop = nullptr;
    size_t maxInsts = 0;

    for (Loop *L : LI) {
      size_t instCount = 0;
      for (BasicBlock *BB : L->blocks())
        instCount += BB->size();
      if (instCount > maxInsts) {
        maxInsts = instCount;
        mainLoop = L;
      }
    }
    return mainLoop;
  }

  static InstClass classifyInstruction(const Instruction &I) {
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (!CI->isInlineAsm()) {
        if (Function *Callee = CI->getCalledFunction()) {
          if (Callee->isIntrinsic()) {
            StringRef Name = Callee->getName();
            if (Name.contains("mfma"))
              return InstClass::MFMA;
            if (Name.contains("wmma"))
              return InstClass::WMMA;
            if (Name.contains("sched.barrier"))
              return InstClass::schedBarrier;
            if (Name.contains("sched.group.barrier"))
              return InstClass::schedGroupBarrier;
            if (Name.contains("s.waitcnt"))
              return InstClass::sWaitCnt;
            if (Name.contains("s.barrier"))
              return InstClass::sBarrier;
            if (Name.contains("llvm.amdgcn.raw.ptr.buffer.load.lds") ||
                Name.contains("llvm.amdgcn.raw.ptr.buffer.load.async.lds"))
              return InstClass::bufferLoadLDS;
            if (Name.contains("llvm.amdgcn.raw.ptr.buffer.store") ||
                Name.contains("tensor.store.from.lds"))
              return InstClass::bufferStore;
            return InstClass::OtherIntrinsic;
          }
        }
      }
      return InstClass::Other;
    }

    if (isa<LoadInst>(I))
      return InstClass::Load;
    if (isa<StoreInst>(I))
      return InstClass::Store;
    if (I.getOpcode() == Instruction::UncondBr ||
        I.getOpcode() == Instruction::CondBr)
      return InstClass::Branch;

    switch (I.getOpcode()) {
    case Instruction::FAdd:
      return InstClass::FAdd;
    case Instruction::FMul:
      return InstClass::FMul;
    default:
      return InstClass::Other;
    }
  }

  static StringRef instClassName(InstClass C) {
    switch (C) {
    case InstClass::MFMA:
      return "MFMA";
    case InstClass::WMMA:
      return "WMMA";
    case InstClass::sBarrier:
      return "s.barrier";
    case InstClass::schedBarrier:
      return "sched.barrier";
    case InstClass::schedGroupBarrier:
      return "sched.group.barrier";
    case InstClass::sWaitCnt:
      return "waitcnt";
    case InstClass::bufferLoadLDS:
      return "buffer_load_lds";
    case InstClass::bufferStore:
      return "buffer_store";
    case InstClass::OtherIntrinsic:
      return "OtherIntrinsic";
    case InstClass::Load:
      return "Load";
    case InstClass::Store:
      return "Store";
    case InstClass::FAdd:
      return "FAdd";
    case InstClass::FMul:
      return "FMul";
    case InstClass::FMA:
      return "FMA";
    case InstClass::Branch:
      return "Branch";
    case InstClass::Other:
      return "Other";
    }
    llvm_unreachable("unknown InstClass");
  }

  static void dumpInstructionHistogram(Loop *L) {
    DenseMap<InstClass, unsigned> hist;

    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        hist[classifyInstruction(I)]++;
      }
    }

    LLVM_DEBUG({
      dbgs() << "=== Pre-RA Main Loop Instruction Histogram ===\n";
      for (auto &KV : hist) {
        dbgs() << "  " << instClassName(KV.first) << ": " << KV.second << "\n";
      }
      dbgs() << "============================================\n";
    });
  }

  static bool isSchedBarrier(const Instruction &I) {
    return InstClass::schedBarrier == classifyInstruction(I);
  }

  static bool isMFMAorWMMA(const Instruction &I) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI || CI->isInlineAsm())
      return false;

    Function *Callee = CI->getCalledFunction();
    if (!Callee || !Callee->isIntrinsic())
      return false;

    StringRef Name = Callee->getName();
    return Name.contains("mfma") || Name.contains("wmma");
  }

  static bool isLDSLoadInst(const Instruction *I) {
    auto *LI = dyn_cast<LoadInst>(I);
    return LI && (LI->getPointerAddressSpace() == 3);
  }

  static bool isHoistTransparentInst(const Instruction &I) {
    return isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I);
  }

  static bool isSinkTransparentInst(const Instruction &I) {
    return isa<ExtractElementInst>(I);
  }

  static SchedKind classifySchedInst(Instruction &I) {
    if (isMFMAorWMMA(I))
      return SchedKind::MFMA;

    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (Function *F = CI->getCalledFunction()) {
        if (F->isIntrinsic()) {
          StringRef Name = F->getName();
          // GR: buffer.load (into regs), buffer.load.lds,
          // buffer.load.async.lds,
          //     tensor.load.to.lds (gfx1250 TDM), tensor.store.from.lds,
          //     raw.ptr.buffer.store (gmem store from regs)
          if (Name.contains("buffer.load") ||
              Name.contains("tensor.load.to.lds") ||
              Name.contains("tensor.store.from.lds") ||
              Name.contains("raw.ptr.buffer.store"))
            return SchedKind::GR;
          // LR: ds_read (ds.read.*) or ds_load (ds.load.*)
          if (Name.contains("ds.read") || Name.contains("ds.load"))
            return SchedKind::LR;
        }
      }
    }

    // LR: load from LDS (addrspace 3)
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (LI->getPointerAddressSpace() == 3)
        return SchedKind::LR;
    }

    // LW: store to LDS (addrspace 3)
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (SI->getPointerAddressSpace() == 3)
        return SchedKind::LW;
    }

    // CVT: fptrunc (f32 -> f16 downcast)
    if (isa<FPTruncInst>(I))
      return SchedKind::CVT;

    return SchedKind::Other;
  }

  static iterator_range<BasicBlock::iterator>
  instructionsInRegion(const BBRegion &R) {
    BasicBlock *BB = R.BB;
    // Begin is now inclusive (region starts at this instruction)
    auto ItBegin = R.Begin ? R.Begin->getIterator() : BB->begin();
    auto ItEnd = R.End ? R.End->getIterator() : BB->end();
    return make_range(ItBegin, ItEnd);
  }

  static Instruction *getLoopHoistInsertPoint(Loop *L) {
    BasicBlock *Header = L->getHeader();
    return &*Header->getFirstInsertionPt();
  }

  static void insertSchedBarrierBefore(Instruction *IP) {
    Function *F = IP->getFunction();
    Module *M = F->getParent();
    Function *BarrierFn =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier);
    IRBuilder<> Builder(F->getContext());
    Builder.SetInsertPoint(IP);
    Value *Zero = Builder.getInt32(0);
    CallInst *CI = Builder.CreateCall(BarrierFn, {Zero});
    CI->setTailCallKind(CallInst::TCK_Tail);
  }

  static bool isGFX12Plus(StringRef Arch) {
    auto family = mlir::triton::AMD::deduceISAFamily(Arch);
    return family == mlir::triton::AMD::ISAFamily::RDNA4 ||
           family == mlir::triton::AMD::ISAFamily::GFX1250;
  }

  static void insertSWaitCntBefore(Instruction *IP, int cnt, StringRef Arch) {
    Function *F = IP->getFunction();
    Module *M = F->getParent();
    IRBuilder<> Builder(F->getContext());
    Builder.SetInsertPoint(IP);

    bool gfx12 = isGFX12Plus(Arch);
    Intrinsic::ID WaitID =
        gfx12 ? Intrinsic::amdgcn_s_wait_dscnt : Intrinsic::amdgcn_s_waitcnt;
    // gfx12+: s_wait_dscnt takes i16, use provided cnt
    // gfx9: s_waitcnt takes i32 49279 (packed vmcnt=15, expcnt=7, lgkmcnt=31)
    Value *Cnt = gfx12 ? Builder.getInt16(cnt) : Builder.getInt32(49279);
    Function *WaitFn = Intrinsic::getOrInsertDeclaration(M, WaitID);
    CallInst *CI = Builder.CreateCall(WaitFn, {Cnt});
    CI->setTailCallKind(CallInst::TCK_Tail);
  }

  static bool containsMFMAAndBufferStore(const BasicBlock *BB) {
    bool HasMFMA = false;
    bool HasStore = false;

    for (const Instruction &I : *BB) {
      auto C = classifyInstruction(I);
      HasMFMA |= (C == InstClass::MFMA || C == InstClass::WMMA);
      HasStore |= (C == InstClass::bufferStore);
      if (HasMFMA && HasStore)
        return true;
    }
    return false;
  }

  static BasicBlock *findEpilogueBlock(Loop *MainLoop, LoopInfo &LI) {
    SmallVector<BasicBlock *, 4> ExitBlocks;
    MainLoop->getExitBlocks(ExitBlocks);

    for (BasicBlock *ExitBB : ExitBlocks) {
      if (MainLoop->contains(ExitBB))
        continue;
      if (containsMFMAAndBufferStore(ExitBB))
        return ExitBB;
    }
    return nullptr;
  }

  static unsigned getMFMACycles(const Instruction &I) {
    if (!isMFMAorWMMA(I))
      return 0;
    const auto *CI = cast<CallInst>(&I);
    const Function *Callee = CI->getCalledFunction();
    if (!Callee)
      return 0;
    StringRef Name = Callee->getName();
    if (Name.contains("mfma.scale.f32.16x16x128.f8f6f4")) {
      // cbsz = operand 3, blgp = operand 4.
      // When cbsz > 1 or blgp > 1, the operand uses a sub-byte format
      // (e.g. e2m1) and the instruction takes 16 cycles.
      // Otherwise (f8 or wider), it takes 32 cycles.
      if (auto *CbszC = dyn_cast<ConstantInt>(CI->getArgOperand(3))) {
        if (auto *BlgpC = dyn_cast<ConstantInt>(CI->getArgOperand(4))) {
          unsigned cbsz = CbszC->getZExtValue();
          unsigned blgp = BlgpC->getZExtValue();
          return (cbsz > 1 || blgp > 1) ? 16 : 32;
        }
      }
      return 32; // Fallback if cbsz/blgp are not constants
    }
    if (Name.contains("mfma.f32.16x16x32.f16"))
      return 16;
    return 0; // Unknown
  }
};

// Region analysis and scheduling logic grouped into a helper class

class PreRAScheduler {
public:
  explicit PreRAScheduler() = default;

  void runOnLoop(Function &F, Loop &MainLoop, LoopInfo &LI, StringRef Arch) {
    LLVM_DEBUG(dbgs() << "Pre-RA scheduler analyzing function: " << F.getName()
                      << "\n");
    Utils::dumpInstructionHistogram(&MainLoop);

    BBMFMAAnalysisMap BBMFMAMap;

    for (BasicBlock *BB : MainLoop.blocks()) {
      LLVM_DEBUG(dbgs() << "BB: " << BB->getName() << "\n");
      analyzeBBMFMA(*BB, BBMFMAMap);
      scheduleBB(*BB, BBMFMAMap, Arch);
    }

    LLVM_DEBUG(dbgs() << "============================================\n");

    if (BasicBlock *epilogue = Utils::findEpilogueBlock(&MainLoop, LI)) {
      LLVM_DEBUG(dbgs() << "Found epilogue block: " << epilogue->getName()
                        << "\n");
      analyzeBBMFMA(*epilogue, BBMFMAMap);
      scheduleEpilogue(*epilogue, BBMFMAMap);
    }
  }

private:
  // Helper: trace back from MFMA operands (0 and 1) to find the first
  // shuffle or insert instruction that prepares the input
  static Instruction *findMFMAInputPrep(CallInst *MFMA) {
    if (!MFMA || MFMA->arg_size() < 2)
      return nullptr;

    SmallPtrSet<Value *, 16> Visited;
    SmallVector<Value *, 8> Worklist;

    // Only check operands 0 and 1 as instructed
    Worklist.push_back(MFMA->getArgOperand(0));
    Worklist.push_back(MFMA->getArgOperand(1));

    Instruction *FirstPrep = nullptr;

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      auto *I = dyn_cast<Instruction>(V);
      if (!I)
        continue;

      // Found a shuffle or insert - candidate for region start
      if (isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I)) {
        // Keep the earliest one (in program order)
        if (!FirstPrep || I->comesBefore(FirstPrep))
          FirstPrep = I;
        // Continue tracing to find earlier ones
        for (Value *Op : I->operands())
          Worklist.push_back(Op);
      } else if (isa<ExtractElementInst>(I)) {
        // Trace through extract as well
        for (Value *Op : I->operands())
          Worklist.push_back(Op);
      }
    }

    return FirstPrep;
  }

  // Automatic region detection based on MFMA + memory operation patterns
  static unsigned assignRegions(BasicBlock &BB, InstRegionMap &RegionMap) {
    unsigned CurRegion = 0;
    bool SeenMemoryOps = false;
    Instruction *RegionStart = nullptr;

    for (Instruction &I : BB) {
      // Check if this is a memory operation (GR, LR, or LW) or a CVT.
      // Treating CVT as a region-boundary anchor matches the v9-style
      // sliced-WMMA epilogue: each sliced WMMA gets its own region, where
      // the region contains: wmma -> cvt(prev-output) -> store(prev-output).
      auto SK = Utils::classifySchedInst(I);
      if (SK == SchedKind::GR || SK == SchedKind::LR || SK == SchedKind::LW ||
          SK == SchedKind::CVT) {
        SeenMemoryOps = true;
      }

      // Check if this is an MFMA/WMMA
      if (Utils::isMFMAorWMMA(I)) {
        // If we've seen memory ops and already have a region, start a new one
        if (SeenMemoryOps && RegionStart != nullptr) {
          CurRegion++;
          SeenMemoryOps = false;
          RegionStart = nullptr;
        }

        // If this is the first MFMA in the current region, find its prep
        // instructions
        if (RegionStart == nullptr) {
          RegionStart = findMFMAInputPrep(cast<CallInst>(&I));
          if (!RegionStart)
            RegionStart = &I; // Fallback: use MFMA itself as region start
        }
      }

      // Assign current instruction to the current region
      if (RegionStart != nullptr) {
        RegionMap[&I] = CurRegion;
      }
    }

    return CurRegion; // number of regions (0-indexed, so actual count is
                      // CurRegion + 1 if any)
  }

  static MFMAInputSource
  traceMFMAOperandSource(Value *StartV, unsigned MFMARegion,
                         const InstRegionMap &RegionMap) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;
    Worklist.push_back(StartV);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      if (isa<PHINode>(V))
        return MFMAInputSource::FullyPrefetched;

      auto *I = dyn_cast<Instruction>(V);
      if (!I)
        continue;

      if (Utils::isLDSLoadInst(I)) {
        auto It = RegionMap.find(I);
        if (It == RegionMap.end())
          return MFMAInputSource::FullyPrefetched;
        unsigned LoadRegion = It->second;
        return (LoadRegion == MFMARegion) ? MFMAInputSource::SameRegionLoad
                                          : MFMAInputSource::FullyPrefetched;
      }

      if (isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I) ||
          isa<ExtractElementInst>(I)) {
        for (Value *Op : I->operands())
          Worklist.push_back(Op);
      }
    }
    return MFMAInputSource::Unknown;
  }

  static void analyzeBBMFMA(BasicBlock &BB, BBMFMAAnalysisMap &Out) {
    InstRegionMap RegionMap;
    unsigned MaxRegion = assignRegions(BB, RegionMap);

    // If no regions detected, return early
    if (RegionMap.empty())
      return;

    MFMARegionList Regions;
    Regions.resize(MaxRegion + 1);

    // Identify region start instructions (barriers)
    // Track which instruction starts each region
    DenseMap<unsigned, Instruction *> RegionStarts;

    for (auto &Entry : RegionMap) {
      const Instruction *I = Entry.first;
      unsigned RegionID = Entry.second;

      // Find the first instruction in each region to use as the "barrier"
      if (RegionStarts.find(RegionID) == RegionStarts.end()) {
        RegionStarts[RegionID] = const_cast<Instruction *>(I);
      } else {
        // Keep the earliest instruction
        if (I->comesBefore(RegionStarts[RegionID]))
          RegionStarts[RegionID] = const_cast<Instruction *>(I);
      }
    }

    // Set the barriers in the region list
    for (auto &Entry : RegionStarts) {
      unsigned RegionID = Entry.first;
      Instruction *StartInst = Entry.second;
      if (RegionID < Regions.size()) {
        Regions[RegionID].Barrier = StartInst;
      }
    }

    // Count MFMA and analyze their inputs
    for (Instruction &I : BB) {
      if (!Utils::isMFMAorWMMA(I))
        continue;

      auto It = RegionMap.find(&I);
      if (It == RegionMap.end())
        continue;

      unsigned R = It->second;
      Regions[R].TotalMFMA++;

      auto *CI = cast<CallInst>(&I);
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);

      auto S0 = traceMFMAOperandSource(Op0, R, RegionMap);
      auto S1 = traceMFMAOperandSource(Op1, R, RegionMap);

      if (S0 != MFMAInputSource::SameRegionLoad &&
          S1 != MFMAInputSource::SameRegionLoad) {
        Regions[R].FullyPrefetchedMFMA++;
      }
    }

    Out[&BB] = std::move(Regions);

    // print info
    for (unsigned i = 0; i < Out[&BB].size(); ++i) {
      const MFMARegionInfo &R = Out[&BB][i];
      if (!R.Barrier)
        continue;

      LLVM_DEBUG(dbgs() << "Region " << i << ": total MFMA: " << R.TotalMFMA
                        << ", fully prefetch: " << R.FullyPrefetchedMFMA
                        << "\n");
    }
  }

  static bool feedsMFMA(Instruction *I) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;

    Worklist.push_back(I);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      for (User *U : V->users()) {
        if (auto *UI = dyn_cast<Instruction>(U)) {
          if (Utils::isMFMAorWMMA(*UI))
            return true;
          if (Utils::isHoistTransparentInst(*UI))
            Worklist.push_back(UI);
        }
      }
    }
    return false;
  }

  static bool definedByMFMA(Instruction *I) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;

    Worklist.push_back(I);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      if (auto *DefI = dyn_cast<Instruction>(V)) {
        if (Utils::isMFMAorWMMA(*DefI))
          return true;

        if (Utils::isSinkTransparentInst(*DefI)) {
          for (Value *Op : DefI->operands())
            Worklist.push_back(Op);
        }
      }
    }
    return false;
  }

  static MFMARegionCollectResult
  collectMFMAAndTransparentInstsInRegion(const BBRegion &R) {
    MFMARegionCollectResult Res;

    // Collect LR instructions in this region so we can check if a shuffle
    // operand is a same-region LDS load.
    SmallPtrSet<Instruction *, 16> RegionLRInsts;
    for (Instruction &I : Utils::instructionsInRegion(R)) {
      SchedKind K = Utils::classifySchedInst(I);
      if (K == SchedKind::LR)
        RegionLRInsts.insert(&I);
    }

    for (Instruction &I : Utils::instructionsInRegion(R)) {
      SchedKind K = Utils::classifySchedInst(I);
      if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW ||
          K == SchedKind::CVT) {
        Res.LastAnchor = &I;
        Res.Anchors.push_back({&I, K});
        continue;
      }

      if (K == SchedKind::MFMA) {
        Res.MFMAInsts.push_back(&I);
        continue;
      }

      if (Utils::isHoistTransparentInst(I)) {
        // Don't hoist shuffles whose operands are LR instructions in this
        // region — hoisting would move the shuffle before its LDS load
        // operand, breaking dominance.
        bool consumesSameRegionLR = false;
        if (isa<ShuffleVectorInst>(I)) {
          for (Value *Op : I.operands()) {
            if (auto *OpI = dyn_cast<Instruction>(Op)) {
              if (RegionLRInsts.count(OpI)) {
                consumesSameRegionLR = true;
                break;
              }
            }
          }
        }
        if (!consumesSameRegionLR && feedsMFMA(&I))
          Res.Hoist.push_back(&I);
        continue;
      }

      if (auto *EI = dyn_cast<ExtractElementInst>(&I)) {
        if (definedByMFMA(&I))
          Res.Sink.push_back(&I);
        (void)EI; // silence unused in release builds
      }
    }

    return Res;
  }

  static MFMARegionCollectResult
  preprocessMFMAInstsInRegion(const BBRegion &R) {
    auto Res = collectMFMAAndTransparentInstsInRegion(R);

    if (Res.Hoist.empty() && Res.Sink.empty())
      return Res;

    Instruction *HoistPos =
        R.Begin; // Region start (shuffle/insert feeding MFMA)
    Instruction *SinkPos = Res.LastAnchor; // last anchor in region

    for (Instruction *I : llvm::reverse(Res.Hoist)) {
      // Don't hoist R.Begin after itself
      if (I != HoistPos)
        I->moveAfter(HoistPos);
    }

    for (Instruction *I : llvm::reverse(Res.Sink))
      I->moveAfter(SinkPos);

    return Res;
  }

  static StringRef schedKindName(SchedKind K) {
    switch (K) {
    case SchedKind::GR:
      return "GR";
    case SchedKind::LR:
      return "LR";
    case SchedKind::LW:
      return "LW";
    case SchedKind::CVT:
      return "CVT";
    case SchedKind::MFMA:
      return "mfma";
    case SchedKind::Other:
      return "other";
    }
    llvm_unreachable("unknown SchedKind");
  }

  // Move LR instructions (and s.waitcnt/s.barrier between LW and LR)
  // from their position between LW and GR to right after the 2nd-to-last GR.
  // This places ds_write (LW) after the ds_read (LR) chunk in each region,
  // so MFMA can hide the long ds_write latency (up to 400 cycles due to
  // LDS port contention with buffer_load_to_lds).
  static void moveAnchors(SmallVectorImpl<AnchorInst> &Anchors,
                          const BBRegion &Region) {
    // Find the range of instructions between the last LW and the first GR.
    // This includes LR, s.waitcnt, and s.barrier instructions.
    // Find the last LW first (needed to filter GR after it)
    // Find the last LW and the first GR *after* the last LW
    Instruction *LastLW = nullptr;
    Instruction *FirstGRAfterLW = nullptr;
    for (auto &A : Anchors) {
      if (A.Kind == SchedKind::LW)
        LastLW = A.I;
    }
    if (!LastLW)
      return;
    bool pastLW = false;
    for (auto &A : Anchors) {
      if (A.I == LastLW) {
        pastLW = true;
        continue;
      }
      if (pastLW && A.Kind == SchedKind::GR) {
        FirstGRAfterLW = A.I;
        break;
      }
    }
    Instruction *FirstGR = FirstGRAfterLW;

    if (!FirstGR)
      return;

    // Collect GR anchors after the last LW (for determining 2nd-to-last target)
    SmallVector<AnchorInst> GR;
    bool afterLW = false;
    for (auto &A : Anchors) {
      if (A.I == LastLW) {
        afterLW = true;
        continue;
      }
      if (afterLW && A.Kind == SchedKind::GR)
        GR.push_back(A);
    }

    if (GR.size() < 2)
      return; // Need at least 2 GR after LW to have a 2nd-to-last

    // Collect LR instructions between LastLW and FirstGR, plus their
    // immediate users (bitcast etc.), s.waitcnt, and s.barrier.
    // Do NOT move make.buffer.rsrc or other GR dependencies.
    SmallVector<Instruction *, 16> ToMove;
    SmallPtrSet<Instruction *, 16> ToMoveSet;
    bool inRange = false;
    for (Instruction &I : Utils::instructionsInRegion(Region)) {
      if (&I == LastLW) {
        inRange = true;
        continue;
      }
      if (&I == FirstGR)
        break;
      if (!inRange)
        continue;

      SchedKind K = Utils::classifySchedInst(I);
      InstClass C = Utils::classifyInstruction(I);
      if (K == SchedKind::LR || C == InstClass::sWaitCnt ||
          C == InstClass::sBarrier || Utils::isSchedBarrier(I)) {
        ToMove.push_back(&I);
        ToMoveSet.insert(&I);
      }
    }
    // Also collect immediate users of moved LR instructions that are
    // in the range (e.g., bitcast of ds_read_tr results)
    SmallVector<Instruction *, 8> ExtraUsers;
    for (Instruction *I : ToMove) {
      if (Utils::classifySchedInst(*I) != SchedKind::LR)
        continue;
      for (User *U : I->users()) {
        if (auto *UI = dyn_cast<Instruction>(U)) {
          if (!ToMoveSet.count(UI)) {
            ExtraUsers.push_back(UI);
            ToMoveSet.insert(UI);
          }
        }
      }
    }
    // Insert extra users right after their defining LR in the move list
    for (Instruction *EU : ExtraUsers) {
      // Find the position: right after the LR that defines it
      for (size_t j = 0; j < ToMove.size(); ++j) {
        if (ToMove[j] == cast<Instruction>(EU->getOperand(0))) {
          ToMove.insert(ToMove.begin() + j + 1, EU);
          break;
        }
      }
    }

    if (ToMove.empty())
      return;

    // Move them right after the 2nd-to-last GR
    Instruction *InsertAfter = GR[GR.size() - 2].I;
    LLVM_DEBUG(
        dbgs() << "  Moving " << ToMove.size()
               << " instructions (LR/waitcnt/barrier) after 2nd-to-last GR\n");

    for (Instruction *I : ToMove) {
      I->moveAfter(InsertAfter);
      InsertAfter = I; // chain them in order
    }

    // Rebuild anchor list in program order
    Anchors.clear();
    for (Instruction &I : Utils::instructionsInRegion(Region)) {
      SchedKind K = Utils::classifySchedInst(I);
      if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW)
        Anchors.push_back({&I, K});
    }

    LLVM_DEBUG({
      dbgs() << "  New anchor order:";
      for (auto &A : Anchors)
        dbgs() << " " << schedKindName(A.Kind);
      dbgs() << "\n";
    });
  }

  // Helper: move N MFMAs after InsertPt using moveAfter.
  // moveAfter naturally produces correct order: each new MFMA goes right
  // after InsertPt, pushing previous ones further away.
  // Result: InsertPt, MFMA[N-K], ..., MFMA[N-2], MFMA[N-1]
  static unsigned moveMFMAsAfter(SmallVectorImpl<Instruction *> &MFMAInsts,
                                 unsigned &MFMAIdx, unsigned Count,
                                 Instruction *InsertPt) {
    unsigned moved = 0;
    for (unsigned j = 0; j < Count && MFMAIdx > 0; ++j) {
      MFMAInsts[--MFMAIdx]->moveAfter(InsertPt);
      moved++;
    }
    return moved;
  }

  // Interleave MFMA with anchor instructions using moveAfter.
  //
  // Step 1: Count needed MFMAs: 4 per GR, 1 per LR, 1 per LW.
  //         leftover = total_mfma - needed
  // Step 2: Process anchors in reverse, inserting MFMAs after each:
  //   - GR: 4 mfma after it
  //   - LR: 1 mfma after it
  //   - LW: first LW seen (reverse) → leftover mfma after it
  //          subsequent LW           → 1 mfma after it
  //   ~1 mfma remains at the front of the region.
  static void scheduleMFMAWithSpacing(SmallVectorImpl<AnchorInst> &Anchors,
                                      SmallVectorImpl<Instruction *> &MFMAInsts,
                                      const BBRegion &Region, StringRef Arch) {
    if (Anchors.empty())
      return;

    // Move LR/waitcnt/barrier from between LW and GR to after 2nd-to-last GR
    moveAnchors(Anchors, Region);

    if (MFMAInsts.empty())
      return;

    unsigned MFMAIdx = MFMAInsts.size();
    unsigned Total = MFMAIdx;

    bool isGFX12 = Utils::isGFX12Plus(Arch);

    // Insert s.waitcnt before the first MFMA in the region (gfx9 only).
    // On gfx1250, the LLVM backend inserts per-load s_wait_dscnt with
    // specific counts. No blanket dscnt is needed from the scheduler.
    if (!isGFX12)
      Utils::insertSWaitCntBefore(MFMAInsts.front(), 0, Arch);

    // Count anchors by kind
    unsigned numGR = 0, numLR = 0, numLW = 0, numGRBeforeLR = 0;
    for (size_t j = 0; j < Anchors.size(); ++j) {
      if (Anchors[j].Kind == SchedKind::GR) {
        numGR++;
        if (j + 1 < Anchors.size() && Anchors[j + 1].Kind == SchedKind::LR)
          numGRBeforeLR++;
      } else if (Anchors[j].Kind == SchedKind::LR) {
        numLR++;
      } else if (Anchors[j].Kind == SchedKind::LW) {
        numLW++;
      }
    }

    if (isGFX12) {
      // GFX1250 scheduling:
      //   1. Reserve 2 WMMAs at the beginning of the region
      //   2. Insert remaining WMMAs at end of region (after last anchor)
      //   3. Process anchors in reverse:
      //      - LR: 1 WMMA before every 2nd LR
      //      - GR (TDM): 1 WMMA before it
      //      - Anchor transition (kind changes): 2 extra WMMAs before it
      //
      // Count anchor transitions: when consecutive anchors have different kinds
      unsigned numTransitions = 0;
      for (size_t j = 0; j + 1 < Anchors.size(); ++j) {
        if (Anchors[j].Kind != Anchors[j + 1].Kind)
          numTransitions++;
      }

      unsigned frontBudget = 2;
      unsigned lrBudget = numLR / 2;
      unsigned grBudget = 1 * numGR;
      unsigned transitionBudget = 2 * numTransitions;
      unsigned needed = frontBudget + lrBudget + grBudget + transitionBudget;
      unsigned remaining = (Total > needed) ? Total - needed : 0;

      LLVM_DEBUG(dbgs() << "  GFX12 budget: total=" << Total
                        << " numLR=" << numLR << " numGR=" << numGR
                        << " numTransitions=" << numTransitions
                        << " front=" << frontBudget << " lrBudget=" << lrBudget
                        << " grBudget=" << grBudget
                        << " transitionBudget=" << transitionBudget
                        << " remaining=" << remaining << "\n");

      // Insert remaining WMMAs at end of region
      moveMFMAsAfter(MFMAInsts, MFMAIdx, remaining, Anchors.back().I);

      // Process anchors in reverse
      unsigned lrCount = 0;
      for (int i = static_cast<int>(Anchors.size()) - 1; i >= 0 && MFMAIdx > 0;
           --i) {
        size_t idx = static_cast<size_t>(i);
        Instruction *InsertPt = Anchors[idx].I;
        SchedKind Kind = Anchors[idx].Kind;

        // Check if the previous anchor (in program order, i.e. idx-1) has a
        // different kind — this anchor is at a transition boundary.
        bool isTransition = (idx > 0 && Anchors[idx - 1].Kind != Kind);

        if (Kind == SchedKind::LR) {
          lrCount++;
          // Count WMMAs to insert before this LR:
          // - 1 WMMA for every 2nd LR
          // - 2 extra WMMAs at anchor transition (e.g. GR->LR boundary)
          unsigned count = 0;
          if (lrCount % 2 == 0)
            count += 1;
          if (isTransition)
            count += 2;
          if (count > 0) {
            Instruction *BeforeLR = InsertPt->getPrevNode();
            if (BeforeLR)
              moveMFMAsAfter(MFMAInsts, MFMAIdx, count, BeforeLR);
          }
        } else if (Kind == SchedKind::GR) {
          // 1 WMMA before TDM in program order
          Instruction *BeforeGR = InsertPt->getPrevNode();
          if (BeforeGR)
            moveMFMAsAfter(MFMAInsts, MFMAIdx, 1, BeforeGR);
        }
      }

      LLVM_DEBUG(dbgs() << "  GFX12 done: " << MFMAIdx
                        << " WMMAs remaining at front\n");
    } else {
      // gfx9 scheduling:
      //   GR: mfmaPerGR MFMAs each (except GR→LR gets 1)
      //   LR: 1 MFMA each
      //   LW: first LW gets leftover, rest get 1
      //   2 MFMAs at end, ~1 at front
      unsigned mfmaPerGR = 4;
      {
        unsigned cycles = Utils::getMFMACycles(*MFMAInsts.front());
        if (cycles == 32)
          mfmaPerGR = 2;
      }

      unsigned lrBudget = numLR;
      unsigned grBudget = mfmaPerGR * (numGR - numGRBeforeLR);
      unsigned needed = grBudget + numGRBeforeLR + lrBudget + numLW + 2;
      unsigned leftover = (Total > needed) ? Total - needed : 0;

      LLVM_DEBUG(dbgs() << "  MFMA budget: total=" << Total << ", needed="
                        << needed << ", leftover=" << leftover << "\n");

      unsigned MFMAAtEnd =
          moveMFMAsAfter(MFMAInsts, MFMAIdx, 2, Anchors.back().I);
      bool seenLW = false;
      DenseMap<SchedKind, unsigned> MFMAPerAnchorKind;

      for (int i = static_cast<int>(Anchors.size()) - 1; i >= 0 && MFMAIdx > 0;
           --i) {
        size_t idx = static_cast<size_t>(i);
        Instruction *InsertPt = Anchors[idx].I;
        SchedKind Kind = Anchors[idx].Kind;

        unsigned Count = 0;
        if (Kind == SchedKind::LR) {
          Count = 1;
        } else if (Kind == SchedKind::GR) {
          bool followedByLR = (idx + 1 < Anchors.size() &&
                               Anchors[idx + 1].Kind == SchedKind::LR);
          Count = followedByLR ? 1 : mfmaPerGR;
        } else if (Kind == SchedKind::LW) {
          if (!seenLW) {
            seenLW = true;
            Count = leftover;
          } else {
            Count = 1;
          }
        }

        unsigned before = MFMAIdx;
        moveMFMAsAfter(MFMAInsts, MFMAIdx, Count, InsertPt);
        MFMAPerAnchorKind[Kind] += before - MFMAIdx;
      }

      LLVM_DEBUG({
        dbgs() << "  MFMA insertion summary: total=" << Total
               << ", at_front=" << MFMAIdx << ", at_end=" << MFMAAtEnd;
        for (auto &KV : MFMAPerAnchorKind) {
          dbgs() << ", after_" << schedKindName(KV.first) << "=" << KV.second;
        }
        dbgs() << "\n";
      });
    }
  }

  // Insert an inline asm comment before the given instruction.
  static void insertAsmComment(Instruction *IP, const std::string &Comment) {
    LLVMContext &Ctx = IP->getContext();
    IRBuilder<> Builder(Ctx);
    Builder.SetInsertPoint(IP);
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), false);
    InlineAsm *IA =
        InlineAsm::get(FTy, ";; " + Comment, "", /*hasSideEffects=*/true);
    Builder.CreateCall(IA);
  }

  static void scheduleBB(BasicBlock &BB, const BBMFMAAnalysisMap &Analysis,
                         StringRef Arch) {
    auto It = Analysis.find(&BB);
    if (It == Analysis.end())
      return;

    const MFMARegionList &Regions = It->second;

    unsigned NumRegions = Regions.size();
    unsigned ScheduledRegionIdx = 0;

    for (unsigned i = 0; i < NumRegions; ++i) {
      const MFMARegionInfo &R = Regions[i];
      if (!R.Barrier)
        continue;

      if (R.hasOnlyPrefetchedMFMA()) {
        BBRegion bbR;
        bbR.BB = &BB;
        bbR.Begin = Regions[i].Barrier;
        bbR.End = (i + 1 < NumRegions) ? Regions[i + 1].Barrier : nullptr;

        MFMARegionCollectResult Res = preprocessMFMAInstsInRegion(bbR);

        // --- Build region comment ---
        std::string Comment;
        raw_string_ostream OS(Comment);

        // Count anchors by kind
        unsigned numGR = 0, numLR = 0;
        for (auto &A : Res.Anchors) {
          if (A.Kind == SchedKind::GR)
            numGR++;
          else if (A.Kind == SchedKind::LR)
            numLR++;
        }
        OS << "Region " << ScheduledRegionIdx << ": " << Res.MFMAInsts.size()
           << " wmma, " << numGR << " GR, " << numLR << " LR";
        ScheduledRegionIdx++;

        insertAsmComment(bbR.Begin, Comment);

        LLVM_DEBUG({
          dbgs() << "Cluster " << i << " structure:";
          SchedKind RunKind = SchedKind::Other;
          unsigned RunCount = 0;
          for (Instruction &Inst : Utils::instructionsInRegion(bbR)) {
            SchedKind K = Utils::classifySchedInst(Inst);
            if (K != SchedKind::MFMA && K != SchedKind::GR &&
                K != SchedKind::LR && K != SchedKind::LW)
              continue;
            if (K == RunKind) {
              RunCount++;
            } else {
              if (RunCount > 0)
                dbgs() << " " << RunCount << " " << schedKindName(RunKind);
              RunKind = K;
              RunCount = 1;
            }
          }
          if (RunCount > 0)
            dbgs() << " " << RunCount << " " << schedKindName(RunKind);
          dbgs() << "\n";
        });

        scheduleMFMAWithSpacing(Res.Anchors, Res.MFMAInsts, bbR, Arch);
      }
    }
  }

  // Schedule WMMAs in an epilogue region.
  // Strategy:
  //   1. 1 WMMA before every 4 CVT
  //   2. 1 WMMA before every 2 LR or 2 LW
  //   3. 1 WMMA before every GR
  //   4. Remaining WMMAs go to the beginning of the region
  // If budget is insufficient, prioritize: CVT > LR/LW > GR
  static void scheduleEpilogueRegion(SmallVectorImpl<AnchorInst> &Anchors,
                                     SmallVectorImpl<Instruction *> &MFMAInsts,
                                     BBRegion &R) {
    if (MFMAInsts.empty() || Anchors.empty())
      return;

    unsigned MFMAIdx = MFMAInsts.size();
    unsigned Total = MFMAIdx;

    // Count anchors by kind
    unsigned numCVT = 0, numLR = 0, numLW = 0, numGR = 0;
    for (auto &A : Anchors) {
      if (A.Kind == SchedKind::CVT)
        numCVT++;
      else if (A.Kind == SchedKind::LR)
        numLR++;
      else if (A.Kind == SchedKind::LW)
        numLW++;
      else if (A.Kind == SchedKind::GR)
        numGR++;
    }

    // Budget calculation with priority: CVT > LR/LW > GR
    unsigned cvtBudget = numCVT / 4;
    unsigned lrBudget = numLR / 2;
    unsigned lwBudget = numLW / 2;
    unsigned grBudget = numGR;

    // Clamp budgets if total is insufficient (prioritize CVT first)
    unsigned needed = cvtBudget + lrBudget + lwBudget + grBudget;
    if (needed > Total) {
      unsigned avail = Total;
      // CVT first
      cvtBudget = std::min(cvtBudget, avail);
      avail -= cvtBudget;
      // LR/LW next
      unsigned lrlwBudget = lrBudget + lwBudget;
      if (lrlwBudget > avail) {
        // Scale down proportionally
        lrBudget = avail * lrBudget / (lrBudget + lwBudget + 1);
        lwBudget = avail - lrBudget;
      }
      avail -= (lrBudget + lwBudget);
      // GR last
      grBudget = std::min(grBudget, avail);
    }
    unsigned remaining =
        Total - std::min(Total, cvtBudget + lrBudget + lwBudget + grBudget);

    LLVM_DEBUG(dbgs() << "  Epilogue schedule: total=" << Total
                      << " cvtBudget=" << cvtBudget << " lrBudget=" << lrBudget
                      << " lwBudget=" << lwBudget << " grBudget=" << grBudget
                      << " remaining=" << remaining << "\n");

    // Place remaining WMMAs at end of region (after last anchor)
    moveMFMAsAfter(MFMAInsts, MFMAIdx, remaining, Anchors.back().I);

    // Process anchors in reverse
    unsigned cvtCount = 0, lrCount = 0, lwCount = 0, grCount = 0;
    for (int i = static_cast<int>(Anchors.size()) - 1; i >= 0 && MFMAIdx > 0;
         --i) {
      size_t idx = static_cast<size_t>(i);
      Instruction *InsertPt = Anchors[idx].I;
      SchedKind Kind = Anchors[idx].Kind;

      unsigned count = 0;
      if (Kind == SchedKind::CVT) {
        cvtCount++;
        if (cvtCount % 4 == 0)
          count = 1;
      } else if (Kind == SchedKind::LR) {
        lrCount++;
        if (lrCount % 2 == 0)
          count = 1;
      } else if (Kind == SchedKind::LW) {
        lwCount++;
        if (lwCount % 2 == 0)
          count = 1;
      } else if (Kind == SchedKind::GR) {
        // Emit 4 WMMAs before every 4th GR (clump the 1:1 GR:WMMA ratio).
        grCount++;
        if (grCount % 4 == 0)
          count = 4;
      }

      if (count > 0) {
        Instruction *Before = InsertPt->getPrevNode();
        if (Before)
          moveMFMAsAfter(MFMAInsts, MFMAIdx, count, Before);
      }
    }

    LLVM_DEBUG(dbgs() << "  Epilogue done: " << MFMAIdx << " WMMAs at front\n");
  }

  // Check if a CVT instruction's input traces back to any WMMA in the given
  // set, walking through shufflevector/extractelement/insertelement
  // intermediaries.
  static bool cvtDependsOnMFMASet(Instruction *CvtInst,
                                  const SmallPtrSetImpl<Instruction *> &MFMAs) {
    SmallVector<Instruction *, 8> Worklist;
    for (Value *Op : CvtInst->operands()) {
      if (auto *OpI = dyn_cast<Instruction>(Op))
        Worklist.push_back(OpI);
    }
    while (!Worklist.empty()) {
      Instruction *Cur = Worklist.pop_back_val();
      if (Utils::isMFMAorWMMA(*Cur)) {
        if (MFMAs.count(Cur))
          return true;
      } else if (isa<ShuffleVectorInst>(Cur) || isa<ExtractElementInst>(Cur) ||
                 isa<InsertElementInst>(Cur)) {
        for (Value *Op : Cur->operands()) {
          if (auto *OpI = dyn_cast<Instruction>(Op))
            Worklist.push_back(OpI);
        }
      }
    }
    return false;
  }

  // Emit an epilogue region comment, debug info, and schedule a sub-region.
  static void emitEpilogueRegionInfo(BBRegion &bbR,
                                     MFMARegionCollectResult &Res,
                                     unsigned &EpiRegionIdx) {
    std::string Comment;
    raw_string_ostream OS(Comment);

    unsigned numGR = 0, numLR = 0, numLW = 0, numCVT = 0;
    for (auto &A : Res.Anchors) {
      if (A.Kind == SchedKind::GR)
        numGR++;
      else if (A.Kind == SchedKind::LR)
        numLR++;
      else if (A.Kind == SchedKind::LW)
        numLW++;
      else if (A.Kind == SchedKind::CVT)
        numCVT++;
    }
    OS << "Epilogue Region " << EpiRegionIdx << ": " << Res.MFMAInsts.size()
       << " wmma, " << numGR << " GR, " << numLR << " LR, " << numLW << " LW, "
       << numCVT << " CVT";
    EpiRegionIdx++;

    insertAsmComment(bbR.Begin, Comment);

    LLVM_DEBUG({
      dbgs() << "  " << OS.str() << "\n";
      dbgs() << "  Structure:";
      SchedKind RunKind = SchedKind::Other;
      unsigned RunCount = 0;
      for (Instruction &Inst : Utils::instructionsInRegion(bbR)) {
        SchedKind K = Utils::classifySchedInst(Inst);
        if (K != SchedKind::MFMA && K != SchedKind::GR && K != SchedKind::LR &&
            K != SchedKind::LW && K != SchedKind::CVT)
          continue;
        if (K == RunKind) {
          RunCount++;
        } else {
          if (RunCount > 0)
            dbgs() << " " << RunCount << " " << schedKindName(RunKind);
          RunKind = K;
          RunCount = 1;
        }
      }
      if (RunCount > 0)
        dbgs() << " " << RunCount << " " << schedKindName(RunKind);
      dbgs() << "\n";
    });

    // Schedule WMMAs in this epilogue region
    scheduleEpilogueRegion(Res.Anchors, Res.MFMAInsts, bbR);
  }

  static void scheduleEpilogue(BasicBlock &BB,
                               const BBMFMAAnalysisMap &Analysis) {
    auto It = Analysis.find(&BB);
    if (It == Analysis.end())
      return;

    const MFMARegionList &Regions = It->second;

    unsigned NumRegions = Regions.size();
    unsigned EpiRegionIdx = 0;

    for (unsigned i = 0; i < NumRegions; ++i) {
      const MFMARegionInfo &R = Regions[i];
      if (!R.Barrier)
        continue;

      Instruction *RegionBegin = Regions[i].Barrier;
      Instruction *RegionEnd =
          (i + 1 < NumRegions) ? Regions[i + 1].Barrier : nullptr;

      // Scan through the barrier-delimited region, splitting at CVT
      // instructions whose inputs come from WMMAs in the current sub-region.
      Instruction *SubRegionBegin = RegionBegin;
      SmallPtrSet<Instruction *, 32> CurrentMFMAs;

      auto ItBegin = SubRegionBegin->getIterator();
      auto ItEnd = RegionEnd ? RegionEnd->getIterator() : BB.end();

      for (auto It = ItBegin; It != ItEnd; ++It) {
        Instruction &Inst = *It;
        SchedKind K = Utils::classifySchedInst(Inst);

        if (K == SchedKind::CVT && cvtDependsOnMFMASet(&Inst, CurrentMFMAs)) {
          // End the current sub-region before this CVT
          BBRegion bbR;
          bbR.BB = &BB;
          bbR.Begin = SubRegionBegin;
          bbR.End = &Inst;

          MFMARegionCollectResult Res = preprocessMFMAInstsInRegion(bbR);
          emitEpilogueRegionInfo(bbR, Res, EpiRegionIdx);

          // Start new sub-region from this CVT
          SubRegionBegin = &Inst;
          CurrentMFMAs.clear();
        }

        if (K == SchedKind::MFMA)
          CurrentMFMAs.insert(&Inst);
      }

      // Emit the last sub-region
      BBRegion bbR;
      bbR.BB = &BB;
      bbR.Begin = SubRegionBegin;
      bbR.End = RegionEnd;

      MFMARegionCollectResult Res = preprocessMFMAInstsInRegion(bbR);
      emitEpilogueRegionInfo(bbR, Res, EpiRegionIdx);
    }
  }
};

// Pass wrapper

struct LLIRSchedulePass : FunctionPass {
  static char ID;
  PreRAScheduler Scheduler;
  std::string Arch;

  LLIRSchedulePass(StringRef Arch = "") : FunctionPass(ID), Arch(Arch.str()) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesAll();
  }

  bool runOnFunction(Function &F) override {
    if (F.isDeclaration())
      return false;

    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    if (LI.empty())
      return false;

    Loop *mainLoop = Utils::findMainLoop(LI);
    if (!mainLoop)
      return false;

    Scheduler.runOnLoop(F, *mainLoop, LI, Arch);
    // Analysis-only pass
    return false;
  }
};

} // end anonymous namespace

char LLIRSchedulePass::ID = 0;

namespace mlir::triton::AMD {

void runLLIRSchedulePass(llvm::Function &F, llvm::StringRef arch) {
  llvm::legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(new LLIRSchedulePass(arch));
  FPM.doInitialization();
  FPM.run(F);
  FPM.doFinalization();

  if (llvm::verifyFunction(F, &llvm::errs())) {
    llvm::errs() << "LLIR schedule pass produced invalid IR!\n";
    assert(false && "expected function to verify successfully");
  }
}

} // namespace mlir::triton::AMD
