// Add a coarse-grained scheduler to help llvm on instruction scheduling.
// This pass currently implements one sched opt: moves LDS loads as close as
// possible to their consuming MFMA instructions. For cases with high register
// pressure, this will reduce live range of dot A & B operand. For cases with
// low register pressure, this will interleave lds with mfma, hiding latency.
#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Pass.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

using namespace llvm;

namespace {

struct CoarseSched : public FunctionPass {
  static char ID;
  CoarseSched() : FunctionPass(ID) {}

  bool isLocalLoad(Instruction *I) {
    if (auto *LI = dyn_cast<LoadInst>(I)) {
      return LI->getPointerAddressSpace() == 3;
    }
    return false;
  }

  bool isMMA(Instruction *I) {
    if (auto *CI = dyn_cast<CallInst>(I)) {
      if (Function *F = CI->getCalledFunction()) {
        StringRef name = F->getName();
        if (name.contains("mfma")) {
          return true;
        }
      }
    }
    return false;
  }

  bool isBarrier(Instruction *I) {
    if (auto *CI = dyn_cast<CallInst>(I)) {
      if (Function *F = CI->getCalledFunction()) {
        StringRef name = F->getName();
        if (name == "llvm.amdgcn.s.barrier" ||
            name == "llvm.amdgcn.sched.barrier") {
          return true;
        }
      }
    }
    return false;
  }

  SmallVector<SmallVector<Instruction *>, 4>
  splitIntoBarrierRegions(BasicBlock &BB) {
    SmallVector<SmallVector<Instruction *>, 4> Regions;
    SmallVector<Instruction *> Current;
    for (Instruction &I : BB) {
      if (isBarrier(&I)) {
        if (!Current.empty()) {
          Regions.push_back(std::move(Current));
          Current.clear();
        }
        continue;
      }
      Current.push_back(&I);
    }
    if (!Current.empty())
      Regions.push_back(std::move(Current));
    return Regions;
  }

  bool runOnFunction(Function &F) override {
    bool changed = false;
    std::unordered_set<Instruction *> moved;

    for (auto &BB : F) {
      auto Regions = splitIntoBarrierRegions(BB);

      for (auto &Region : Regions) {
        // Collect MMAs
        SmallVector<Instruction *> MMAs;
        for (Instruction *I : Region) {
          if (isMMA(I))
            MMAs.push_back(I);
        }
        if (MMAs.empty())
          continue;
        std::unordered_set<Instruction *> RegionSet(Region.begin(),
                                                    Region.end());

        for (Instruction *MMA : MMAs) {
          SmallVector<Instruction *> workList;
          if (auto I = dyn_cast<Instruction>(MMA->getOperand(0)))
            workList.push_back(I);
          if (auto I = dyn_cast<Instruction>(MMA->getOperand(1)))
            workList.push_back(I);

          // Backtrace to get all the local loads that can reach mma
          std::unordered_set<Instruction *> backTrace;
          SmallVector<Instruction *> localLoads;
          while (!workList.empty()) {
            Instruction *curr = workList.pop_back_val();
            if (backTrace.count(curr) || !RegionSet.count(curr) ||
                moved.count(curr))
              continue;
            backTrace.insert(curr);
            if (isLocalLoad(curr)) {
              localLoads.push_back(curr);
              continue;
            }
            for (Value *Op : curr->operand_values()) {
              if (Instruction *opInst = dyn_cast<Instruction>(Op))
                workList.push_back(opInst);
            }
          }
          // Forward trace to get all load users
          std::unordered_set<Instruction *> loadUsers;
          while (!localLoads.empty()) {
            Instruction *curr = localLoads.pop_back_val();
            if (loadUsers.count(curr) || !RegionSet.count(curr))
              continue;
            loadUsers.insert(curr);
            for (User *U : curr->users()) {
              if (Instruction *UInst = dyn_cast<Instruction>(U))
                localLoads.push_back(UInst);
            }
          }
          // Move instructions in original program order
          for (Instruction *I : Region) {
            if (I == MMA)
              break;
            if (loadUsers.count(I) && !moved.count(I)) {
              I->moveBefore(MMA->getIterator());
              moved.insert(I);
              changed = true;
            }
          }
        }
      }
    }

    return changed;
  }
};
} // end anonymous namespace

char CoarseSched::ID = 0;

namespace mlir::triton::AMD {
void runCoarseSched(Function &F) {
  CoarseSched pass;
  pass.runOnFunction(F);
}
} // namespace mlir::triton::AMD
