//===----------------------------------------------------------------------===//
// Remove unused lanes of packed FP32 arithmetic before NVPTX code generation.
//===----------------------------------------------------------------------===//
#include "LLVMPasses.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

// Only ordinary, lane-wise arithmetic is safe to split. In particular, do not
// split constrained FP intrinsics or look through freeze or arbitrary calls.
static bool isSupported(Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FNeg:
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
    return true;
  default:
    auto *II = dyn_cast<IntrinsicInst>(&I);
    return II && !II->hasOperandBundles() && !II->isMustTailCall() &&
           (II->getIntrinsicID() == Intrinsic::fma ||
            II->getIntrinsicID() == Intrinsic::fmuladd);
  }
}

static APInt getDemandedElts(Instruction &I) {
  unsigned Width = cast<FixedVectorType>(I.getType())->getNumElements();
  APInt Demanded(Width, 0);
  for (Use &U : I.uses()) {
    if (auto *Extract = dyn_cast<ExtractElementInst>(U.getUser())) {
      auto *Index = dyn_cast<ConstantInt>(Extract->getIndexOperand());
      if (!Index || Index->getValue().uge(Width))
        return APInt::getAllOnes(Width);
      Demanded.setBit(Index->getZExtValue());
    } else if (auto *Shuffle = dyn_cast<ShuffleVectorInst>(U.getUser())) {
      unsigned Begin = U.getOperandNo() * Width;
      for (int Index : Shuffle->getShuffleMask())
        if (Index >= 0 && unsigned(Index) >= Begin &&
            unsigned(Index) < Begin + Width)
          Demanded.setBit(Index - Begin);
    } else {
      return APInt::getAllOnes(Width);
    }
  }
  return Demanded;
}

// Resolve routing instructions while extracting. This exposes multiply inputs
// to the scalar add/sub operations, preserving opportunities for contraction.
static Value *extractLane(IRBuilder<NoFolder> &Builder, Value *V,
                          unsigned Lane) {
  while (true) {
    if (auto *Shuffle = dyn_cast<ShuffleVectorInst>(V)) {
      int Index = Shuffle->getMaskValue(Lane);
      if (Index < 0)
        return PoisonValue::get(Builder.getFloatTy());
      unsigned Width = cast<FixedVectorType>(Shuffle->getOperand(0)->getType())
                           ->getNumElements();
      V = Shuffle->getOperand(Index / Width);
      Lane = Index % Width;
    } else if (auto *Insert = dyn_cast<InsertElementInst>(V)) {
      auto *Index = dyn_cast<ConstantInt>(Insert->getOperand(2));
      if (!Index)
        break;
      unsigned Width = cast<FixedVectorType>(V->getType())->getNumElements();
      if (Index->getValue().uge(Width))
        return PoisonValue::get(Builder.getFloatTy());
      if (Index->equalsInt(Lane))
        return Insert->getOperand(1);
      V = Insert->getOperand(0);
    } else {
      break;
    }
  }
  return Builder.CreateExtractElement(V, Builder.getInt64(Lane));
}

static Value *createOperation(IRBuilder<NoFolder> &Builder, Instruction &I,
                              ArrayRef<Value *> Operands) {
  Value *Result;
  if (auto *II = dyn_cast<IntrinsicInst>(&I))
    Result = Builder.CreateIntrinsic(
        II->getIntrinsicID(), {Operands.front()->getType()}, Operands, &I);
  else if (I.getOpcode() == Instruction::FNeg)
    Result = Builder.CreateFNeg(Operands.front());
  else
    Result = Builder.CreateBinOp(Instruction::BinaryOps(I.getOpcode()),
                                 Operands[0], Operands[1]);
  if (auto *NewI = dyn_cast<Instruction>(Result)) {
    NewI->copyIRFlags(&I);
    NewI->copyMetadata(I);
    if (auto *Call = dyn_cast<CallInst>(NewI)) {
      Call->setCallingConv(cast<CallInst>(I).getCallingConv());
      Call->setAttributes(cast<CallInst>(I).getAttributes());
    }
  }
  return Result;
}

static bool scalarizePackedFOp(Instruction &I,
                               SmallPtrSetImpl<Instruction *> &ScalarAddSubs,
                               bool ScalarizeAll = false) {
  auto *Ty = dyn_cast<FixedVectorType>(I.getType());
  if (!Ty || !Ty->getElementType()->isFloatTy() || Ty->getNumElements() < 2 ||
      Ty->getNumElements() % 2 || !isSupported(I))
    return false;

  APInt Demanded = getDemandedElts(I);
  unsigned Width = Ty->getNumElements();
  bool HasSingleLane = false;
  for (unsigned Lane = 0; Lane < Width; Lane += 2)
    HasSingleLane |= Demanded[Lane] != Demanded[Lane + 1];
  if (!HasSingleLane && !ScalarizeAll)
    return false;

  // SLP may have formed wider vectors. NVPTX splits these into adjacent f32x2
  // pairs, so retain fully demanded pairs and scalarize only half-used pairs.
  IRBuilder<NoFolder> Builder(&I);
  Builder.setFastMathFlags(I.getFastMathFlags());
  unsigned NumOperands = isa<IntrinsicInst>(I)
                             ? cast<IntrinsicInst>(I).arg_size()
                             : I.getNumOperands();
  Value *Result = PoisonValue::get(Ty);
  for (unsigned Lane = 0; Lane < Width;) {
    if (!Demanded[Lane]) {
      ++Lane;
      continue;
    }
    bool Packed = !ScalarizeAll && Lane % 2 == 0 && Demanded[Lane + 1];
    SmallVector<Value *> Operands;
    for (unsigned Op = 0; Op < NumOperands; ++Op) {
      Value *V = I.getOperand(Op);
      if (Packed)
        V = Builder.CreateShuffleVector(V, {int(Lane), int(Lane + 1)});
      else
        V = extractLane(Builder, V, Lane);
      Operands.push_back(V);
    }
    Value *Part = createOperation(Builder, I, Operands);
    if (!Packed && (I.getOpcode() == Instruction::FAdd ||
                    I.getOpcode() == Instruction::FSub))
      ScalarAddSubs.insert(cast<Instruction>(Part));
    if (Packed) {
      for (unsigned Offset = 0; Offset < 2; ++Offset)
        Result = Builder.CreateInsertElement(
            Result,
            Builder.CreateExtractElement(Part, Builder.getInt64(Offset)),
            Builder.getInt64(Lane + Offset));
    } else {
      Result =
          Builder.CreateInsertElement(Result, Part, Builder.getInt64(Lane));
    }
    Lane += Packed ? 2 : 1;
  }
  I.replaceAllUsesWith(Result);
  I.eraseFromParent();
  return true;
}

PreservedAnalyses ScalarizePackedFOpsPass::run(Function &F,
                                               FunctionAnalysisManager &AM) {
  SmallVector<Instruction *> Worklist;
  for (Instruction &I : instructions(F))
    if (isSupported(I))
      Worklist.push_back(&I);

  // Visit consumers before producers, exposing single-lane inputs in chains.
  bool Changed = false;
  SmallPtrSet<Instruction *, 16> ScalarAddSubs;
  for (Instruction *I : llvm::reverse(Worklist))
    Changed |= scalarizePackedFOp(*I, ScalarAddSubs);
  if (!Changed)
    return PreservedAnalyses::all();

  // Remove dead routing instructions before checking the remaining uses of
  // multiplies. Keep weak handles because deleting one may delete another.
  SmallVector<WeakTrackingVH> Dead;
  for (Instruction &I : instructions(F))
    if (isInstructionTriviallyDead(&I))
      Dead.push_back(&I);
  for (WeakTrackingVH &V : Dead)
    if (auto *I = dyn_cast_or_null<Instruction>(V))
      RecursivelyDeleteTriviallyDeadInstructions(I);

  // Splitting add/sub while leaving their multiply inputs packed prevents DAG
  // contraction. Split those multiplies as well when *all* their uses feed the
  // scalar add/sub operations just created. No multiply is duplicated, and
  // codegen still decides whether fusion is permitted.
  Worklist.clear();
  for (Instruction &I : instructions(F)) {
    if (I.getOpcode() != Instruction::FMul || I.use_empty())
      continue;
    bool ScalarUsers = llvm::all_of(I.users(), [&](User *U) {
      auto *Extract = dyn_cast<ExtractElementInst>(U);
      return Extract && !Extract->use_empty() &&
             isa<ConstantInt>(Extract->getIndexOperand()) &&
             llvm::all_of(Extract->users(), [&](User *User) {
               return ScalarAddSubs.contains(dyn_cast<Instruction>(User));
             });
    });
    if (ScalarUsers)
      Worklist.push_back(&I);
  }
  for (Instruction *I : Worklist)
    scalarizePackedFOp(*I, ScalarAddSubs, /*ScalarizeAll=*/true);
  return PreservedAnalyses::none();
}
