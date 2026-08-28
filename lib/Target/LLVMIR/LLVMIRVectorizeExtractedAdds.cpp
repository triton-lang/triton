//===----------------------------------------------------------------------===//
// Re-form full-lane vector additions before NVPTX code generation.
//===----------------------------------------------------------------------===//
#include "LLVMPasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static bool vectorizeExtractedAdds(BinaryOperator &Mul) {
  auto *Ty = dyn_cast<FixedVectorType>(Mul.getType());
  if (Mul.getOpcode() != Instruction::FMul || !Ty ||
      !Ty->getElementType()->isFloatTy() || Ty->getNumElements() < 2 ||
      Ty->getNumElements() > 16)
    return false;

  unsigned Width = Ty->getNumElements();
  SmallVector<BinaryOperator *, 16> Adds(Width, nullptr);
  SmallVector<ExtractElementInst *, 16> Extracts;
  ConstantFP *CommonAddend = nullptr;
  BinaryOperator *FirstAdd = nullptr;
  for (User *U : Mul.users()) {
    auto *Extract = dyn_cast<ExtractElementInst>(U);
    if (!Extract || !Extract->hasOneUse() ||
        Extract->getParent() != Mul.getParent())
      return false;

    auto *Index = dyn_cast<ConstantInt>(Extract->getIndexOperand());
    if (!Index || Index->getValue().uge(Width))
      return false;
    unsigned Lane = Index->getZExtValue();

    auto *Add = dyn_cast<BinaryOperator>(*Extract->user_begin());
    if (!Add || Add->getOpcode() != Instruction::FAdd ||
        Add->getParent() != Mul.getParent() || Adds[Lane])
      return false;

    Value *Other = Add->getOperand(Add->getOperand(0) == Extract ? 1 : 0);
    auto *Addend = dyn_cast<ConstantFP>(Other);
    if (!Addend || (CommonAddend && Addend != CommonAddend) ||
        (FirstAdd && Add->getFastMathFlags() != FirstAdd->getFastMathFlags()))
      return false;

    CommonAddend = Addend;
    if (!FirstAdd || Add->comesBefore(FirstAdd))
      FirstAdd = Add;
    Adds[Lane] = Add;
    Extracts.push_back(Extract);
  }
  if (Extracts.size() != Width)
    return false;

  IRBuilder<> Builder(FirstAdd);
  Builder.setFastMathFlags(FirstAdd->getFastMathFlags());
  Builder.SetCurrentDebugLocation(FirstAdd->getDebugLoc());
  Value *Splat =
      ConstantVector::getSplat(ElementCount::getFixed(Width), CommonAddend);
  Value *PackedAdd = Builder.CreateFAdd(&Mul, Splat, "repacked.add");
  for (unsigned Lane = 0; Lane != Width; ++Lane) {
    IRBuilder<> LaneBuilder(Adds[Lane]);
    LaneBuilder.SetCurrentDebugLocation(Adds[Lane]->getDebugLoc());
    Value *Result = LaneBuilder.CreateExtractElement(PackedAdd, Lane);
    Adds[Lane]->replaceAllUsesWith(Result);
  }
  for (BinaryOperator *Add : Adds)
    Add->eraseFromParent();
  for (ExtractElementInst *Extract : Extracts)
    Extract->eraseFromParent();
  return true;
}

PreservedAnalyses VectorizeExtractedAddsPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  // Do not change evaluation or contraction in strict floating-point code.
  if (F.hasFnAttribute(Attribute::StrictFP))
    return PreservedAnalyses::all();

  SmallVector<BinaryOperator *> Multiplies;
  for (Instruction &I : instructions(F))
    if (auto *Mul = dyn_cast<BinaryOperator>(&I);
        Mul && Mul->getOpcode() == Instruction::FMul &&
        Mul->getType()->isVectorTy())
      Multiplies.push_back(Mul);

  bool Changed = false;
  for (BinaryOperator *Mul : Multiplies)
    Changed |= vectorizeExtractedAdds(*Mul);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
