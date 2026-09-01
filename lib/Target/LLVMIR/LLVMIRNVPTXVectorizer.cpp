//===----------------------------------------------------------------------===//
// Form profitable packed arithmetic supported by the selected NVIDIA GPU.
//===----------------------------------------------------------------------===//

#include "LLVMPasses.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Utils/Local.h"

#include <algorithm>
#include <optional>
#include <utility>

using namespace llvm;

namespace {

using InstructionPair = std::pair<Instruction *, Instruction *>;
using ValuePair = std::pair<Value *, Value *>;

struct ExtractedVector {
  Value *vector;
  unsigned firstLane;
};

struct PackedResultUse {
  InsertElementInst *first;
  InsertElementInst *second;
};

struct PackedLoopCarry {
  PHINode *first;
  PHINode *second;
  Instruction *firstUpdate;
  Instruction *secondUpdate;
  unsigned backedgeIndex;
};

struct LoopAccumulator {
  PHINode *phi;
  BinaryOperator *update;
  CallBase *contribution;
  unsigned backedgeIndex;
};

struct LoadedAccumulator {
  BinaryOperator *update;
  PHINode *phi;
  Value *source;
  unsigned lane;
};

struct ArithmeticPlan {
  unsigned scalarCost = 0;
  unsigned vectorCost = 0;
  SmallVector<InstructionPair, 8> instructions;
  SmallVector<ValuePair, 8> packedInputs;
  SmallVector<PackedLoopCarry, 4> loopCarries;
};

class NVPTXCostModel {
public:
  explicit NVPTXCostModel(unsigned computeCapability)
      : computeCapability(computeCapability) {}

  bool supportsPackedArithmetic(unsigned opcode, Type *elementType) const {
    if (opcode != Instruction::FAdd && opcode != Instruction::FSub &&
        opcode != Instruction::FMul)
      return false;
    return supportsPackedType(elementType);
  }

  bool supportsPackedFMA(Type *elementType) const {
    return supportsPackedType(elementType);
  }

  // Packing a 32-bit register tuple is cheaper than floating-point arithmetic,
  // but still consumes register bandwidth and can extend live ranges.
  unsigned getScalarOperationCost(const Instruction &instruction,
                                  unsigned fusedConversionChainLength) const {
    Type *elementType = instruction.getType();
    // Scalar zero adds can use the zero register, without keeping a packed
    // constant live. Signed-zero semantics can prevent eliminating these adds.
    if (isFloatZeroAdd(instruction))
      return 1;
    if (computeCapability >= 100 && computeCapability < 107 &&
        elementType->isFloatTy() && fusedConversionChainLength > 1 &&
        fusedConversionChainLength <= 8)
      return 2;
    return 3;
  }
  unsigned getPackedOperationCost(const Instruction &instruction,
                                  bool registerChain) const {
    // An existing register pair feeding packed arithmetic can use a broadcast
    // zero register without materializing a constant or repacking its result.
    if (registerChain && isFloatZeroAdd(instruction))
      return 1;
    return 3;
  }

  unsigned getPackCost(Type *elementType) const {
    return elementType->isFloatTy() ? 1 : 2;
  }

  unsigned getUnpackCost(Type *elementType) const {
    return elementType->isFloatTy() ? 1 : 2;
  }

private:
  static bool isFloatZeroAdd(const Instruction &instruction) {
    return instruction.getType()->isFloatTy() &&
           instruction.getOpcode() == Instruction::FAdd &&
           llvm::any_of(instruction.operands(), [](const Use &operand) {
             auto *constant = dyn_cast<ConstantFP>(operand.get());
             return constant && constant->isZero();
           });
  }

  bool supportsPackedType(Type *elementType) const {
    if (elementType->isHalfTy())
      return computeCapability >= 53;
    if (elementType->isBFloatTy())
      return computeCapability >= 90;
    return elementType->isFloatTy() && computeCapability >= 100;
  }

  unsigned computeCapability;
};

class NVPTXVectorizer {
public:
  NVPTXVectorizer(Function &function, unsigned computeCapability)
      : function(function), costModel(computeCapability) {}

  bool run() {
    collectFusedConversionChains();
    bool changed = vectorizeLoopAccumulators();
    for (BasicBlock &block : function) {
      changed |= vectorizeInterleavedAccumulators(block);
      changed |= vectorizeArithmetic(block);
    }
    return changed;
  }

private:
  bool isFusedHalfAddition(Instruction &instruction) const {
    if (instruction.getOpcode() != Instruction::FAdd ||
        !instruction.getType()->isFloatTy())
      return false;

    return llvm::any_of(instruction.operands(), [](Value *operand) {
      auto *extension = dyn_cast<FPExtInst>(operand);
      return extension && extension->getSrcTy()->isHalfTy();
    });
  }

  void collectFusedConversionChains() {
    for (BasicBlock &block : function) {
      for (Instruction &instruction : block) {
        if (!isFusedHalfAddition(instruction) ||
            fusedConversionChainLengths.contains(&instruction))
          continue;

        SmallVector<Instruction *, 8> chain;
        Instruction *current = &instruction;
        while (current && current->getParent() == &block &&
               isFusedHalfAddition(*current)) {
          chain.push_back(current);
          if (!current->hasOneUse())
            break;
          current = dyn_cast<Instruction>(*current->user_begin());
        }

        for (Instruction *operation : chain)
          fusedConversionChainLengths[operation] = chain.size();
      }
    }
  }

  bool isPackedOperation(Instruction &instruction) const {
    Type *elementType = instruction.getType();
    if (!elementType->isFloatingPointTy())
      return false;

    if (auto *arithmetic = dyn_cast<BinaryOperator>(&instruction))
      return costModel.supportsPackedArithmetic(arithmetic->getOpcode(),
                                                elementType);

    auto *intrinsic = dyn_cast<IntrinsicInst>(&instruction);
    return intrinsic && intrinsic->getIntrinsicID() == Intrinsic::fma &&
           costModel.supportsPackedFMA(elementType);
  }

  bool canPair(Instruction &first, Instruction &second) const {
    if (&first == &second || first.getParent() != second.getParent() ||
        first.getType() != second.getType() ||
        first.getOpcode() != second.getOpcode() ||
        first.getDebugLoc() != second.getDebugLoc() ||
        !isPackedOperation(first) || !isPackedOperation(second))
      return false;

    if (auto *firstIntrinsic = dyn_cast<IntrinsicInst>(&first)) {
      auto *secondIntrinsic = dyn_cast<IntrinsicInst>(&second);
      return secondIntrinsic && firstIntrinsic->getIntrinsicID() ==
                                    secondIntrinsic->getIntrinsicID();
    }

    return isa<BinaryOperator>(second);
  }

  unsigned getOperandCount(Instruction &instruction) const {
    return isa<BinaryOperator>(instruction) ? 2 : 3;
  }

  std::optional<LoopAccumulator> getLoopAccumulator(PHINode &phi) const {
    if (!costModel.supportsPackedArithmetic(Instruction::FAdd, phi.getType()) ||
        !phi.hasOneUse() || phi.getNumIncomingValues() != 2)
      return std::nullopt;

    auto *update = dyn_cast<BinaryOperator>(*phi.user_begin());
    if (!update || update->getOpcode() != Instruction::FAdd)
      return std::nullopt;

    unsigned backedgeIndex = phi.getNumIncomingValues();
    for (unsigned index = 0; index < phi.getNumIncomingValues(); ++index) {
      if (phi.getIncomingValue(index) == update &&
          phi.getIncomingBlock(index) == update->getParent()) {
        backedgeIndex = index;
      } else if (!isa<Constant>(phi.getIncomingValue(index))) {
        return std::nullopt;
      }
    }
    if (backedgeIndex == phi.getNumIncomingValues())
      return std::nullopt;

    Value *contribution = nullptr;
    if (update->getOperand(0) == &phi)
      contribution = update->getOperand(1);
    else if (update->getOperand(1) == &phi)
      contribution = update->getOperand(0);
    auto *call = dyn_cast_or_null<CallBase>(contribution);
    if (!call || !call->getCalledFunction() || call->arg_empty())
      return std::nullopt;

    return LoopAccumulator{&phi, update, call, backedgeIndex};
  }

  bool canPairLoopAccumulators(const LoopAccumulator &first,
                               const LoopAccumulator &second) const {
    if (first.phi->getParent() != second.phi->getParent() ||
        first.phi->getType() != second.phi->getType() ||
        first.update->getParent() != second.update->getParent() ||
        first.contribution->getCalledFunction() !=
            second.contribution->getCalledFunction() ||
        first.contribution->arg_size() != second.contribution->arg_size())
      return false;

    for (unsigned index = 1; index < first.contribution->arg_size(); ++index)
      if (first.contribution->getArgOperand(index) !=
          second.contribution->getArgOperand(index))
        return false;

    for (unsigned index = 0; index < first.phi->getNumIncomingValues(); ++index)
      if (second.phi->getBasicBlockIndex(first.phi->getIncomingBlock(index)) <
          0)
        return false;

    return getInsertionPoint(*first.update, *second.update) != nullptr;
  }

  bool vectorizeLoopAccumulatorPair(const LoopAccumulator &first,
                                    const LoopAccumulator &second) const {
    Instruction *insertion = getInsertionPoint(*first.update, *second.update);
    if (!insertion)
      return false;

    auto *vectorType = FixedVectorType::get(first.phi->getType(), 2);
    IRBuilder<> phiBuilder(first.phi);
    PHINode *packedPhi = phiBuilder.CreatePHI(
        vectorType, first.phi->getNumIncomingValues(), "nvptx.accumulator");
    for (unsigned index = 0; index < first.phi->getNumIncomingValues();
         ++index) {
      BasicBlock *block = first.phi->getIncomingBlock(index);
      int otherIndex = second.phi->getBasicBlockIndex(block);
      if (index == first.backedgeIndex) {
        packedPhi->addIncoming(PoisonValue::get(vectorType), block);
        continue;
      }

      ValuePair initial{first.phi->getIncomingValue(index),
                        second.phi->getIncomingValue(otherIndex)};
      packedPhi->addIncoming(buildVector(initial, phiBuilder), block);
    }

    IRBuilder<> builder(insertion);
    IRBuilder<>::FastMathFlagGuard guard(builder);
    builder.setFastMathFlags(first.update->getFastMathFlags() &
                             second.update->getFastMathFlags());
    Value *contributions =
        buildVector({first.contribution, second.contribution}, builder);
    Value *packed = builder.CreateFAdd(packedPhi, contributions,
                                       "nvptx.accumulator.update");
    packedPhi->setIncomingValue(first.backedgeIndex, packed);

    Value *firstLane =
        builder.CreateExtractElement(packed, uint64_t(0), "nvptx.extract");
    Value *secondLane =
        builder.CreateExtractElement(packed, uint64_t(1), "nvptx.extract");
    first.update->replaceAllUsesWith(firstLane);
    second.update->replaceAllUsesWith(secondLane);
    first.update->eraseFromParent();
    second.update->eraseFromParent();
    first.phi->eraseFromParent();
    second.phi->eraseFromParent();
    return true;
  }

  bool vectorizeLoopAccumulators() const {
    bool changed = false;
    for (BasicBlock &block : function) {
      SmallVector<LoopAccumulator, 8> candidates;
      for (PHINode &phi : block.phis())
        if (auto accumulator = getLoopAccumulator(phi))
          candidates.push_back(*accumulator);

      llvm::sort(candidates, [](const LoopAccumulator &first,
                                const LoopAccumulator &second) {
        if (first.update->getParent() != second.update->getParent())
          return first.update->getParent()->getNumber() <
                 second.update->getParent()->getNumber();
        return first.update->comesBefore(second.update);
      });

      SmallVector<bool, 8> paired(candidates.size(), false);
      for (unsigned index = 0; index < candidates.size(); ++index) {
        if (paired[index])
          continue;
        for (unsigned other = index + 1; other < candidates.size(); ++other) {
          if (paired[other] ||
              !canPairLoopAccumulators(candidates[index], candidates[other]))
            continue;
          if (vectorizeLoopAccumulatorPair(candidates[index],
                                           candidates[other])) {
            paired[index] = true;
            paired[other] = true;
            changed = true;
          }
          break;
        }
      }
    }
    return changed;
  }

  std::optional<LoadedAccumulator>
  getLoadedAccumulator(BinaryOperator &update) const {
    if (update.getOpcode() != Instruction::FAdd ||
        !update.getType()->isFloatTy() ||
        !costModel.supportsPackedArithmetic(update.getOpcode(),
                                            update.getType()))
      return std::nullopt;

    auto *phi = dyn_cast<PHINode>(update.getOperand(0));
    auto *cast = dyn_cast<BitCastInst>(update.getOperand(1));
    if (!phi || !cast || phi->getParent() != update.getParent() ||
        !phi->hasOneUse())
      return std::nullopt;

    auto *lane = dyn_cast<ExtractValueInst>(cast->getOperand(0));
    if (!lane || lane->getNumIndices() != 1)
      return std::nullopt;
    auto *load = dyn_cast<CallBase>(lane->getAggregateOperand());
    auto *assembly =
        load ? dyn_cast<InlineAsm>(load->getCalledOperand()) : nullptr;
    if (!assembly || !assembly->getAsmString().contains("ld.global.v4.b32"))
      return std::nullopt;

    Value *pointer = nullptr;
    for (Value *argument : load->args()) {
      if (!argument->getType()->isPointerTy())
        continue;
      if (pointer)
        return std::nullopt;
      pointer = argument;
    }
    if (!pointer)
      return std::nullopt;

    return LoadedAccumulator{&update, phi, getUnderlyingObject(pointer),
                             lane->getIndices().front()};
  }

  bool vectorizeInterleavedAccumulators(BasicBlock &block) const {
    SmallVector<SmallVector<LoadedAccumulator, 8>, 8> groups;
    for (Instruction &instruction : block) {
      auto *arithmetic = dyn_cast<BinaryOperator>(&instruction);
      if (!arithmetic)
        continue;
      auto accumulator = getLoadedAccumulator(*arithmetic);
      if (!accumulator)
        continue;

      auto group = llvm::find_if(groups, [&](const auto &candidate) {
        return candidate.front().source == accumulator->source;
      });
      if (group == groups.end()) {
        groups.emplace_back();
        groups.back().push_back(*accumulator);
      } else {
        group->push_back(*accumulator);
      }
    }

    // Keep adjacent-lane packing for small groups. Pairing across four
    // independent load streams exposes memory parallelism while retaining
    // native-width arithmetic.
    if (groups.size() < 4)
      return false;

    bool changed = false;
    for (unsigned start = 0; start + 1 < groups.size(); start += 4) {
      unsigned end = std::min<unsigned>(start + 4, groups.size());
      for (unsigned left = start, right = end - 1; left < right;
           ++left, --right) {
        auto &firstGroup = groups[left];
        auto &secondGroup = groups[right];
        unsigned count = std::min(firstGroup.size(), secondGroup.size());
        for (unsigned index = 0; index < count; ++index) {
          const LoadedAccumulator &first = firstGroup[index];
          const LoadedAccumulator &second = secondGroup[index];
          if (first.lane != second.lane ||
              !getLoopCarry({first.phi, second.phi}, *first.update,
                            *second.update))
            continue;
          changed |= vectorizePair(*first.update, *second.update);
        }
      }
    }
    return changed;
  }

  std::optional<ExtractedVector> getExtractedVector(ValuePair values) const {
    auto *first = dyn_cast<ExtractElementInst>(values.first);
    auto *second = dyn_cast<ExtractElementInst>(values.second);
    if (!first || !second ||
        first->getVectorOperand() != second->getVectorOperand())
      return std::nullopt;

    auto *firstIndex = dyn_cast<ConstantInt>(first->getIndexOperand());
    auto *secondIndex = dyn_cast<ConstantInt>(second->getIndexOperand());
    auto *vectorType = dyn_cast<FixedVectorType>(first->getVectorOperandType());
    if (!firstIndex || !secondIndex || !vectorType)
      return std::nullopt;

    unsigned firstLane = firstIndex->getZExtValue();
    if (firstLane % 2 != 0 || secondIndex->getZExtValue() != firstLane + 1 ||
        firstLane + 1 >= vectorType->getNumElements())
      return std::nullopt;

    return ExtractedVector{first->getVectorOperand(), firstLane};
  }

  bool isRegisterTuple(ValuePair values) const {
    if (!values.first->getType()->isFloatTy())
      return false;

    auto stripBitcast = [](Value *value) {
      if (auto *bitcast = dyn_cast<BitCastInst>(value))
        return bitcast->getOperand(0);
      return value;
    };

    auto *first = dyn_cast<ExtractValueInst>(stripBitcast(values.first));
    auto *second = dyn_cast<ExtractValueInst>(stripBitcast(values.second));
    if (!first || !second ||
        first->getAggregateOperand() != second->getAggregateOperand() ||
        first->getNumIndices() != 1 || second->getNumIndices() != 1)
      return false;

    unsigned firstLane = first->getIndices().front();
    return firstLane % 2 == 0 && second->getIndices().front() == firstLane + 1;
  }

  std::optional<PackedResultUse> getPackedResultUse(Instruction &first,
                                                    Instruction &second) const {
    if (!first.hasOneUse() || !second.hasOneUse())
      return std::nullopt;

    auto *firstInsert = dyn_cast<InsertElementInst>(*first.user_begin());
    auto *secondInsert = dyn_cast<InsertElementInst>(*second.user_begin());
    if (!firstInsert || !secondInsert || !firstInsert->hasOneUse() ||
        secondInsert->getOperand(0) != firstInsert)
      return std::nullopt;

    auto *firstIndex = dyn_cast<ConstantInt>(firstInsert->getOperand(2));
    auto *secondIndex = dyn_cast<ConstantInt>(secondInsert->getOperand(2));
    auto *vectorType = dyn_cast<FixedVectorType>(firstInsert->getType());
    if (!firstIndex || !secondIndex || !vectorType ||
        vectorType->getNumElements() != 2 || !firstIndex->isZero() ||
        !secondIndex->isOne())
      return std::nullopt;

    return PackedResultUse{firstInsert, secondInsert};
  }

  bool feedsOnlyScalarDivision(Instruction &instruction) const {
    if (instruction.use_empty())
      return false;

    return llvm::all_of(instruction.users(), [](User *user) {
      auto *call = dyn_cast<CallBase>(user);
      Function *callee = call ? call->getCalledFunction() : nullptr;
      return callee && callee->getName() == "llvm.nvvm.div.full";
    });
  }

  bool usesStayPacked(Instruction &first, Instruction &second) const {
    if (first.use_empty() || second.use_empty())
      return false;

    auto hasMatchingUser = [&](Instruction &source, Instruction &other) {
      return llvm::all_of(source.users(), [&](User *user) {
        auto *instruction = dyn_cast<Instruction>(user);
        if (!instruction || !isPackedOperation(*instruction))
          return false;
        return llvm::any_of(other.users(), [&](User *otherUser) {
          auto *paired = dyn_cast<Instruction>(otherUser);
          if (!paired || !canPair(*instruction, *paired))
            return false;
          for (unsigned index = 0, count = getOperandCount(*instruction);
               index < count; ++index)
            if (instruction->getOperand(index) == &source &&
                paired->getOperand(index) == &other)
              return true;
          return false;
        });
      });
    };

    return hasMatchingUser(first, second) && hasMatchingUser(second, first);
  }

  std::optional<PackedLoopCarry> getLoopCarry(ValuePair values,
                                              Instruction &first,
                                              Instruction &second) const {
    auto *firstPhi = dyn_cast<PHINode>(values.first);
    auto *secondPhi = dyn_cast<PHINode>(values.second);
    if (!firstPhi || !secondPhi ||
        firstPhi->getParent() != secondPhi->getParent() ||
        !firstPhi->hasOneUse() || !secondPhi->hasOneUse() ||
        firstPhi->getNumIncomingValues() != secondPhi->getNumIncomingValues())
      return std::nullopt;

    std::optional<unsigned> backedge;
    for (unsigned index = 0; index < firstPhi->getNumIncomingValues();
         ++index) {
      BasicBlock *block = firstPhi->getIncomingBlock(index);
      int otherIndex = secondPhi->getBasicBlockIndex(block);
      if (otherIndex < 0)
        return std::nullopt;

      Value *firstValue = firstPhi->getIncomingValue(index);
      Value *secondValue = secondPhi->getIncomingValue(otherIndex);
      if (firstValue == &first && secondValue == &second) {
        if (backedge || block != first.getParent())
          return std::nullopt;
        backedge = index;
      } else if (!isa<Constant>(firstValue) || !isa<Constant>(secondValue)) {
        // An already packed initial value is as cheap as a constant. Keep it
        // packed across the loop instead of rebuilding it on every iteration.
        ValuePair incoming{firstValue, secondValue};
        if (!getExtractedVector(incoming) && !isRegisterTuple(incoming))
          return std::nullopt;
      }
    }

    if (!backedge)
      return std::nullopt;
    return PackedLoopCarry{firstPhi, secondPhi, &first, &second, *backedge};
  }

  Instruction *getInsertionPoint(Instruction &first,
                                 Instruction &second) const {
    Instruction *insertion = &first;
    for (Use &operand : second.operands()) {
      auto *definition = dyn_cast<Instruction>(operand.get());
      if (!definition || definition->getParent() != first.getParent())
        continue;
      if (definition == &first)
        return nullptr;
      if (insertion == definition || insertion->comesBefore(definition))
        insertion = definition->getNextNode();
    }

    if (!insertion || insertion == &first)
      return insertion;

    for (User *user : first.users()) {
      auto *use = dyn_cast<Instruction>(user);
      if (auto *phi = dyn_cast_or_null<PHINode>(use)) {
        if (llvm::any_of(phi->incoming_values(), [&](const Use &incoming) {
              return incoming.get() == &first &&
                     phi->getIncomingBlock(incoming) == first.getParent();
            }))
          continue;
      }
      if (use && use->getParent() == first.getParent() &&
          use->comesBefore(insertion))
        return nullptr;
    }
    return insertion;
  }

  std::optional<ValuePair> getNegatedOperands(ValuePair values) const {
    auto *first = dyn_cast<UnaryOperator>(values.first);
    auto *second = dyn_cast<UnaryOperator>(values.second);
    if (!first || !second || first->getOpcode() != Instruction::FNeg ||
        second->getOpcode() != Instruction::FNeg || !first->hasOneUse() ||
        !second->hasOneUse())
      return std::nullopt;
    return ValuePair{first->getOperand(0), second->getOperand(0)};
  }

  void addInputCost(ValuePair values, ArithmeticPlan &plan) const {
    // Negating a packed input replaces the two existing scalar negations. It
    // also keeps the sign visible to arithmetic combines, without extra packs.
    if (auto operands = getNegatedOperands(values)) {
      addInputCost(*operands, plan);
      return;
    }
    if (values.first == values.second ||
        (isa<Constant>(values.first) && isa<Constant>(values.second)) ||
        getExtractedVector(values) || isRegisterTuple(values))
      return;

    if (llvm::is_contained(plan.packedInputs, values))
      return;

    plan.packedInputs.push_back(values);
    plan.vectorCost += costModel.getPackCost(values.first->getType());
  }

  unsigned getScalarPairCost(Instruction &first, Instruction &second) const {
    return costModel.getScalarOperationCost(
               first, fusedConversionChainLengths.lookup(&first)) +
           costModel.getScalarOperationCost(
               second, fusedConversionChainLengths.lookup(&second));
  }

  unsigned getPackedPairCost(Instruction &first, Instruction &second) const {
    bool registerChain = usesStayPacked(first, second);
    for (unsigned index = 0, count = getOperandCount(first); index < count;
         ++index) {
      ValuePair operands{first.getOperand(index), second.getOperand(index)};
      registerChain &=
          operands.first == operands.second ||
          (isa<Constant>(operands.first) && isa<Constant>(operands.second)) ||
          getExtractedVector(operands).has_value() || isRegisterTuple(operands);
    }
    return std::max(costModel.getPackedOperationCost(first, registerChain),
                    costModel.getPackedOperationCost(second, registerChain));
  }

  void addInstructionCost(Instruction &first, Instruction &second,
                          ArithmeticPlan &plan, unsigned depth = 0) const {
    InstructionPair pair{&first, &second};
    if (llvm::is_contained(plan.instructions, pair))
      return;

    plan.instructions.push_back(pair);
    plan.scalarCost += getScalarPairCost(first, second);
    plan.vectorCost += getPackedPairCost(first, second);

    for (unsigned index = 0, count = getOperandCount(first); index < count;
         ++index) {
      ValuePair operands{first.getOperand(index), second.getOperand(index)};
      if (auto carry = getLoopCarry(operands, first, second)) {
        plan.loopCarries.push_back(*carry);
        continue;
      }
      auto *firstOperand = dyn_cast<Instruction>(operands.first);
      auto *secondOperand = dyn_cast<Instruction>(operands.second);
      if (depth < 8 && firstOperand && secondOperand &&
          firstOperand->hasOneUse() && secondOperand->hasOneUse() &&
          *firstOperand->user_begin() == &first &&
          *secondOperand->user_begin() == &second &&
          canPair(*firstOperand, *secondOperand) &&
          getPackedPairCost(*firstOperand, *secondOperand) +
                  costModel.getPackCost(firstOperand->getType()) <
              getScalarPairCost(*firstOperand, *secondOperand)) {
        addInstructionCost(*firstOperand, *secondOperand, plan, depth + 1);
        continue;
      }
      addInputCost(operands, plan);
    }
  }

  Value *buildVector(ValuePair values, IRBuilder<> &builder) const {
    if (auto operands = getNegatedOperands(values)) {
      Value *packed = buildVector(*operands, builder);
      IRBuilder<>::FastMathFlagGuard guard(builder);
      builder.setFastMathFlags(
          cast<Instruction>(values.first)->getFastMathFlags() &
          cast<Instruction>(values.second)->getFastMathFlags());
      return builder.CreateFNeg(packed, "nvptx.negate");
    }

    auto *vectorType = FixedVectorType::get(values.first->getType(), 2);

    if (auto extracted = getExtractedVector(values)) {
      if (extracted->vector->getType() == vectorType)
        return extracted->vector;
      SmallVector<int, 2> lanes{static_cast<int>(extracted->firstLane),
                                static_cast<int>(extracted->firstLane + 1)};
      return builder.CreateShuffleVector(extracted->vector, lanes,
                                         "nvptx.subvector");
    }

    if (isa<Constant>(values.first) && isa<Constant>(values.second))
      return ConstantVector::get(
          {cast<Constant>(values.first), cast<Constant>(values.second)});

    Value *vector = builder.CreateInsertElement(
        PoisonValue::get(vectorType), values.first, uint64_t(0), "nvptx.pack");
    return builder.CreateInsertElement(vector, values.second, uint64_t(1),
                                       "nvptx.pack");
  }

  Value *buildPackedArithmetic(
      Instruction &first, Instruction &second, IRBuilder<> &builder,
      const ArithmeticPlan &plan,
      SmallVectorImpl<std::pair<InstructionPair, Value *>> &built,
      SmallVectorImpl<std::pair<ValuePair, Value *>> &builtInputs) const {
    InstructionPair pair{&first, &second};
    auto existing = llvm::find_if(
        built, [&](const auto &entry) { return entry.first == pair; });
    if (existing != built.end())
      return existing->second;

    SmallVector<Value *, 3> operands;
    for (unsigned index = 0, count = getOperandCount(first); index < count;
         ++index) {
      ValuePair lanes{first.getOperand(index), second.getOperand(index)};
      auto *firstOperand = dyn_cast<Instruction>(lanes.first);
      auto *secondOperand = dyn_cast<Instruction>(lanes.second);
      InstructionPair child{firstOperand, secondOperand};
      if (firstOperand && secondOperand &&
          llvm::is_contained(plan.instructions, child)) {
        operands.push_back(buildPackedArithmetic(
            *firstOperand, *secondOperand, builder, plan, built, builtInputs));
      } else {
        auto packedInput = llvm::find_if(builtInputs, [&](const auto &entry) {
          return entry.first == lanes;
        });
        if (packedInput != builtInputs.end()) {
          operands.push_back(packedInput->second);
        } else {
          Value *input = buildVector(lanes, builder);
          builtInputs.emplace_back(lanes, input);
          operands.push_back(input);
        }
      }
    }

    IRBuilder<>::FastMathFlagGuard guard(builder);
    builder.setFastMathFlags(first.getFastMathFlags() &
                             second.getFastMathFlags());
    Value *packed = nullptr;
    if (auto *arithmetic = dyn_cast<BinaryOperator>(&first)) {
      packed = builder.CreateBinOp(
          static_cast<Instruction::BinaryOps>(arithmetic->getOpcode()),
          operands[0], operands[1], "nvptx.packed");
    } else {
      packed = builder.CreateFMA(operands[0], operands[1], operands[2], {},
                                 "nvptx.packed.fma");
    }
    built.emplace_back(pair, packed);
    return packed;
  }

  bool vectorizePair(Instruction &first, Instruction &second) const {
    if (first.use_empty() || second.use_empty() ||
        ((first.getOpcode() == Instruction::FAdd ||
          first.getOpcode() == Instruction::FSub) &&
         feedsOnlyScalarDivision(first) && feedsOnlyScalarDivision(second)))
      return false;

    Instruction *insertion = getInsertionPoint(first, second);
    if (!insertion)
      return false;

    ArithmeticPlan plan;
    addInstructionCost(first, second, plan);
    std::optional<PackedResultUse> packedUse =
        getPackedResultUse(first, second);
    if (!packedUse && !usesStayPacked(first, second))
      plan.vectorCost += costModel.getUnpackCost(first.getType());
    if (plan.vectorCost >= plan.scalarCost)
      return false;

    SmallVector<WeakTrackingVH, 16> deadInstructions;
    for (InstructionPair pair : plan.instructions) {
      deadInstructions.push_back(pair.first);
      deadInstructions.push_back(pair.second);
      for (Instruction *instruction : {pair.first, pair.second})
        for (Use &operand : instruction->operands())
          if (auto *definition = dyn_cast<Instruction>(operand.get()))
            deadInstructions.push_back(definition);
    }

    IRBuilder<> builder(insertion);
    SmallVector<std::pair<InstructionPair, Value *>, 8> built;
    SmallVector<std::pair<ValuePair, Value *>, 8> builtInputs;
    SmallVector<std::pair<PackedLoopCarry, PHINode *>, 4> packedCarries;
    for (const PackedLoopCarry &carry : plan.loopCarries) {
      IRBuilder<> phiBuilder(carry.first);
      auto *vectorType = FixedVectorType::get(carry.first->getType(), 2);
      PHINode *packedPhi = phiBuilder.CreatePHI(
          vectorType, carry.first->getNumIncomingValues(), "nvptx.accumulator");
      for (unsigned index = 0; index < carry.first->getNumIncomingValues();
           ++index) {
        BasicBlock *block = carry.first->getIncomingBlock(index);
        if (index == carry.backedgeIndex) {
          packedPhi->addIncoming(PoisonValue::get(vectorType), block);
          continue;
        }

        int otherIndex = carry.second->getBasicBlockIndex(block);
        ValuePair incoming{carry.first->getIncomingValue(index),
                           carry.second->getIncomingValue(otherIndex)};
        IRBuilder<> incomingBuilder(block->getTerminator());
        packedPhi->addIncoming(buildVector(incoming, incomingBuilder), block);
      }
      builtInputs.emplace_back(ValuePair{carry.first, carry.second}, packedPhi);
      packedCarries.emplace_back(carry, packedPhi);
    }
    Value *packed =
        buildPackedArithmetic(first, second, builder, plan, built, builtInputs);
    for (const auto &packedCarry : packedCarries) {
      const PackedLoopCarry &carry = packedCarry.first;
      PHINode *phi = packedCarry.second;
      auto update = llvm::find_if(built, [&](const auto &entry) {
        return entry.first ==
               InstructionPair{carry.firstUpdate, carry.secondUpdate};
      });
      assert(update != built.end() && "missing packed loop accumulator update");
      phi->setIncomingValue(carry.backedgeIndex, update->second);
    }

    if (packedUse) {
      packedUse->second->replaceAllUsesWith(packed);
      packedUse->second->eraseFromParent();
      packedUse->first->eraseFromParent();
    } else {
      Value *firstLane =
          builder.CreateExtractElement(packed, uint64_t(0), "nvptx.extract");
      Value *secondLane =
          builder.CreateExtractElement(packed, uint64_t(1), "nvptx.extract");
      first.replaceAllUsesWith(firstLane);
      second.replaceAllUsesWith(secondLane);
    }

    first.eraseFromParent();
    second.eraseFromParent();
    for (WeakTrackingVH &instruction : deadInstructions)
      if (instruction)
        RecursivelyDeleteTriviallyDeadInstructions(instruction);
    return true;
  }

  bool vectorizeArithmetic(BasicBlock &block) const {
    SmallVector<WeakTrackingVH, 32> candidates;
    for (Instruction &instruction : block)
      if (isPackedOperation(instruction))
        candidates.push_back(&instruction);

    bool changed = false;
    for (unsigned index = 0; index + 1 < candidates.size(); ++index) {
      auto *first = dyn_cast_or_null<Instruction>(candidates[index]);
      if (!first || !isPackedOperation(*first))
        continue;

      unsigned limit = std::min<unsigned>(candidates.size(), index + 33);
      for (unsigned other = index + 1; other < limit; ++other) {
        auto *second = dyn_cast_or_null<Instruction>(candidates[other]);
        if (!second || !canPair(*first, *second))
          continue;
        changed |= vectorizePair(*first, *second);
        break;
      }
    }
    return changed;
  }

  Function &function;
  NVPTXCostModel costModel;
  DenseMap<const Instruction *, unsigned> fusedConversionChainLengths;
};

} // namespace

PreservedAnalyses NVPTXVectorizerPass::run(Function &function,
                                           FunctionAnalysisManager &) {
  const Triple &triple = function.getParent()->getTargetTriple();
  if (!triple.getTriple().empty() && !triple.isNVPTX())
    return PreservedAnalyses::all();

  bool changed = NVPTXVectorizer(function, computeCapability).run();
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
