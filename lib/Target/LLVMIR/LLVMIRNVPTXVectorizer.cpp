//===----------------------------------------------------------------------===//
// Form profitable packed arithmetic supported by the selected NVIDIA GPU.
//===----------------------------------------------------------------------===//

#include "LLVMPasses.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
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

struct PackNode {
  ValuePair values{nullptr, nullptr};
  Instruction *operation = nullptr;
  Instruction *insertion = nullptr;
  SmallVector<unsigned, 3> operands;
  bool horizontal = false;
  Value *emitted = nullptr;
};

struct PackPlan {
  SmallVector<PackNode, 32> nodes;
  DenseMap<ValuePair, unsigned> cached;
  DenseSet<Instruction *> replaced;
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

  bool supportsPackedIntegerMax(Type *elementType) const {
    return computeCapability >= 100 && elementType->isIntegerTy(16);
  }

  // Packing a 32-bit register tuple is cheaper than floating-point arithmetic,
  // but still consumes register bandwidth and can extend live ranges.
  unsigned getScalarOperationCost(const Instruction &instruction,
                                  unsigned fusedConversionChainLength) const {
    Type *elementType = instruction.getType();
    if (isa<BitCastInst>(instruction) ||
        instruction.getOpcode() == Instruction::FNeg)
      return 0;
    if (isa<FPTruncInst>(instruction))
      return 2;
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
    if (isa<BitCastInst>(instruction) ||
        (instruction.getOpcode() == Instruction::FNeg &&
         instruction.getType()->isFloatTy()))
      return 0;
    // An existing register pair feeding packed arithmetic can use a broadcast
    // zero register without materializing a constant or repacking its result.
    if (registerChain && isFloatZeroAdd(instruction))
      return 1;
    return 3;
  }

  unsigned getPackCost(Type *elementType) const {
    return elementType->getScalarSizeInBits() == 32 ? 1 : 2;
  }

  unsigned getUnpackCost(Type *elementType) const {
    return elementType->getScalarSizeInBits() == 32 ? 1 : 2;
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
      : function(function), costModel(computeCapability), dominators(function) {
  }

  bool run() {
    collectFusedConversionChains();
    bool changed = vectorizeLoopAccumulators();
    for (BasicBlock &block : function) {
      changed |= vectorizeInterleavedAccumulators(block);
      changed |= vectorizeArithmetic(block);
      changed |= vectorizeIntegerMaxReductions(block);
    }
    return changed;
  }

private:
  static IntrinsicInst *getUnsignedMax(Value *value, BasicBlock &block) {
    auto *intrinsic = dyn_cast<IntrinsicInst>(value);
    if (!intrinsic || intrinsic->getParent() != &block ||
        intrinsic->getIntrinsicID() != Intrinsic::umax ||
        !intrinsic->getType()->isIntegerTy(16))
      return nullptr;
    return intrinsic;
  }

  static bool collectIntegerMaxLeaves(Value *value, BasicBlock &block,
                                      SmallVectorImpl<Value *> &leaves) {
    SmallVector<std::pair<Value *, unsigned>, 32> pending{{value, 0}};
    while (!pending.empty()) {
      auto [current, depth] = pending.pop_back_val();
      if (depth > 256)
        return false;
      auto *maximum = getUnsignedMax(current, block);
      if (maximum && maximum->hasOneUse()) {
        pending.emplace_back(maximum->getArgOperand(1), depth + 1);
        pending.emplace_back(maximum->getArgOperand(0), depth + 1);
      } else {
        leaves.push_back(current);
        if (leaves.size() > 1024)
          return false;
      }
    }
    return true;
  }

  bool isFreePack(ValuePair values) const {
    return values.first == values.second ||
           (isa<Constant>(values.first) && isa<Constant>(values.second)) ||
           getExtractedVector(values).has_value() || isRegisterTuple(values);
  }

  bool insertionDominates(Instruction *anchor, Instruction *use) const {
    return anchor == use || dominators.dominates(anchor, use);
  }

  std::optional<unsigned> planPack(ValuePair values, Instruction *use,
                                   PackPlan &plan, unsigned depth = 0,
                                   Instruction *rootInsertion = nullptr) const {
    if (values.first->getType() != values.second->getType() ||
        values.first->getType()->isVectorTy() || plan.nodes.size() >= 4096)
      return std::nullopt;
    auto found = plan.cached.find(values);
    if (found != plan.cached.end()) {
      PackNode &node = plan.nodes[found->second];
      if (!insertionDominates(node.insertion, use)) {
        if (node.operation || !insertionDominates(use, node.insertion) ||
            !dominators.dominates(values.first, use) ||
            !dominators.dominates(values.second, use))
          return std::nullopt;
        node.insertion = use;
      }
      return found->second;
    }

    PackNode node;
    node.values = values;
    node.insertion = use;
    auto *first = dyn_cast<Instruction>(values.first);
    auto *second = dyn_cast<Instruction>(values.second);
    if (!isFreePack(values) && depth < 8 && first && second &&
        canPair(*first, *second)) {
      Instruction *last = first->comesBefore(second) ? second : first;
      Instruction *insertion =
          rootInsertion ? rootInsertion : last->getNextNode();
      if (insertion && insertionDominates(insertion, use)) {
        node.operation = first;
        node.insertion = insertion;
        for (unsigned operand = 0; operand < getOperandCount(*first);
             ++operand) {
          ValuePair inputs{first->getOperand(operand),
                           second->getOperand(operand)};
          if (auto carry = getLoopCarry(inputs, *first, *second)) {
            if (!plan.cached.contains(inputs)) {
              unsigned index = plan.nodes.size();
              PackNode input;
              input.values = inputs;
              input.insertion = insertion;
              plan.nodes.push_back(input);
              plan.cached[inputs] = index;
              plan.loopCarries.push_back(*carry);
            }
            node.operands.push_back(plan.cached.lookup(inputs));
          } else {
            auto input = planPack(inputs, insertion, plan, depth + 1);
            if (!input)
              return std::nullopt;
            node.operands.push_back(*input);
          }
        }
      }
    }
    if (!node.operation && (!dominators.dominates(values.first, use) ||
                            !dominators.dominates(values.second, use)))
      return std::nullopt;
    unsigned index = plan.nodes.size();
    plan.nodes.push_back(std::move(node));
    plan.cached[values] = index;
    return index;
  }

  SmallVector<char, 64> reachableNodes(const PackPlan &plan, unsigned root) const {
    SmallVector<char, 64> live(plan.nodes.size(), false);
    live[root] = true;
    for (unsigned index = plan.nodes.size(); index-- > 0;)
      if (live[index])
        for (unsigned input : plan.nodes[index].operands)
          live[input] = true;
    return live;
  }

  int getPlanCost(const PackPlan &plan, unsigned root,
                  unsigned outputCost) const {
    auto live = reachableNodes(plan, root);
    DenseSet<Instruction *> removed = plan.replaced;
    DenseSet<Value *> retainedInputs;
    for (unsigned index = 0; index < plan.nodes.size(); ++index) {
      const PackNode &node = plan.nodes[index];
      if (live[index] && !node.operation &&
          llvm::none_of(plan.loopCarries, [&](const PackedLoopCarry &carry) {
            return node.values == ValuePair{carry.first, carry.second};
          })) {
        retainedInputs.insert(node.values.first);
        retainedInputs.insert(node.values.second);
      }
    }
    for (unsigned index = plan.nodes.size(); index-- > 0;) {
      const PackNode &node = plan.nodes[index];
      if (!live[index] || !node.operation || !node.values.first)
        continue;
      for (Value *value : {node.values.first, node.values.second}) {
        auto *instruction = cast<Instruction>(value);
        if (!retainedInputs.contains(instruction) &&
            llvm::all_of(instruction->users(), [&](User *user) {
              return removed.contains(dyn_cast<Instruction>(user));
            }))
          removed.insert(instruction);
      }
    }
    unsigned scalarCost = 0;
    for (Instruction *instruction : removed)
      scalarCost += costModel.getScalarOperationCost(
          *instruction, fusedConversionChainLengths.lookup(instruction));
    unsigned vectorCost = outputCost;
    bool materializesInputPair = false;
    for (unsigned index = 0; index < plan.nodes.size(); ++index) {
      if (!live[index])
        continue;
      const PackNode &node = plan.nodes[index];
      if (node.operation) {
        if (node.horizontal) {
          vectorCost += costModel.getScalarOperationCost(*node.operation, 0) +
                        costModel.getUnpackCost(node.operation->getType());
        } else {
          bool registerChain = llvm::all_of(node.operands, [&](unsigned input) {
            const PackNode &operand = plan.nodes[input];
            return operand.operation || isFreePack(operand.values);
          });
          vectorCost +=
              costModel.getPackedOperationCost(*node.operation, registerChain);
        }
      } else {
        bool carried =
            llvm::any_of(plan.loopCarries, [&](const PackedLoopCarry &carry) {
              return node.values == ValuePair{carry.first, carry.second};
            });
        if (!carried && !isFreePack(node.values)) {
          vectorCost += costModel.getPackCost(node.values.first->getType());
          materializesInputPair = true;
        }
      }
    }
    bool livePackedCarry =
        llvm::any_of(plan.loopCarries, [&](const PackedLoopCarry &carry) {
          return live[plan.cached.lookup({carry.first, carry.second})];
        });
    bool scalarExit = outputCost != 0 || plan.nodes[root].horizontal;
    // Require an extra profitability margin for scalar-to-packed-to-scalar
    // regions: forming a temporary register pair can constrain scheduling.
    // Live packed carries amortize that boundary.
    if (materializesInputPair && scalarExit && !livePackedCarry)
      ++vectorCost;
    return static_cast<int>(vectorCost) - static_cast<int>(scalarCost);
  }

  bool isProfitable(PackPlan &plan, unsigned root,
                    unsigned outputCost = 0) const {
    int cost = getPlanCost(plan, root, outputCost);
    unsigned work = 0;
    for (unsigned index = 0; index < plan.nodes.size(); ++index) {
      PackNode original = plan.nodes[index];
      if (!original.operation || !original.values.first ||
          plan.replaced.contains(cast<Instruction>(original.values.first)) ||
          plan.replaced.contains(cast<Instruction>(original.values.second)))
        continue;
      if (!dominators.dominates(original.values.first, original.insertion) ||
          !dominators.dominates(original.values.second, original.insertion))
        continue;
      work += plan.nodes.size();
      if (work > 1024 * 1024)
        break;
      plan.nodes[index].operation = nullptr;
      plan.nodes[index].operands.clear();
      int boundaryCost = getPlanCost(plan, root, outputCost);
      if (boundaryCost <= cost)
        cost = boundaryCost;
      else
        plan.nodes[index] = std::move(original);
    }
    auto live = reachableNodes(plan, root);
    llvm::erase_if(plan.loopCarries, [&](const PackedLoopCarry &carry) {
      return !live[plan.cached.lookup({carry.first, carry.second})];
    });
    return cost < 0;
  }

  Value *emitPack(unsigned index, PackPlan &plan) const {
    PackNode &node = plan.nodes[index];
    if (node.emitted)
      return node.emitted;
    IRBuilder<> builder(node.insertion);
    if (!node.operation)
      return node.emitted = buildVector(node.values, builder);
    SmallVector<Value *, 3> operands;
    for (unsigned input : node.operands)
      operands.push_back(emitPack(input, plan));
    Instruction &operation = *node.operation;
    if (node.horizontal)
      return node.emitted = builder.CreateIntMaxReduce(operands.front(), false);
    if (isa<FPMathOperator>(operation) && node.values.first) {
      builder.setFastMathFlags(
          operation.getFastMathFlags() &
          cast<Instruction>(node.values.second)->getFastMathFlags());
    }
    auto *type = FixedVectorType::get(operation.getType(), 2);
    auto *intrinsic = dyn_cast<IntrinsicInst>(&operation);
    bool absolute = intrinsic && intrinsic->getIntrinsicID() == Intrinsic::fabs;
    bool negate16 = operation.getOpcode() == Instruction::FNeg &&
                    operation.getType()->getScalarSizeInBits() == 16;
    if (absolute || negate16) {
      // Packed floating sign operations can canonicalize NaN payloads in PTXAS.
      // Keep their masks integer even when later InstCombine revisits them.
      Type *bitsType = builder.getInt32Ty();
      auto *signature = FunctionType::get(bitsType, {bitsType}, false);
      auto *mask = InlineAsm::get(signature,
                                  absolute ? "and.b32 $0, $1, 0x7fff7fff;"
                                           : "xor.b32 $0, $1, 0x80008000;",
                                  "=r,r", /*hasSideEffects=*/false);
      Value *bits = builder.CreateBitCast(operands[0], bitsType);
      Value *result =
          builder.CreateCall(signature, mask, {bits}, "nvptx.sign.bits");
      return node.emitted = builder.CreateBitCast(
                 result, type, absolute ? "nvptx.abs" : "nvptx.negate");
    }
    if (isa<CastInst>(operation))
      node.emitted = builder.CreateCast(
          static_cast<Instruction::CastOps>(operation.getOpcode()), operands[0],
          type, isa<FPTruncInst>(operation) ? "nvptx.narrow" : "nvptx.bits");
    else if (operation.getOpcode() == Instruction::FNeg)
      node.emitted = builder.CreateFNeg(operands[0], "nvptx.negate");
    else if (intrinsic) {
      StringRef name = intrinsic->getIntrinsicID() == Intrinsic::umax
                           ? "nvptx.max"
                           : "nvptx.packed.fma";
      node.emitted = builder.CreateIntrinsic(intrinsic->getIntrinsicID(),
                                             {type}, operands, nullptr, name);
    } else
      node.emitted = builder.CreateBinOp(
          static_cast<Instruction::BinaryOps>(operation.getOpcode()),
          operands[0], operands[1], "nvptx.packed");
    return node.emitted;
  }

  unsigned planReductionNode(Instruction *operation,
                             ArrayRef<unsigned> operands, PackPlan &plan,
                             bool horizontal = false) const {
    PackNode node;
    node.operation = operation;
    node.insertion = operation;
    node.operands.append(operands.begin(), operands.end());
    node.horizontal = horizontal;
    unsigned index = plan.nodes.size();
    plan.nodes.push_back(std::move(node));
    return index;
  }

  bool vectorizeIntegerMaxReductions(BasicBlock &block) const {
    if (!costModel.supportsPackedIntegerMax(
            Type::getInt16Ty(function.getContext())))
      return false;
    SmallVector<WeakTrackingVH, 8> roots;
    for (Instruction &instruction : block) {
      auto *maximum = getUnsignedMax(&instruction, block);
      if (maximum && !maximum->use_empty() &&
          llvm::none_of(maximum->users(), [&](User *user) {
            return getUnsignedMax(user, block) != nullptr;
          }))
        roots.push_back(maximum);
    }
    bool changed = false;
    for (WeakTrackingVH &handle : roots) {
      auto *root = dyn_cast_or_null<IntrinsicInst>(handle);
      if (!root)
        continue;
      SmallVector<Value *, 64> leaves;
      if (!collectIntegerMaxLeaves(root->getArgOperand(0), block, leaves) ||
          !collectIntegerMaxLeaves(root->getArgOperand(1), block, leaves) ||
          leaves.size() < 4)
        continue;
      BasicBlock *inputBlock = nullptr;
      bool validInputs = true;
      for (Value *leaf : leaves) {
        if (isa<Constant>(leaf) || isa<Argument>(leaf))
          continue;
        auto *instruction = dyn_cast<Instruction>(leaf);
        if (!instruction ||
            (inputBlock && instruction->getParent() != inputBlock)) {
          validInputs = false;
          break;
        }
        inputBlock = instruction->getParent();
      }
      if (!inputBlock)
        inputBlock = &block;
      if (!validInputs || !dominators.dominates(inputBlock, &block))
        continue;
      DenseMap<Value *, unsigned> order;
      unsigned index = 0;
      for (Argument &argument : function.args())
        order[&argument] = index++;
      for (Instruction &instruction : *inputBlock)
        order[&instruction] = index++;
      llvm::stable_sort(leaves, [&](Value *first, Value *second) {
        return order.lookup(first) < order.lookup(second);
      });
      PackPlan plan;
      SmallVector<Value *, 64> pending{root};
      while (!pending.empty()) {
        Value *value = pending.pop_back_val();
        auto *maximum = getUnsignedMax(value, block);
        if (maximum && (maximum == root || maximum->hasOneUse())) {
          plan.replaced.insert(maximum);
          pending.push_back(maximum->getArgOperand(0));
          pending.push_back(maximum->getArgOperand(1));
        }
      }
      SmallVector<unsigned, 32> level;
      bool validPlan = true;
      for (unsigned lane = 0; lane < leaves.size(); lane += 2) {
        Value *second = lane + 1 < leaves.size()
                            ? leaves[lane + 1]
                            : ConstantInt::get(root->getType(), 0);
        auto input = planPack({leaves[lane], second}, root, plan);
        if (!input) {
          validPlan = false;
          break;
        }
        level.push_back(*input);
      }
      if (!validPlan)
        continue;
      while (level.size() > 1) {
        SmallVector<unsigned, 32> next;
        for (unsigned lane = 0; lane + 1 < level.size(); lane += 2)
          next.push_back(planReductionNode(root, {level[lane], level[lane + 1]}, plan));
        if (level.size() % 2)
          next.push_back(level.back());
        level = std::move(next);
      }
      unsigned resultNode = planReductionNode(root, level, plan, true);
      if (!isProfitable(plan, resultNode))
        continue;
      Value *result = emitPack(resultNode, plan);
      root->replaceAllUsesWith(result);
      RecursivelyDeleteTriviallyDeadInstructions(root);
      changed = true;
    }
    return changed;
  }

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
    Type *type = instruction.getType();
    if (type->isVectorTy())
      return false;
    if (isa<BitCastInst>(instruction)) {
      Type *source = instruction.getOperand(0)->getType();
      return !source->isVectorTy() &&
             (source->isFloatingPointTy() || source->isIntegerTy()) &&
             (type->isFloatingPointTy() || type->isIntegerTy());
    }
    if (isa<FPTruncInst>(instruction))
      return instruction.getOperand(0)->getType()->isFloatTy() &&
             (type->isHalfTy() || type->isBFloatTy()) &&
             costModel.supportsPackedFMA(type);
    if (instruction.getOpcode() == Instruction::FNeg)
      return costModel.supportsPackedFMA(type);
    if (auto *arithmetic = dyn_cast<BinaryOperator>(&instruction))
      return costModel.supportsPackedArithmetic(arithmetic->getOpcode(), type);
    auto *intrinsic = dyn_cast<IntrinsicInst>(&instruction);
    if (!intrinsic)
      return false;
    switch (intrinsic->getIntrinsicID()) {
    case Intrinsic::fma:
      return costModel.supportsPackedFMA(type);
    case Intrinsic::fabs:
      return (type->isHalfTy() || type->isBFloatTy()) &&
             costModel.supportsPackedFMA(type);
    case Intrinsic::umax:
      return costModel.supportsPackedIntegerMax(type);
    default:
      return false;
    }
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

    if (isa<CastInst>(first))
      return first.getOperand(0)->getType() == second.getOperand(0)->getType();
    return isa<BinaryOperator>(second) || isa<UnaryOperator>(second);
  }

  unsigned getOperandCount(Instruction &instruction) const {
    if (auto *intrinsic = dyn_cast<IntrinsicInst>(&instruction))
      return intrinsic->arg_size();
    return instruction.getNumOperands();
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

  Value *buildVector(ValuePair values, IRBuilder<> &builder) const {
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

  bool vectorizePair(Instruction &first, Instruction &second) const {
    if (first.use_empty() || second.use_empty() ||
        ((first.getOpcode() == Instruction::FAdd ||
          first.getOpcode() == Instruction::FSub) &&
         feedsOnlyScalarDivision(first) && feedsOnlyScalarDivision(second)))
      return false;

    Instruction *insertion = getInsertionPoint(first, second);
    if (!insertion)
      return false;

    PackPlan plan;
    plan.replaced.insert(&first);
    plan.replaced.insert(&second);
    auto root = planPack({&first, &second}, insertion, plan, 0, insertion);
    if (!root || !plan.nodes[*root].operation)
      return false;
    std::optional<PackedResultUse> packedUse =
        getPackedResultUse(first, second);
    unsigned outputCost =
        packedUse ? 0 : costModel.getUnpackCost(first.getType());
    if (!isProfitable(plan, *root, outputCost))
      return false;

    SmallVector<WeakTrackingVH, 16> deadInstructions;
    for (const PackNode &node : plan.nodes)
      for (Value *value : {node.values.first, node.values.second})
        if (auto *instruction = dyn_cast_or_null<Instruction>(value))
          deadInstructions.push_back(instruction);
    IRBuilder<> builder(insertion);
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
      unsigned input = plan.cached.lookup(ValuePair{carry.first, carry.second});
      plan.nodes[input].emitted = packedPhi;
      packedCarries.emplace_back(carry, packedPhi);
    }
    Value *packed = emitPack(*root, plan);
    for (const auto &packedCarry : packedCarries) {
      const PackedLoopCarry &carry = packedCarry.first;
      auto update = plan.cached.find({carry.firstUpdate, carry.secondUpdate});
      assert(update != plan.cached.end() &&
             "missing packed loop accumulator update");
      packedCarry.second->setIncomingValue(carry.backedgeIndex,
                                           emitPack(update->second, plan));
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
      if (instruction.getType()->isFloatingPointTy() &&
          isPackedOperation(instruction))
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
  DominatorTree dominators;
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
