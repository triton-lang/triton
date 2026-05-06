#include "OffsetUniformitySplit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace mlir::LLVM::AMD {
namespace {

// Triton's tensor offsets reach the buffer-load conversion as scalar
// `llvm.extractvalue` ops sitting on top of a chain of `llvm.insertvalue`
// ops that pack the per-element offsets into a struct. Walk that chain to
// recover the actual offset SSA value at the requested position so the
// additive splitter and uniformity checker can see through it.
Value lookThroughExtractValue(Value v) {
  while (auto extract = v.getDefiningOp<LLVM::ExtractValueOp>()) {
    auto position = extract.getPosition();
    if (position.size() != 1)
      return v;
    int64_t targetIdx = position[0];
    Value cur = extract.getContainer();
    bool found = false;
    for (int steps = 0; steps < 4096 && cur; ++steps) {
      auto insert = cur.getDefiningOp<LLVM::InsertValueOp>();
      if (!insert)
        break;
      auto insertPos = insert.getPosition();
      if (insertPos.size() == 1 && insertPos[0] == targetIdx) {
        v = insert.getValue();
        found = true;
        break;
      }
      cur = insert.getContainer();
    }
    if (!found)
      return v;
  }
  return v;
}

class UniformityChecker {
public:
  bool isUniform(Value v) {
    auto it = cache.find(v);
    if (it != cache.end())
      return it->second;
    // Cycle break: optimistically assume uniform if we hit `v` mid-recursion
    // (e.g. `%k = phi [0, ...], [%k_next, ...]` where `%k_next = add(%k, c)`).
    // The result is finalized once the chain closes.
    if (!inFlight.insert(v).second)
      return true;
    bool result = compute(v);
    inFlight.erase(v);
    cache[v] = result;
    return result;
  }

private:
  bool compute(Value v) {
    v = lookThroughExtractValue(v);

    if (auto blockArg = dyn_cast<BlockArgument>(v))
      return computeBlockArg(blockArg);

    Operation *def = v.getDefiningOp();
    if (!def)
      return false;

    if (isa<LLVM::ConstantOp, arith::ConstantOp, ROCDL::ReadfirstlaneOp,
            ROCDL::BlockIdXOp, ROCDL::BlockIdYOp, ROCDL::BlockIdZOp,
            ROCDL::WaveId>(def))
      return true;

    if (isa<ROCDL::ThreadIdXOp, ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>(def))
      return false;

    if (isa<gpu::ThreadIdOp, gpu::LaneIdOp>(def))
      return false;

    // Pure arithmetic / cast ops are uniform iff every operand is uniform.
    if (isa<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::ShlOp, LLVM::LShrOp,
            LLVM::AShrOp, LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp, LLVM::SExtOp,
            LLVM::ZExtOp, LLVM::TruncOp, LLVM::SelectOp, LLVM::ICmpOp,
            LLVM::URemOp, LLVM::SRemOp, LLVM::UDivOp, LLVM::SDivOp,
            LLVM::BitcastOp, LLVM::SMinOp, LLVM::SMaxOp, LLVM::UMinOp,
            LLVM::UMaxOp, LLVM::AbsOp, LLVM::PtrToIntOp, LLVM::IntToPtrOp,
            arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp,
            arith::ShRSIOp, arith::ShRUIOp, arith::AndIOp, arith::OrIOp,
            arith::XOrIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
            arith::SelectOp, arith::CmpIOp, arith::IndexCastOp,
            arith::IndexCastUIOp, arith::BitcastOp,
            arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
            arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp>(
            def)) {
      for (Value op : def->getOperands())
        if (!isUniform(op))
          return false;
      return true;
    }

    return false;
  }

  bool computeBlockArg(BlockArgument blockArg) {
    Block *block = blockArg.getOwner();
    Operation *parent = block->getParentOp();
    if (block->isEntryBlock()) {
      // Entry block of a function: kernel arguments are uniform by
      // construction in Triton.
      if (isa<FunctionOpInterface>(parent))
        return true;
      // Entry block of a non-function region (e.g. an scf.for body): be
      // conservative; the producer may carry per-lane state.
      return false;
    }

    unsigned argIdx = blockArg.getArgNumber();
    for (Block *pred : block->getPredecessors()) {
      Operation *term = pred->getTerminator();
      if (!areIncomingValuesUniform(term, block, argIdx))
        return false;
    }
    return true;
  }

  bool areIncomingValuesUniform(Operation *term, Block *target,
                                unsigned argIdx) {
    if (auto br = dyn_cast<LLVM::BrOp>(term)) {
      auto operands = br.getDestOperands();
      if (argIdx < operands.size())
        return isUniform(operands[argIdx]);
      return false;
    }
    if (auto cb = dyn_cast<LLVM::CondBrOp>(term)) {
      bool sawTargetEdge = false;
      if (cb.getTrueDest() == target) {
        sawTargetEdge = true;
        auto operands = cb.getTrueDestOperands();
        if (argIdx >= operands.size() || !isUniform(operands[argIdx]))
          return false;
      }
      if (cb.getFalseDest() == target) {
        sawTargetEdge = true;
        auto operands = cb.getFalseDestOperands();
        if (argIdx >= operands.size() || !isUniform(operands[argIdx]))
          return false;
      }
      return sawTargetEdge;
    }
    return false;
  }

  llvm::DenseMap<Value, bool> cache;
  llvm::DenseSet<Value> inFlight;
};

bool isAddOrDisjointOr(Operation *op) {
  if (!op)
    return false;
  if (isa<LLVM::AddOp, arith::AddIOp>(op))
    return true;
  if (auto orOp = dyn_cast<LLVM::OrOp>(op))
    return orOp.getIsDisjoint();
  return false;
}

void collectAddTreeLeaves(Value v, UniformityChecker &uniformity,
                          SmallVectorImpl<Value> &uniformLeaves,
                          SmallVectorImpl<Value> &perLaneLeaves) {
  v = lookThroughExtractValue(v);
  Operation *def = v.getDefiningOp();
  // Walk through every additive node we can recognize. Shared sub-trees
  // get re-summed in the partitioned output; downstream CSE/DCE folds the
  // duplicates back into a single computation.
  if (def && isAddOrDisjointOr(def)) {
    collectAddTreeLeaves(def->getOperand(0), uniformity, uniformLeaves,
                         perLaneLeaves);
    collectAddTreeLeaves(def->getOperand(1), uniformity, uniformLeaves,
                         perLaneLeaves);
    return;
  }
  if (uniformity.isUniform(v))
    uniformLeaves.push_back(v);
  else
    perLaneLeaves.push_back(v);
}

bool isLiteralZero(Value v) {
  auto def = v.getDefiningOp<LLVM::ConstantOp>();
  if (!def)
    return false;
  if (auto attr = dyn_cast<IntegerAttr>(def.getValue()))
    return attr.getValue().isZero();
  return false;
}

Value sumValues(ArrayRef<Value> vs, RewriterBase &rewriter, Location loc) {
  if (vs.empty())
    return Value();
  Value sum = vs.front();
  for (Value v : vs.drop_front())
    sum = LLVM::AddOp::create(rewriter, loc, sum, v).getResult();
  return sum;
}

} // namespace

std::pair<Value, Value> splitUniformAdditive(Value offset,
                                             RewriterBase &rewriter,
                                             Location loc) {
  UniformityChecker uniformity;
  SmallVector<Value> uniformLeaves;
  SmallVector<Value> perLaneLeaves;
  collectAddTreeLeaves(offset, uniformity, uniformLeaves, perLaneLeaves);

  // Drop literal-zero uniform leaves; they add no value to soffset.
  uniformLeaves.erase(std::remove_if(uniformLeaves.begin(),
                                     uniformLeaves.end(), isLiteralZero),
                      uniformLeaves.end());

  if (uniformLeaves.empty())
    return {Value(), offset};

  Value uniform = sumValues(uniformLeaves, rewriter, loc);
  if (perLaneLeaves.empty()) {
    Value zero = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(0))
                     .getResult();
    return {uniform, zero};
  }
  Value perLane = sumValues(perLaneLeaves, rewriter, loc);
  return {uniform, perLane};
}

} // namespace mlir::LLVM::AMD
