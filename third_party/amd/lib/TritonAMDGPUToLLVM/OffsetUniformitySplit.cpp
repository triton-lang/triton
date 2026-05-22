#include "OffsetUniformitySplit.h"

#include "TritonAMDGPUToLLVM/UniformityAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>

namespace mlir::LLVM::AMD {
namespace {

// From UniformityAnalysis: peels extractvalue/insertvalue chains.
using mlir::triton::AMD::lookThroughExtractValue;

bool isAddOrDisjointOr(Operation *op) {
  if (!op)
    return false;
  if (isa<LLVM::AddOp, arith::AddIOp>(op))
    return true;
  if (auto orOp = dyn_cast<LLVM::OrOp>(op))
    return orOp.getIsDisjoint();
  return false;
}

template <typename IsUniformFn>
void collectAddTreeLeavesWith(Value v, IsUniformFn isUniform,
                              SmallVectorImpl<Value> &uniformLeaves,
                              SmallVectorImpl<Value> &perLaneLeaves) {
  v = lookThroughExtractValue(v);
  Operation *def = v.getDefiningOp();
  if (def && isAddOrDisjointOr(def)) {
    collectAddTreeLeavesWith(def->getOperand(0), isUniform, uniformLeaves,
                             perLaneLeaves);
    collectAddTreeLeavesWith(def->getOperand(1), isUniform, uniformLeaves,
                             perLaneLeaves);
    return;
  }
  if (isUniform(v))
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

std::pair<Value, Value>
splitUniformAdditiveImpl(Value offset, RewriterBase &rewriter, Location loc,
                         llvm::function_ref<bool(Value)> isUniform) {
  SmallVector<Value> uniformLeaves;
  SmallVector<Value> perLaneLeaves;
  collectAddTreeLeavesWith(offset, isUniform, uniformLeaves, perLaneLeaves);

  llvm::erase_if(uniformLeaves, isLiteralZero);

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

} // namespace

std::pair<Value, Value> splitUniformAdditive(Value offset,
                                             RewriterBase &rewriter,
                                             Location loc,
                                             const DataFlowSolver *solver) {
  assert(solver && "splitUniformAdditive requires a non-null DataFlowSolver");
  if (!solver)
    return {Value(), offset};
  auto isUniform = [&](Value v) {
    return mlir::triton::AMD::isUniformValue(v, *solver);
  };
  return splitUniformAdditiveImpl(offset, rewriter, loc, isUniform);
}

} // namespace mlir::LLVM::AMD
