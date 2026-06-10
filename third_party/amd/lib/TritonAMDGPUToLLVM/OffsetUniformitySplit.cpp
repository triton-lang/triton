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

// Walk the additive tree of `v` (rooted at `add` ops). Recurse into
// both operands of each `add`; classify every leaf as uniform or
// per-lane via `isUniform`.
void collectAddTreeLeaves(Value v, llvm::function_ref<bool(Value)> isUniform,
                          SmallVectorImpl<Value> &uniformLeaves,
                          SmallVectorImpl<Value> &perLaneLeaves) {
  v = lookThroughExtractValue(v);
  Operation *def = v.getDefiningOp();
  if (def && isa<LLVM::AddOp, arith::AddIOp>(def)) {
    collectAddTreeLeaves(def->getOperand(0), isUniform, uniformLeaves,
                         perLaneLeaves);
    collectAddTreeLeaves(def->getOperand(1), isUniform, uniformLeaves,
                         perLaneLeaves);
    return;
  }
  (isUniform(v) ? uniformLeaves : perLaneLeaves).push_back(v);
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
                                             Location loc,
                                             const DataFlowSolver *solver) {
  assert(solver && "splitUniformAdditive requires a non-null DataFlowSolver");

  auto isUniform = [&](Value v) {
    return mlir::triton::AMD::isUniformValue(v, *solver);
  };

  SmallVector<Value> uniformLeaves;
  SmallVector<Value> perLaneLeaves;
  collectAddTreeLeaves(offset, isUniform, uniformLeaves, perLaneLeaves);

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
  return {uniform, sumValues(perLaneLeaves, rewriter, loc)};
}

} // namespace mlir::LLVM::AMD
