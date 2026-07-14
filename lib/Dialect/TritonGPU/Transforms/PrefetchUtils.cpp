#include "triton/Dialect/TritonGPU/Transforms/PrefetchUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-prefetch-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {

namespace {

// Internal helper: find the LocalLoadOp at the bottom of a single-operand
// chain rooted at `v`. Less restrictive than findLocalLoadForDotOperand: does
// not require single-use links and does not check the load encoding. Used
// only by getLocalLoadToken below; clients that need the chain itself should
// use findLocalLoadForDotOperand.
LocalLoadOp findLocalLoadOpInChain(Value v) {
  Operation *op = v.getDefiningOp();
  while (op) {
    if (auto load = dyn_cast<LocalLoadOp>(op))
      return load;
    if (op->getNumOperands() != 1)
      break;
    op = op->getOperand(0).getDefiningOp();
  }
  return nullptr;
}

} // namespace

FailureOr<SmallVector<Value>> findLocalLoadForDotOperand(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return failure();
  bool foundLocalLoad = false;
  SmallVector<Value> rets;
  rets.push_back(op->getResult(0));
  LDBG("Looking for local_load starting at: " << *op);
  while (op) {
    if (!op->getResult(0).hasOneUse())
      return failure();
    if (auto ll = dyn_cast<LocalLoadOp>(op)) {
      if (isa<DotOperandEncodingAttr>(ll.getType().getEncoding())) {
        rets.push_back(op->getOperand(0));
        foundLocalLoad = true;
        break;
      }
      return failure();
    }
    if (op->getNumOperands() != 1)
      return failure();
    rets.push_back(op->getOperand(0));
    op = op->getOperand(0).getDefiningOp();
    if (op)
      LDBG("op between dot and local_load: " << *op);
  }
  std::reverse(rets.begin(), rets.end());

  if (foundLocalLoad)
    return rets;
  return failure();
}

Value getLocalLoadToken(Value dotOperand) {
  if (auto load = findLocalLoadOpInChain(dotOperand))
    return load.getToken();
  return Value();
}

void clonePrefetchElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                 OpBuilder &builder) {
  // Nothing to clone if the chain has no intermediate ops; `ret` already
  // points at the freshly created LocalLoadOp.
  if (vals.size() <= 2)
    return;
  IRMapping mapping;
  // vals[1] is the original LocalLoadOp result; substitute it with the new
  // prefetched result so that downstream clones pick up the new value.
  mapping.map(vals[1], ret);
  for (size_t i = 2; i < vals.size(); ++i) {
    Value v = vals[i];
    Value curr = builder.clone(*v.getDefiningOp(), mapping)->getResult(0);
    if (isa<RankedTensorType>(curr.getType())) {
      // The prefetched slice has a smaller shape than the original tensor;
      // rewrite the cloned op's result type to match the new shape while
      // preserving the cloned op's element type and operand encoding.
      auto retType = RankedTensorType::get(
          cast<RankedTensorType>(ret.getType()).getShape(),
          cast<RankedTensorType>(curr.getType()).getElementType(),
          cast<RankedTensorType>(curr.getDefiningOp()->getOperand(0).getType())
              .getEncoding());
      curr.setType(retType);
    }
    mapping.map(v, curr);
  }
  ret = mapping.lookup(vals.back());
}

bool isLoopCarriedValue(scf::ForOp forOp, Value v) {
  auto arg = dyn_cast_if_present<BlockArgument>(v);
  return arg && arg.getOwner() == forOp.getBody() &&
         arg.getArgNumber() >= forOp.getNumInductionVars();
}

Value getYieldOperand(scf::ForOp forOp, scf::YieldOp yieldOp, Value v) {
  auto arg = mlir::cast<BlockArgument>(v);
  unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
  return yieldOp.getOperand(yieldIdx);
}

bool isPromotableValue(scf::ForOp forOp, Value v) {
  // Null operands are treated as trivially promotable so callers can pass
  // optional values (e.g. local_load's async-wait token) uniformly.
  if (!v)
    return true;
  if (auto arg = dyn_cast<BlockArgument>(v))
    // Block args owned by another region are already available where we
    // materialize. Loop-carried iter_args and the induction var are remapped
    // by the caller.
    return arg.getOwner() != forOp.getBody() ||
           isLoopCarriedValue(forOp, arg) || arg == forOp.getInductionVar();
  // Loop-carried iter_args reach here as their next-iter values; they will be
  // remapped to either the init value or the yielded value during rewrite.
  if (isLoopCarriedValue(forOp, v))
    return true;
  Operation *op = v.getDefiningOp();
  // Other values without a defining op (block arguments of outer blocks) are
  // assumed safe.
  if (!op)
    return true;
  // Values defined outside this loop body are already available where we
  // materialize the prologue/yield expressions, so they do not need cloning.
  if (op->getBlock() != forOp.getBody())
    return true;
  // Nested control flow is not handled by the cloning logic below.
  if (op->getNumRegions() != 0)
    return false;
  // Only clone simple elementwise/constant ops plus the specific loop-local
  // ops needed to rebuild the async-wait + memdesc-index chain.
  if (!op->hasTrait<OpTrait::Elementwise>() &&
      !op->hasTrait<OpTrait::ConstantLike>() &&
      !isa<AsyncWaitOp, MemDescIndexOp>(op))
    return false;
  // Every operand must also be promotable, otherwise the whole expression is
  // rejected (we cannot partially clone).
  return llvm::all_of(op->getOperands(), [&](Value operand) {
    return isPromotableValue(forOp, operand);
  });
}

Value cloneLoopValue(scf::ForOp forOp, Value v, OpBuilder &builder,
                     llvm::function_ref<Value(BlockArgument)> mapBlockArg,
                     DenseMap<Value, Value> &cache) {
  // Null values are allowed for optional operands such as local_load tokens.
  if (!v)
    return Value();
  // Reuse previously cloned values when reconstructing a shared expression
  // DAG (e.g. an async_wait feeding both A and B).
  if (auto it = cache.find(v); it != cache.end())
    return it->second;
  // Block arguments are remapped by the caller depending on whether we are
  // materializing the loop init or the yielded next-iteration value.
  if (auto arg = dyn_cast<BlockArgument>(v))
    return cache[v] = mapBlockArg(arg);
  Operation *op = v.getDefiningOp();
  // Values defined outside this loop body can be reused directly.
  if (op->getBlock() != forOp.getBody())
    return cache[v] = v;

  // Recursively rebuild the loop-local expression with remapped operands.
  IRMapping operandMapping;
  for (Value operand : op->getOperands())
    operandMapping.map(
        operand, cloneLoopValue(forOp, operand, builder, mapBlockArg, cache));
  Operation *clonedOp = builder.clone(*op, operandMapping);
  for (auto [result, clonedResult] :
       llvm::zip(op->getResults(), clonedOp->getResults()))
    cache[result] = clonedResult;
  return cache[v];
}

Value materializeInitValue(scf::ForOp forOp, Value v, OpBuilder &builder,
                           DenseMap<Value, Value> &cache) {
  return cloneLoopValue(
      forOp, v, builder,
      [&](BlockArgument arg) -> Value {
        if (arg.getOwner() != forOp.getBody())
          return arg;
        if (arg == forOp.getInductionVar())
          return forOp.getLowerBound();
        return forOp.getTiedLoopInit(arg)->get();
      },
      cache);
}

Value materializeYieldValue(scf::ForOp forOp, scf::YieldOp yieldOp, Value v,
                            OpBuilder &builder, IRMapping &mapping,
                            DenseMap<Value, Value> &cache) {
  return cloneLoopValue(
      forOp, v, builder,
      [&](BlockArgument arg) -> Value {
        if (arg.getOwner() != forOp.getBody())
          return arg;
        if (arg == forOp.getInductionVar())
          return arith::AddIOp::create(builder, forOp.getLoc(),
                                       mapping.lookupOrDefault(arg),
                                       forOp.getStep());
        return mapping.lookupOrDefault(getYieldOperand(forOp, yieldOp, arg));
      },
      cache);
}

bool isBroadcastedAlongCTABlock(Value v) {
  auto type = dyn_cast<TensorOrMemDesc>(v.getType());
  if (!type)
    return false;
  auto kBlock = StringAttr::get(v.getContext(), "block");
  auto cgaLayout = getCGALayout(type.getEncoding()).getLinearLayout();
  if (!cgaLayout.hasInDim(kBlock))
    return false;
  return cgaLayout.getFreeVariableMasks()[kBlock] != 0;
}

//===----------------------------------------------------------------------===//
// Per-dot prefetch tracking
//===----------------------------------------------------------------------===//

Value DotPrefetchSources::get(Operation *dot, bool isA, bool isToken) const {
  if (isToken)
    return isA ? aToken.lookup(dot) : bToken.lookup(dot);
  return isA ? aSource.lookup(dot) : bSource.lookup(dot);
}

const DenseMap<Operation *, unsigned> &
DotPrefetchCarriedArgs::get(bool isA, bool isToken) const {
  if (isToken)
    return isA ? aToken : bToken;
  return isA ? aSource : bSource;
}

bool DotPrefetchCarriedArgs::contains(Operation *dot, bool isA,
                                      bool isToken) const {
  return get(isA, isToken).contains(dot);
}

void appendMaterializedLoopArgIfNeeded(scf::ForOp forOp, Operation *dot,
                                       Value value,
                                       DenseMap<Operation *, unsigned> &argMap,
                                       SmallVector<Value> &loopArgs,
                                       OpBuilder &builder,
                                       DenseMap<Value, Value> &cache) {
  if (!value || isLoopCarriedValue(forOp, value))
    return;
  argMap[dot] = loopArgs.size();
  loopArgs.push_back(materializeInitValue(forOp, value, builder, cache));
}

Value getCurrentTrackedValue(scf::ForOp forOp, Operation *dot, bool isA,
                             bool isToken, scf::ForOp newForOp,
                             IRMapping &mapping,
                             const DotPrefetchSources &sources,
                             const DotPrefetchCarriedArgs &carriedArgs) {
  Value value = sources.get(dot, isA, isToken);
  if (!value)
    return Value();
  // If the original source/token is loop-carried, the local_load is done
  // outside the loop and we can directly use the mapped iter_arg.
  if (isLoopCarriedValue(forOp, value))
    return mapping.lookupOrDefault(value);
  const auto &argMap = carriedArgs.get(isA, isToken);
  auto it = argMap.find(dot);
  if (it == argMap.end())
    // No iter_arg was allocated for this value (e.g. null token); fall back
    // to the cloned-in-body version for sources, or null for tokens.
    return isToken ? Value() : mapping.lookupOrDefault(value);
  // The iter_arg was initialized outside the loop and threaded through as a
  // new region iter_arg.
  return newForOp.getRegionIterArgs()[it->second];
}

Value getNextTrackedValue(scf::ForOp forOp, scf::YieldOp yieldOp,
                          Operation *dot, bool isA, bool isToken,
                          OpBuilder &builder, IRMapping &mapping,
                          const DotPrefetchSources &sources) {
  Value value = sources.get(dot, isA, isToken);
  if (!value)
    return Value();
  if (isLoopCarriedValue(forOp, value))
    return mapping.lookupOrDefault(getYieldOperand(forOp, yieldOp, value));

  DenseMap<Value, Value> yieldCache;
  return materializeYieldValue(forOp, yieldOp, value, builder, mapping,
                               yieldCache);
}

} // namespace mlir::triton::gpu
