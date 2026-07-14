#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PREFETCHUTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PREFETCHUTILS_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

namespace mlir::triton::gpu {

class LocalLoadOp;

//===----------------------------------------------------------------------===//
// Local-load chain discovery
//===----------------------------------------------------------------------===//

// Walk backwards from a dot operand through single-operand, single-use
// elementwise ops until reaching a LocalLoadOp with DotOperandEncodingAttr.
// On success, returns the chain of values from the LocalLoadOp's memdesc
// source through to the dot operand, in def-use order:
//
//   vals[0]            = LocalLoadOp source memdesc
//   vals[1]            = LocalLoadOp result
//   vals[2..size()-1]  = intermediate elementwise op results, ending at the
//                        value the dot consumes
//
// Returns failure if any link in the chain has multiple uses, multiple
// operands, or if the chain does not end at a LocalLoadOp whose result has
// DotOperandEncodingAttr.
FailureOr<SmallVector<Value>> findLocalLoadForDotOperand(Value v);

// Returns the async-wait token consumed by the LocalLoadOp that feeds a
// dot operand, or a null Value if none is found.
Value getLocalLoadToken(Value dotOperand);

//===----------------------------------------------------------------------===//
// Elementwise chain re-cloning
//===----------------------------------------------------------------------===//

// Clones the elementwise ops between the LocalLoadOp and the dot operand,
// remapping them to a new prefetched slice value.
//
// `vals` MUST be the chain returned by findLocalLoadForDotOperand: vals[1] is
// the LocalLoadOp result that is being substituted, and vals[2..] is the
// downstream chain (each op is single-operand and is cloned once).
// `ret` enters as the new prefetched LocalLoadOp result and is updated
// in-place to the cloned tail of the chain. If the chain has no intermediate
// ops (vals.size() == 2), `ret` is left unchanged.
void clonePrefetchElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                 OpBuilder &builder);

//===----------------------------------------------------------------------===//
// SCF for-loop helpers
//===----------------------------------------------------------------------===//

// True if `v` is the next-iteration value of a loop-carried iter_arg of
// `forOp` (i.e. a non-induction-var block argument of forOp's body). Returns
// false for null values, induction vars, and arguments belonging to other
// blocks.
bool isLoopCarriedValue(scf::ForOp forOp, Value v);

// Given a loop-carried iter_arg of `forOp`, returns the corresponding yielded
// (next-iteration) operand from `yieldOp`. The caller must ensure `v` is a
// loop-carried block argument; passing the induction var or an unrelated
// value asserts.
Value getYieldOperand(scf::ForOp forOp, scf::YieldOp yieldOp, Value v);

//===----------------------------------------------------------------------===//
// Promotable expressions
//===----------------------------------------------------------------------===//

// True if the expression rooted at `v` can be re-cloned into the prologue
// (with loop-carried args replaced by their inits and the induction var by
// the lower bound) or at the end of the loop body (with loop-carried args
// replaced by their yielded next-iteration values and the induction var
// advanced).
//
// Null values are trivially promotable so callers can pass optional operands
// such as a local_load's async-wait token uniformly. The set of supported
// in-body ops is intentionally small: Elementwise, ConstantLike, plus the
// loop-local AsyncWaitOp / MemDescIndexOp that appear in pipelined async
// memdesc-index chains.
bool isPromotableValue(scf::ForOp forOp, Value v);

// Recursively clone the expression DAG rooted at `v`, remapping block
// arguments via the provided callback. Values defined outside `forOp`'s body
// and null values are returned as-is. `cache` is shared across calls so that
// shared subexpressions (e.g. an async_wait feeding both A and B) are cloned
// exactly once.
Value cloneLoopValue(scf::ForOp forOp, Value v, OpBuilder &builder,
                     llvm::function_ref<Value(BlockArgument)> mapBlockArg,
                     DenseMap<Value, Value> &cache);

// Materialize `v` (expected to be promotable per isPromotableValue) at the
// current builder insertion point, replacing the induction var with `forOp`'s
// lower bound and loop-carried args with their init values.
Value materializeInitValue(scf::ForOp forOp, Value v, OpBuilder &builder,
                           DenseMap<Value, Value> &cache);

// Materialize `v` (expected to be promotable) at the current builder
// insertion point as the yielded (next-iteration) value: the induction var is
// replaced with iv+step and loop-carried args with their yielded operands,
// looked up through `mapping`.
Value materializeYieldValue(scf::ForOp forOp, scf::YieldOp yieldOp, Value v,
                            OpBuilder &builder, IRMapping &mapping,
                            DenseMap<Value, Value> &cache);

//===----------------------------------------------------------------------===//
// Layout queries
//===----------------------------------------------------------------------===//

// Returns true if `v`'s encoding broadcasts along the "block" dimension of
// its CGA layout (i.e. the value is replicated across CTAs). Returns false if
// `v`'s type is not a TensorOrMemDesc or its encoding has no "block" input
// dimension. Prefetching such values is incorrect: they would duplicate a
// broadcast.
bool isBroadcastedAlongCTABlock(Value v);

//===----------------------------------------------------------------------===//
// Per-dot prefetch tracking
//===----------------------------------------------------------------------===//

// Maps populated during `initialize()` of a prefetch pass: for each dot we
// remember the shared-memory source (either a loop-carried iter_arg or a
// promotable in-body expression like memdesc_index), the optional async-wait
// token feeding the local_load, and the chain of intermediate elementwise
// ops between the local_load and the dot operand (used to re-clone through
// clonePrefetchElementwiseOps).
struct DotPrefetchSources {
  DenseMap<Operation *, Value> aSource;
  DenseMap<Operation *, Value> bSource;
  DenseMap<Operation *, Value> aToken;
  DenseMap<Operation *, Value> bToken;
  DenseMap<Operation *, SmallVector<Value>> aVals;
  DenseMap<Operation *, SmallVector<Value>> bVals;

  // Returns the source memdesc (isToken=false) or async-wait token
  // (isToken=true) tracked for `dot`. Returns a null Value if not tracked.
  Value get(Operation *dot, bool isA, bool isToken) const;
};

// For each dot whose source/token is NOT a loop-carried iter_arg in the
// original loop, we materialize the in-body expression as a new iter_arg of
// the rewritten loop and remember the iter-arg index here. Empty entries
// mean the original value is already loop-carried (or null) and no new
// iter_arg was needed.
struct DotPrefetchCarriedArgs {
  DenseMap<Operation *, unsigned> aSource;
  DenseMap<Operation *, unsigned> bSource;
  DenseMap<Operation *, unsigned> aToken;
  DenseMap<Operation *, unsigned> bToken;

  const DenseMap<Operation *, unsigned> &get(bool isA, bool isToken) const;
  bool contains(Operation *dot, bool isA, bool isToken) const;
};

// If `value` is non-null and not already a loop-carried iter_arg of `forOp`,
// materializes it into the prologue (via materializeInitValue + `cache`),
// records its index in `argMap`, and appends it to `loopArgs`. Otherwise a
// no-op.
void appendMaterializedLoopArgIfNeeded(scf::ForOp forOp, Operation *dot,
                                       Value value,
                                       DenseMap<Operation *, unsigned> &argMap,
                                       SmallVector<Value> &loopArgs,
                                       OpBuilder &builder,
                                       DenseMap<Value, Value> &cache);

// Returns the value to use INSIDE the new loop body for a tracked
// source/token:
//   - If the original value is already a loop-carried iter_arg, returns the
//     mapped iter_arg of the new loop.
//   - Else, if a new iter_arg was materialized in `carriedArgs`, returns it.
//   - Else (e.g. a null token), returns a null Value for tokens or the cloned
//     in-body version for sources via `mapping.lookupOrDefault`.
Value getCurrentTrackedValue(scf::ForOp forOp, Operation *dot, bool isA,
                             bool isToken, scf::ForOp newForOp,
                             IRMapping &mapping,
                             const DotPrefetchSources &sources,
                             const DotPrefetchCarriedArgs &carriedArgs);

// Returns the value to YIELD (i.e. pass to the next iteration) for a tracked
// source/token. For loop-carried originals this is the mapped yield operand;
// for promoted expressions, the expression is cloned with the induction var
// advanced and iter_args substituted by their yielded successors.
Value getNextTrackedValue(scf::ForOp forOp, scf::YieldOp yieldOp,
                          Operation *dot, bool isA, bool isToken,
                          OpBuilder &builder, IRMapping &mapping,
                          const DotPrefetchSources &sources);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_PREFETCHUTILS_H_
