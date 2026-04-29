#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_OFFSETUNIFORMITYSPLIT_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_OFFSETUNIFORMITYSPLIT_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include <utility>

namespace mlir::LLVM::AMD {

// Split an i32 offset value into a (uniformPart, perLanePart) pair such that
// `offset == uniformPart + perLanePart` (a null component is treated as
// zero). The caller is expected to route `uniformPart` to the AMD raw
// buffer intrinsic's `soffset` argument and `perLanePart` to the `voffset`
// argument; this collapses uniform address arithmetic (e.g. `k * stride` in
// a K loop) from per-lane VGPR adds into a single SGPR feed.
//
// The split is built by walking the additive tree of `offset`. Both
// `llvm.add` and `llvm.or disjoint` are treated as additive nodes (the
// disjoint flag guarantees the bit ranges do not overlap). Leaves are
// classified as wave-uniform iff their defining-op chain reaches only:
//   - `llvm.mlir.constant`
//   - kernel function arguments (block arguments of a function entry block)
//   - `rocdl.workgroup.id.{x,y,z}` (`ROCDL::BlockId{X,Y,Z}Op`)
//   - `rocdl.wave.id`
//   - `rocdl.readfirstlane`
//   - pure arithmetic / casts on top of the above
//   - block arguments whose every cross-block incoming value is uniform
//     (handles loop-carried scalar IVs)
// Anything reaching `rocdl.workitem.id.*` (`ROCDL::ThreadId{X,Y,Z}Op`) or
// `gpu.thread_id` (without an intervening `readfirstlane`) is divergent.
// Unknown / opaque ops are treated conservatively as divergent.
//
// `uniformPart` is null when no uniform leaf is found, when the uniform
// leaves sum to a literal zero, or when the only uniform leaf is the
// trivial constant zero. In that case the original `offset` is returned
// as `perLanePart` and the caller should fall back to the previous
// `soffset = 0` behavior.
//
// New `llvm.add` ops that recombine the partitioned leaves are inserted at
// the current insertion point of `rewriter`. The original chain is left
// in place; downstream DCE is expected to clean up dead intermediate
// computations.
std::pair<Value, Value> splitUniformAdditive(Value offset,
                                             RewriterBase &rewriter,
                                             Location loc);

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_OFFSETUNIFORMITYSPLIT_H_
