#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_OFFSETUNIFORMITYSPLIT_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_OFFSETUNIFORMITYSPLIT_H_

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include <utility>

namespace mlir::LLVM::AMD {

// Split an i32 offset into (uniformPart, perLanePart) where
// offset == uniformPart + perLanePart. Route uniformPart to the buffer
// intrinsic's soffset (SGPR) and perLanePart to voffset (VGPR).
//
// Walks the additive tree (llvm.add, llvm.or with disjoint flag).
// Leaves are uniform if they trace back to constants, kernel args,
// block/wave IDs, readfirstlane, pure arith/casts, or uniform phis.
// Thread IDs and unknown ops are divergent.
//
// Returns null uniformPart when nothing uniform is found or when the
// uniform leaves sum to zero. The caller falls back to soffset=0.
//
// Inserts new llvm.add ops at the rewriter's insertion point. DCE
// cleans up dead intermediates.
std::pair<Value, Value> splitUniformAdditive(Value offset,
                                             RewriterBase &rewriter,
                                             Location loc,
                                             const DataFlowSolver *solver);

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_OFFSETUNIFORMITYSPLIT_H_
