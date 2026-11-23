#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_

#include "triton/Analysis/Membar/SmemModel.h"
#include "llvm/ADT/UniqueVector.h"
#include <tuple>

namespace mlir::triton::AMD {

class SMemOrdering final : public membar::SMemOrdering {
public:
  void classify(membar::SMemAccess &access, Operation *op,
                const MemoryAccess &effect, membar::ThreadEffects threads,
                Allocation &allocation) override;
  bool canSkip(const membar::SMemAccess &source, const membar::SMemAccess &sink,
               Allocation &allocation) const override;

private:
  // Token ordering requires a common payload buffer in the analyzed frame.
  // Intern sets so revisiting an op in the forward analysis keeps its key.
  using Buffers = SmallVector<Allocation::BufferId, 1>;
  // Keep an unannotated load only to diagnose it if a copy matches.
  llvm::UniqueVector<std::tuple<Allocation *, Buffers, Operation *>> payloads;
};

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
