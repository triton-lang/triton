#ifndef TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_UTILITY_H
#define TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_UTILITY_H

#include "triton/Analysis/Membar/SmemModel.h"

namespace mlir::triton::NVIDIA {

class SMemOrdering final : public membar::SMemOrdering {
public:
  explicit SMemOrdering(ModuleAllocation &module);

  void classify(membar::SMemAccess &access, Operation *op,
                const MemoryAccess &effect, membar::ThreadEffects threads,
                Allocation &allocation) override;
  bool canSkip(const membar::SMemAccess &source, const membar::SMemAccess &sink,
               Allocation &allocation) const override;

private:
  bool hasOnlySourceAliases(const membar::SMemAccess &source,
                            const membar::SMemAccess &sink,
                            Allocation &allocation) const;

  // Only entry allocations have one lifetime across different calls.
  DenseMap<Operation *, Operation *> entryKernels;
  // Zero excludes counters that lack a participating issuer in each CTA.
  DenseMap<Value, uint32_t> counterKeys;
  // Commutativity requires both the same operation and the same element type.
  DenseMap<std::pair<unsigned, Type>, uint32_t> atomicKeys;
};

} // namespace mlir::triton::NVIDIA

#endif // TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_UTILITY_H
