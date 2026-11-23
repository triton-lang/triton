#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_

namespace mlir {
class Allocation;
namespace triton::membar {
struct SMemAccess;
}
} // namespace mlir

namespace mlir::triton::AMD {

/// Exempt accesses ordered by AMD async-copy and LDS barrier protocols.
/// A token can order a load after a copy. A subsequent refill must still
/// wait for all readers.
bool membarFilter(const membar::SMemAccess &before,
                  const membar::SMemAccess &after, Allocation &allocation);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
