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
/// Membar passes the earlier access first. The token orders a LocalLoad
/// after the copies it waits for even when the subviews used for pipelining
/// may alias. A loop that refills the tile still needs both barriers below.
///
///     %copy = AsyncCopyGlobalToLocal %ptr, %tile
///     scf.for ... iter_args(%pending = %copy) {
///       %wait = AsyncWait %pending
///       ttg.barrier local
///       %value = LocalLoad %tile token %wait
///       ttg.barrier local
///       %next = AsyncCopyGlobalToLocal %next_ptr, %tile
///       scf.yield %next
///     }
///
/// The first joins the per-wave waits (RAW). The second waits for every
/// thread to finish reading the tile (WAR). See amdgpu_membar.mlir.
bool membarFilter(const membar::SMemAccess &source,
                  const membar::SMemAccess &sink, Allocation &allocation);
} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
