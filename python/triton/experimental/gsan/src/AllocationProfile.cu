#include "AllocationProfile.h"
#include "TMA.cuh"

namespace gsan_profile {
using namespace gsan;

// The hidden state argument is the process-local GSan reservation base.
// Reuse the allocation's real/shadow mapping, including imported aliases.
GSAN_DEVICE uint32_t atomicOr(uint32_t *ptr, uint32_t bits) {
  return __scoped_atomic_fetch_or(ptr, bits, __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_SYSTEM);
}

GSAN_DEVICE uint32_t forbiddenGranularities(uintptr_t begin, uintptr_t end,
                                            int atomicBytes) {
  uint32_t result = 0;
  for (int i = 0; i < kNumGranularities; ++i) {
    int granularity = 1 << i;
    if ((begin & (granularity - 1)) || (end & (granularity - 1)) ||
        (atomicBytes &&
         (granularity > atomicBytes || granularity == kGranularity)))
      result |= 1u << (kGranularityShift + i);
  }
  return result;
}

GSAN_DEVICE void recordCell(uintptr_t cell, uintptr_t begin, uintptr_t end,
                            uint32_t flags) {
  unsigned first = begin > cell ? begin - cell : 0;
  unsigned last = end < cell + kGranularity ? end - cell : kGranularity;
  uint32_t mask = ((1u << last) - 1) & ~((1u << first) - 1);
  // The profiling pool has one contiguous four-byte cell per 16 data bytes.
  auto *shadow = reinterpret_cast<uint32_t *>(metadataAddress(cell));
  uint32_t old = atomicOr(shadow, flags | ((flags & kWrite) ? mask : 0));
  uint32_t violations = 0;
  if ((flags & kWrite) && (old & mask))
    violations |= kRepeatedWrite;
  if ((flags & kRead) && (old & mask) != mask)
    violations |= kReadBeforeWrite;
  if (violations)
    atomicOr(shadow, violations);
}

GSAN_DEVICE void recordRange(void *state, uintptr_t begin, int nbytes,
                             uint32_t flags, int atomicBytes = 0,
                             bool distribute = false) {
  if (nbytes <= 0)
    return;
  uintptr_t end = begin + nbytes;
  auto reserveBase = reinterpret_cast<uintptr_t>(state);
  // As in GSan, unmanaged memory is outside the instrumentation domain.
  if (!isManaged(begin, reserveBase))
    return;
  // Do not derive a shadow pointer outside this pool for an invalid range.
  if (end <= begin || !isManaged(end - 1, reserveBase))
    return;
  flags |= forbiddenGranularities(begin, end, atomicBytes);
  // TMA checks distribute complete 16-byte cells across a warp. In particular,
  // do not infer byte-sized accesses from the lanes used by instrumentation.
  uintptr_t cell = begin & ~uintptr_t(kGranularity - 1);
  if (distribute)
    cell += kGranularity * __nvvm_read_ptx_sreg_laneid();
  for (; cell < end; cell += distribute ? kGranularity * 32 : kGranularity)
    recordCell(cell, begin, end, flags);
}

GSAN_DEVICE void tensor(void *state, const char *stack, int count, int nbytes,
                        uint32_t flags) {
  const auto *pointers = reinterpret_cast<const uintptr_t *>(stack);
  const char *masks = stack + count * sizeof(uintptr_t);
  for (int i = 0; i < count; ++i)
    if (masks[i])
      recordRange(state, pointers[i], nbytes, flags);
}

GSAN_DEVICE void descriptor(void *state, const void *descriptor,
                            const int *coords, int rank, const int16_t *shape,
                            int bytesPerElem, int warpId, int numWarps,
                            uint32_t flags) {
  auto desc = gsan::decodeTensorDesc(descriptor, rank, bytesPerElem);
  int first[5], extent[5];
  int64_t rows = 1;
  for (int dim = 0; dim < rank; ++dim) {
    first[dim] = coords[dim] > 0 ? coords[dim] : 0;
    int64_t end = static_cast<int64_t>(coords[dim]) + shape[dim];
    end = end < desc.shape(dim) ? end : desc.shape(dim);
    if (end <= first[dim])
      return;
    extent[dim] = end - first[dim];
    if (dim + 1 < rank)
      rows *= extent[dim];
  }
  for (int64_t row = warpId; row < rows; row += numWarps) {
    int64_t index = row;
    uintptr_t address = desc.base + first[rank - 1] * bytesPerElem;
    for (int dim = rank - 2; dim >= 0; --dim) {
      address += (first[dim] + index % extent[dim]) * desc.byteStride(dim);
      index /= extent[dim];
    }
    recordRange(state, address, extent[rank - 1] * bytesPerElem, flags,
                (flags & kAtomic) ? bytesPerElem : 0, true);
  }
}

GSAN_DEVICE void indexed(void *state, const void *descriptor,
                         const int *indices, int count, int y, int columns,
                         int bytesPerElem, uint32_t flags) {
  auto desc = gsan::decodeTensorDesc(descriptor, 2, bytesPerElem);
  int64_t first = y > 0 ? y : 0;
  int64_t end = static_cast<int64_t>(y) + columns;
  end = end < desc.shape(1) ? end : desc.shape(1);
  if (end <= first)
    return;
  for (int i = 0; i < count; ++i) {
    if (indices[i] < 0 || indices[i] >= desc.shape(0))
      continue;
    recordRange(state,
                desc.base + indices[i] * desc.byteStride(0) +
                    first * bytesPerElem,
                (end - first) * bytesPerElem, flags, 0, true);
  }
}
} // namespace gsan_profile

extern "C" GSAN_DEVICE void __triton_gsan_load_tensor(void *state,
                                                      const char *stack,
                                                      int count, int bytes,
                                                      const char *, unsigned) {
  gsan_profile::tensor(state, stack, count, bytes, gsan_profile::kRead);
}
extern "C" GSAN_DEVICE void __triton_gsan_store_tensor(void *state,
                                                       const char *stack,
                                                       int count, int bytes,
                                                       const char *, unsigned) {
  gsan_profile::tensor(state, stack, count, bytes, gsan_profile::kWrite);
}
extern "C" GSAN_DEVICE void
__triton_gsan_load_tensor_desc(void *state, const void *desc, const int *coords,
                               int rank, const gsan::int16_t *shape, int bytes,
                               int pred, int warp, int warps, const char *,
                               unsigned) {
  if (pred)
    gsan_profile::descriptor(state, desc, coords, rank, shape, bytes, warp,
                             warps, gsan_profile::kRead);
}
extern "C" GSAN_DEVICE void
__triton_gsan_store_tensor_desc(void *state, const void *desc,
                                const int *coords, int rank,
                                const gsan::int16_t *shape, int bytes, int pred,
                                int warp, int warps, const char *, unsigned) {
  if (pred)
    gsan_profile::descriptor(state, desc, coords, rank, shape, bytes, warp,
                             warps, gsan_profile::kWrite);
}
extern "C" GSAN_DEVICE void __triton_gsan_load_indexed_tensor_desc(
    void *state, const void *desc, const int *indices, int count, int y,
    int columns, int bytes, int pred, const char *, unsigned) {
  if (pred)
    gsan_profile::indexed(state, desc, indices, count, y, columns, bytes,
                          gsan_profile::kRead);
}
extern "C" GSAN_DEVICE void __triton_gsan_store_indexed_tensor_desc(
    void *state, const void *desc, const int *indices, int count, int y,
    int columns, int bytes, int pred, const char *, unsigned) {
  if (pred)
    gsan_profile::indexed(state, desc, indices, count, y, columns, bytes,
                          gsan_profile::kWrite);
}
extern "C" GSAN_DEVICE void __triton_gsan_atomic_tensor_desc(
    void *state, const void *desc, const int *coords, int rank,
    const gsan::int16_t *shape, int bytes, int, int, int warp, int warps,
    const char *, unsigned) {
  gsan_profile::descriptor(state, desc, coords, rank, shape, bytes, warp, warps,
                           gsan_profile::kAtomic);
}
extern "C" GSAN_DEVICE void __triton_gsan_atomic_begin_scalar(
    void *state, void *, int pred, gsan::uintptr_t address, int bytes,
    int bytesPerElem, int, int, int, const char *, unsigned) {
  // A vectorized atomic load/store spans multiple atomic objects. Limit the
  // recommendation by the element width, independently of the access width.
  if (pred)
    gsan_profile::recordRange(state, address, bytes, gsan_profile::kAtomic,
                              bytesPerElem);
}
extern "C" GSAN_DEVICE void __triton_gsan_atomic_end_scalar(void *, int, int,
                                                            int, int, int,
                                                            const char *,
                                                            unsigned) {}

// The original hardware synchronization is retained by the compiler. Only the
// race detector's additional clock bookkeeping becomes a no-op in this mode.
extern "C" GSAN_DEVICE void __triton_gsan_init(void *, const void *,
                                               gsan::uint64_t, unsigned,
                                               unsigned, unsigned, const char *,
                                               unsigned) {}
extern "C" GSAN_DEVICE void __triton_gsan_kernel_exit(void *, const void *,
                                                      gsan::uint64_t, unsigned,
                                                      unsigned, unsigned,
                                                      const char *, unsigned) {}
extern "C" GSAN_DEVICE void
__triton_gsan_grid_dependency_wait(void *, const void *, gsan::uint64_t,
                                   unsigned, unsigned, unsigned) {}
extern "C" GSAN_DEVICE void __triton_gsan_cluster_barrier_init(void *, int) {}
extern "C" GSAN_DEVICE void __triton_gsan_cluster_barrier_sync(void *, void *,
                                                               int, int, int,
                                                               const char *,
                                                               unsigned) {}
extern "C" GSAN_DEVICE void __triton_gsan_mbarrier_table_init(void *, int,
                                                              unsigned) {}
extern "C" GSAN_DEVICE void __triton_gsan_mbarrier_init(void *, unsigned, int,
                                                        unsigned, unsigned,
                                                        const char *,
                                                        unsigned) {}
extern "C" GSAN_DEVICE void
__triton_gsan_mbarrier_arrive(void *, void *, unsigned, int, unsigned, unsigned,
                              unsigned, int, const char *, unsigned) {}
extern "C" GSAN_DEVICE void __triton_gsan_mbarrier_wait(void *, void *,
                                                        unsigned, int, unsigned,
                                                        unsigned, const char *,
                                                        unsigned) {}
