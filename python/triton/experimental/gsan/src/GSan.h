#pragma once

#include "Compat.h"

namespace gsan {

// The table and its pointer lists are immutable while in flight.
// Clocks use u32 so all CTAs can publish with a native atomic maximum.
struct LaunchState {
  const uint32_t *const *entryClocks;
  uint64_t numEntryClocks;
  const uint32_t *const *waitClocks;
  uint64_t numWaitClocks;
  uint32_t *completionClock;
};

// Most pools reserve 8 TiB. Byte-granular normal shadow needs 24 bytes
// per data byte, so give it 24 TiB of shadow and 1 TiB of data.
static constexpr size_t kPoolReserveSize = 1ull << 43;
// A normal/write-once pair for each power-of-two granularity from 1 to 16.
// The low pool-index bit selects the mode; the remaining bits encode log2
// granularity. Pool offsets are relative to the reservation base.
static constexpr int kNumPools = 10;
static constexpr size_t kByteDataSize = 1ull << 40;
static constexpr size_t kBytePoolSize = 25 * kByteDataSize;
static constexpr size_t kReserveSize =
    kBytePoolSize + (kNumPools - 1) * kPoolReserveSize;
// Default normal granularity. Write-once allocations default to one byte.
static constexpr int kShadowMemGranularityBytes = 4;

using thread_id_t = uint16_t;

enum class AtomicScope : uint8_t {
  NonAtomic,
  CTA,
  GPU,
  System,
  MAX_VALUE = System,
};

using epoch_t = uint16_t;

// The release flag applies only to writes in normal shadow cells. Keep it
// alongside the read count so stored clocks retain all 16 epoch bits.
struct ScalarClock {
  epoch_t epoch;
  thread_id_t threadId;
  AtomicScope scope;
  bool isRelease;
};

struct alignas(4) PackedClock {
  uint32_t epoch : 16;
  uint32_t threadId : 14;
  AtomicScope scope : 2;
};

inline GSAN_HOST_DEVICE PackedClock packClock(ScalarClock clock) {
  PackedClock packed{};
  packed.epoch = clock.epoch;
  packed.threadId = clock.threadId;
  packed.scope = clock.scope;
  return packed;
}

inline GSAN_HOST_DEVICE ScalarClock unpackClock(PackedClock clock,
                                                bool isRelease = false) {
  return {static_cast<epoch_t>(clock.epoch),
          static_cast<thread_id_t>(clock.threadId), clock.scope, isRelease};
}
static constexpr int kMaxThreads = 1 << 14;
static_assert(sizeof(PackedClock) == 4);
static_assert(static_cast<int>(AtomicScope::MAX_VALUE) == 3);

// TODO: Change to struct-of-array for better coalescing?
struct alignas(4) ShadowCell {
  static constexpr int kReadClockSize = 4;
  static constexpr uint32_t kLockBit = 1u << 16;
  static constexpr uint32_t kReadCountMask = kLockBit - 1;
  static constexpr uint32_t kWriteReleaseBit = 1u << 17;
  PackedClock readClocks[kReadClockSize];
  PackedClock writeClock;
  // Keep the saturating 16-bit read count and lock in one native atomic word.
  uint32_t readCountAndLock;
};
static_assert(sizeof(ShadowCell) == 24);
static_assert(alignof(ShadowCell) == 4);

// The entire scalar clock is accessed atomically. There is no lock or
// reader state, and zero denotes a cell without an instrumented write.
struct WriteOnceShadowCell {
  PackedClock writeClock;
};
static_assert(sizeof(WriteOnceShadowCell) == sizeof(PackedClock));
static_assert(alignof(WriteOnceShadowCell) == alignof(PackedClock));

inline GSAN_HOST_DEVICE constexpr size_t getShadowCellSize(bool writeOnce) {
  return writeOnce ? sizeof(WriteOnceShadowCell) : sizeof(ShadowCell);
}

struct GlobalState {
  // Base address of gsan managed memory
  uintptr_t reserveBase;
  uintptr_t globalsBase;

  uint32_t rngSeed;

  thread_id_t numSms;
  thread_id_t numDevices;
  // numThreads = numSms * numDevices
  thread_id_t numThreads;

  uint16_t clockBufferSize;
};

struct ThreadState {
  GlobalState *globals;
  uintptr_t reserveBase;

  // monotonic counter, used for stochastic read clock updates
  uint32_t numReads;

  uint32_t gdcWaitCalled : 1;

  // Whether the vector clock changed since the last snapshot (to allow reuse).
  uint32_t clockBufferDirty : 1;

  // Keep the head separate from flags modified without the clock-buffer lock.
  uint32_t clockBufferHead;

  // Per-SM lock serializing vector-clock changes and snapshot publication.
  // Remote snapshot readers use clockBufferLock instead.
  uint32_t lock;

  // System-scope reader-writer lock. Readers pin snapshots while importing
  // without a shadow lock; writers hold it only while publishing a snapshot.
  uint32_t clockBufferLock;

  thread_id_t threadId;

  // Local vector clock, shape [numThreads]
  // Followed by the clock buffer, shape [clockBufferSize, numThreads]
  epoch_t vectorClock[];
};

// The widest supported scalar atomic is eight bytes, including in byte pools.
static constexpr int kMaxAtomicShadowCells = 8;

struct AtomicEventState {
  ThreadState *threadState;
  ShadowCell *cells[kMaxAtomicShadowCells];
  uint8_t numCells;
};

static constexpr int kMaxClusterCTAs = 16;
static constexpr int kClusterBarrierScratchBytes = 128;

struct alignas(16) ClusterBarrierState {
  uint32_t lock;
  uint32_t arrivalCount;
  uint32_t publicationCount;
  uint32_t generation;
  uint32_t registrationGeneration;
  thread_id_t threadIds[kMaxClusterCTAs];
  epoch_t tokens[kMaxClusterCTAs];
};
static_assert(sizeof(ClusterBarrierState) <= kClusterBarrierScratchBytes);

struct MBarrierPublishedClock {
  // An immutable snapshot whose publisher has not advanced its epoch. A wait
  // must exclude that thread's current epoch when acquiring the snapshot.
  thread_id_t threadId;
  epoch_t token;
};
static_assert(sizeof(MBarrierPublishedClock) == 4);

struct MBarrierPhaseState {
  // One-based generation tag. The initially completed phase 1 has tag zero.
  uint32_t generation;
  uint32_t complete;
  MBarrierPublishedClock clocks[kMaxClusterCTAs];
};
static_assert(sizeof(MBarrierPhaseState) == 72);

static constexpr uint32_t kEmptyMBarrierKey = 0xffffffffu;

struct alignas(16) MBarrierState {
  // Key uniquely identifying the barrier. kEmptyMBarrierKey represents empty.
  uint32_t key;
  uint32_t lock;
  uint32_t expectedCount;
  uint32_t pendingCount;
  // Number of completed phases. The active phase has this zero-based index.
  uint32_t generation;
  MBarrierPhaseState phases[2];
};
static_assert(sizeof(MBarrierState) == 176);

struct alignas(16) MBarrierTable {
  uint32_t lock;
  uint32_t capacity;
  MBarrierState states[];
};
static_assert(sizeof(MBarrierTable) == 16);

// Place the thread state for each device at a fixed stride for ease of
// address calculation.
static constexpr uintptr_t kPerDeviceStateStride = 1ull << 32;
static constexpr uintptr_t kMaxGPUs = 64;
static constexpr uintptr_t kGlobalsReserveSize =
    kPerDeviceStateStride * kMaxGPUs;

inline GSAN_HOST_DEVICE GlobalState *getGlobalState(ThreadState *threadState) {
  auto threadAddr = (uintptr_t)threadState;
  return (GlobalState *)(threadAddr & ~(kPerDeviceStateStride - 1));
}

inline GSAN_HOST_DEVICE bool isValidShadowGranularity(int granularity) {
  return granularity > 0 && granularity <= 16 &&
         (granularity & (granularity - 1)) == 0;
}

inline GSAN_HOST_DEVICE int getPoolIndexForGranularity(int granularity,
                                                       bool writeOnce = false) {
  int log2Granularity = granularity == 1   ? 0
                        : granularity == 2 ? 1
                        : granularity == 4 ? 2
                        : granularity == 8 ? 3
                                           : 4;
  return 2 * log2Granularity + writeOnce;
}

inline GSAN_HOST_DEVICE int getPoolIndex(uintptr_t address,
                                         uintptr_t reserveBase) {
  auto offset = address - reserveBase;
  return offset < kBytePoolSize
             ? 0
             : 1 + (offset - kBytePoolSize) / kPoolReserveSize;
}

inline GSAN_HOST_DEVICE constexpr size_t poolOffset(int pool) {
  return pool == 0 ? 0 : kBytePoolSize + (pool - 1) * kPoolReserveSize;
}

inline GSAN_HOST_DEVICE constexpr size_t poolDataOffset(int pool) {
  // Normal four-byte cells can back 1 TiB of data with 6 TiB of shadow.
  if (pool == 0)
    return kBytePoolSize - kByteDataSize;
  if (pool == 4)
    return kPoolReserveSize - (1ull << 40);
  return kPoolReserveSize / 2;
}

// Assumes address is in this process's GSan reservation.
inline GSAN_HOST_DEVICE int getShadowGranularity(uintptr_t address,
                                                 uintptr_t reserveBase) {
  int pool = getPoolIndex(address, reserveBase);
  return 1 << (pool >> 1);
}

inline GSAN_HOST_DEVICE uintptr_t getRealBaseAddress(
    uintptr_t reserveBase, int granularity = kShadowMemGranularityBytes,
    bool writeOnce = false) {
  int pool = getPoolIndexForGranularity(granularity, writeOnce);
  return reserveBase + poolOffset(pool) + poolDataOffset(pool);
}

// Assumes address is in this process's GSan reservation.
inline GSAN_HOST_DEVICE bool isWriteOnceAddress(uintptr_t addr,
                                                uintptr_t reserveBase) {
  return (getPoolIndex(addr, reserveBase) & 1) != 0;
}

inline GSAN_HOST_DEVICE size_t getShadowCellSize(uintptr_t addr,
                                                 uintptr_t reserveBase) {
  return getShadowCellSize(isWriteOnceAddress(addr, reserveBase));
}

// Assumes address is in gsan-managed memory
inline GSAN_HOST_DEVICE uintptr_t getShadowAddress(uintptr_t virtualAddress,
                                                   uintptr_t reserveBase) {
  int pool = getPoolIndex(virtualAddress, reserveBase);
  auto poolBase = reserveBase + poolOffset(pool);
  auto realBase = poolBase + poolDataOffset(pool);
  auto byteOffset = virtualAddress - realBase;
  auto cellOffset =
      byteOffset / getShadowGranularity(virtualAddress, reserveBase);
  return poolBase + getShadowCellSize(virtualAddress, reserveBase) * cellOffset;
}

inline GSAN_HOST_DEVICE bool isGsanManaged(uintptr_t addr,
                                           uintptr_t reserveBase) {
  auto offset = addr - reserveBase;
  if (offset >= kReserveSize)
    return false;
  int pool = getPoolIndex(addr, reserveBase);
  return offset - poolOffset(pool) >= poolDataOffset(pool);
}

enum class AddressCategory { Unmanaged, Normal, WriteOnce };

inline GSAN_HOST_DEVICE AddressCategory categorize(uintptr_t addr,
                                                   uintptr_t reserveBase) {
  if (!isGsanManaged(addr, reserveBase))
    return AddressCategory::Unmanaged;
  return isWriteOnceAddress(addr, reserveBase) ? AddressCategory::WriteOnce
                                               : AddressCategory::Normal;
}

inline GSAN_HOST_DEVICE bool isAtomicScope(AtomicScope scope) {
  return scope != AtomicScope::NonAtomic;
}

} // namespace gsan
