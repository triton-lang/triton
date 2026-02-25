#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#define GSAN_HOST_DEVICE __host__ __device__
#else
#define GSAN_HOST_DEVICE
#endif

namespace gsan {

// Reserve 1 PiB, should be big enough for a while :)
static constexpr size_t kReserveSize = 1ull << 40;
static constexpr int kShadowMemGranularityBytes = 4;
static_assert((kReserveSize & (kReserveSize - 1)) == 0,
              "kReserveSize must be a power of 2");

using thread_id_t = uint16_t;

enum class AtomicScope : uint8_t {
  NonAtomic,
  CTA,
  GPU,
  System,
  MAX_VALUE = System,
};

using epoch_t = uint16_t;

struct alignas(4) ScalarClock {
  epoch_t epoch;
  thread_id_t threadId : 12; // Supports 4096 threads
  AtomicScope scope : 2;
};
static constexpr int kMaxThreads = 1 << 12;
static_assert(sizeof(ScalarClock) == 4);
static_assert(static_cast<int>(AtomicScope::MAX_VALUE) == 3);

// TODO: Change to struct-of-array for better coalescing?
struct alignas(4) ShadowCell {
  static constexpr int kReadClockSize = 4;
  ScalarClock readClocks[kReadClockSize];
  ScalarClock writeClock;
  uint16_t numReads;
  uint16_t lock;
};
static_assert(sizeof(ShadowCell) == 24);
static_assert(alignof(ShadowCell) == 4);

static constexpr int kShadowSizeBytes = sizeof(ShadowCell);

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

  // Index to head of the circular clock buffer, plus a bit to mark if the
  // vector clock has changed since the last clock buffer write (to allow reuse)
  uint32_t clockBufferDirty : 1;
  uint32_t clockBufferHead : 31;

  // Reader-writer lock controlling access to the vector clock and clock buffer
  uint32_t lock;

  thread_id_t threadId;

  // Local vector clock, shape [numThreads]
  // Followed by the clock buffer, shape [clockBufferSize, numThreads]
  epoch_t vectorClock[];
};

// Place the thread state for each device at a fixed stride for ease of
// address calculation.
static constexpr uintptr_t kPerDeviceStateStride = 1ull << 30;
static constexpr uintptr_t kMaxGPUs = 16;
static constexpr uintptr_t kGlobalsReserveSize =
    kPerDeviceStateStride * kMaxGPUs;

inline GSAN_HOST_DEVICE GlobalState *getGlobalState(ThreadState *threadState) {
  auto threadAddr = (uintptr_t)threadState;
  return (GlobalState *)(threadAddr & ~(kPerDeviceStateStride - 1));
}

inline GSAN_HOST_DEVICE uintptr_t getRealBaseAddress(uintptr_t reserveBase) {
  return reserveBase + kReserveSize / 2;
}

inline GSAN_HOST_DEVICE uintptr_t getReserveBaseFromAddress(uintptr_t addr) {
  return addr & ~(kReserveSize - 1);
}

// Assumes address is in gsan-managed memory
inline GSAN_HOST_DEVICE uintptr_t getShadowAddress(uintptr_t virtualAddress) {
  auto reserveBase = getReserveBaseFromAddress(virtualAddress);
  auto realBase = getRealBaseAddress(reserveBase);
  auto byteOffset = virtualAddress - realBase;
  auto wordOffset = byteOffset / kShadowMemGranularityBytes;
  return reserveBase + kShadowSizeBytes * wordOffset;
}

inline GSAN_HOST_DEVICE bool isGsanManaged(uintptr_t addr,
                                           uintptr_t reserveBase) {
  return getReserveBaseFromAddress(addr) == reserveBase;
}

} // namespace gsan
