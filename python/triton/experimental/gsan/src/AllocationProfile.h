#pragma once

#include "Compat.h"

namespace gsan_profile {
using gsan::uint32_t;
using gsan::uintptr_t;

// A separate CUDA reservation, independent of every GSan race-checking pool.
// The lower half contains compact metadata; the upper half contains data.
constexpr uintptr_t kReserveSize = 1ull << 40;
constexpr uintptr_t kDataSize = kReserveSize / 2;
constexpr int kGranularity = 16;

// ProfileCell encoding shared by instrumentation and the Python decoder.
// The low bits track written bytes; the high bits are monotonic observations
// and forbidden shadow granularities (1, 2, 4, 8, 16).
constexpr uint32_t kWrittenMask = (1u << kGranularity) - 1;
constexpr uint32_t kFlagsMask = ~kWrittenMask;
constexpr uint32_t kRepeatedWrite = 1u << 16;
constexpr uint32_t kReadBeforeWrite = 1u << 17;
constexpr uint32_t kAtomic = 1u << 18;
constexpr uint32_t kRead = 1u << 19;
constexpr uint32_t kWrite = 1u << 20;
constexpr int kGranularityShift = 24;
constexpr int kNumGranularities = 5;

struct ProfileCell {
  uint32_t observations;
};
static_assert(sizeof(ProfileCell) == 4);

inline GSAN_HOST_DEVICE bool isManaged(uintptr_t address, uintptr_t reserve) {
  auto offset = address - reserve;
  return offset >= kDataSize && offset < kReserveSize;
}

inline GSAN_HOST_DEVICE uintptr_t metadataAddress(uintptr_t address) {
  auto base = address & ~(kReserveSize - 1);
  return base +
         (address - base - kDataSize) / kGranularity * sizeof(ProfileCell);
}
} // namespace gsan_profile
