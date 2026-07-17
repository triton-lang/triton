#ifndef TRITONINSTRUMENT_CONSAN_CONSTANTS_H
#define TRITONINSTRUMENT_CONSAN_CONSTANTS_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir::triton::instrument {

inline constexpr llvm::StringLiteral kConSanExtraCaptureBytesAttr =
    "consan.extra_capture_bytes";

// Shared-memory addresses are represented by their 24-bit object-relative
// address throughout the instrumentation dialect. Keep static allocator
// regions within the same representable range used by memdesc materialization.
inline constexpr uint32_t kSharedMemoryObjectMask = (1u << 24) - 1;

} // namespace mlir::triton::instrument

#endif // TRITONINSTRUMENT_CONSAN_CONSTANTS_H
