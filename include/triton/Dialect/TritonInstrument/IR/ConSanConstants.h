#ifndef TRITONINSTRUMENT_CONSAN_CONSTANTS_H
#define TRITONINSTRUMENT_CONSAN_CONSTANTS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::instrument {

inline constexpr llvm::StringLiteral kConSanExtraCaptureBytesAttr =
    "consan.extra_capture_bytes";

} // namespace mlir::triton::instrument

#endif // TRITONINSTRUMENT_CONSAN_CONSTANTS_H
