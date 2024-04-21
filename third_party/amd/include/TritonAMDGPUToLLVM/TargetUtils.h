#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETUTILS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETUTILS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::AMD {

// A list of ISA families we care about.
enum class ISAFamily {
  Unknown,
  CDNA1,
  CDNA2,
  CDNA3,
  RDNA1,
  RDNA2,
  RDNA3,
};

// Deduces the corresponding ISA family for the given target gfx |arch|.
ISAFamily deduceISAFamily(llvm::StringRef arch);

} // namespace mlir::triton::AMD

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETUTILS_H
