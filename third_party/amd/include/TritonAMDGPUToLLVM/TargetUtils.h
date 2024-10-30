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

// Here is a partial definition of DppCtrl enums. For the complete definition,
// please check:
// https://github.com/llvm/llvm-project/blob/llvmorg-19.1.3/llvm/lib/Target/AMDGPU/SIDefines.h
enum DppCtrl : uint32_t {
  ROW_SHL0 = 0x100,
  ROW_SHR0 = 0x110,
  BCAST15 = 0x142,
  BCAST31 = 0x143
};

} // namespace mlir::triton::AMD

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETUTILS_H
