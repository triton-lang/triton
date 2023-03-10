#ifndef TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_
#define TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_

#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {
class ConversionPatternRewriter;
class Location;

namespace triton {
using llvm::StringRef;

inline std::string strJoin(llvm::ArrayRef<std::string> strs,
                           llvm::StringRef delimiter) {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);
  for (size_t i = 0; !strs.empty() && i < strs.size() - 1; ++i)
    os << strs[i] << delimiter;
  if (!strs.empty())
    os << strs.back();
  os.flush();
  return osStr;
}

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_
