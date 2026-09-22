#pragma once

#include <optional>
#include <utility>

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::nvidia_gpu {

struct NvmmaSmemAttrs {
  unsigned swizzlingByteWidth = 0;
  bool transposed = false;
  bool fp4Padded = false;
};

// Internal helper for creating SMEM descriptors for MMA instructions. The
// nvmmaSmemLL input must map two logical dims to offset/block outputs.
std::optional<std::pair<NvmmaSmemAttrs, LinearLayout>>
getNvmmaSmemAttrs(const LinearLayout &nvmmaSmemLL, unsigned bitwidth);

std::optional<NvmmaSmemAttrs> getNvmmaSmemAttrs(gpu::MemDescType memTy);

} // namespace mlir::triton::nvidia_gpu
