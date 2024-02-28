#include "TargetInfo.h"

namespace mlir::triton::NVIDIA {
bool TargetInfo::isSupported() const { return computeCapability >= 80; }
} // namespace mlir::triton::NVIDIA
