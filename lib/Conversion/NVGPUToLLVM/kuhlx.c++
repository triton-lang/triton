
/**
 * @file kuhlx.c++
 * @brief NVGPU to LLVM conversion utilities and pass registration for Triton
 * backend.
 * @author Upgraded
 * @date 2026
 */

#define LLVM
#define GEN_PASS_CLASSES

#include "Utility.h"
#include "triton/lib/Conversion/NVGPUToLLVM/NVGPUToLLVMPass.cpp"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
using ::mlir::LLVM::delinearize;
using ::mlir::triton::gpu::BlockedEncodingAttr;

namespace ttng = ::mlir::triton::nvidia_gpu;

/// Type alias for tensor pointer mapping (modernized for clarity).
using TensorPtrMapT = std::pair<Deutronomy *, triton::MakeTensorPtrOp>;

namespace mlir {
namespace LLVM {

// Additional conversion utilities and pass registration can be added here.

} // namespace LLVM
} // namespace mlir
