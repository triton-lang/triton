#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_WARPSPECIALIZEUTILITY_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_WARPSPECIALIZEUTILITY_H

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include <functional>
#include <optional>

namespace mlir {
namespace triton {

// Forward declaration
class TritonLLVMIRRewriter;

//===----------------------------------------------------------------------===//
// lowerWarpSpecializeBarriers
//===----------------------------------------------------------------------===//

class WarpSpecializeBarrierHelper {
public:
  virtual ~WarpSpecializeBarrierHelper() = default;

  virtual bool isBarrierOp(Operation *op) const = 0;
  virtual bool isBarrierHandleOp(Operation *op) const { return false; }
  virtual Type getBarrierHandleType(MLIRContext *ctx) const = 0;
  virtual FailureOr<Value>
  getBarrierHandle(TritonLLVMIRRewriter &b,
                   std::optional<unsigned> partitionIdx) = 0;
  virtual void createBarrier(TritonLLVMIRRewriter &b, unsigned numWarps,
                             Value handle) = 0;
  LogicalResult createBarrier(TritonLLVMIRRewriter &b, unsigned numWarps,
                              std::optional<unsigned> partitionIdx);
};

// Assign hardware barriers to each warp group and rewrite warp group barriers
// into named barrier instructions. There is a maximum number of named barriers.
LogicalResult
lowerWarpSpecializeBarriers(ModuleOp module,
                            WarpSpecializeBarrierHelper &barrierHelper);

//===----------------------------------------------------------------------===//
// convertOpTypes
//===----------------------------------------------------------------------===//

/// Convert operand types, region argument types, and result types of a
/// an operation using the provided type converter. This is used for
/// WarpSpecializeOp and related operations during lowering to LLVM.
void convertOpTypes(Operation *op, const TypeConverter &typeConverter);

//===----------------------------------------------------------------------===//
// elideTrivialCaptures
//===----------------------------------------------------------------------===//

/// Attempt to eliminate captures by rematerializing trivial computations into
/// each partition region.
void elideTrivialCaptures(LLVM::LLVMFuncOp func,
                          ArrayRef<gpu::WarpSpecializeOp> wsOps);

//===----------------------------------------------------------------------===//
// lowerWarpSpecializeCommon
//===----------------------------------------------------------------------===//

/// Callbacks for backend-specific operations during warp specialization
/// lowering.
struct WarpSpecializeCallbacks {
  /// Create a barrier to synchronize threads across the whole CTA
  std::function<void(TritonLLVMIRRewriter &)> createAllBarrier;

  /// Effective worker cap, or -1 when no adjustment is needed.
  std::function<int32_t(gpu::WarpSpecializeOp, unsigned)> getWorkerRegisterCap =
      [](gpu::WarpSpecializeOp, unsigned) { return -1; };

  std::function<void(TritonLLVMIRRewriter &, int32_t cap, bool acquire)>
      reallocWorkerRegisters = [](TritonLLVMIRRewriter &, int32_t, bool) {};
  std::function<void(TritonLLVMIRRewriter &, gpu::WarpSpecializeOp,
                     bool acquire)>
      reallocDefaultRegisters =
          [](TritonLLVMIRRewriter &, gpu::WarpSpecializeOp, bool) {};

  /// Number of consecutive worker warps sharing a command byte.
  unsigned warpsPerCommand = 1;
};

/// A single site can dispatch directly if every returning path executes it
/// once.
bool canDirectDispatch(ArrayRef<gpu::WarpSpecializeOp> wsOps);

/// Common implementation of warp specialize lowering.
/// Uses callbacks for backend-specific barrier and register reallocation
/// operations.
LogicalResult lowerWarpSpecializeCommon(
    LLVM::LLVMFuncOp func, ArrayRef<gpu::WarpSpecializeOp> wsOps, bool direct,
    Block *switchLoop, Value wid, unsigned defaultNumWarps,
    unsigned totalNumWarps, const TargetInfoBase &targetInfo,
    const WarpSpecializeCallbacks &callbacks);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_WARPSPECIALIZEUTILITY_H
