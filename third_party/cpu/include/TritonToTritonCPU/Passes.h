#ifndef TRITONTOTRITONCPU_CONVERSION_PASSES_H
#define TRITONTOTRITONCPU_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/AxisInfo.h"
#include "llvm/ADT/TypeSwitch.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace cpu {

#define GEN_PASS_DECL
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertElementwiseOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertElemManipOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertMemoryOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertPtrOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertDotOp();
std::unique_ptr<OperationPass<ModuleOp>> createConvertControlFlowOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertHistogramOp();
std::unique_ptr<OperationPass<ModuleOp>> createConvertReductionOp();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertReductionOp(bool useReductionOp, bool useMultiDimReductionOp);
std::unique_ptr<OperationPass<ModuleOp>> createConvertScanOp();
std::unique_ptr<OperationPass<ModuleOp>> createConvertAtomicOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertDebugOps();

std::unique_ptr<OperationPass<ModuleOp>> createScalarizeUsingForOpPass();

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"

template <typename T, typename... Ts>
constexpr bool is_one_of_v = (std::is_same_v<T, Ts> || ...);

template <class OpTy>
constexpr bool is_memory_op_v =
    is_one_of_v<OpTy, triton::LoadOp, triton::StoreOp>;

inline mlir::Type getMemoryOpType(triton::LoadOp operation) {
  return operation.getType();
}

inline mlir::Type getMemoryOpType(triton::StoreOp operation) {
  return operation.getValue().getType();
}

inline ArrayRef<int64_t> getShape(mlir::Type type) {
  return llvm::TypeSwitch<Type, ArrayRef<int64_t>>(type)
      .Case([](ShapedType t) { return t.getShape(); })
      .Case([](RankedTensorType t) { return t.getShape(); })
      .Default([](Type t) {
        llvm::errs() << "Attempt to getShape from unknow type: " << t << "\n";
        llvm_unreachable("Unsupported type in getShape");
        return ArrayRef<int64_t>();
      });
}

inline bool hasShape(mlir::Type type) {
  return isa<ShapedType, RankedTensorType>(type);
}

template <class OpTy,
          typename std::enable_if_t<is_memory_op_v<OpTy>, bool> = true>
bool isContiguousRowMajorAccess(AxisInfo *axisInfo, OpTy op) {
  if (!axisInfo)
    return false;

  mlir::Type type = getMemoryOpType(op);
  if (!hasShape(type)) {
    return false;
  }
  auto shape = getShape(type);
  auto contiguity = axisInfo->getContiguity();
  return (shape.back() > 1 && shape.back() == contiguity.back());
}

} // namespace cpu
} // namespace triton

} // namespace mlir

#endif
