#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_UTILS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_UTILS_H_

#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {

// Adapt the encoding for the given tensor type.
SharedEncodingTrait updateEncodingForShape(Operation *op,
                                           SharedEncodingTrait encoding,
                                           RankedTensorType tensorType);

/// Backend-specific callbacks during descriptor encoding assignment
struct DescriptorAnalysisCallbacks {
  /// Callback to check for compatible shared encoding
  llvm::function_ref<bool(Attribute)> isCompatibleSharedEncoding;

  /// create a fallback encoding given the shape, order, cga layout and
  /// element type
  llvm::function_ref<Attribute(mlir::MLIRContext *, ArrayRef<int64_t>,
                               ArrayRef<unsigned>, CGAEncodingAttr, Type)>
      buildFallbackSharedEncoding;
};

/// Utility class to assign memory layouts to tensor descriptors in a module.
class AssignDescriptorMemoryLayouts {
public:
  AssignDescriptorMemoryLayouts() = default;
  explicit AssignDescriptorMemoryLayouts(
      const DescriptorAnalysisCallbacks &callbacks);
  void assignMemoryLayouts(ModuleOp &mod);

private:
  void runOnFunction(FuncOp &func);

  DescriptorAnalysisCallbacks callbacks;
};
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_UTILS_H_
