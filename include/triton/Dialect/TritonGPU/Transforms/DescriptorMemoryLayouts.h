#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_MEMORY_LAYOUTS_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_MEMORY_LAYOUTS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include <unordered_set>

namespace mlir::triton::gpu {
struct UseInfo;
struct EncodingInfo;

/// Update shared encoding given a new shape
SharedEncodingTrait updateEncodingForShape(Operation *op,
                                           SharedEncodingTrait encoding,
                                           RankedTensorType tensorType);

//===----------------------------------------------------------------------===//
// AssignDescriptorMemoryLayouts
//===----------------------------------------------------------------------===//

/// Assign memory layouts to tensor descriptors in a module.
class AssignDescriptorMemoryLayouts {
public:
  AssignDescriptorMemoryLayouts() = default;
  void assignMemoryLayouts(ModuleOp &mod);

private:
  void runOnFunction(FuncOp &func);
  const EncodingInfo *
  internEncoding(std::unordered_set<EncodingInfo> &encodings,
                 EncodingInfo info);
  EncodingInfo combineEncodings(const EncodingInfo &lhs,
                                const EncodingInfo &rhs, unsigned rank);
  Attribute findLoadEncodingFromUsers(Operation *op);
  std::optional<UseInfo> getUseInfo(Operation *op);
  Attribute getFallbackSharedEncoding(RankedTensorType tensorType,
                                      CGAEncodingAttr cgaLayout,
                                      ArrayRef<int64_t> usageShape,
                                      unsigned numCTAs);
  // Override with backend specific implementation
  virtual Attribute buildFallbackSharedEncoding(mlir::MLIRContext *,
                                                ArrayRef<int64_t>,
                                                ArrayRef<unsigned>,
                                                CGAEncodingAttr, Type) = 0;
  virtual bool isCompatibleSharedEncoding(Attribute) = 0;
};

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_DESCRIPTOR_MEMORY_LAYOUTS_H_
