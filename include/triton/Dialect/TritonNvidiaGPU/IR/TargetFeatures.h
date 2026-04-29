#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_TARGETFEATURES_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_TARGETFEATURES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <cassert>

namespace mlir::triton::nvidia_gpu {

class TargetFeatures {
public:
  explicit TargetFeatures(int computeCapability)
      : computeCapability(computeCapability) {}

  static TargetFeatures fromModuleOp(ModuleOp moduleOp) {
    auto targetAttr =
        moduleOp->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
    assert(targetAttr && "Expected a target attribute on the module operation");

    StringRef targetName = targetAttr.getValue();
    assert(targetName.starts_with(kTargetPrefix) &&
           "expected target attribute to be prefixed with \"cuda:\"");

    int computeCapability;
    bool parseError = targetName.drop_front(sizeof(kTargetPrefix) - 1)
                          .getAsInteger(10, computeCapability);
    assert(!parseError &&
           "invalid compute capability string in target attribute");

    return TargetFeatures(computeCapability);
  }

  int getComputeCapability() const { return computeCapability; }

  bool supportClusterOps() const {
    return computeCapability >= 90 && computeCapability / 10 != 12;
  }

  bool supportMaximumMinimum() const { return computeCapability >= 80; }

  bool supportLdMatrix() const { return computeCapability >= 75; }
  bool supportStMatrix() const { return computeCapability >= 90; }
  bool supportLdStMatrixB8() const { return computeCapability >= 100; }

  bool supportBitwidth16Elementwise() const {
    // Hopper (sm90) and newer.
    return computeCapability >= 90;
  }

  bool supportBitwidth32Elementwise() const {
    // Blackwell (sm100) and newer.
    return computeCapability >= 100;
  }

private:
  static constexpr char kTargetPrefix[] = "cuda:";

  int computeCapability;
};

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_TARGETFEATURES_H_
