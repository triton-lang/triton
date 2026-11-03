#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

class NvidiaGPUAssignDescriptorMemoryLayouts
    : public ttg::AssignDescriptorMemoryLayouts {
public:
  NvidiaGPUAssignDescriptorMemoryLayouts() = default;

private:
  Attribute buildFallbackSharedEncoding(mlir::MLIRContext *ctx,
                                        ArrayRef<int64_t> shape,
                                        ArrayRef<unsigned> order,
                                        ttg::CGAEncodingAttr cgaLayout,
                                        Type elementType) override;
  Attribute getCompatibleSharedEncoding(Attribute enc, ArrayRef<int64_t> shape,
                                        Type elementType) override;
  bool isCompatibleSharedEncoding(Attribute enc) override;
};

bool NvidiaGPUAssignDescriptorMemoryLayouts::isCompatibleSharedEncoding(
    Attribute enc) {
  if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(enc))
    // Only 128-byte swizzling is supported for padded FP4 TMA descriptors.
    // MMA operands can still use padded layouts with smaller swizzles.
    return !nvmma.getTransposed() &&
           (!nvmma.getFp4Padded() || nvmma.getSwizzlingByteWidth() == 128);
  return false;
}

Attribute NvidiaGPUAssignDescriptorMemoryLayouts::getCompatibleSharedEncoding(
    Attribute enc, ArrayRef<int64_t> shape, Type elementType) {
  if (isCompatibleSharedEncoding(enc))
    return enc;

  auto sharedLinear = dyn_cast<ttg::SharedLinearEncodingAttr>(enc);
  if (!sharedLinear)
    return {};

  auto *ctx = enc.getContext();
  auto cgaLayout = ttg::getCGALayout(sharedLinear);
  auto order = ttg::getOrder(sharedLinear, shape);
  auto sharedLinearLayout = ttg::toLinearLayout(shape, sharedLinear);
  auto isEquivalent = [&](ttg::NVMMASharedEncodingAttr candidate) {
    auto candidateLayout = ttg::nvmmaSharedToLinearLayout(
        shape, candidate, ttg::TMAMode::Tiled, /*disableSwizzle=*/false,
        /*emitErrors=*/false);
    return succeeded(candidateLayout) && *candidateLayout == sharedLinearLayout;
  };

  // TMA descriptors only support non-transposed layouts. Preserve Triton's
  // default shape/order-based choice when it already matches this
  // shared_linear layout. The full candidate scan below is only a fallback for
  // equivalent non-transposed layouts not selected by the heuristic builder.
  unsigned elementBitWidth = std::max(8u, elementType.getIntOrFloatBitWidth());
  unsigned preferredSwizzle =
      ttg::NVMMASharedEncodingAttr::getDefaultSwizzlingByteWidth(
          shape, order, cgaLayout, elementBitWidth, /*fp4Padded=*/false);
  auto preferred = ttg::NVMMASharedEncodingAttr::get(
      ctx, preferredSwizzle, /*transposed=*/false, elementBitWidth,
      /*fp4Padded=*/false, cgaLayout);
  if (isEquivalent(preferred))
    return preferred;

  for (unsigned swizzle : {0u, 32u, 64u, 128u}) {
    if (swizzle == preferredSwizzle)
      continue;
    auto candidate = ttg::NVMMASharedEncodingAttr::get(
        ctx, swizzle, /*transposed=*/false, elementBitWidth,
        /*fp4Padded=*/false, cgaLayout);
    if (isEquivalent(candidate))
      return candidate;
  }

  // Only 128-byte swizzling is supported for padded FP4 TMA descriptors.
  auto fp4Padded = ttg::NVMMASharedEncodingAttr::get(
      ctx, /*swizzlingByteWidth=*/128, /*transposed=*/false, elementBitWidth,
      /*fp4Padded=*/true, cgaLayout);
  if (isEquivalent(fp4Padded))
    return fp4Padded;

  return {};
}

// Build fallback encoding given shape, order, cga layout and element type
Attribute NvidiaGPUAssignDescriptorMemoryLayouts::buildFallbackSharedEncoding(
    mlir::MLIRContext *ctx, ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
    ttg::CGAEncodingAttr cgaLayout, Type elementType) {
  return ttg::NVMMASharedEncodingAttr::get(ctx, shape, order, cgaLayout,
                                           elementType, /*fp4Padded*/ false);
}

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZEDESCRIPTORENCODINGPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

class TritonNvidiaGPUOptimizeDescriptorEncodingPass
    : public impl::TritonNvidiaGPUOptimizeDescriptorEncodingPassBase<
          TritonNvidiaGPUOptimizeDescriptorEncodingPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeDescriptorEncodingPassBase<
      TritonNvidiaGPUOptimizeDescriptorEncodingPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    NvidiaGPUAssignDescriptorMemoryLayouts assignMemoryLayouts;
    assignMemoryLayouts.assignMemoryLayouts(m);
  }
};

} // namespace mlir::triton::nvidia_gpu
