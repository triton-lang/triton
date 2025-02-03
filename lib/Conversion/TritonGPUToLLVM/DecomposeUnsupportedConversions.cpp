#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

static void addAttrs(Operation *op, ArrayRef<mlir::NamedAttribute> attrs) {
  for (const NamedAttribute attr : attrs)
    op->setAttr(attr.getName(), attr.getValue());
}

} // namespace

namespace mlir::triton::gpu {

void decomposeTensorCoreToDotLayoutConversion(ModuleOp module,
                                              ShortcutFn shortcutFn) {
  int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(module);
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(module);
  int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);

  module.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
    OpBuilder builder(cvtOp);
    auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
    auto dstType = cast<RankedTensorType>(cvtOp.getType());
    auto srcMma = dyn_cast<MmaEncodingTrait>(srcType.getEncoding());
    auto dstDotOp =
        dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
    if (srcMma && dstDotOp && !shortcutFn(srcType, dstType)) {
      auto tmpType = RankedTensorType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::BlockedEncodingAttr::get(
              module.getContext(), srcType.getShape(), getSizePerThread(srcMma),
              getOrder(srcMma), numWarps, threadsPerWarp, numCTAs));
      auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), tmpType, cvtOp.getSrc());
      addAttrs(tmp, cvtOp->getAttrs());
      auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), dstType, tmp);
      addAttrs(newConvert, cvtOp->getAttrs());
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    }
  });
}

void decomposeBlockedToDotLayoutConversion(ModuleOp module) {
  int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(module);
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(module);
  int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);
  module.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
    OpBuilder builder(cvtOp);
    auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
    auto dstType = cast<RankedTensorType>(cvtOp.getType());
    if (!cvtNeedsSharedMemory(srcType, dstType))
      return;
    auto srcBlocked =
        dyn_cast<triton::gpu::BlockedEncodingAttr>(srcType.getEncoding());
    auto dstDotOp =
        dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
    if (srcBlocked && dstDotOp) {
      auto dotParent = dyn_cast<NvidiaMmaEncodingAttr>(dstDotOp.getParent());
      if (dotParent) {
        return;
      }
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      auto tmpType = MemDescType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::SwizzledSharedEncodingAttr::get(
              module.getContext(), dstDotOp, srcType.getShape(),
              srcBlocked.getOrder(), srcBlocked.getCTALayout(),
              srcType.getElementType()),
          sharedMemorySpace);
      auto tmp = builder.create<triton::gpu::LocalAllocOp>(
          cvtOp.getLoc(), tmpType, cvtOp.getSrc());
      addAttrs(tmp, cvtOp->getAttrs());
      auto newConvert = builder.create<triton::gpu::LocalLoadOp>(cvtOp.getLoc(),
                                                                 dstType, tmp);
      addAttrs(newConvert, cvtOp->getAttrs());
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    }
  });
}

} // namespace mlir::triton::gpu
