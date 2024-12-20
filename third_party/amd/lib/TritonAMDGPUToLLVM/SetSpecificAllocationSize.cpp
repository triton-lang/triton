#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <numeric>

using namespace mlir;
namespace mlir::triton {
#define GEN_PASS_DEF_SETSPECIFICALLOCATIONSIZE
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

using namespace mlir;
using namespace mlir::triton;

struct SetSpecificAllocationSize
    : public mlir::triton::impl::SetSpecificAllocationSizeBase<
          SetSpecificAllocationSize> {
  explicit SetSpecificAllocationSize(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  void runOnOperation() override {
    triton::AMD::TargetInfo targetInfo(this->arch.getValue());
    ModuleOp mod = getOperation();
    mod.walk([&](triton::AtomicRMWOp atomOp) -> void {
      Value res = atomOp.getResult();
      Value value = atomOp.getVal();
      auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
      Type elemTy = tensorTy ? tensorTy.getElementType() : value.getType();
      size_t elemSize = elemTy.getIntOrFloatBitWidth();
      if (!LLVM::AMD::isRuntimeLdsReductionforAtomicApplicable(
              atomOp, targetInfo.getISAFamily()))
        return;

      auto layout = dyn_cast<RankedTensorType>(res.getType()).getEncoding();
      size_t elemsToAlloc = triton::gpu::getWarpSize(layout) *
                            triton::gpu::getNumWarpsPerCTA(layout);
      size_t size = elemsToAlloc * std::max<int>(8, elemSize) / 8;
      atomOp->setAttr(
          "allocation.size",
          IntegerAttr::get(IntegerType::get(atomOp.getContext(), 32), size));
    });
  }
};

} // namespace

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createSetSpecificAllocationSizePass(StringRef targetArch) {
  return std::make_unique<SetSpecificAllocationSize>(targetArch);
}

} // namespace mlir::triton::AMD
