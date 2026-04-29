#include "TritonMetalGPUTransforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

namespace tt = mlir::triton;

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonmetal-inject-tensor-stride-args"

namespace mlir {
#define GEN_PASS_DEF_TRITONMETALGPUINJECTTENSORSTRIDEARGS
#include "TritonMetalGPUTransforms/Passes.h.inc"

struct TritonMetalGPUInjectTensorStrideArgsPass
    : impl::TritonMetalGPUInjectTensorStrideArgsBase<
          TritonMetalGPUInjectTensorStrideArgsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    mod.walk([&](tt::FuncOp funcOp) {
      if (!triton::isKernel(funcOp))
        return;

      auto *ctx = funcOp.getContext();
      // stride arg is a pointer to a packed array of i64 strides
      auto stridePtrTy =
          tt::PointerType::get(IntegerType::get(ctx, 64), /*addressSpace=*/1);

      // collect ptr arg indices before modification
      SmallVector<unsigned> ptrArgIndices;
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        if (isa<tt::PointerType>(funcOp.getArgument(i).getType()))
          ptrArgIndices.push_back(i);
      }

      // insert stride arg immediately after each tensor ptr arg
      // track insertOffset since insertion shifts later indices
      unsigned insertOffset = 0;
      for (unsigned origIdx : ptrArgIndices) {
        unsigned insertIdx = origIdx + 1 + insertOffset;
        (void)funcOp.insertArgument(insertIdx, stridePtrTy,
                                    DictionaryAttr::get(ctx), funcOp.getLoc());
        // Record which original ptr arg this stride belongs to.
        funcOp.setArgAttr(insertIdx, "metal.implicit_stride_for",
                          IntegerAttr::get(IntegerType::get(ctx, 32),
                                           origIdx + insertOffset));
        insertOffset++;
      }
    });
  }
};

} // namespace mlir
