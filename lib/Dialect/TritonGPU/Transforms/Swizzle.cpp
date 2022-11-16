#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

struct SwizzlePass : public TritonGPUSwizzleBase<SwizzlePass> {
  SwizzlePass() = default;

  struct SwizzleInfo {
    int vec;
    int perPhase;
    int maxPhase;
  };

  SwizzleInfo getSwizzleMMA(int opIdx, triton::gpu::MmaEncodingAttr retEncoding,
                            RankedTensorType ty) {
    SwizzleInfo noSwizzling = {1, 1, 1};
    int version = retEncoding.getVersion();
    auto tyEncoding = ty.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto order = tyEncoding.getOrder();
    // number of rows per phase
    int perPhase = 128 / (ty.getShape()[order[0]] *
                          (ty.getElementType().getIntOrFloatBitWidth() / 8));
    perPhase = std::max<int>(perPhase, 1);
    // index of the inner dimension in `order`
    size_t inner = (opIdx == 0) ? 0 : 1;
    if (version == 1) {
      int maxPhase = (order[inner] == 1 ? 8 : 4) / perPhase;
      // TODO: handle rep (see
      // https://github.com/openai/triton/blob/master/lib/codegen/analysis/layout.cc#L209)
      int vec = 1;
      return SwizzleInfo{vec, perPhase, maxPhase};
    } else if (version == 2) {
      auto eltTy = ty.getElementType();
      std::vector<size_t> mat_shape = {8, 8,
                                       2 * 64 / eltTy.getIntOrFloatBitWidth()};
      // for now, disable swizzle when using transposed int8 tensor cores
      bool is_int8_mma = ty.getElementType().isInteger(8);
      if (is_int8_mma && order[0] == inner)
        return noSwizzling;
      // compute swizzling for A operand
      if (opIdx == 0) {
        int vec = order[0] == 1 ? mat_shape[2] : mat_shape[0]; // k : m
        int mmaStride = order[0] == 1 ? mat_shape[0] : mat_shape[2];
        int maxPhase = mmaStride / perPhase;
        return SwizzleInfo{vec, perPhase, maxPhase};
      }
      // compute swizzling for B operand
      else if (opIdx == 1) {
        int vec = order[0] == 1 ? mat_shape[1] : mat_shape[2]; // n : k
        int mmaStride = order[0] == 1 ? mat_shape[2] : mat_shape[1];
        int maxPhase = mmaStride / perPhase;
        return SwizzleInfo{vec, perPhase, maxPhase};
      } else {
        llvm_unreachable("invalid operand index");
      }
    } else
      llvm_unreachable("unsupported swizzling for provided MMA version");
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    op->walk([&](triton::DotOp dotOp) -> void {
      OpBuilder builder(dotOp);
      auto _retEncoding =
          dotOp.getResult().getType().cast<RankedTensorType>().getEncoding();
      auto retEncoding = _retEncoding.dyn_cast<triton::gpu::MmaEncodingAttr>();
      if (!retEncoding)
        return;
      for (int opIdx : {0, 1}) {
        Value op = dotOp.getOperand(opIdx);
        auto ty = op.getType().template cast<RankedTensorType>();
        // compute new swizzled encoding
        SwizzleInfo swizzle = getSwizzleMMA(opIdx, retEncoding, ty);
        auto newEncoding = triton::gpu::SharedEncodingAttr::get(
            &getContext(), swizzle.vec, swizzle.perPhase, swizzle.maxPhase,
            ty.getEncoding()
                .cast<triton::gpu::SharedEncodingAttr>()
                .getOrder());
        // create conversion
        auto newType = RankedTensorType::get(ty.getShape(), ty.getElementType(),
                                             newEncoding);
        Operation *newOp = builder.create<triton::gpu::ConvertLayoutOp>(
            op.getLoc(), newType, op);
        // bind new op to dot operand
        dotOp->replaceUsesOfWith(op, newOp->getResult(0));
      }
    });
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUSwizzlePass() {
  return std::make_unique<SwizzlePass>();
}