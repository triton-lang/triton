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
    int inner = (opIdx == 0) ? 0 : 1;
    if (version == 1) {
      int maxPhase = (order[inner] == 1 ? 8 : 4) / perPhase;
      // TODO: handle rep (see
      // https://github.com/openai/triton/blob/master/lib/codegen/analysis/layout.cc#L209)
      int vec = 1;
      return SwizzleInfo{vec, perPhase, maxPhase};
    } else if (version == 2) {
      auto eltTy = ty.getElementType();
      std::vector<size_t> matShape = {8, 8,
                                      2 * 64 / eltTy.getIntOrFloatBitWidth()};
      // for now, disable swizzle when using transposed int8 tensor cores
      bool isInt8Mma = ty.getElementType().isInteger(8);
      if (isInt8Mma && order[0] == inner)
        return noSwizzling;
      // compute swizzling for A operand
      if (opIdx == 0) {
        int vec = order[0] == 1 ? matShape[2] : matShape[0]; // k : m
        int mmaStride = order[0] == 1 ? matShape[0] : matShape[2];
        int maxPhase = mmaStride / perPhase;
        return SwizzleInfo{vec, perPhase, maxPhase};
      }
      // compute swizzling for B operand
      else if (opIdx == 1) {
        int vec = order[0] == 1 ? matShape[1] : matShape[2]; // n : k
        int mmaStride = order[0] == 1 ? matShape[2] : matShape[1];
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
    MLIRContext *context = &getContext();
    // replace blocked -> dot_op with
    // blocked -> shared -> dot_op in order to
    // expose opportunities for swizzling
    op->walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      if (srcType.getEncoding().isa<triton::gpu::BlockedEncodingAttr>() &&
          dstType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>()) {
        auto tmpType =
            RankedTensorType::get(dstType.getShape(), dstType.getElementType(),
                                  triton::gpu::SharedEncodingAttr::get(
                                      op->getContext(), 1, 1, 1, {1, 0}));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
      }
    });

    op->walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto arg = cvtOp.getOperand();
      auto argType = arg.getType().cast<RankedTensorType>();
      auto retType = cvtOp.getResult().getType().cast<RankedTensorType>();
      auto argEncoding =
          argType.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      auto retEncoding =
          retType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (!argEncoding || !retEncoding)
        return;
      auto opIdx = retEncoding.getOpIdx();
      // compute new swizzled encoding
      auto parentEncoding =
          retEncoding.getParent().dyn_cast<triton::gpu::MmaEncodingAttr>();
      if (!parentEncoding)
        return;
      SwizzleInfo swizzle = getSwizzleMMA(opIdx, parentEncoding, argType);
      auto newEncoding = triton::gpu::SharedEncodingAttr::get(
          &getContext(), swizzle.vec, swizzle.perPhase, swizzle.maxPhase,
          argEncoding.getOrder());
      // create conversion
      auto newType = RankedTensorType::get(
          argType.getShape(), argType.getElementType(), newEncoding);
      Operation *newArg = builder.create<triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), newType, arg);
      // bind new op to cvt operand
      cvtOp->replaceUsesOfWith(arg, newArg->getResult(0));
    });
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUSwizzlePass() {
  return std::make_unique<SwizzlePass>();
}