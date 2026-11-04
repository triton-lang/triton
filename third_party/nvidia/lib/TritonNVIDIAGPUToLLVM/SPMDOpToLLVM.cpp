#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

static Value getNumPrograms(OpBuilder &rewriter, int numCTAs, Location loc,
                            ProgramIDDim axis) {
  switch (axis) {
  case ProgramIDDim::X: {
    Value gridDim = NVVM::GridDimXOp::create(rewriter, loc, i32_ty);
    if (numCTAs == 1)
      return gridDim;
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    return b.udiv(gridDim, b.i32_val(numCTAs));
  }
  case ProgramIDDim::Y:
    return NVVM::GridDimYOp::create(rewriter, loc, i32_ty);
  case ProgramIDDim::Z:
    return NVVM::GridDimZOp::create(rewriter, loc, i32_ty);
  }
  llvm_unreachable("invalid axis");
}

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Clusters are launched with dimensions (numCTAs, 1, 1).
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(
        op->getParentOfType<ModuleOp>());

    rewriter.replaceOp(
        op, getNumPrograms(rewriter, numCTAs, op.getLoc(), op.getAxis()));
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
}
