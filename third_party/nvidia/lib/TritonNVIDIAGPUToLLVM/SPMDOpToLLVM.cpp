#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

static Value getNumPrograms(OpBuilder &rewriter, int numCTAs, Location loc,
                            ProgramIDDim axis) {
  if (numCTAs == 1) {
    switch (axis) {
    case ProgramIDDim::X:
      return NVVM::GridDimXOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Y:
      return NVVM::GridDimYOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Z:
      return NVVM::GridDimZOp::create(rewriter, loc, i32_ty);
    }
  } else {
    switch (axis) {
    case ProgramIDDim::X:
      return NVVM::ClusterDimXOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Y:
      return NVVM::ClusterDimYOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Z:
      return NVVM::ClusterDimZOp::create(rewriter, loc, i32_ty);
    }
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
    // It is not easy to get the compute capability here, so we use numCTAs to
    // decide the semantic of GetNumProgramsOp. If numCTAs = 1, then
    // GetNumProgramsOp is converted to "%nctaid", otherwise it is converted to
    // "%nclusterid".
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
