#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

static Value getNumPrograms(OpBuilder &rewriter, ModuleOp moduleOp,
                            Location loc, ProgramIDDim axis) {
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);
  bool preferredFallback = LLVM::NVIDIA::usePreferredClusterFallback(moduleOp);
  switch (axis) {
  case ProgramIDDim::X: {
    // Multi-CTA launches expand the CUDA grid in X. Fixed cluster launches can
    // read the logical program grid from %nclusterid.x; preferred fallback must
    // read %nctaid.x and divide by the static preferred cluster size.
    Value ret = numCTAs == 1 || preferredFallback
                    ? Value(NVVM::GridDimXOp::create(rewriter, loc, i32_ty))
                    : Value(NVVM::ClusterDimXOp::create(rewriter, loc, i32_ty));

    if (preferredFallback) {
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      ret = b.udiv(ret, b.i32_val(numCTAs));
    }
    return ret;
  }
  case ProgramIDDim::Y:
    // Clusters are always launched as {numCTAs, 1, 1}, so Y/Z are already the
    // logical program grid for all cluster modes.
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
    // getNumPrograms handles the X-only CUDA grid expansion used for multi-CTA
    // launches.
    rewriter.replaceOp(op,
                       getNumPrograms(rewriter, op->getParentOfType<ModuleOp>(),
                                      op.getLoc(), op.getAxis()));
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
}
