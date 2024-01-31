#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertOpToLLVMPattern<triton::GetProgramIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value programId = llGetPid(op.getAxisAsInt(), op->getLoc(),
                               op->getParentOfType<ModuleOp>(), rewriter);
    rewriter.replaceOp(op, programId);
    return success();
  }
};

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
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for GetProgramIdOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Location loc = op->getLoc();
    assert(op.getAxis() < 3);
    std::string sreg = numCTAs == 1 ? "%nctaid." : "%nclusterid.";
    sreg.append(1, 'x' + op.getAxis()); // 0 -> 'x', 1 -> 'y', 2 -> 'z'

    Value numPrograms = LLVM::getSRegValue(rewriter, loc, sreg);
    rewriter.replaceOp(op, numPrograms);
    return success();
  }
};

// TODO[goostavz]: GetThreadIdOp/GetClusterCTAIdOp is a temporary solution
// before async dialect is done. These concepts should appear in ttgpu
// level, and they are planned to be deprecated along with ttgpu.mbarrier_xxx
// ops.
struct GetThreadIdOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::GetThreadIdOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::GetThreadIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, getThreadId(rewriter, op->getLoc()));
    return success();
  }
};

struct GetCanonicalWarpIdConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::GetCanonicalWarpId> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::GetCanonicalWarpId>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetCanonicalWarpId op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, GetCanonicalWarpId(rewriter, op->getLoc()));
    return success();
  }
};

struct GetClusterCTAIdOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::GetClusterCTAIdOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::GetClusterCTAIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetClusterCTAIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, getClusterCTAId(rewriter, op->getLoc()));
    return success();
  }
};

} // namespace

void mlir::triton::populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<GetThreadIdOpConversion>(typeConverter, benefit);
  patterns.add<GetCanonicalWarpIdConversion>(typeConverter, benefit);
  patterns.add<GetClusterCTAIdOpConversion>(typeConverter, benefit);
}
