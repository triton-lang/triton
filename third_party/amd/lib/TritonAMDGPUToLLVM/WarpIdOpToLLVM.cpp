#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using mlir::triton::AMD::ISAFamily;

namespace {

class WarpIdOpPattern : public ConvertOpToLLVMPattern<WarpIdOp> {
public:
  WarpIdOpPattern(LLVMTypeConverter &typeConverter,
                  const AMD::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<WarpIdOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // These are runtime constant values so insert ops at the beginning of the
    // function to help LLVM uniformity analysis, unless we are in a warp
    // specialized partition region where we need to keep ops in their
    // respective regions.
    std::optional<int> startWarpId = getWarpGroupStartWarpId(op->getBlock());
    if (!startWarpId) {
      auto funcOp = op->getParentOfType<FunctionOpInterface>();
      rewriter.setInsertionPoint(
          &funcOp.getFunctionBody().getBlocks().front().front());
    }

    auto isaFamily = targetInfo.getISAFamily();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value warpId;

    if (ISAFamily::RDNA4 == isaFamily || ISAFamily::GFX1250 == isaFamily) {
      auto warpIdOp = LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc, "llvm.amdgcn.wave.id", {i32_ty}, ValueRange{});
      warpId = warpIdOp.getResult(0);
    } else {
      int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
      Value warpSizeVal = b.i32_val(threadsPerWarp);
      Value tid = getThreadId(rewriter, loc);
      warpId = b.udiv(tid, warpSizeVal);
      if (ISAFamily::CDNA3 == isaFamily || ISAFamily::CDNA4 == isaFamily) {
        // On GFX9, there is no dedicated hardware instruction to read
        // `wave_id`. The value is instead computed from `workitem.id.x`. Per
        // the GFX9 ABI, `workitem.id.x` is initialized in a vector register,
        // and vector instructions are generated for IR operations that depend
        // on `wave_id`.
        //
        // A `v_readfirstlane` instruction is inserted at the end of these
        // vector sequences to transfer the value from a vector register to a
        // scalar register, initializing `$m0`.
        auto call =
            ROCDL::ReadfirstlaneOp::create(rewriter, loc, {i32_ty}, warpId);
        warpId = call.getRes();
      }
    }

    if (startWarpId) {
      warpId = b.sub(warpId, b.i32_val(*startWarpId));
    }

    rewriter.replaceOp(op, warpId);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::AMD::populateWarpIdOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<WarpIdOpPattern>(typeConverter, targetInfo, benefit);
}
