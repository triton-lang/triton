#include "Dialect/TritonMetalGPU/IR/Dialect.h"
#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::gpu;

namespace ttmetalgpu = mlir::triton::metalgpu;

namespace mlir::triton::metal {

namespace {

struct SimdgroupWaitOpConversion
    : public ConvertOpToLLVMPattern<ttmetalgpu::SimdgroupWaitOp> {
  explicit SimdgroupWaitOpConversion(
      LLVMTypeConverter &typeConverter,
      const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<ttmetalgpu::SimdgroupWaitOp>(typeConverter,
                                                            benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttmetalgpu::SimdgroupWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto events = adaptor.getEvents();
    int64_t N = static_cast<int64_t>(events.size());
    assert(N > 0 && "SimdgroupWaitOp must have at least one event");

    auto p0Ty = LLVM::LLVMPointerType::get(ctx, 0);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto arrTy = LLVM::LLVMArrayType::get(p0Ty, N);

    // alloca [N x ptr] at function entry block
    auto func = rewriter.getInsertionBlock()
                    ->getParent()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
    Value eventAlloca;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&func.getBody().front());
      eventAlloca = LLVM::AllocaOp::create(rewriter, loc, p0Ty, arrTy,
                                           b.i32_val(1), /*alignment=*/8);
    }

    // store event into slot
    for (int64_t i = 0; i < N; ++i) {
      Value slot = LLVM::GEPOp::create(rewriter, loc, p0Ty, arrTy, eventAlloca,
                                       ArrayRef<LLVM::GEPArg>{0, (int32_t)i});
      LLVM::StoreOp::create(rewriter, loc, events[i], slot);
    }

    // get ptr to slot 0 for wait call
    Value slot0 = LLVM::GEPOp::create(rewriter, loc, p0Ty, arrTy, eventAlloca,
                                      ArrayRef<LLVM::GEPArg>{0, 0});

    // declare and call air.wait_simdgroup_events(i32 N, ptr slot0)
    auto funcType = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, p0Ty});
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    auto waitFuncOp = appendOrGetExternFuncOp(
        rewriter, parentOp, "air.wait_simdgroup_events", funcType);
    LLVM::createLLVMCallOp(rewriter, loc, waitFuncOp,
                           ValueRange{b.i32_val(N), slot0});

    rewriter.eraseOp(op);
    return success();
  }

protected:
  const mlir::triton::metal::TargetInfo &targetInfo;
};

} // namespace

void populateSimdgroupWaitOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<SimdgroupWaitOpConversion>(typeConverter, targetInfo, benefit);
}

} // namespace mlir::triton::metal
