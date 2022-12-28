#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;


class LoadConvertToInsertSlice : public mlir::RewritePattern{

public:
  explicit LoadConvertToInsertSlice(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
      auto cvt = cast<triton::gpu::ConvertLayoutOp>(op);
      auto origRetType = cvt.getResult().getType().cast<RankedTensorType>();
      auto shape = origRetType.getShape();
      auto eltType = origRetType.getElementType();
      auto dotOpEncoding = origRetType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if(!dotOpEncoding){
        return failure();
      }
      auto cvtArg = cvt.getOperand().getDefiningOp();
      if(!cvtArg)
        return failure();
      auto loadOp = dyn_cast<triton::LoadOp>(*cvtArg);
      if(!loadOp){
        return failure();
      }
      auto blockedEncoding = loadOp.getType().cast<RankedTensorType>().getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      if(!blockedEncoding)
        return failure();
      auto sharedEncoding = triton::gpu::SharedEncodingAttr::get(getContext(), dotOpEncoding, shape, 
                                                                 blockedEncoding.getOrder(), eltType);
      auto srcTy = RankedTensorType::get({1, shape[0], shape[1]}, 
                                          eltType, 
                                          sharedEncoding); 
      auto loadTensor = rewriter.create<triton::gpu::AllocTensorOp>(op->getLoc(), srcTy);
      
      auto newOp = rewriter.create<triton::gpu::InsertSliceAsyncOp>(
              op->getLoc(), loadTensor.getType(),
              loadOp.ptr(),
              loadTensor, rewriter.create<arith::ConstantIntOp>(op->getLoc(), 0, 32), 
              loadOp.mask(),
              loadOp.other(), loadOp.cache(),
              loadOp.evict(), loadOp.isVolatile(), /*axis*/ 0);
      
      rewriter.create<triton::gpu::AsyncWaitOp>(op->getLoc(), 0);
      auto tmpType = RankedTensorType::get({shape[0], shape[1]}, eltType, sharedEncoding);
      auto _0 = rewriter.getI64IntegerAttr(0);
      auto _1 = rewriter.getI64IntegerAttr(1);
      auto tmp = rewriter.create<tensor::ExtractSliceOp>(op->getLoc(), tmpType, newOp,
        SmallVector<OpFoldResult>{_0, _0, _0},
        SmallVector<OpFoldResult>{_1,
                                  rewriter.getI64IntegerAttr(shape[0]),
                                  rewriter.getI64IntegerAttr(shape[1])},
        SmallVector<OpFoldResult>{_1, _1, _1});
      rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(op, origRetType, tmp);
      return success();

  }
    
};

class TritonGPUOptimizeLoadConvertPass
    : public TritonGPUOptimizeLoadConvertBase<TritonGPUOptimizeLoadConvertPass> {
public:
  TritonGPUOptimizeLoadConvertPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<LoadConvertToInsertSlice>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonGPUOptimizeLoadConvertPass() {
  return std::make_unique<TritonGPUOptimizeLoadConvertPass>();
}
