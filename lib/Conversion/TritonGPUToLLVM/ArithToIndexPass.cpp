#include "triton/Conversion/TritonGPUToLLVM/ArithToIndexPass.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM//ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/Passes.h.inc"

namespace {
class TritonArithToIndexConversionTarget : public mlir::ConversionTarget {
public:
  static bool hasIndexResultOrOperand(Operation *op) {
    if (!op)
      return false;
    bool hasRetIndex = llvm::find_if(op->getResultTypes(), [](Type type) {
                         return type.isIndex();
                       }) != op->getResultTypes().end();
    bool hasArgIndex = llvm::find_if(op->getOperandTypes(), [](Type type) {
                         return type.isIndex();
                       }) != op->getOperandTypes().end();
    return !hasRetIndex && !hasArgIndex;
  }

  explicit TritonArithToIndexConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addDynamicallyLegalDialect<arith::ArithDialect>(hasIndexResultOrOperand);
  }
};

template <class SrcOp, class DstOp>
LogicalResult replaceArithWithIndex(SrcOp op, PatternRewriter &rewriter) {
  // if (!hasIndexResultOrOperand(&*op))
  //   return failure();
  rewriter.replaceOpWithNewOp<DstOp>(op, op->getResultTypes(),
                                     op->getOperands(), op->getAttrs());
  return success();
}

LogicalResult replaceArithCmpWithIndexCmp(arith::CmpIOp op,
                                          PatternRewriter &rewriter) {
  // if (!hasIndexResultOrOperand(&*op))
  //   return failure();
  rewriter.replaceOpWithNewOp<index::CmpOp>(
      op, op.getResult().getType(), (index::IndexCmpPredicate)op.getPredicate(),
      op.getOperand(0), op.getOperand(1));
  return success();
}

class ArithToIndex : public TritonConvertArithToIndexBase<ArithToIndex> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TritonArithToIndexConversionTarget target(*context);
    RewritePatternSet patterns(context);
    patterns.add(replaceArithWithIndex<arith::IndexCastOp, index::CastSOp>);
    patterns.add(replaceArithWithIndex<arith::ConstantOp, index::ConstantOp>);
    patterns.add(replaceArithWithIndex<arith::AddIOp, index::AddOp>);
    patterns.add(replaceArithCmpWithIndexCmp);
    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonConvertArithToIndexPass() {
  return std::make_unique<::ArithToIndex>();
}

} // namespace triton
} // namespace mlir