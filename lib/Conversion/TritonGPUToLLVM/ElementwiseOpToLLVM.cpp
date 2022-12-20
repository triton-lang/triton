#include "ElementwiseOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

LLVM::ICmpPredicate
CmpIOpConversion::ArithCmpIPredicateToLLVM(arith::CmpIPredicate predicate) {
  switch (predicate) {
#define __PRED_ENUM(item__)                                                    \
  case arith::CmpIPredicate::item__:                                           \
    return LLVM::ICmpPredicate::item__

    __PRED_ENUM(eq);
    __PRED_ENUM(ne);
    __PRED_ENUM(sgt);
    __PRED_ENUM(sge);
    __PRED_ENUM(slt);
    __PRED_ENUM(sle);
    __PRED_ENUM(ugt);
    __PRED_ENUM(uge);
    __PRED_ENUM(ult);
    __PRED_ENUM(ule);

#undef __PRED_ENUM
  }
  return LLVM::ICmpPredicate::eq;
}

LLVM::FCmpPredicate
CmpFOpConversion::ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) {
  switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

    __PRED_ENUM(OEQ, oeq);
    __PRED_ENUM(ONE, one);
    __PRED_ENUM(OGT, ogt);
    __PRED_ENUM(OGE, oge);
    __PRED_ENUM(OLT, olt);
    __PRED_ENUM(OLE, ole);
    __PRED_ENUM(ORD, ord);
    __PRED_ENUM(UEQ, ueq);
    __PRED_ENUM(UGT, ugt);
    __PRED_ENUM(UGE, uge);
    __PRED_ENUM(ULT, ult);
    __PRED_ENUM(ULE, ule);
    __PRED_ENUM(UNE, une);
    __PRED_ENUM(UNO, uno);
    __PRED_ENUM(AlwaysTrue, _true);
    __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
  }
  return LLVM::FCmpPredicate::_true;
}

Value ExtElemwiseOpConversion::createDestOp(
    triton::ExtElemwiseOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter, Type elemTy,
    ValueRange operands, Location loc) const {
  StringRef funcName = op.symbol();
  if (funcName.empty())
    llvm::errs() << "ExtElemwiseOpConversion";

  Type funcType = getFunctionType(elemTy, operands);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetFuncOp(rewriter, op, funcName, funcType);
  return rewriter.create<LLVM::CallOp>(loc, funcOp, operands).getResult(0);
}

LLVM::LLVMFuncOp ExtElemwiseOpConversion::appendOrGetFuncOp(
    ConversionPatternRewriter &rewriter, triton::ExtElemwiseOp op,
    StringRef funcName, Type funcType) const {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(op->getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  mlir::OpBuilder b(op->getParentOfType<LLVMFuncOp>());
  auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  ret.getOperation()->setAttr(
      "libname", StringAttr::get(op->getContext(), op.libname()));
  ret.getOperation()->setAttr(
      "libpath", StringAttr::get(op->getContext(), op.libpath()));
  return ret;
}

Value FDivOpConversion::createDestOp(
    mlir::arith::DivFOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter, Type elemTy,
    ValueRange operands, Location loc) const {

  PTXBuilder ptxBuilder;
  auto &fdiv = *ptxBuilder.create<PTXInstr>("div");
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  if (32 == bitwidth) {
    fdiv.o("full").o("f32");
  } else if (64 == bitwidth) {
    fdiv.o("rn").o("f64");
  } else {
    assert(0 && bitwidth && "not supported");
  }

  auto res = ptxBuilder.newOperand(bitwidth == 32 ? "=r" : "=l");
  auto lhs = ptxBuilder.newOperand(operands[0], bitwidth == 32 ? "r" : "l");
  auto rhs = ptxBuilder.newOperand(operands[1], bitwidth == 32 ? "r" : "l");
  fdiv(res, lhs, rhs);

  Value ret = ptxBuilder.launch(rewriter, loc, elemTy, false);
  return ret;
}

Value ExpOpConversionApprox::createDestOp(
    mlir::math::ExpOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter, Type elemTy,
    ValueRange operands, Location loc) const {
  // For FP64 input, call __nv_expf for higher-precision calculation
  if (elemTy.getIntOrFloatBitWidth() == 64)
    return {};

  const double log2e = 1.4426950408889634;
  Value prod = fmul(f32_ty, operands[0], f32_val(log2e));

  PTXBuilder ptxBuilder;
  auto &exp2 = ptxBuilder.create<PTXInstr>("ex2")->o("approx").o("f32");
  auto output = ptxBuilder.newOperand("=f");
  auto input = ptxBuilder.newOperand(prod, "f");
  exp2(output, input);
  return ptxBuilder.launch(rewriter, loc, f32_ty, false);
}

void populateElementwiseOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, int numWarps,
    AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    PatternBenefit benefit) {
  patterns.add<CmpIOpConversion>(typeConverter, benefit);
  patterns.add<CmpFOpConversion>(typeConverter, benefit);
  patterns.add<FDivOpConversion>(typeConverter, benefit);
  patterns.add<ExtElemwiseOpConversion>(typeConverter, benefit);
  // ExpOpConversionApprox will try using ex2.approx if the input type is FP32.
  // For FP64 input type, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, benefit);
}
