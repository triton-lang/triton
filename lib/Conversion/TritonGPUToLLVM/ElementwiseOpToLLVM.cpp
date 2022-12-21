#include "ElementwiseOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getElementsFromStruct;
using ::mlir::LLVM::getStructFromElements;
using ::mlir::triton::gpu::getElemsPerThread;

template <typename SourceOp, typename ConcreteT>
class ElementwiseOpConversionBase
    : public ConvertTritonGPUOpToLLVMPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ElementwiseOpConversionBase(LLVMTypeConverter &typeConverter,
                                       PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<SourceOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    Location loc = op->getLoc();

    unsigned elems = getElemsPerThread(resultTy);
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    SmallVector<Type> types(elems, elemTy);
    Type structTy = this->getTypeConverter()->convertType(resultTy);

    auto *concreteThis = static_cast<const ConcreteT *>(this);
    auto operands = getOperands(rewriter, adaptor, elems, loc);
    SmallVector<Value> resultVals(elems);
    for (unsigned i = 0; i < elems; ++i) {
      resultVals[i] = concreteThis->createDestOp(op, adaptor, rewriter, elemTy,
                                                 operands[i], loc);
      if (!bool(resultVals[i]))
        return failure();
    }
    Value view = getStructFromElements(loc, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, view);

    return success();
  }

protected:
  SmallVector<SmallVector<Value>>
  getOperands(ConversionPatternRewriter &rewriter, OpAdaptor adaptor,
              const unsigned elems, Location loc) const {
    SmallVector<SmallVector<Value>> operands(elems);
    for (auto operand : adaptor.getOperands()) {
      auto sub_operands = getElementsFromStruct(loc, operand, rewriter);
      for (size_t i = 0; i < elems; ++i) {
        operands[i].push_back(sub_operands[i]);
      }
    }
    return operands;
  }
};

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  explicit ElementwiseOpConversion(LLVMTypeConverter &typeConverter,
                                   PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase<SourceOp, ElementwiseOpConversion>(
            typeConverter, benefit) {}

  // An interface to support variant DestOp builder.
  DestOp createDestOp(SourceOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Type elemTy,
                      ValueRange operands, Location loc) const {
    return rewriter.create<DestOp>(loc, elemTy, operands,
                                   adaptor.getAttributes().getValue());
  }
};

struct CmpIOpConversion
    : public ElementwiseOpConversionBase<triton::gpu::CmpIOp,
                                         CmpIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<triton::gpu::CmpIOp, CmpIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  LLVM::ICmpOp createDestOp(triton::gpu::CmpIOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter, Type elemTy,
                            ValueRange operands, Location loc) const {
    return rewriter.create<LLVM::ICmpOp>(
        loc, elemTy, ArithCmpIPredicateToLLVM(op.predicate()), operands[0],
        operands[1]);
  }

  static LLVM::ICmpPredicate
  ArithCmpIPredicateToLLVM(arith::CmpIPredicate predicate) {
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
};

struct CmpFOpConversion
    : public ElementwiseOpConversionBase<triton::gpu::CmpFOp,
                                         CmpFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<triton::gpu::CmpFOp, CmpFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  static LLVM::FCmpOp createDestOp(triton::gpu::CmpFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, ValueRange operands,
                                   Location loc) {
    return rewriter.create<LLVM::FCmpOp>(
        loc, elemTy, ArithCmpFPredicateToLLVM(op.predicate()), operands[0],
        operands[1]);
  }

  static LLVM::FCmpPredicate
  ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) {
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
};

struct ExtElemwiseOpConversion
    : public ElementwiseOpConversionBase<triton::ExtElemwiseOp,
                                         ExtElemwiseOpConversion> {
  using Base = ElementwiseOpConversionBase<triton::ExtElemwiseOp,
                                           ExtElemwiseOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(triton::ExtElemwiseOp op, OpAdaptor adaptor,
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

private:
  Type getFunctionType(Type resultType, ValueRange operands) const {
    SmallVector<Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter,
                                     triton::ExtElemwiseOp op,
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
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::DivFOp op, OpAdaptor adaptor,
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
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::math::ExpOp op, OpAdaptor adaptor,
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
};

void populateElementwiseOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         int numWarps,
                                         AxisInfoAnalysis &axisInfoAnalysis,
                                         const Allocation *allocation,
                                         Value smem, PatternBenefit benefit) {
#define POPULATE_TERNARY_OP(SRC_OP, DST_OP)                                    \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(typeConverter, benefit);
  POPULATE_TERNARY_OP(triton::gpu::SelectOp, LLVM::SelectOp)
#undef POPULATE_TERNARY_OP

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(typeConverter, benefit);
  POPULATE_BINARY_OP(arith::SubIOp, LLVM::SubOp) // -
  POPULATE_BINARY_OP(arith::SubFOp, LLVM::FSubOp)
  POPULATE_BINARY_OP(arith::AddIOp, LLVM::AddOp) // +
  POPULATE_BINARY_OP(arith::AddFOp, LLVM::FAddOp)
  POPULATE_BINARY_OP(arith::MulIOp, LLVM::MulOp) // *
  POPULATE_BINARY_OP(arith::MulFOp, LLVM::FMulOp)
  POPULATE_BINARY_OP(arith::DivFOp, LLVM::FDivOp) // /
  POPULATE_BINARY_OP(arith::DivSIOp, LLVM::SDivOp)
  POPULATE_BINARY_OP(arith::DivUIOp, LLVM::UDivOp)
  POPULATE_BINARY_OP(arith::RemFOp, LLVM::FRemOp) // %
  POPULATE_BINARY_OP(arith::RemSIOp, LLVM::SRemOp)
  POPULATE_BINARY_OP(arith::RemUIOp, LLVM::URemOp)
  POPULATE_BINARY_OP(arith::AndIOp, LLVM::AndOp)   // &
  POPULATE_BINARY_OP(arith::OrIOp, LLVM::OrOp)     // |
  POPULATE_BINARY_OP(arith::XOrIOp, LLVM::XOrOp)   // ^
  POPULATE_BINARY_OP(arith::ShLIOp, LLVM::ShlOp)   // <<
  POPULATE_BINARY_OP(arith::ShRSIOp, LLVM::AShrOp) // >>
  POPULATE_BINARY_OP(arith::ShRUIOp, LLVM::LShrOp) // >>
#undef POPULATE_BINARY_OP

#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(typeConverter, benefit);
  POPULATE_UNARY_OP(arith::TruncIOp, LLVM::TruncOp)
  POPULATE_UNARY_OP(arith::TruncFOp, LLVM::FPTruncOp)
  POPULATE_UNARY_OP(arith::ExtSIOp, LLVM::SExtOp)
  POPULATE_UNARY_OP(arith::ExtUIOp, LLVM::ZExtOp)
  POPULATE_UNARY_OP(arith::FPToUIOp, LLVM::FPToUIOp)
  POPULATE_UNARY_OP(arith::FPToSIOp, LLVM::FPToSIOp)
  POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)
  POPULATE_UNARY_OP(arith::SIToFPOp, LLVM::SIToFPOp)
  POPULATE_UNARY_OP(arith::ExtFOp, LLVM::FPExtOp)
  POPULATE_UNARY_OP(math::LogOp, math::LogOp)
  POPULATE_UNARY_OP(math::CosOp, math::CosOp)
  POPULATE_UNARY_OP(math::SinOp, math::SinOp)
  POPULATE_UNARY_OP(math::SqrtOp, math::SqrtOp)
  POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)
  POPULATE_UNARY_OP(triton::BitcastOp, LLVM::BitcastOp)
  POPULATE_UNARY_OP(triton::IntToPtrOp, LLVM::IntToPtrOp)
  POPULATE_UNARY_OP(triton::PtrToIntOp, LLVM::PtrToIntOp)
#undef POPULATE_UNARY_OP

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
