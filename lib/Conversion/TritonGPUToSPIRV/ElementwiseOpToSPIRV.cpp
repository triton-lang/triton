#include "ElementwiseOpToSPIRV.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::getElementsFromStruct;
using ::mlir::spirv::getStructFromElements;
using ::mlir::triton::gpu::getElemsPerThread;

struct FpToFpOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::FpToFpOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::FpToFpOp>::ConvertTritonGPUOpToSPIRVPattern;

  static SmallVector<Value>
  convertFp8x4ToFp16x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    assert(0 && "no convertFp8x4ToFp16x4");
  }

  static SmallVector<Value>
  convertFp16x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    assert(0 && "no convertFp8x4ToFp16x4");
  }

  static SmallVector<Value>
  convertFp8x4ToBf16x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    assert(0 && "no convertFp8x4ToFp16x4");
  }

  static SmallVector<Value>
  convertBf16x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    assert(0 && "no convertFp8x4ToFp16x4");
  }

  static SmallVector<Value>
  convertFp8x4ToFp32x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto fp16Values = convertFp8x4ToFp16x4(loc, rewriter, v0, v1, v2, v3);
    return {rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[0]),
            rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[1]),
            rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[2]),
            rewriter.create<LLVM::FPExtOp>(loc, f32_ty, fp16Values[3])};
  }

  static SmallVector<Value>
  convertFp32x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto c0 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v0);
    auto c1 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v1);
    auto c2 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v2);
    auto c3 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v3);
    return convertFp16x4ToFp8x4(loc, rewriter, c0, c1, c2, c3);
  }

  static SmallVector<Value>
  convertFp8x4ToFp64x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto fp16Values = convertFp8x4ToFp16x4(loc, rewriter, v0, v1, v2, v3);
    return {rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[0]),
            rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[1]),
            rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[2]),
            rewriter.create<LLVM::FPExtOp>(loc, f64_ty, fp16Values[3])};
  }

  static SmallVector<Value>
  convertFp64x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto c0 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v0);
    auto c1 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v1);
    auto c2 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v2);
    auto c3 = rewriter.create<LLVM::FPTruncOp>(loc, f16_ty, v3);
    return convertFp16x4ToFp8x4(loc, rewriter, c0, c1, c2, c3);
  }

  static Value convertBf16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    assert(0 && "no convertBf16ToFp32");
    return nullptr;
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    assert(0 && "no convertFp32ToBf16");
    return nullptr;
  }

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTensorType = op.getFrom().getType().cast<mlir::RankedTensorType>();
    auto dstTensorType = op.getResult().getType().cast<mlir::RankedTensorType>();
    auto srcEltType = srcTensorType.getElementType();
    auto dstEltType = dstTensorType.getElementType();
    auto loc = op->getLoc();
    auto elems = getElemsPerThread(dstTensorType);
    SmallVector<Value> resultVals;

    // Select convertor
    if (srcEltType.isa<triton::Float8Type>() ||
        dstEltType.isa<triton::Float8Type>()) {
      std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                       const Value &, const Value &,
                                       const Value &, const Value &)>
          convertor;
      if (srcEltType.isa<triton::Float8Type>() && dstEltType.isF16()) {
        convertor = convertFp8x4ToFp16x4;
      } else if (srcEltType.isF16() && dstEltType.isa<triton::Float8Type>()) {
        convertor = convertFp16x4ToFp8x4;
      } else if (srcEltType.isa<triton::Float8Type>() && dstEltType.isBF16()) {
        convertor = convertFp8x4ToBf16x4;
      } else if (srcEltType.isBF16() && dstEltType.isa<triton::Float8Type>()) {
        convertor = convertBf16x4ToFp8x4;
      } else if (srcEltType.isa<triton::Float8Type>() && dstEltType.isF32()) {
        convertor = convertFp8x4ToFp32x4;
      } else if (srcEltType.isF32() && dstEltType.isa<triton::Float8Type>()) {
        convertor = convertFp32x4ToFp8x4;
      } else if (srcEltType.isa<triton::Float8Type>() && dstEltType.isF64()) {
        convertor = convertFp8x4ToFp64x4;
      } else if (srcEltType.isF64() && dstEltType.isa<triton::Float8Type>()) {
        convertor = convertFp64x4ToFp8x4;
      } else {
        assert(false && "unsupported fp8 casting");
      }

      // Vectorized casting
      assert(elems % 4 == 0 &&
             "FP8 casting only support tensors with 4-aligned sizes");
      auto elements = getElementsFromStruct(loc, adaptor.getFrom(), rewriter);
      for (size_t i = 0; i < elems; i += 4) {
        auto converted = convertor(loc, rewriter, elements[i], elements[i + 1],
                                   elements[i + 2], elements[i + 3]);
        resultVals.append(converted);
      }
    } else if (srcEltType.isBF16() && dstEltType.isF32()) {
      resultVals.emplace_back(convertBf16ToFp32(loc, rewriter, adaptor.getFrom()));
    } else if (srcEltType.isF32() && dstEltType.isBF16()) {
      resultVals.emplace_back(convertFp32ToBf16(loc, rewriter, adaptor.getFrom()));
    } else {
      assert(false && "unsupported type casting");
    }

    assert(resultVals.size() == elems);
    auto convertedDstTensorType =
        this->getTypeConverter()->convertType(dstTensorType);
    auto result = getStructFromElements(loc, resultVals, rewriter,
                                        convertedDstTensorType);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SourceOp, typename ConcreteT>
class ElementwiseOpSPIRVConversionBase
    : public ConvertTritonGPUOpToSPIRVPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ElementwiseOpSPIRVConversionBase(SPIRVTypeConverter &converter,
                                       MLIRContext *context,
                                       PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToSPIRVPattern<SourceOp>(converter, context, benefit) {}

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
struct ElementwiseOpSPIRVConversion
    : public ElementwiseOpSPIRVConversionBase<
          SourceOp, ElementwiseOpSPIRVConversion<SourceOp, DestOp>> {
  using Base =
          ElementwiseOpSPIRVConversionBase<SourceOp,
              ElementwiseOpSPIRVConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  DestOp createDestOp(SourceOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Type elemTy,
                      ValueRange operands, Location loc) const {
    return rewriter.create<DestOp>(loc, elemTy, operands,
                                   adaptor.getAttributes().getValue());
  }
};

struct CmpIOpSPIRVConversion
    : public ElementwiseOpSPIRVConversionBase<triton::gpu::CmpIOp,
            CmpIOpSPIRVConversion> {
  using Base =
          ElementwiseOpSPIRVConversionBase<triton::gpu::CmpIOp, CmpIOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  Value createDestOp(triton::gpu::CmpIOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {

    switch (op.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    return rewriter.create<spirvOp>(loc, operands[0], operands[1]);

      DISPATCH(arith::CmpIPredicate::eq, spirv::IEqualOp);
      DISPATCH(arith::CmpIPredicate::ne, spirv::INotEqualOp);
      DISPATCH(arith::CmpIPredicate::slt, spirv::SLessThanOp);
      DISPATCH(arith::CmpIPredicate::sle, spirv::SLessThanEqualOp);
      DISPATCH(arith::CmpIPredicate::sgt, spirv::SGreaterThanOp);
      DISPATCH(arith::CmpIPredicate::sge, spirv::SGreaterThanEqualOp);
      DISPATCH(arith::CmpIPredicate::ult, spirv::ULessThanOp);
      DISPATCH(arith::CmpIPredicate::ule, spirv::ULessThanEqualOp);
      DISPATCH(arith::CmpIPredicate::ugt, spirv::UGreaterThanOp);
      DISPATCH(arith::CmpIPredicate::uge, spirv::UGreaterThanEqualOp);

#undef DISPATCH

      default:
        break;
    }
    return nullptr;
  }
};

struct CmpFOpSPIRVConversion
    : public ElementwiseOpSPIRVConversionBase<triton::gpu::CmpFOp,
            CmpFOpSPIRVConversion> {
  using Base =
          ElementwiseOpSPIRVConversionBase<triton::gpu::CmpFOp, CmpFOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  Value createDestOp(triton::gpu::CmpFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, ValueRange operands,
                                   Location loc) const {
    switch (op.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                         \
  case cmpPredicate:                                                            \
    return rewriter.create<spirvOp>(loc, operands[0], operands[1]);

      // Ordered.
      DISPATCH(arith::CmpFPredicate::OEQ, spirv::FOrdEqualOp);
      DISPATCH(arith::CmpFPredicate::OGT, spirv::FOrdGreaterThanOp);
      DISPATCH(arith::CmpFPredicate::OGE, spirv::FOrdGreaterThanEqualOp);
      DISPATCH(arith::CmpFPredicate::OLT, spirv::FOrdLessThanOp);
      DISPATCH(arith::CmpFPredicate::OLE, spirv::FOrdLessThanEqualOp);
      DISPATCH(arith::CmpFPredicate::ONE, spirv::FOrdNotEqualOp);
      // Unordered.
      DISPATCH(arith::CmpFPredicate::UEQ, spirv::FUnordEqualOp);
      DISPATCH(arith::CmpFPredicate::UGT, spirv::FUnordGreaterThanOp);
      DISPATCH(arith::CmpFPredicate::UGE, spirv::FUnordGreaterThanEqualOp);
      DISPATCH(arith::CmpFPredicate::ULT, spirv::FUnordLessThanOp);
      DISPATCH(arith::CmpFPredicate::ULE, spirv::FUnordLessThanEqualOp);
      DISPATCH(arith::CmpFPredicate::UNE, spirv::FUnordNotEqualOp);

#undef DISPATCH

      default:
        break;
    }
    return nullptr;
  }
};

struct ExtElemwiseOpSPIRVConversion
    : public ElementwiseOpSPIRVConversionBase<triton::ExtElemwiseOp,
            ExtElemwiseOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<triton::ExtElemwiseOp,
          ExtElemwiseOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(triton::ExtElemwiseOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    StringRef funcName = op.getSymbol();
    if (funcName.empty())
      llvm::errs() << "ExtElemwiseOpSPIRVConversion";

    Type funcType = getFunctionType(elemTy, operands);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetFuncOp(rewriter, op, funcName, funcType);
    return rewriter.create<LLVM::CallOp>(loc, funcOp, operands).getResult();
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
        "libname", StringAttr::get(op->getContext(), op.getLibname()));
    ret.getOperation()->setAttr(
        "libpath", StringAttr::get(op->getContext(), op.getLibpath()));
    return ret;
  }
};

struct FDivOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::DivFOp, FDivOpSPIRVConversion> {
  using Base =
          ElementwiseOpSPIRVConversionBase<mlir::arith::DivFOp, FDivOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::DivFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
//    PTXBuilder ptxBuilder;
//    auto &fdiv = *ptxBuilder.create<PTXInstr>("div");
//    unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
//    if (32 == bitwidth) {
//      fdiv.o("full").o("f32");
//    } else if (64 == bitwidth) {
//      fdiv.o("rn").o("f64");
//    } else {
//      assert(0 && bitwidth && "not supported");
//    }
//
//    auto res = ptxBuilder.newOperand(bitwidth == 32 ? "=r" : "=l");
//    auto lhs = ptxBuilder.newOperand(operands[0], bitwidth == 32 ? "r" : "l");
//    auto rhs = ptxBuilder.newOperand(operands[1], bitwidth == 32 ? "r" : "l");
//    fdiv(res, lhs, rhs);
//
//    Value ret = ptxBuilder.launch(rewriter, loc, elemTy, false);
//    return ret;
    return rewriter.create<spirv::FDivOp>(loc, elemTy, operands);
  }
};

struct FMulOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::MulFOp, FMulOpSPIRVConversion> {
  using Base =
          ElementwiseOpSPIRVConversionBase<mlir::arith::MulFOp, FMulOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::MulFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
//    auto lhsElemTy = getElementType(op.getLhs());
//    auto rhsElemTy = getElementType(op.getRhs());
//    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
//      PTXBuilder builder;
//      auto ptxAsm = " { .reg .b16 c;        \n"
//                    "    mov.b16 c, 0x8000U; \n" // 0.0
//                    "    fma.rn.bf16 $0, $1, $2, c; } \n";
//      auto &fMul = *builder.create<PTXInstr>(ptxAsm);
//      auto res = builder.newOperand("=h");
//      auto lhs = builder.newOperand(operands[0], "h");
//      auto rhs = builder.newOperand(operands[1], "h");
//      fMul({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
//      return builder.launch(rewriter, loc, i16_ty, false);
//    } else {
//      return rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0],
//                                           operands[1]);
//    }
    return rewriter.create<spirv::FMulOp>(loc, elemTy, operands);
  }
};

struct FAddOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::AddFOp, FAddOpSPIRVConversion> {
  using Base =
          ElementwiseOpSPIRVConversionBase<mlir::arith::AddFOp, FAddOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::AddFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
//    auto lhsElemTy = getElementType(op.getLhs());
//    auto rhsElemTy = getElementType(op.getRhs());
//    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
//      PTXBuilder builder;
//      auto ptxAsm = "{ .reg .b16 c;         \n"
//                    "   mov.b16 c, 0x3f80U; \n" // 1.0
//                    "   fma.rn.bf16 $0, $1, c, $2; } \n";
//      auto &fAdd = *builder.create<PTXInstr>(ptxAsm);
//      auto res = builder.newOperand("=h");
//      auto lhs = builder.newOperand(operands[0], "h");
//      auto rhs = builder.newOperand(operands[1], "h");
//      fAdd({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
//      return builder.launch(rewriter, loc, i16_ty, false);
//    } else {
//      return rewriter.create<LLVM::FAddOp>(loc, elemTy, operands[0],
//                                           operands[1]);
//    }
    return rewriter.create<spirv::FAddOp>(loc, elemTy, operands);
  }
};

struct FSubOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::SubFOp, FSubOpSPIRVConversion> {
  using Base =
          ElementwiseOpSPIRVConversionBase<mlir::arith::SubFOp, FSubOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::SubFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
//    auto lhsElemTy = getElementType(op.getLhs());
//    auto rhsElemTy = getElementType(op.getRhs());
//    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
//      PTXBuilder builder;
//      auto ptxAsm = " { .reg .b16 c;         \n"
//                    "    mov.b16 c, 0xbf80U; \n" // -1.0
//                    "    fma.rn.bf16 $0, $2, c, $1;} \n";
//      auto &fSub = *builder.create<PTXInstr>(ptxAsm);
//      auto res = builder.newOperand("=h");
//      auto lhs = builder.newOperand(operands[0], "h");
//      auto rhs = builder.newOperand(operands[1], "h");
//      fSub({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
//      return builder.launch(rewriter, loc, i16_ty, false);
//    } else {
//      return rewriter.create<LLVM::FSubOp>(loc, elemTy, operands[0],
//                                           operands[1]);
//    }

    return rewriter.create<spirv::FSubOp>(loc, elemTy, operands);
  }
};

struct SIToFPOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::SIToFPOp, SIToFPOpSPIRVConversion> {
  using Base =
      ElementwiseOpSPIRVConversionBase<mlir::arith::SIToFPOp, SIToFPOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::SIToFPOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto value = rewriter.create<arith::SIToFPOp>(loc, f32_ty, operands[0]);
      return FpToFpOpSPIRVConversion::convertFp32ToBf16(loc, rewriter, value);
    } else {
      return rewriter.create<arith::SIToFPOp>(loc, elemTy, operands[0]);
    }
  }
};

struct FPToSIOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::FPToSIOp, FPToSIOpSPIRVConversion> {
  using Base =
      ElementwiseOpSPIRVConversionBase<mlir::arith::FPToSIOp, FPToSIOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::FPToSIOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto value =
          FpToFpOpSPIRVConversion::convertBf16ToFp32(loc, rewriter, operands[0]);
      return rewriter.create<arith::FPToSIOp>(loc, elemTy, value);
    } else {
      return rewriter.create<arith::FPToSIOp>(loc, elemTy, operands[0]);
    }
  }
};

struct ExtFOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::ExtFOp, ExtFOpSPIRVConversion> {
  using Base =
      ElementwiseOpSPIRVConversionBase<mlir::arith::ExtFOp, ExtFOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::ExtFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return FpToFpOpSPIRVConversion::convertBf16ToFp32(loc, rewriter, operands[0]);
    } else {
      return rewriter.create<arith::ExtFOp>(loc, elemTy, operands[0]);
    }
  }
};

struct TruncFOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::TruncFOp, TruncFOpSPIRVConversion> {
  using Base =
      ElementwiseOpSPIRVConversionBase<mlir::arith::TruncFOp, TruncFOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::TruncFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto inElemTy = getElementType(op.getIn());
      assert(inElemTy.isF32() && "unsupported conversion");
      return FpToFpOpSPIRVConversion::convertFp32ToBf16(loc, rewriter, operands[0]);
    } else {
      return rewriter.create<arith::TruncFOp>(loc, elemTy, operands[0]);
    }
  }
};

void populateElementwiseOpToSPIRVPatterns(mlir::SPIRVTypeConverter &typeConverter,
                                          mlir::MLIRContext *context,
                                          RewritePatternSet &patterns,
                                          int numWarps,
                                          AxisInfoAnalysis &axisInfoAnalysis,
                                          const Allocation *allocation,
                                          Value smem, PatternBenefit benefit) {
#define POPULATE_TERNARY_OP(SRC_OP, DST_OP)                                    \
  patterns.add<ElementwiseOpSPIRVConversion<SRC_OP, DST_OP>>(typeConverter, context, benefit);
  POPULATE_TERNARY_OP(triton::gpu::SelectOp, spirv::SelectOp)
#undef POPULATE_TERNARY_OP

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpSPIRVConversion<SRC_OP, DST_OP>>(typeConverter, context, benefit);
  POPULATE_BINARY_OP(arith::SubIOp, spirv::ISubOp) // -
  POPULATE_BINARY_OP(arith::AddIOp, spirv::IAddOp) // +
  POPULATE_BINARY_OP(arith::MulIOp, spirv::IMulOp) // *
  POPULATE_BINARY_OP(arith::DivSIOp, spirv::SDivOp)
  POPULATE_BINARY_OP(arith::DivUIOp, spirv::UDivOp)
  POPULATE_BINARY_OP(arith::RemFOp, spirv::FRemOp) // %
  POPULATE_BINARY_OP(arith::RemSIOp, spirv::SRemOp)
  POPULATE_BINARY_OP(arith::RemUIOp, spirv::UModOp)
  POPULATE_BINARY_OP(arith::AndIOp, arith::AndIOp)   // &
  POPULATE_BINARY_OP(arith::OrIOp, arith::OrIOp)     // |
  POPULATE_BINARY_OP(arith::XOrIOp, spirv::BitwiseXorOp)   // ^
  POPULATE_BINARY_OP(arith::ShLIOp, spirv::ShiftLeftLogicalOp)   // <<
  POPULATE_BINARY_OP(arith::ShRSIOp, spirv::ShiftRightArithmeticOp) // >>
  POPULATE_BINARY_OP(arith::ShRUIOp, spirv::ShiftRightLogicalOp) // >>
#undef POPULATE_BINARY_OP

#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpSPIRVConversion<SRC_OP, DST_OP>>(typeConverter, context, benefit);
  POPULATE_UNARY_OP(arith::TruncIOp, arith::TruncIOp)
  POPULATE_UNARY_OP(arith::ExtSIOp, arith::ExtSIOp)
  POPULATE_UNARY_OP(arith::ExtUIOp, arith::ExtUIOp)
  POPULATE_UNARY_OP(arith::FPToUIOp, arith::FPToUIOp)
  POPULATE_UNARY_OP(arith::UIToFPOp, arith::UIToFPOp)
  POPULATE_UNARY_OP(math::LogOp, math::LogOp)
  POPULATE_UNARY_OP(math::CosOp, math::CosOp)
  POPULATE_UNARY_OP(math::SinOp, math::SinOp)
  POPULATE_UNARY_OP(math::SqrtOp, math::SqrtOp)
  POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)
  POPULATE_UNARY_OP(triton::BitcastOp, spirv::BitcastOp)
  POPULATE_UNARY_OP(triton::IntToPtrOp, spirv::IntToPtrOp)
  POPULATE_UNARY_OP(triton::PtrToIntOp, spirv::PtrToIntOp)
#undef POPULATE_UNARY_OP

  patterns.add<CmpIOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<CmpFOpSPIRVConversion>(typeConverter, context, benefit);

  patterns.add<FDivOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<FSubOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<FAddOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<FMulOpSPIRVConversion>(typeConverter, context, benefit);

  patterns.add<ExtFOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<TruncFOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<FPToSIOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<SIToFPOpSPIRVConversion>(typeConverter, context, benefit);

//  patterns.add<FpToFpOpSPIRVConversion>(typeConverter, context, benefit);

//  patterns.add<ExtElemwiseOpSPIRVConversion>(typeConverter, benefit);
}
