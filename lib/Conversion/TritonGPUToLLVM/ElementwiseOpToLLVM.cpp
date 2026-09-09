#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir::triton::gpu;

namespace {

struct AddPtrOpConversion : public ConvertOpToLLVMPattern<AddPtrOp> {
  using ConvertOpToLLVMPattern<AddPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resultTy = op.getType();
    auto typeConverter = getTypeConverter();
    auto resultTensorTy = dyn_cast<RankedTensorType>(resultTy);
    if (resultTensorTy) {
      unsigned elems = getUniqueElemsPerThread(resultTy);
      Type elemTy = typeConverter->convertType(
          cast<PointerType>(resultTensorTy.getElementType()).getPointeeType());
      Type ptrTy = typeConverter->convertType(resultTensorTy.getElementType());
      auto ptrs = unpackUniqueTensorElements(loc, adaptor.getPtr(), rewriter);
      auto offsets =
          unpackUniqueTensorElements(loc, adaptor.getOffset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = b.gep(ptrTy, elemTy, ptrs[i], offsets[i]);
      }
      Value view = packUniqueTensorElements(loc, typeConverter, resultVals,
                                            rewriter, resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(isa<PointerType>(resultTy));
      auto resultPtrTy = typeConverter->convertType(resultTy);
      auto resultElemTy = typeConverter->convertType(
          cast<PointerType>(resultTy).getPointeeType());
      Value result = b.gep(resultPtrTy, resultElemTy, adaptor.getPtr(),
                           adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct CmpIOpConversion
    : public ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<LLVM::ICmpOp> createDestOps(arith::CmpIOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          Type elemTy,
                                          MultipleOperandsRange operands,
                                          Location loc) const {
    return {LLVM::ICmpOp::create(rewriter, loc, elemTy,
                                 ArithCmpIPredicateToLLVM(op.getPredicate()),
                                 operands[0][0], operands[0][1])};
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
    llvm_unreachable("Unknown arith::CmpIPredicate");
  }
};

struct CmpFOpConversion
    : public ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  static SmallVector<LLVM::FCmpOp>
  createDestOps(arith::CmpFOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                MultipleOperandsRange operands, Location loc) {
    return {LLVM::FCmpOp::create(rewriter, loc, elemTy,
                                 ArithCmpFPredicateToLLVM(op.getPredicate()),
                                 operands[0][0], operands[0][1])};
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
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }
};

struct MulhiUIOpConversion
    : public ElementwiseOpConversionBase<MulhiUIOp, MulhiUIOpConversion> {
  using Base = ElementwiseOpConversionBase<MulhiUIOp, MulhiUIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  explicit MulhiUIOpConversion(LLVMTypeConverter &typeConverter,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               const TargetInfoBase &targetInfo,
                               PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        targetInfo(targetInfo) {}

  SmallVector<Value> createDestOps(MulhiUIOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {

    Type resultElementTy = getElementTypeOrSelf(op.getResult().getType());
    assert(resultElementTy.isInteger(32) || resultElementTy.isInteger(64));

    auto funcName = targetInfo.getMulhiFuncName(resultElementTy);
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    return {
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }

protected:
  const TargetInfoBase &targetInfo;
};

struct ExternElementwiseOpConversion
    : public ElementwiseOpConversionBase<ExternElementwiseOp,
                                         ExternElementwiseOpConversion> {
  using Base = ElementwiseOpConversionBase<ExternElementwiseOp,
                                           ExternElementwiseOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  SmallVector<Value> createDestOps(ExternElementwiseOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    StringRef funcName = op.getSymbol();
    if (funcName.empty())
      llvm::errs() << "ExternElementwiseOpConversion";

    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp = appendOrGetExternFuncOp(
        rewriter, op, funcName, funcType, op.getLibname(), op.getLibpath());
    return {
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }
};

template <typename Op>
struct InlineAsmOpConversion : public ConvertOpToLLVMPattern<Op> {
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;
  using typename ConvertOpToLLVMPattern<Op>::OpAdaptor;
  static constexpr bool elementwise =
      std::is_same_v<Op, ElementwiseInlineAsmOp>;

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = op.getContext();
    auto *converter = this->getTypeConverter();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    unsigned pack = 1, numElems = 1;
    if constexpr (elementwise) {
      pack = op.getPackedElement();
      numElems = getUniqueElemsPerThread(op->getResult(0).getType());
    }
    auto resultSize = [&](Type type) {
      return elementwise ? pack : getTotalElemsPerThread(type);
    };
    auto registerWidth = [&](Type type) {
      unsigned bitwidth = getIntOrFloatOrPtrBitWidth(type);
      return elementwise ? std::min(pack, std::max(32 / bitwidth, 1u)) : 1u;
    };

    SmallVector<SmallVector<Value>> inputs;
    for (auto [original, lowered] :
         llvm::zip(op.getOperands(), adaptor.getOperands())) {
      if (auto desc = dyn_cast<MemDescType>(original.getType())) {
        inputs.push_back(
            {getMemDescAddress(rewriter, loc, converter, desc, lowered)});
      } else {
        auto values = elementwise
                          ? unpackUniqueTensorElements(loc, lowered, rewriter)
                          : unpackTensorElements(loc, lowered, rewriter,
                                                 original.getType());
        // Elementwise packs may extend beyond the unique values in a thread.
        if constexpr (elementwise)
          values.resize(llvm::alignTo(values.size(), pack),
                        b.undef(values.front().getType()));
        inputs.push_back(std::move(values));
      }
    }

    SmallVector<Type> resultTypes;
    for (Type type : op.getResultTypes()) {
      Type elemTy = converter->convertType(getElementTypeOrSelf(type));
      unsigned width = registerWidth(elemTy);
      Type regTy = width == 1 ? elemTy : vec_ty(elemTy, width);
      resultTypes.append(resultSize(type) / width, regTy);
    }
    Type returnType = resultTypes.empty()       ? LLVM::LLVMVoidType::get(ctx)
                      : resultTypes.size() == 1 ? resultTypes.front()
                                                : struct_ty(resultTypes);
    SmallVector<SmallVector<Value>> outputs(op.getNumResults());
    for (unsigned offset = 0; offset < numElems; offset += pack) {
      SmallVector<Value> operands;
      for (auto [original, input] : llvm::zip(op.getOperands(), inputs)) {
        ArrayRef<Value> values(input);
        if (elementwise && !isa<MemDescType>(original.getType()))
          values = values.slice(offset, pack);
        unsigned width = registerWidth(values.front().getType());
        for (unsigned i = 0; i < values.size(); i += width)
          operands.push_back(
              width == 1 ? values[i]
                         : packLLVector(loc, values.slice(i, width), rewriter));
      }
      auto call = LLVM::InlineAsmOp::create(
          rewriter, loc, returnType, operands, op.getAsmString(),
          op.getConstraints(), !op.getPure(), false, LLVM::TailCallKind::None,
          LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
          ArrayAttr());
      if (resultTypes.empty())
        continue;
      auto results = resultTypes.size() == 1
                         ? SmallVector<Value>{call.getResult(0)}
                         : unpackLLElements(loc, call.getResult(0), rewriter);
      unsigned cursor = 0;
      for (auto [type, output] : llvm::zip(op.getResultTypes(), outputs)) {
        unsigned width =
            registerWidth(converter->convertType(getElementTypeOrSelf(type)));
        for (unsigned i = 0; i < resultSize(type); i += width)
          llvm::append_range(output,
                             unpackLLVector(loc, results[cursor++], rewriter));
      }
    }

    SmallVector<Value> results;
    for (auto [type, output] : llvm::zip(op.getResultTypes(), outputs)) {
      if constexpr (elementwise)
        output.resize(numElems);
      auto packResult =
          elementwise ? packUniqueTensorElements : packTensorElements;
      results.push_back(packResult(loc, converter, output, rewriter, type));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct AbsIOpConversion
    : ElementwiseOpConversionBase<math::AbsIOp, AbsIOpConversion> {
  using Base = ElementwiseOpConversionBase<math::AbsIOp, AbsIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::AbsIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    return {LLVM::AbsOp::create(rewriter, loc, elemTy, operands[0][0],
                                /*is_int_min_poison=*/false)};
  }
};

struct AbsFOpConversion
    : ElementwiseOpConversionBase<math::AbsFOp, AbsFOpConversion> {
  using Base = ElementwiseOpConversionBase<math::AbsFOp, AbsFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::AbsFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (llvm::isa<IntegerType>(elemTy)) {
      // Mask out the sign bit
      auto num_bits =
          getElementTypeOrSelf(op.getType()).getIntOrFloatBitWidth();
      assert(num_bits <= 16);
      auto mask = (1u << (num_bits - 1u)) - 1u;
      auto maskAttr = rewriter.getIntegerAttr(elemTy, mask);
      auto maskConst = LLVM::ConstantOp::create(rewriter, loc, maskAttr);
      return {b.and_(operands[0][0], maskConst)};
    }

    return {LLVM::FAbsOp::create(rewriter, loc, elemTy, operands[0][0])};
  }
};

struct SelectOpConversion
    : ElementwiseOpConversionBase<arith::SelectOp, SelectOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SelectOp, SelectOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::SelectOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    std::array<Value, 3> llvmOperands;
    if (operands[0].size() == 2) {
      // Case of scalar condition with tensor operands.
      assert(op.getCondition().getType().isInteger(1));
      llvmOperands = {adaptor.getCondition(), operands[0][0], operands[0][1]};
    } else {
      llvmOperands = {operands[0][0], operands[0][1], operands[0][2]};
    }
    return {LLVM::SelectOp::create(rewriter, loc, llvmOperands[1].getType(),
                                   llvmOperands,
                                   adaptor.getAttributes().getValue())};
  }
};
template <typename OpTy>
struct MinMaxFOpConversion
    : ElementwiseOpConversionBase<OpTy, MinMaxFOpConversion<OpTy>> {
  using Base = ElementwiseOpConversionBase<OpTy, MinMaxFOpConversion<OpTy>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  static_assert(std::is_same<OpTy, arith::MinimumFOp>::value ||
                    std::is_same<OpTy, arith::MaximumFOp>::value,
                "OpTy must be arith::MinimumFOp or arith::MaximumFOp");

  // Choose the destination op based on the OpTy.
  using DestOpNanProp =
      typename std::conditional<std::is_same<OpTy, arith::MinimumFOp>::value,
                                LLVM::MinimumOp, LLVM::MaximumOp>::type;
  using DestOpNoNanProp =
      typename std::conditional<std::is_same<OpTy, arith::MinimumFOp>::value,
                                LLVM::MinNumOp, LLVM::MaxNumOp>::type;

  explicit MinMaxFOpConversion(LLVMTypeConverter &typeConverter,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               bool hwNanPropagationSupported,
                               PatternBenefit benefit = 1)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        hwNanPropagationSupported(hwNanPropagationSupported) {}

  SmallVector<Value> createDestOps(OpTy op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (hwNanPropagationSupported) {
      return {DestOpNanProp::create(rewriter, loc, elemTy, operands[0][0],
                                    operands[0][1])};
    }
    // Handle workaround for NaN propagation, i.e. software emulation of NaN
    // propagation. If any of the operands is NaN, return NaN.
    auto lhs = operands[0][0];
    auto rhs = operands[0][1];
    auto lhsIsNan =
        LLVM::FCmpOp::create(rewriter, loc, LLVM::FCmpPredicate::une, lhs, lhs);
    auto rhsIsNan =
        LLVM::FCmpOp::create(rewriter, loc, LLVM::FCmpPredicate::une, rhs, rhs);
    auto isNan = LLVM::OrOp::create(rewriter, loc, lhsIsNan, rhsIsNan);
    auto nonNanRes = DestOpNoNanProp::create(rewriter, loc, elemTy, lhs, rhs);

    auto nan = LLVM::createNaNConstant(loc, rewriter, elemTy);

    // Select the result based on the isNan flag.
    return {LLVM::SelectOp::create(rewriter, loc, isNan, nan, nonNanRes)};
  }

private:
  bool hwNanPropagationSupported;
};

struct ClampFOpConversion
    : ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion> {
  using Base = ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit ClampFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        targetInfo(targetInfo) {}

  SmallVector<Value> createDestOps(ClampFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // Clip pattern not found, use min/max.
    if (op.getPropagateNan() == PropagateNan::ALL) {
      if (targetInfo.supportMaximumMinimum()) {
        auto v = LLVM::MaximumOp::create(rewriter, loc, elemTy, operands[0][0],
                                         operands[0][1]);
        return {LLVM::MinimumOp::create(rewriter, loc, v, operands[0][2])};
      }
      // On pre-80 compute capability, we need to handle NaN propagation
      // manually. We need to check only the first operand for clamp.
      auto lhs = operands[0][0];
      auto isNan = LLVM::FCmpOp::create(rewriter, loc, LLVM::FCmpPredicate::une,
                                        lhs, lhs);
      auto v = LLVM::MaxNumOp::create(rewriter, loc, elemTy, operands[0][0],
                                      operands[0][1]);
      auto nonNanRes = LLVM::MinNumOp::create(rewriter, loc, v, operands[0][2]);
      auto nan = LLVM::createNaNConstant(loc, rewriter, elemTy);
      // Select the result based on the isNan flag.
      return {LLVM::SelectOp::create(rewriter, loc, isNan, nan, nonNanRes)};
    }

    // No NaN propagation.
    assert(op.getPropagateNan() == PropagateNan::NONE);
    auto v = LLVM::MaxNumOp::create(rewriter, loc, elemTy, operands[0][0],
                                    operands[0][1]);
    return {LLVM::MinNumOp::create(rewriter, loc, v, operands[0][2])};
  }

protected:
  const TargetInfoBase &targetInfo;
};

struct MapElementwiseOpConversion
    : public ConvertOpToLLVMPattern<MapElementwiseOp> {
  using Base = ConvertOpToLLVMPattern<MapElementwiseOp>;
  using Adaptor = typename Base::OpAdaptor;

  using Base::Base;

  LogicalResult matchAndRewrite(MapElementwiseOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();

    auto operands = adaptor.getOperands();
    const auto nOperands = operands.size();
    const auto nElems = getUniqueElemsPerThread(op->getOperand(0).getType());
    const auto nElemsPerPack = op.getPack();
    if (nElems % nElemsPerPack != 0)
      return op->emitError()
             << "pack size must be a divisor of the number of elements per "
                "thread, but got pack = "
             << nElemsPerPack << ", elements per thread = " << nElems << "\n";

    const auto nPacks = nElems / nElemsPerPack;
    auto nArgsUnpacked = nElemsPerPack * nOperands;

    SmallVector<Value> scalarOperands(nOperands * nElems);
    for (auto iOp : llvm::seq(nOperands)) {
      auto elems = unpackUniqueTensorElements(loc, operands[iOp], rewriter);
      assert(elems.size() == nElems);
      for (auto iPack : llvm::seq(nPacks)) {
        auto *packOperands =
            &scalarOperands[iPack * nArgsUnpacked + iOp * nElemsPerPack];
        auto *packElems = &elems[iPack * nElemsPerPack];
        for (auto iElem : llvm::seq(nElemsPerPack)) {
          packOperands[iElem] = packElems[iElem];
        }
      }
    }

    auto &scalarOp = op.getScalarOp();

    auto nOutputs = op.getNumResults();
    SmallVector<Value> scalarOutputs(nOutputs * nElems);
    for (auto iPack : llvm::seq(nPacks)) {
      ArrayRef<Value> packedArgs(&scalarOperands[iPack * nArgsUnpacked],
                                 nArgsUnpacked);
      auto packResults = inlineRegion<triton::MapElementwiseReturnOp>(
          rewriter, scalarOp, packedArgs, loc);
      assert(packResults.size() == nOutputs * nElemsPerPack);
      for (auto iOut : llvm::seq(nOutputs)) {
        auto *packOutputs =
            &scalarOutputs[iOut * nElems + iPack * nElemsPerPack];
        for (auto iElem : llvm::seq(nElemsPerPack)) {
          packOutputs[iElem] = packResults[iOut * nElemsPerPack + iElem];
        }
      }
    }

    SmallVector<Value> packedOutputs(nOutputs);
    for (auto iOut : llvm::seq(nOutputs)) {
      ArrayRef<Value> vals(&scalarOutputs[iOut * nElems], nElems);
      packedOutputs[iOut] = packUniqueTensorElements(
          loc, typeConverter, vals, rewriter, op.getType(iOut));
    }
    rewriter.replaceOp(op, packedOutputs);
    return success();
  }
};

} // namespace

void mlir::triton::populateMinMaxFOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, bool hwNanPropagationSupported,
    PatternBenefit benefit) {
  patterns.add<MinMaxFOpConversion<arith::MinimumFOp>>(
      typeConverter, axisInfoAnalysis, hwNanPropagationSupported, benefit);
  patterns.add<MinMaxFOpConversion<arith::MaximumFOp>>(
      typeConverter, axisInfoAnalysis, hwNanPropagationSupported, benefit);
}

void mlir::triton::populateClampFOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit) {
  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis, targetInfo,
                                   benefit);
}

void mlir::triton::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit) {
#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);

  POPULATE_UNARY_OP(arith::TruncIOp, LLVM::TruncOp)
  POPULATE_UNARY_OP(arith::ExtSIOp, LLVM::SExtOp)
  POPULATE_UNARY_OP(arith::ExtUIOp, LLVM::ZExtOp)
  POPULATE_UNARY_OP(arith::FPToUIOp, LLVM::FPToUIOp)
  POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)
  POPULATE_UNARY_OP(arith::NegFOp, LLVM::FNegOp)
  POPULATE_UNARY_OP(math::FloorOp, math::FloorOp)
  POPULATE_UNARY_OP(math::CeilOp, math::CeilOp)
  POPULATE_UNARY_OP(math::LogOp, math::LogOp)
  POPULATE_UNARY_OP(math::Log2Op, math::Log2Op)
  POPULATE_UNARY_OP(math::CosOp, math::CosOp)
  POPULATE_UNARY_OP(math::SinOp, math::SinOp)
  POPULATE_UNARY_OP(math::SqrtOp, math::SqrtOp)
  POPULATE_UNARY_OP(math::RsqrtOp, math::RsqrtOp)
  POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)
  POPULATE_UNARY_OP(math::Exp2Op, math::Exp2Op)
  POPULATE_UNARY_OP(math::ErfOp, math::ErfOp)
  POPULATE_UNARY_OP(triton::BitcastOp, LLVM::BitcastOp)
  POPULATE_UNARY_OP(triton::IntToPtrOp, LLVM::IntToPtrOp)
  POPULATE_UNARY_OP(triton::PtrToIntOp, LLVM::PtrToIntOp)
#undef POPULATE_UNARY_OP

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);

  POPULATE_BINARY_OP(arith::SubIOp, LLVM::SubOp) // -
  POPULATE_BINARY_OP(arith::AddIOp, LLVM::AddOp) // +
  POPULATE_BINARY_OP(arith::MulIOp, LLVM::MulOp) // *
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
  // fmin (return non-NaN if either op is non-NaN)
  POPULATE_BINARY_OP(arith::MinNumFOp, LLVM::MinNumOp)
  // fmax (return non-NaN if either op is non-NaN)
  POPULATE_BINARY_OP(arith::MaxNumFOp, LLVM::MaxNumOp)
  POPULATE_BINARY_OP(arith::MinSIOp, LLVM::SMinOp) // smin
  POPULATE_BINARY_OP(arith::MaxSIOp, LLVM::SMaxOp) // smax
  POPULATE_BINARY_OP(arith::MinUIOp, LLVM::UMinOp) // umin
  POPULATE_BINARY_OP(arith::MaxUIOp, LLVM::UMaxOp) // umax
#undef POPULATE_BINARY_OP

  patterns.add<ElementwiseOpConversion<math::FmaOp, LLVM::FMAOp>>(
      typeConverter, axisInfoAnalysis, benefit);

  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<CmpIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<CmpFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<MulhiUIOpConversion>(typeConverter, axisInfoAnalysis, targetInfo,
                                    benefit);
  patterns.add<ExternElementwiseOpConversion>(typeConverter, axisInfoAnalysis,
                                              benefit);
  patterns.add<InlineAsmOpConversion<ElementwiseInlineAsmOp>,
               InlineAsmOpConversion<triton::gpu::InlineAsmOp>>(typeConverter,
                                                                benefit);
  patterns.add<AbsIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<AbsFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SelectOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<MapElementwiseOpConversion>(typeConverter, benefit);
}
