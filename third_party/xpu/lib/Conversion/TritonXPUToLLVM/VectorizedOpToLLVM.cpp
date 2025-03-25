//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
// clang-format off
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"
// clang-format on

namespace {
// TODO[dyq]: add to head file
enum class ElemState {
  SS = 0, /*00*/
  SV = 1, /*01*/
  VS = 2, /*10*/
  VV = 3  /*11*/
};

template <typename OP> struct SVOp2Str;

#define SVOp2ASMStr(SrcType, ASM_STR)                                          \
  template <> struct SVOp2Str<SrcType> {                                       \
    static const llvm::StringRef value;                                        \
  };                                                                           \
  const llvm::StringRef SVOp2Str<SrcType>::value = ASM_STR;

SVOp2ASMStr(triton::xpu::SvaddFOp, "vadd.f.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStr(triton::xpu::SvmulFOp, "vmul.f.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStr(triton::xpu::SvsubFOp, "vsub.f.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStr(triton::xpu::SvmaxFOp, "vmax.f.mz $0{mr1}, $1, $2");

template <typename OP> struct SVOp2StrFP16;

#define SVOp2ASMStrFP16(SrcType, ASM_STR)                                      \
  template <> struct SVOp2StrFP16<SrcType> {                                   \
    static const llvm::StringRef value;                                        \
  };                                                                           \
  const llvm::StringRef SVOp2StrFP16<SrcType>::value = ASM_STR;

SVOp2ASMStrFP16(triton::xpu::SvaddFOp, "vadd.hf.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStrFP16(triton::xpu::SvmulFOp, "vmul.hf.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStrFP16(triton::xpu::SvsubFOp, "vsub.hf.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStrFP16(triton::xpu::SvmaxFOp, "vmax.hf.mz $0{mr1}, $1, $2");

template <typename OP> struct VLibOp;

#define VLibOp2DevCall(SrcType, ASM_STR)                                       \
  template <> struct VLibOp<SrcType> {                                         \
    static const llvm::StringRef value;                                        \
  };                                                                           \
  const llvm::StringRef VLibOp<SrcType>::value = ASM_STR;

VLibOp2DevCall(triton::xpu::VSinFOp, "_ZN3xpu5vsinfEDv16_f");
VLibOp2DevCall(triton::xpu::VCosFOp, "_ZN3xpu5vcosfEDv16_f");

template <typename OP, int ARCH> struct VLibOpFP16;

#define VLibOpFP162DevCall(SrcType, ARCH, ASM_STR)                             \
  template <> struct VLibOpFP16<SrcType, ARCH> {                               \
    static const llvm::StringRef value;                                        \
  };                                                                           \
  const llvm::StringRef VLibOpFP16<SrcType, ARCH>::value = ASM_STR;

VLibOpFP162DevCall(triton::xpu::VSinFOp, 2, "_ZN3xpu5vsinfEDv32_t");
VLibOpFP162DevCall(triton::xpu::VCosFOp, 2, "_ZN3xpu5vcosfEDv32_t");
VLibOpFP162DevCall(triton::xpu::VSinFOp, 3, "_ZN3xpu5vsinfEDv32_DF16_");
VLibOpFP162DevCall(triton::xpu::VCosFOp, 3, "_ZN3xpu5vcosfEDv32_DF16_");

} // namespace

namespace {

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::getTotalElemsPerThread;

struct XPUVectorizedOpsConversionBase {

  explicit XPUVectorizedOpsConversionBase(
      const triton::xpu::TargetInfo &targetInfo) {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      xpuArch = 2;
      break;
    }
    case XPUArch::XPU3: {
      xpuArch = 3;
      break;
    }
    default:
      llvm_unreachable(
          "Failed to create GM2LMOp with unsupported xpu architecture.");
    }
  }

  unsigned getVectorSize(Type type) const {
    auto vectorTy = mlir::dyn_cast<mlir::VectorType>(type);
    if (!vectorTy)
      return 1;
    auto elemTy = vectorTy.getElementType();
    auto width = elemTy.getIntOrFloatBitWidth();

    auto shape = vectorTy.getShape();
    if (shape[0] != 16) { // return vecSize = numElems for vector<numElemsxf32>
      return shape[0];
    }

    return 512 / width;
  }

  Type convertVectorType(Type type) const {
    auto vectorType = mlir::cast<mlir::VectorType>(type);
    auto ctx = vectorType.getContext();
    auto elemTy = vectorType.getElementType();
    if (elemTy.isF16())
      return LLVM::getFixedVectorType(LLVM::type::f16Ty(ctx),
                                      getVectorSize(type));
    else if (elemTy.isF32())
      return LLVM::getFixedVectorType(LLVM::type::f32Ty(ctx),
                                      getVectorSize(type));
    else if (elemTy.isInteger(16))
      return LLVM::getFixedVectorType(LLVM::type::i16Ty(ctx),
                                      getVectorSize(type));
    else if (elemTy.isInteger(32))
      return LLVM::getFixedVectorType(LLVM::type::i32Ty(ctx),
                                      getVectorSize(type));
    else if (elemTy.isBF16())
      return LLVM::getFixedVectorType(LLVM::type::bf16Ty(ctx),
                                      getVectorSize(type));

    llvm_unreachable("Not implemented.");
  }

protected:
  int xpuArch = 3;
};

template <typename SrcOp, typename DstOp>
struct VVBinOpsConversion : public ConvertOpToLLVMPattern<SrcOp>,
                            public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VVBinOpsConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                     const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    Value lllhs = adaptor.getLhs();
    Value llrhs = adaptor.getRhs();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = lhs.getType();

    Type valueElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(valueTy));
    unsigned numElems = getTotalElemsPerThread(valueTy);

    auto lhsElems = unpackLLElements(loc, lllhs, rewriter);
    auto rhsElems = unpackLLElements(loc, llrhs, rewriter);
    assert(lhsElems.size() == rhsElems.size());

    SmallVector<Value> calculatedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += 1) {
      Value vaddOp =
          rewriter.create<DstOp>(loc, convertVectorType(valueElemTy),
                                 lhsElems[vecStart], rhsElems[vecStart]);
      calculatedVals.push_back(vaddOp);
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

template <typename SrcOp>
struct SVBinOpsConversion : public ConvertOpToLLVMPattern<SrcOp>,
                            public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  SVBinOpsConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                     const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    int32_t elemStateInt = op.getElemState();
    ElemState elemState = static_cast<ElemState>(elemStateInt);

    Value lllhs = adaptor.getLhs();
    Value llrhs = adaptor.getRhs();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Type valueTy;

    if (elemState == ElemState::SV) {
      valueTy = rhs.getType();
    } else if (elemState == ElemState::VS) {
      valueTy = lhs.getType();
    }

    Type valueElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(valueTy));
    unsigned numElems = getTotalElemsPerThread(valueTy);

    // Get data from a struct
    auto lhsElems = unpackLLElements(loc, lllhs, rewriter);
    auto rhsElems = unpackLLElements(loc, llrhs, rewriter);

    // Create LLVM Op
    SmallVector<Value> calculatedVals;
    Type vecTy = getElementTypeOrSelf(valueTy);
    Type elemTy = getElementTypeOrSelf(vecTy);
    StringRef asm_string;
    if (elemTy.isF32()) {
      asm_string = SVOp2Str<SrcOp>::value;
    } else if (elemTy.isF16()) {
      asm_string = SVOp2StrFP16<SrcOp>::value;
    } else {
      llvm_unreachable("Only FP16 and FP32 are supported in SVBinary!");
    }
    StringRef constraints = "=v,r,v";
    for (size_t vecStart = 0; vecStart < numElems; vecStart += 1) {
      if (elemState == ElemState::SV) {
        SmallVector<Value, 4> operands({lhsElems[0], rhsElems[vecStart]});
        auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
            loc, valueElemTy, operands, asm_string, constraints,
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        calculatedVals.push_back(asmOp.getRes());
      } else if (elemState == ElemState::VS) {
        SmallVector<Value, 4> operands({rhsElems[0], lhsElems[vecStart]});
        auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
            loc, valueElemTy, operands, asm_string, constraints,
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        calculatedVals.push_back(asmOp.getRes());
      }
    }

    // Wrap data into a struct
    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

template <typename SrcOp, typename DstOp>
struct UnaryOpConversion : public ConvertOpToLLVMPattern<SrcOp>,
                           public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  UnaryOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                    const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value value = op.getValue();
    Value result = op.getResult();

    Value llvalue = adaptor.getValue();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto resultTy = result.getType();
    Type resultElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(resultTy));
    unsigned numElems = getTotalElemsPerThread(value.getType());

    auto valueElems = unpackLLElements(loc, llvalue, rewriter);

    SmallVector<Value> calculatedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += 1) {
      Value vexpOp = rewriter.create<DstOp>(
          loc, convertVectorType(resultElemTy), valueElems[vecStart]);
      calculatedVals.push_back(vexpOp);
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(resultTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

template <typename SrcOp>
struct VOpConversionLibCall : public ConvertOpToLLVMPattern<SrcOp>,
                              public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VOpConversionLibCall(LLVMTypeConverter &converter, PatternBenefit benefit,
                       const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    Location loc = op->getLoc();
    unsigned vecElems = getTotalElemsPerThread(resultTy);
    auto resultVecTy = getElementTypeOrSelf(resultTy);
    Type vecTy = this->getTypeConverter()->convertType(resultVecTy);
    auto elemTy = getElementTypeOrSelf(vecTy);
    SmallVector<Type> types(vecElems, vecTy);
    Type structTy = this->getTypeConverter()->convertType(resultTy);

    auto operands = getOperands(rewriter, adaptor, vecElems, loc);
    SmallVector<Value> resultVals(vecElems);
    for (unsigned i = 0; i < vecElems; ++i) {
      ValueRange singleOperandRange(operands[i]);
      if (elemTy.isF32()) {
        Value devCall = mlir::LLVM::XPU::createDeviceCall(
            VLibOp<SrcOp>::value, rewriter, op, vecTy, singleOperandRange, loc);
        resultVals[i] = devCall;
      } else if (elemTy.isF16()) {
        Value devCall;
        switch (xpuArch) {
        case 2:
          devCall = mlir::LLVM::XPU::createDeviceCall(
              VLibOpFP16<SrcOp, 2>::value, rewriter, op, vecTy,
              singleOperandRange, loc);
          break;
        case 3:
          devCall = mlir::LLVM::XPU::createDeviceCall(
              VLibOpFP16<SrcOp, 3>::value, rewriter, op, vecTy,
              singleOperandRange, loc);
          break;
        default:
          llvm_unreachable("Failed to create device call with unsupported xpu "
                           "architecture.");
        }
        resultVals[i] = devCall;
      } else {
        llvm_unreachable("Only FP16 and FP32 are supported in LibDevice!");
      }
      if (!bool(resultVals[i]))
        return failure();
    }
    Value view =
        packLLElements(loc, getTypeConverter(), resultVals, rewriter, structTy);
    rewriter.replaceOp(op, view);

    return success();
  }

private:
  SmallVector<SmallVector<Value>>
  getOperands(ConversionPatternRewriter &rewriter, OpAdaptor adaptor,
              const unsigned elems, Location loc) const {
    SmallVector<SmallVector<Value>> operands(elems);
    for (auto operand : adaptor.getOperands()) {
      auto sub_operands = unpackLLElements(loc, operand, rewriter);
      for (size_t i = 0; i < elems; ++i) {
        operands[i].push_back(sub_operands[i]);
      }
    }
    return operands;
  }
};

template <typename SrcOp, typename DstOp>
struct VConstOpConversion : public ConvertOpToLLVMPattern<SrcOp>,
                            public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VConstOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                     const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value res = op.getResult();
    Attribute attr = adaptor.getValueAttr();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto resTy = res.getType();
    Type resElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(resTy));
    unsigned numElems = getTotalElemsPerThread(res.getType());

    SmallVector<Value> calculatedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += 1) {
      Value vconstOp =
          rewriter.create<DstOp>(loc, convertVectorType(resElemTy), attr);
      calculatedVals.push_back(vconstOp);
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

template <typename SrcOp>
struct VSplatOpConversion : public ConvertOpToLLVMPattern<SrcOp>,
                            public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VSplatOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                     const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  Value convertSplatLikeOp(Type resTy, Value llsrc, Type llvmResultStructTy,
                           ConversionPatternRewriter &rewriter,
                           Location loc) const {
    auto resElemTy =
        this->getTypeConverter()->convertType(getElementTypeOrSelf(resTy));
    size_t elemsPerThread = getTotalElemsPerThread(resTy);

    auto valueElems = unpackLLElements(loc, llsrc, rewriter);

    Value vector_1xTy = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
    vector_1xTy =
        insert_element(resElemTy, vector_1xTy, valueElems[0], i32_val(0));

    int32_t vecSize = cast<mlir::VectorType>(resElemTy).getNumElements();
    SmallVector<int32_t, 16> zeroValues(vecSize, 0);
    // TODO[dyq]: check getI32ArrayAttr -> getDenseI32ArrayAttr
    auto zeroAttrs = rewriter.getDenseI32ArrayAttr(zeroValues);
    Value shuffleVectorOp = rewriter.create<LLVM::ShuffleVectorOp>(
        loc, resElemTy, vector_1xTy, vector_1xTy, zeroAttrs);

    llvm::SmallVector<Value> elems(elemsPerThread, shuffleVectorOp);

    return packLLElements(loc, this->getTypeConverter(), elems, rewriter,
                          llvmResultStructTy);
  }

  LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto llsrc = adaptor.getSrc();
    auto llvmResultStructTy =
        this->getTypeConverter()->convertType(op.getType());
    auto llStruct = convertSplatLikeOp(op.getType(), llsrc, llvmResultStructTy,
                                       rewriter, loc);

    rewriter.replaceOp(op, {llStruct});
    return success();
  }
};

template <typename SrcOp>
struct VSelectOpConversion : public ConvertOpToLLVMPattern<SrcOp>,
                             public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VSelectOpConversion(LLVMTypeConverter &converter,

                      PatternBenefit benefit,
                      const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // original values
    Value condition = op.getCondition();
    Value true_value = op.getTrueValue();
    Value false_value = op.getFalseValue();

    // adaptor values
    Value llCondition = adaptor.getCondition();
    Value llTrue_value = adaptor.getTrueValue();
    Value llFalse_value = adaptor.getFalseValue();

    MLIRContext *ctx = rewriter.getContext();
    auto loc = op->getLoc();
    auto resTy = op.getType();

    Type resElemTy = getTypeConverter()->convertType(
        getElementTypeOrSelf(resTy)); // vector<16xf32>
    unsigned numElems = getTotalElemsPerThread(resTy);

    // Get data from a struct
    auto conditionElems = unpackLLElements(loc, llCondition, rewriter);
    auto trueValElems = unpackLLElements(loc, llTrue_value, rewriter);
    auto falseValElems = unpackLLElements(loc, llFalse_value, rewriter);

    // Create LLVM Op
    Type elemTy = getElementTypeOrSelf(resElemTy);
    unsigned elemBits = elemTy.getIntOrFloatBitWidth();
    unsigned vecSize = mlir::cast<VectorType>(resElemTy).getNumElements();
    SmallVector<Value> resVals;

    for (size_t elemIter = 0; elemIter < numElems; ++elemIter) {
      // Step 1. Convert Condition To v32i1/v16i1 Mask
      Value orV = i32_val(0);
      for (size_t conditionIter = 0; conditionIter < vecSize; ++conditionIter) {
        Value boolVal =
            isa<VectorType>(conditionElems[0].getType())
                ? extract_element(i1_ty, conditionElems[elemIter],
                                  i32_val(conditionIter))
                : conditionElems[elemIter * vecSize + conditionIter];
        Value extV = zext(i32_ty, boolVal);
        Value shlV = shl(extV, i32_val(conditionIter));
        orV = or_(orV, shlV);
      }
      VectorType maskTy = VectorType::get(32, i1_ty);
      Value maskV = bitcast(orV, maskTy);

      if (elemTy.isF32()) {
        // Step 2. vset_zero()
        StringRef xor_asm_string = "vxor.s.mz $0{mr1}, $0, $0";
        StringRef xor_constraints = "=v";
        SmallVector<Value, 4> xor_operands({});
        auto zerosIAsmOp = rewriter.create<LLVM::InlineAsmOp>(
            loc, resElemTy, xor_operands, xor_asm_string, xor_constraints,
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        Value zerosFAsmOp = bitcast(zerosIAsmOp.getRes(), resElemTy);
        // Step 3. vvor_float32x16_mh(mask, zero, a, b)
        Value vvorFOp = rewriter.create<mlir::LLVM::XPU::VVOR_F_MHOp>(
            loc, resElemTy, maskV, zerosFAsmOp, trueValElems[elemIter],
            falseValElems[elemIter]);
        resVals.push_back(vvorFOp);
      } else if (elemTy.isInteger(32)) {
        // Step 2. vset_zero()
        StringRef xor_asm_string = "vxor.s.mz $0{mr1}, $0, $0";
        StringRef xor_constraints = "=v";
        SmallVector<Value, 4> xor_operands({});
        auto zerosIAsmOp = rewriter.create<LLVM::InlineAsmOp>(
            loc, resElemTy, xor_operands, xor_asm_string, xor_constraints,
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        Value zerosFAsmOp = bitcast(zerosIAsmOp.getRes(), resElemTy);
        // Step 3. vvor_int32x16_mh(mask, zero, a, b)
        Value vvorFOp = rewriter.create<mlir::LLVM::XPU::VVOR_S_MHOp>(
            loc, resElemTy, maskV, zerosFAsmOp, trueValElems[elemIter],
            falseValElems[elemIter]);
        resVals.push_back(vvorFOp);
      } else if (elemTy.isF16()) {
        // Step 2. vset_zero()
        StringRef xor_asm_string = "vxor.hf.mz $0{mr1}, $0, $0";
        StringRef xor_constraints = "=v";
        SmallVector<Value, 4> xor_operands({});
        auto zerosFAsmOp = rewriter.create<LLVM::InlineAsmOp>(
            loc, resElemTy, xor_operands, xor_asm_string, xor_constraints,
            /*has_side_effects=*/true,
            /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        // Step 3. vvor_float16x32_mh(mask, zero, a, b)
        Value vvorFOp = rewriter.create<mlir::LLVM::XPU::VVOR_HF_MHOp>(
            loc, resElemTy, maskV, zerosFAsmOp.getRes(), trueValElems[elemIter],
            falseValElems[elemIter]);
        resVals.push_back(vvorFOp);
      } else {
        llvm_unreachable("Only FP16 and FP32 are supported in VSelect!");
      }
    }

    // Wrap data into a struct
    auto llvmResultStructTy = getTypeConverter()->convertType(resTy);
    auto llStruct = packLLElements(loc, getTypeConverter(), resVals, rewriter,
                                   llvmResultStructTy);
    rewriter.replaceOp(op, {llStruct});

    return success();
  }
};

template <typename SrcOp>
struct VMacFOpConversion : public ConvertOpToLLVMPattern<SrcOp>,
                           public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VMacFOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                    const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {

    // original values
    Value value = op.getValue();
    Value mulData = op.getMulData();
    Value addData = op.getAddData();

    // adaptor values
    Value llValue = adaptor.getValue();
    Value llMulData = adaptor.getMulData();
    Value llAddData = adaptor.getAddData();
    auto attrs = adaptor.getAttributes();

    MLIRContext *ctx = rewriter.getContext();
    auto loc = op->getLoc();
    auto resTy = op.getType();

    auto resElemTy = getTypeConverter()->convertType(
        getElementTypeOrSelf(resTy)); // vector<16xf32>
    unsigned numElems = getTotalElemsPerThread(resTy);

    // Get data from a struct
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    auto mulElems = unpackLLElements(loc, llMulData, rewriter);
    auto addElems = unpackLLElements(loc, llAddData, rewriter);

    // Create LLVM Op
    SmallVector<Value> calculatedVals;
    auto elemTy = getElementTypeOrSelf(resElemTy);
    StringRef asm_string;
    if (elemTy.isF32()) {
      asm_string = "vmac.f.mz.rn $0{mr1}, $1, $2";
    } else if (elemTy.isF16()) {
      asm_string = "vmac.hf.mz.rn $0{mr1}, $1, $2";
    } else {
      llvm_unreachable("Only FP16 and FP32 are supported in VMac!");
    }
    StringRef constraints = "=v,v,v,0";
    for (size_t vecStart = 0; vecStart < numElems; vecStart += 1) {
      SmallVector<Value, 4> operands(
          {valueElems[vecStart], mulElems[vecStart], addElems[vecStart]});
      auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
          loc, resElemTy, operands, asm_string, constraints,
          /*has_side_effects=*/true,
          /*is_align_stack=*/false,
          LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
          ArrayAttr());
      calculatedVals.push_back(asmOp.getRes());
    }

    // Wrap data into a struct
    auto llvmResultStructTy = getTypeConverter()->convertType(resTy);
    auto llStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                   rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {llStruct});

    return success();
  }
};

struct VExtFOpConversion : public ConvertOpToLLVMPattern<triton::xpu::VExtFOp>,
                           public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<triton::xpu::VExtFOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<triton::xpu::VExtFOp>::getTypeConverter;
  using OpAdaptor = typename triton::xpu::VExtFOp::Adaptor;

  VExtFOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                    const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<triton::xpu::VExtFOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  Value convertFp16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                          Value val, Value res, Type valElemTy, Type resElemTy,
                          Value llVal, Type llvmResultStructTy) const {
    auto ctx = rewriter.getContext();
    auto llVals = unpackLLElements(loc, llVal, rewriter);
    unsigned numElems = getTotalElemsPerThread(val.getType());

    SmallVector<Value, 8> fp32x16Vecs;
    for (int i = 0; i < numElems; ++i) {
      auto asml = rewriter.create<LLVM::InlineAsmOp>(
          loc, resElemTy, ValueRange{llVals[i]}, // operands
          "vfp162float_l.rn $0, $1",             // asm_string
          "=&v,v",                               // constraints
          false,                                 // has_size_effects
          false,                                 // is_align_stack
          LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
          ArrayAttr::get(ctx, {}));
      fp32x16Vecs.emplace_back(asml.getRes());
      auto asmh = rewriter.create<LLVM::InlineAsmOp>(
          loc, resElemTy, ValueRange{llVals[i]}, // operands
          "vfp162float_h.rn $0, $1",             // asm_string
          "=&v,v",                               // constraints
          false,                                 // has_size_effects
          false,                                 // is_align_stack
          LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
          ArrayAttr::get(ctx, {}));
      fp32x16Vecs.emplace_back(asmh.getRes());
    }

    Value resultStruct = packLLElements(loc, getTypeConverter(), fp32x16Vecs,
                                        rewriter, llvmResultStructTy);
    return resultStruct;
  }

  Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                          Value val, Value res, Type valElemTy, Type resElemTy,
                          Value llVal, Type llvmResultStructTy) const {
    auto ctx = rewriter.getContext();
    auto llVals = unpackLLElements(loc, llVal, rewriter);
    unsigned numElems = getTotalElemsPerThread(val.getType());

    VectorType vecFp16Ty = VectorType::get(32, f16_ty);
    Value padVec = rewriter.create<LLVM::UndefOp>(loc, vecFp16Ty);
    for (size_t elemIdx = 0; elemIdx < 32; ++elemIdx) {
      padVec = insert_element(vecFp16Ty, padVec, f16_val(0), i16_val(elemIdx));
    }

    SmallVector<Value, 8> fp32x16Vecs;
    for (int i = 0; i < numElems; ++i) {
      Value val = bitcast(llVals[i], vecFp16Ty);
      Value vl = rewriter.create<mlir::LLVM::XPU::VMERGE_L_HFOp>(loc, vecFp16Ty,
                                                                 padVec, val);
      vl = bitcast(vl, resElemTy);
      fp32x16Vecs.emplace_back(vl);
      Value vh = rewriter.create<mlir::LLVM::XPU::VMERGE_H_HFOp>(loc, vecFp16Ty,
                                                                 padVec, val);
      vh = bitcast(vh, resElemTy);
      fp32x16Vecs.emplace_back(vh);
    }

    Value resultStruct = packLLElements(loc, getTypeConverter(), fp32x16Vecs,
                                        rewriter, llvmResultStructTy);
    return resultStruct;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::VExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto val = op.getValue();
    auto res = op.getResult();
    auto llVal = adaptor.getValue();

    Type valTy = val.getType();
    Type resTy = res.getType();
    auto valElemTy = getElementTypeOrSelf(valTy);
    auto _valElemTy = getElementTypeOrSelf(valElemTy);
    auto resElemTy = getElementTypeOrSelf(resTy);
    auto _resElemTy = getElementTypeOrSelf(resElemTy);
    auto llValElemTy = typeConverter->convertType(valElemTy);
    auto llResElemTy = typeConverter->convertType(resElemTy);
    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    assert(_resElemTy.isF32() && "Only support F32 as target dtype inVExtF!");
    if (_valElemTy.isF16()) {
      auto result = convertFp16ToFp32(loc, rewriter, val, res, valElemTy,
                                      resElemTy, llVal, llvmResultStructTy);
      rewriter.replaceOp(op, {result});
    } else if (_valElemTy.isBF16()) {
      auto result = convertBf16ToFp32(loc, rewriter, val, res, valElemTy,
                                      resElemTy, llVal, llvmResultStructTy);
      rewriter.replaceOp(op, {result});
    } else {
      assert(0 && "Only support FP16 as source dtype in VExtF!");
    }
    return success();
  }
};

struct VTruncFOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::VTruncFOp>,
      public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<triton::xpu::VTruncFOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<triton::xpu::VTruncFOp>::getTypeConverter;
  using OpAdaptor = typename triton::xpu::VTruncFOp::Adaptor;

  VTruncFOpConversion(LLVMTypeConverter &converter,

                      PatternBenefit benefit,
                      const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<triton::xpu::VTruncFOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  Value convertFp32ToFp16(Location loc, ConversionPatternRewriter &rewriter,
                          Value val, Value res, Type valElemTy, Type resElemTy,
                          Value llVal, Type llvmResultStructTy) const {
    auto ctx = rewriter.getContext();
    auto llVals = unpackLLElements(loc, llVal, rewriter);
    unsigned numElems = getTotalElemsPerThread(val.getType());

    SmallVector<Value, 8> fp16x32Vecs;
    for (int i = 0; i < numElems; i += 2) {
      auto asmlh = rewriter.create<LLVM::InlineAsmOp>(
          loc, resElemTy, ValueRange{llVals[i], llVals[i + 1]}, // operands
          "vfloat2fp16_l.rn $0, $1\nvfloat2fp16_h.rn $0, $2",   // asm_string
          "=&v,v,v",                                            // constraints
          false, // has_size_effects
          false, // is_align_stack
          LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
          ArrayAttr::get(ctx, {}));
      fp16x32Vecs.push_back(asmlh.getRes());
    }

    Value resultStruct = packLLElements(loc, getTypeConverter(), fp16x32Vecs,
                                        rewriter, llvmResultStructTy);
    return resultStruct;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::VTruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto val = op.getValue();
    auto res = op.getResult();
    auto llVal = adaptor.getValue();

    Type valTy = val.getType();
    Type resTy = res.getType();
    auto valElemTy = getElementTypeOrSelf(valTy);
    auto _valElemTy = getElementTypeOrSelf(valElemTy);
    auto resElemTy = getElementTypeOrSelf(resTy);
    auto _resElemTy = getElementTypeOrSelf(resElemTy);
    auto llValElemTy = typeConverter->convertType(valElemTy);
    auto llResElemTy = typeConverter->convertType(resElemTy);
    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    assert(_valElemTy.isF32() &&
           "Only support F32 as source dtype in VTruncF!");
    if (_resElemTy.isF16()) {
      auto result = convertFp32ToFp16(loc, rewriter, val, res, valElemTy,
                                      resElemTy, llVal, llvmResultStructTy);
      rewriter.replaceOp(op, {result});
    } else {
      assert(0 && "Only support FP16 as target dtype in VTruncF!");
    }
    return success();
  }
};

struct VCmpFOpConversion : public ConvertOpToLLVMPattern<triton::xpu::VCmpFOp>,
                           public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<triton::xpu::VCmpFOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<triton::xpu::VCmpFOp>::getTypeConverter;
  using OpAdaptor = typename triton::xpu::VCmpFOp::Adaptor;

  VCmpFOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                    const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<triton::xpu::VCmpFOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult matchAndRewrite(triton::xpu::VCmpFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    Value lllhs = adaptor.getLhs();
    Value llrhs = adaptor.getRhs();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = lhs.getType();

    Type valueElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(valueTy));
    unsigned numElems = getTotalElemsPerThread(valueTy);

    auto lhsElems = unpackLLElements(loc, lllhs, rewriter);
    auto rhsElems = unpackLLElements(loc, llrhs, rewriter);
    assert(lhsElems.size() == rhsElems.size());

    auto resTy = op.getResult().getType();
    Type resElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(resTy));

    SmallVector<Value> calculatedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += 1) {
      Value vcmpfOp = rewriter.create<LLVM::FCmpOp>(
          loc, resElemTy, ArithCmpFPredicateToLLVM(op.getPredicate()),
          lhsElems[vecStart], rhsElems[vecStart]);
      calculatedVals.push_back(vcmpfOp);
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
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

} // namespace

void mlir::triton::xpu::populateTTXPUVectorizedOpToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, const triton::xpu::TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<VVBinOpsConversion<triton::xpu::VvaddFOp, LLVM::FAddOp>,
               VVBinOpsConversion<triton::xpu::VvsubFOp, LLVM::FSubOp>,
               VVBinOpsConversion<triton::xpu::VvmulFOp, LLVM::FMulOp>,
               VVBinOpsConversion<triton::xpu::VvdivFOp, LLVM::FDivOp>,
               VVBinOpsConversion<triton::xpu::VvmaxFOp, LLVM::MaximumOp>,
               VVBinOpsConversion<triton::xpu::VvminFOp, LLVM::MinimumOp>,
               VVBinOpsConversion<triton::xpu::VvmaxNumFOp, LLVM::MaxNumOp>,
               VVBinOpsConversion<triton::xpu::VvminNumFOp, LLVM::MinNumOp>,
               VVBinOpsConversion<triton::xpu::VvorIOp, LLVM::OrOp>,
               VVBinOpsConversion<triton::xpu::VvxorIOp, LLVM::XOrOp>,
               VVBinOpsConversion<triton::xpu::VvandIOp, LLVM::AndOp>,
               VVBinOpsConversion<triton::xpu::VvaddIOp, LLVM::AddOp>,
               VVBinOpsConversion<triton::xpu::VvsubIOp, LLVM::SubOp>,
               VVBinOpsConversion<triton::xpu::VvmulIOp, LLVM::MulOp>>(
      typeConverter, benefit, targetInfo);
  patterns.add<SVBinOpsConversion<triton::xpu::SvaddFOp>,
               SVBinOpsConversion<triton::xpu::SvmulFOp>,
               SVBinOpsConversion<triton::xpu::SvsubFOp>,
               SVBinOpsConversion<triton::xpu::SvmaxFOp>>(typeConverter,
                                                          benefit, targetInfo);
  patterns.add<UnaryOpConversion<triton::xpu::VExpFOp, LLVM::Exp2Op>,
               UnaryOpConversion<triton::xpu::VSqrtFOp, LLVM::SqrtOp>,
               UnaryOpConversion<triton::xpu::VAbsFOp, LLVM::FAbsOp>,
               UnaryOpConversion<triton::xpu::VSIToFPOp, LLVM::SIToFPOp>>(
      typeConverter, benefit, targetInfo);
  patterns.add<VOpConversionLibCall<triton::xpu::VSinFOp>,
               VOpConversionLibCall<triton::xpu::VCosFOp>>(typeConverter,
                                                           benefit, targetInfo);
  patterns.add<VConstOpConversion<triton::xpu::VConstOp, LLVM::ConstantOp>>(
      typeConverter, benefit, targetInfo);
  patterns.add<VSplatOpConversion<triton::xpu::VSplatOp>>(typeConverter,
                                                          benefit, targetInfo);
  patterns.add<VSelectOpConversion<triton::xpu::VSelectOp>>(
      typeConverter, benefit, targetInfo);
  patterns.add<VMacFOpConversion<triton::xpu::VMacFOp>>(typeConverter, benefit,
                                                        targetInfo);
  patterns.add<VExtFOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<VTruncFOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<VCmpFOpConversion>(typeConverter, benefit, targetInfo);
}
