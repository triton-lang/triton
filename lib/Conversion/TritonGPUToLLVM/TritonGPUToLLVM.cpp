#include "TritonGPUToLLVM.h"
#include "DotOpHelpers.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getElementsFromStruct;
using ::mlir::LLVM::getStructFromElements;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::getElemsPerThread;

struct ReturnOpConversion : public ConvertOpToLLVMPattern<::mlir::ReturnOp> {
  using ConvertOpToLLVMPattern<ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                op->getAttrs());
    return success();
  }
};

struct BroadcastOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::BroadcastOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::BroadcastOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Following the order of indices in the legacy code, a broadcast of:
    //   [s(0), s(1) ... s(k-1),    1, s(k+1), s(k+2) ... s(n-1)]
    // =>
    //   [s(0), s(1) ... s(k-1), s(k), s(k+1), s(k+2) ... s(n-1)]
    //
    // logically maps to a broadcast within a thread's scope:
    //   [cta(0)..cta(k-1),     1,cta(k+1)..cta(n-1),spt(0)..spt(k-1),
    //   1,spt(k+1)..spt(n-1)]
    // =>
    //   [cta(0)..cta(k-1),cta(k),cta(k+1)..cta(n-1),spt(0)..spt(k-1),spt(k),spt(k+1)..spt(n-1)]
    //
    // regardless of the order of the layout
    //
    Location loc = op->getLoc();
    Value src = adaptor.src();
    Value result = op.result();
    auto srcTy = op.src().getType().cast<RankedTensorType>();
    auto resultTy = result.getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();
    assert(rank == resultTy.getRank());
    auto order = triton::gpu::getOrder(srcLayout);
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcShape);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultShape);
    SmallVector<Value> srcVals = getElementsFromStruct(loc, src, rewriter);
    DenseMap<SmallVector<unsigned>, Value, SmallVectorKeyInfo> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); j++)
        if (srcShape[j] == 1)
          offset[j] = 0;
      resultVals.push_back(srcValues.lookup(offset));
    }
    auto llvmStructTy = getTypeConverter()->convertType(resultTy);
    Value resultStruct =
        getStructFromElements(loc, resultVals, rewriter, llvmStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct PrintfOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::PrintfOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::PrintfOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::PrintfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    SmallVector<Value, 16> operands;
    for (auto operand : adaptor.getOperands()) {
      auto sub_operands = getElementsFromStruct(loc, operand, rewriter);
      for (auto elem : sub_operands) {
        operands.push_back(elem);
      }
    }
    std::string formatStr;
    llvm::raw_string_ostream os(formatStr);
    os << op.prefix();
    if (!operands.empty()) {
      os << getFormatSubstr(operands[0]);
    }

    for (size_t i = 1; i < operands.size(); ++i) {
      os << ", " << getFormatSubstr(operands[i]);
    }
    llPrintf(formatStr, operands, rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  std::string getFormatSubstr(Value value) const {
    Type type = value.getType();
    if (type.isa<LLVM::LLVMPointerType>()) {
      return "%p";
    } else if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
      return "%f";
    } else if (type.isSignedInteger()) {
      return "%i";
    } else if (type.isUnsignedInteger() || type.isSignlessInteger()) {
      return "%u";
    }
    assert(false && "not supported type");
    return "";
  }

  // declare vprintf(i8*, i8*) as external function
  static LLVM::LLVMFuncOp
  getVprintfDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("vprintf");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto *context = rewriter.getContext();

    SmallVector<Type> argsType{ptr_ty(IntegerType::get(context, 8)),
                               ptr_ty(IntegerType::get(context, 8))};
    auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                             funcType);
  }

  // extend integer to int32, extend float to float64
  // this comes from vprintf alignment requirements.
  static std::pair<Type, Value>
  promoteValue(ConversionPatternRewriter &rewriter, Value value) {
    auto *context = rewriter.getContext();
    auto type = value.getType();
    Value newOp = value;
    Type newType = type;

    bool bUnsigned = type.isUnsignedInteger();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
      if (bUnsigned) {
        newType = ui32_ty;
        newOp = rewriter.create<LLVM::ZExtOp>(UnknownLoc::get(context), newType,
                                              value);
      } else {
        newType = i32_ty;
        newOp = rewriter.create<LLVM::SExtOp>(UnknownLoc::get(context), newType,
                                              value);
      }
    } else if (type.isBF16() || type.isF16() || type.isF32()) {
      newType = f64_ty;
      newOp = rewriter.create<LLVM::FPExtOp>(UnknownLoc::get(context), newType,
                                             value);
    }

    return {newType, newOp};
  }

  static void llPrintf(StringRef msg, ValueRange args,
                       ConversionPatternRewriter &rewriter) {
    static const char formatStringPrefix[] = "printfFormat_";
    assert(!msg.empty() && "printf with empty string not support");
    Type int8Ptr = ptr_ty(i8_ty);

    auto *context = rewriter.getContext();
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto funcOp = getVprintfDeclaration(rewriter);

    Value one = rewriter.create<LLVM::ConstantOp>(
        UnknownLoc::get(context), i32_ty, rewriter.getI32IntegerAttr(1));
    Value zero = rewriter.create<LLVM::ConstantOp>(
        UnknownLoc::get(context), i32_ty, rewriter.getI32IntegerAttr(0));

    unsigned stringNumber = 0;
    SmallString<16> stringConstName;
    do {
      stringConstName.clear();
      (formatStringPrefix + Twine(stringNumber++)).toStringRef(stringConstName);
    } while (moduleOp.lookupSymbol(stringConstName));

    llvm::SmallString<64> formatString(msg);
    formatString.push_back('\n');
    formatString.push_back('\0');
    size_t formatStringSize = formatString.size_in_bytes();
    auto globalType = LLVM::LLVMArrayType::get(i8_ty, formatStringSize);

    LLVM::GlobalOp global;
    {
      ConversionPatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      global = rewriter.create<LLVM::GlobalOp>(
          UnknownLoc::get(context), globalType,
          /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
          rewriter.getStringAttr(formatString));
    }

    Value globalPtr =
        rewriter.create<LLVM::AddressOfOp>(UnknownLoc::get(context), global);
    Value stringStart = rewriter.create<LLVM::GEPOp>(
        UnknownLoc::get(context), int8Ptr, globalPtr,
        SmallVector<Value>({zero, zero}));

    Value bufferPtr =
        rewriter.create<LLVM::NullOp>(UnknownLoc::get(context), int8Ptr);

    SmallVector<Value, 16> newArgs;
    if (args.size() >= 1) {
      SmallVector<Type> argTypes;
      for (auto arg : args) {
        Type newType;
        Value newArg;
        std::tie(newType, newArg) = promoteValue(rewriter, arg);
        argTypes.push_back(newType);
        newArgs.push_back(newArg);
      }

      Type structTy = LLVM::LLVMStructType::getLiteral(context, argTypes);
      auto allocated = rewriter.create<LLVM::AllocaOp>(UnknownLoc::get(context),
                                                       ptr_ty(structTy), one,
                                                       /*alignment=*/0);

      for (const auto &entry : llvm::enumerate(newArgs)) {
        auto index = rewriter.create<LLVM::ConstantOp>(
            UnknownLoc::get(context), i32_ty,
            rewriter.getI32IntegerAttr(entry.index()));
        auto fieldPtr = rewriter.create<LLVM::GEPOp>(
            UnknownLoc::get(context), ptr_ty(argTypes[entry.index()]),
            allocated, ArrayRef<Value>{zero, index});
        rewriter.create<LLVM::StoreOp>(UnknownLoc::get(context), entry.value(),
                                       fieldPtr);
      }
      bufferPtr = rewriter.create<LLVM::BitcastOp>(UnknownLoc::get(context),
                                                   int8Ptr, allocated);
    }

    SmallVector<Value> operands{stringStart, bufferPtr};
    rewriter.create<LLVM::CallOp>(UnknownLoc::get(context), funcOp, operands);
  }
};

struct MakeRangeOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::MakeRangeOp> {

  MakeRangeOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::MakeRangeOp>(converter,
                                                             benefit) {}

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto rankedTy = op.result().getType().dyn_cast<RankedTensorType>();
    auto shape = rankedTy.getShape();
    auto layout = rankedTy.getEncoding();

    auto elemTy = rankedTy.getElementType();
    assert(elemTy.isInteger(32));
    Value start = createIndexAttrConstant(rewriter, loc, elemTy, op.start());
    auto idxs = emitIndices(loc, rewriter, layout, shape);
    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }
    SmallVector<Type> types(elems, elemTy);
    Type structTy = LLVM::LLVMStructType::getLiteral(getContext(), types);
    Value result = getStructFromElements(loc, retVals, rewriter, structTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GetProgramIdOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GetProgramIdOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(op.axis() < 3);

    Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(
        loc, rewriter.getIndexType(), dims[op.axis()]);
    auto llvmIndexTy = getTypeConverter()->getIndexType();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{llvmIndexTy}, ValueRange{blockId});
    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

struct GetNumProgramsOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(op.axis() < 3);

    Value blockId = rewriter.create<::mlir::gpu::GridDimOp>(
        loc, rewriter.getIndexType(), dims[op.axis()]);
    auto llvmIndexTy = getTypeConverter()->getIndexType();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{llvmIndexTy}, ValueRange{blockId});
    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

struct AddPtrOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AddPtrOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AddPtrOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto resultTensorTy = resultTy.dyn_cast<RankedTensorType>();
    if (resultTensorTy) {
      unsigned elems = getElemsPerThread(resultTy);
      Type elemTy =
          getTypeConverter()->convertType(resultTensorTy.getElementType());
      SmallVector<Type> types(elems, elemTy);
      Type structTy = LLVM::LLVMStructType::getLiteral(getContext(), types);
      auto ptrs = getElementsFromStruct(loc, adaptor.ptr(), rewriter);
      auto offsets = getElementsFromStruct(loc, adaptor.offset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(elemTy, ptrs[i], offsets[i]);
      }
      Value view = getStructFromElements(loc, resultVals, rewriter, structTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(resultTy.isa<triton::PointerType>());
      Type llResultTy = getTypeConverter()->convertType(resultTy);
      Value result = gep(llResultTy, adaptor.ptr(), adaptor.offset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct AllocTensorOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AllocTensorOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AllocTensorOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getResult());
    auto resultTy = op.getType().dyn_cast<RankedTensorType>();
    auto llvmElemTy =
        getTypeConverter()->convertType(resultTy.getElementType());
    auto elemPtrTy = ptr_ty(llvmElemTy, 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto order = resultTy.getEncoding().cast<SharedEncodingAttr>().getOrder();
    // Workaround for 3D tensors
    // TODO: we need to modify the pipeline pass to give a proper shared
    // encoding to 3D tensors
    SmallVector<unsigned> newOrder;
    if (resultTy.getShape().size() == 3)
      newOrder = {1 + order[0], 1 + order[1], 0};
    else
      newOrder = SmallVector<unsigned>(order.begin(), order.end());

    auto smemObj = SharedMemoryObject(smemBase, resultTy.getShape(), newOrder,
                                      loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct ExtractSliceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<tensor::ExtractSliceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      tensor::ExtractSliceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = extract_slice %src[%offsets]
    Location loc = op->getLoc();
    auto srcTy = op.source().getType().dyn_cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(srcLayout && "Unexpected resultLayout in ExtractSliceOpConversion");
    assert(op.hasUnitStride() &&
           "Only unit stride supported by ExtractSliceOpConversion");

    // newBase = base + offset
    // Triton supports either static and dynamic offsets
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.source(), rewriter);
    SmallVector<Value, 4> opOffsetVals;
    SmallVector<Value, 4> offsetVals;
    auto mixedOffsets = op.getMixedOffsets();
    for (auto i = 0; i < mixedOffsets.size(); ++i) {
      if (op.isDynamicOffset(i))
        opOffsetVals.emplace_back(adaptor.offsets()[i]);
      else
        opOffsetVals.emplace_back(i32_val(op.getStaticOffset(i)));
      offsetVals.emplace_back(add(smemObj.offsets[i], opOffsetVals[i]));
    }
    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, opOffsetVals, smemObj.strides);
    // newShape = rank_reduce(shape)
    // Triton only supports static tensor sizes
    SmallVector<Value, 4> strideVals;
    for (auto i = 0; i < op.static_sizes().size(); ++i) {
      if (op.getStaticSize(i) == 1) {
        offsetVals.erase(offsetVals.begin() + i);
      } else {
        strideVals.emplace_back(smemObj.strides[i]);
      }
    }

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto elemPtrTy = ptr_ty(llvmElemTy, 3);
    auto resTy = op.getType().dyn_cast<RankedTensorType>();
    smemObj = SharedMemoryObject(gep(elemPtrTy, smemObj.base, offset),
                                 strideVals, offsetVals);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct FpToFpOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::FpToFpOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::FpToFpOp>::ConvertTritonGPUOpToLLVMPattern;

  static SmallVector<Value>
  convertFp8x4ToFp16x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto ctx = rewriter.getContext();
    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    Value fp8x4Vec = undef(fp8x4VecTy);
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v0, i32_val(0));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v1, i32_val(1));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v2, i32_val(2));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v3, i32_val(3));
    fp8x4Vec = bitcast(fp8x4Vec, i32_ty);

    PTXBuilder builder;
    auto *ptxAsm = "{                                      \n"
                   ".reg .b32 a<2>, b<2>;                  \n"
                   "prmt.b32 a0, 0, $2, 0x5040;            \n"
                   "prmt.b32 a1, 0, $2, 0x7060;            \n"
                   "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n"
                   "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n"
                   "shr.b32  b0, b0, 1;                    \n"
                   "shr.b32  b1, b1, 1;                    \n"
                   "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n"
                   "lop3.b32 $1, b1, 0x80008000, a1, 0xf8; \n"
                   "}";
    auto &call = *builder.create(ptxAsm);

    auto *o0 = builder.newOperand("=r");
    auto *o1 = builder.newOperand("=r");
    auto *i = builder.newOperand(fp8x4Vec, "r");
    call({o0, o1, i}, /*onlyAttachMLIRArgs=*/true);

    auto fp16x2VecTy = vec_ty(f16_ty, 2);
    auto fp16x2x2StructTy =
        struct_ty(SmallVector<Type>{fp16x2VecTy, fp16x2VecTy});
    auto fp16x2x2Struct =
        builder.launch(rewriter, loc, fp16x2x2StructTy, false);
    auto fp16x2Vec0 =
        extract_val(fp16x2VecTy, fp16x2x2Struct, rewriter.getI32ArrayAttr({0}));
    auto fp16x2Vec1 =
        extract_val(fp16x2VecTy, fp16x2x2Struct, rewriter.getI32ArrayAttr({1}));
    return {extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
            extract_element(f16_ty, fp16x2Vec0, i32_val(1)),
            extract_element(f16_ty, fp16x2Vec1, i32_val(0)),
            extract_element(f16_ty, fp16x2Vec1, i32_val(1))};
  }

  static SmallVector<Value>
  convertFp16x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto ctx = rewriter.getContext();
    auto fp16x2VecTy = vec_ty(f16_ty, 2);
    Value fp16x2Vec0 = undef(fp16x2VecTy);
    Value fp16x2Vec1 = undef(fp16x2VecTy);
    fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v0, i32_val(0));
    fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v1, i32_val(1));
    fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v2, i32_val(0));
    fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v3, i32_val(1));
    fp16x2Vec0 = bitcast(fp16x2Vec0, i32_ty);
    fp16x2Vec1 = bitcast(fp16x2Vec1, i32_ty);

    PTXBuilder builder;
    auto *ptxAsm = "{                                      \n"
                   ".reg .b32 a<2>, b<2>;                  \n"
                   "shl.b32 a0, $1, 1;                     \n"
                   "shl.b32 a1, $2, 1;                     \n"
                   "lop3.b32 a0, a0, 0x7fff7fff, 0, 0xc0;  \n"
                   "lop3.b32 a1, a1, 0x7fff7fff, 0, 0xc0;  \n"
                   "add.u32 a0, a0, 0x00800080;            \n"
                   "add.u32 a1, a1, 0x00800080;            \n"
                   "lop3.b32 b0, $1, 0x80008000, a0, 0xea; \n"
                   "lop3.b32 b1, $2, 0x80008000, a1, 0xea; \n"
                   "prmt.b32 $0, b0, b1, 0x7531;           \n"
                   "}";
    auto &call = *builder.create(ptxAsm);

    auto *o = builder.newOperand("=r");
    auto *i0 = builder.newOperand(fp16x2Vec0, "r");
    auto *i1 = builder.newOperand(fp16x2Vec1, "r");
    call({o, i0, i1}, /*onlyAttachMLIRArgs=*/true);

    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    auto fp8x4Vec = builder.launch(rewriter, loc, fp8x4VecTy, false);
    return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
            extract_element(i8_ty, fp8x4Vec, i32_val(1)),
            extract_element(i8_ty, fp8x4Vec, i32_val(2)),
            extract_element(i8_ty, fp8x4Vec, i32_val(3))};
  }

  static SmallVector<Value>
  convertFp8x4ToBf16x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto ctx = rewriter.getContext();
    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    Value fp8x4Vec = undef(fp8x4VecTy);
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v0, i32_val(0));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v1, i32_val(1));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v2, i32_val(2));
    fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v3, i32_val(3));
    fp8x4Vec = bitcast(fp8x4Vec, i32_ty);

    PTXBuilder builder;
    auto *ptxAsm = "{                                          \n"
                   ".reg .b32 a<2>, sign<2>, nosign<2>, b<2>;  \n"
                   "prmt.b32 a0, 0, $2, 0x5040;                \n"
                   "prmt.b32 a1, 0, $2, 0x7060;                \n"
                   "and.b32 sign0, a0, 0x80008000;             \n"
                   "and.b32 sign1, a1, 0x80008000;             \n"
                   "and.b32 nosign0, a0, 0x7fff7fff;           \n"
                   "and.b32 nosign1, a1, 0x7fff7fff;           \n"
                   "shr.b32 nosign0, nosign0, 4;               \n"
                   "shr.b32 nosign1, nosign1, 4;               \n"
                   "add.u32 nosign0, nosign0, 0x38003800;      \n"
                   "add.u32 nosign1, nosign1, 0x38003800;      \n"
                   "or.b32 $0, sign0, nosign0;                 \n"
                   "or.b32 $1, sign1, nosign1;                 \n"
                   "}";
    auto &call = *builder.create(ptxAsm);

    auto *o0 = builder.newOperand("=r");
    auto *o1 = builder.newOperand("=r");
    auto *i = builder.newOperand(fp8x4Vec, "r");
    call({o0, o1, i}, /* onlyAttachMLIRArgs */ true);

    auto bf16x2VecTy = vec_ty(bf16_ty, 2);
    auto bf16x2x2StructTy =
        struct_ty(SmallVector<Type>{bf16x2VecTy, bf16x2VecTy});
    auto bf16x2x2Struct =
        builder.launch(rewriter, loc, bf16x2x2StructTy, false);
    auto bf16x2Vec0 =
        extract_val(bf16x2VecTy, bf16x2x2Struct, rewriter.getI32ArrayAttr({0}));
    auto bf16x2Vec1 =
        extract_val(bf16x2VecTy, bf16x2x2Struct, rewriter.getI32ArrayAttr({1}));
    return {extract_element(bf16_ty, bf16x2Vec0, i32_val(0)),
            extract_element(bf16_ty, bf16x2Vec0, i32_val(1)),
            extract_element(bf16_ty, bf16x2Vec1, i32_val(0)),
            extract_element(bf16_ty, bf16x2Vec1, i32_val(1))};
  }

  static SmallVector<Value>
  convertBf16x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    auto ctx = rewriter.getContext();
    auto bf16x2VecTy = vec_ty(bf16_ty, 2);
    Value bf16x2Vec0 = undef(bf16x2VecTy);
    Value bf16x2Vec1 = undef(bf16x2VecTy);
    bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v0, i32_val(0));
    bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v1, i32_val(1));
    bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v2, i32_val(0));
    bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v3, i32_val(1));
    bf16x2Vec0 = bitcast(bf16x2Vec0, i32_ty);
    bf16x2Vec1 = bitcast(bf16x2Vec1, i32_ty);

    PTXBuilder builder;
    auto *ptxAsm = "{                                            \n"
                   ".reg .u32 sign, sign<2>, nosign, nosign<2>;  \n"
                   ".reg .u32 fp8_min, fp8_max, rn_, zero;       \n"
                   "mov.u32 fp8_min, 0x38003800;                 \n"
                   "mov.u32 fp8_max, 0x3ff03ff0;                 \n"
                   "mov.u32 rn_, 0x80008;                        \n"
                   "mov.u32 zero, 0;                             \n"
                   "and.b32 sign0, $1, 0x80008000;               \n"
                   "and.b32 sign1, $2, 0x80008000;               \n"
                   "prmt.b32 sign, sign0, sign1, 0x7531;         \n"
                   "and.b32 nosign0, $1, 0x7fff7fff;             \n"
                   "and.b32 nosign1, $2, 0x7fff7fff;             \n"
                   ".reg .u32 nosign_0_<2>, nosign_1_<2>;        \n"
                   "and.b32 nosign_0_0, nosign0, 0xffff0000;     \n"
                   "max.u32 nosign_0_0, nosign_0_0, 0x38000000;  \n"
                   "min.u32 nosign_0_0, nosign_0_0, 0x3ff00000;  \n"
                   "and.b32 nosign_0_1, nosign0, 0x0000ffff;     \n"
                   "max.u32 nosign_0_1, nosign_0_1, 0x3800;      \n"
                   "min.u32 nosign_0_1, nosign_0_1, 0x3ff0;      \n"
                   "or.b32 nosign0, nosign_0_0, nosign_0_1;      \n"
                   "and.b32 nosign_1_0, nosign1, 0xffff0000;     \n"
                   "max.u32 nosign_1_0, nosign_1_0, 0x38000000;  \n"
                   "min.u32 nosign_1_0, nosign_1_0, 0x3ff00000;  \n"
                   "and.b32 nosign_1_1, nosign1, 0x0000ffff;     \n"
                   "max.u32 nosign_1_1, nosign_1_1, 0x3800;      \n"
                   "min.u32 nosign_1_1, nosign_1_1, 0x3ff0;      \n"
                   "or.b32 nosign1, nosign_1_0, nosign_1_1;      \n"
                   "add.u32 nosign0, nosign0, rn_;               \n"
                   "add.u32 nosign1, nosign1, rn_;               \n"
                   "sub.u32 nosign0, nosign0, 0x38003800;        \n"
                   "sub.u32 nosign1, nosign1, 0x38003800;        \n"
                   "shr.u32 nosign0, nosign0, 4;                 \n"
                   "shr.u32 nosign1, nosign1, 4;                 \n"
                   "prmt.b32 nosign, nosign0, nosign1, 0x6420;   \n"
                   "or.b32 $0, nosign, sign;                     \n"
                   "}";
    auto &call = *builder.create(ptxAsm);

    auto *o = builder.newOperand("=r");
    auto *i0 = builder.newOperand(bf16x2Vec0, "r");
    auto *i1 = builder.newOperand(bf16x2Vec1, "r");
    call({o, i0, i1}, /*onlyAttachMLIRArgs=*/true);

    auto fp8x4VecTy = vec_ty(i8_ty, 4);
    auto fp8x4Vec = builder.launch(rewriter, loc, fp8x4VecTy, false);
    return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
            extract_element(i8_ty, fp8x4Vec, i32_val(1)),
            extract_element(i8_ty, fp8x4Vec, i32_val(2)),
            extract_element(i8_ty, fp8x4Vec, i32_val(3))};
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

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTensorType = op.from().getType().cast<mlir::RankedTensorType>();
    auto dstTensorType = op.result().getType().cast<mlir::RankedTensorType>();
    auto srcEltType = srcTensorType.getElementType();
    auto dstEltType = dstTensorType.getElementType();
    assert(srcEltType.isa<triton::Float8Type>() ||
           dstEltType.isa<triton::Float8Type>());
    auto convertedDstTensorType =
        this->getTypeConverter()->convertType(dstTensorType);
    auto convertedDstEleType =
        this->getTypeConverter()->convertType(dstEltType);

    // Select convertor
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
      assert(false && "unsupported type casting");
    }

    // Vectorized casting
    auto loc = op->getLoc();
    auto elems = getElemsPerThread(dstTensorType);
    assert(elems % 4 == 0 &&
           "FP8 casting only support tensors with 4-aligned sizes");
    auto elements = getElementsFromStruct(loc, adaptor.from(), rewriter);
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < elems; i += 4) {
      auto converted = convertor(loc, rewriter, elements[i], elements[i + 1],
                                 elements[i + 2], elements[i + 3]);
      resultVals.append(converted);
    }
    assert(resultVals.size() == elems);
    auto result = getStructFromElements(loc, resultVals, rewriter,
                                        convertedDstTensorType);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    auto &asyncWaitOp = *ptxBuilder.create<>("cp.async.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);

    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

void populateTritonGPUToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns, int numWarps,
                                     AxisInfoAnalysis &axisInfoAnalysis,
                                     const Allocation *allocation, Value smem,
                                     PatternBenefit benefit) {
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<AllocTensorOpConversion>(typeConverter, allocation, smem,
                                        benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, benefit);
  patterns.add<FpToFpOpConversion>(typeConverter, benefit);
  patterns.add<BroadcastOpConversion>(typeConverter, benefit);

  patterns.add<ExtractSliceOpConversion>(typeConverter, allocation, smem,
                                         benefit);
  patterns.add<GetProgramIdOpConversion>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<MakeRangeOpConversion>(typeConverter, benefit);
  patterns.add<ReturnOpConversion>(typeConverter, benefit);
  patterns.add<PrintfOpConversion>(typeConverter, benefit);
}
