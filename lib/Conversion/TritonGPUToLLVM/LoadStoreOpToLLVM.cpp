#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "LoadStoreOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getElementsFromStruct;
using ::mlir::LLVM::getStructFromElements;
using ::mlir::triton::gpu::getElemsPerThread;

// Get corresponding LLVM element values of \param value.
SmallVector<Value> LoadStoreConversionBase::getLLVMElems(
    Value value, Value llValue,
    ConversionPatternRewriter &rewriter, Location loc) {
  if (!value)
    return {};
  if (!llValue.getType().isa<LLVM::LLVMStructType>())
    return {llValue};
  // Here, we assume that all inputs should have a blockedLayout
  auto valueVals = getElementsFromStruct(loc, llValue, rewriter);
  return valueVals;
}

LogicalResult LoadOpConversion::matchAndRewrite(
    triton::LoadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op->getLoc();

  // original values
  Value ptr = op.ptr();
  Value mask = op.mask();
  Value other = op.other();

  // adaptor values
  Value llPtr = adaptor.ptr();
  Value llMask = adaptor.mask();
  Value llOther = adaptor.other();

  // Determine the vectorization size
  Type valueTy = op.getResult().getType();
  Type valueElemTy =
      typeConverter->convertType(getElementTypeOrSelf(valueTy));
  unsigned vec = getVectorSize(ptr);
  unsigned numElems = getElemsPerThread(ptr.getType());
  if (llMask)
    vec = std::min<size_t>(vec, getMaskAlignment(mask));

  // Get the LLVM values for pointers
  auto ptrElems = getLLVMElems(ptr, llPtr, rewriter, loc);
  assert(ptrElems.size() == numElems);

  // Get the LLVM values for mask
  SmallVector<Value> maskElems;
  if (llMask) {
    maskElems = getLLVMElems(mask, llMask, rewriter, loc);
    assert(maskElems.size() == numElems);
  }

  // Get the LLVM values for `other`
  // TODO: (goostavz) handle when other is const but not splat, which
  //       should be rarely seen
  bool otherIsSplatConstInt = false;
  DenseElementsAttr constAttr;
  int64_t splatVal = 0;
  if (other && valueElemTy.isa<IntegerType>() &&
      matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat()) {
    otherIsSplatConstInt = true;
    splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
  }
  auto otherElems = getLLVMElems(other, llOther, rewriter, loc);

  // vectorized iteration through all the pointer/mask/other elements
  const int valueElemNbits =
      std::max(8u, valueElemTy.getIntOrFloatBitWidth());
  const int numVecs = numElems / vec;

  SmallVector<Value> loadedVals;
  for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
    // TODO: optimization when ptr is GEP with constant offset
    size_t in_off = 0;

    const size_t maxWordWidth = std::max<size_t>(32, valueElemNbits);
    const size_t totalWidth = valueElemNbits * vec;
    const size_t width = std::min(totalWidth, maxWordWidth);
    const size_t nWords = std::max<size_t>(1, totalWidth / width);
    const size_t wordNElems = width / valueElemNbits;
    assert(wordNElems * nWords * numVecs == numElems);

    // TODO(Superjomn) Add cache policy fields to StoreOp.
    // TODO(Superjomn) Deal with cache policy here.
    const bool hasL2EvictPolicy = false;

    PTXBuilder ptxBuilder;

    Value pred = mask ? maskElems[vecStart] : int_val(1, 1);

    const std::string readConstraint =
        (width == 64) ? "l" : ((width == 32) ? "r" : "c");
    const std::string writeConstraint =
        (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");

    // prepare asm operands
    auto *dstsOpr = ptxBuilder.newListOperand();
    for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
      auto *opr = ptxBuilder.newOperand(writeConstraint); // =r operations
      dstsOpr->listAppend(opr);
    }

    auto *addrOpr =
        ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

    // Define the instruction opcode
    auto &ld = ptxBuilder.create<>("ld")
                   ->o("volatile", op.isVolatile())
                   .global()
                   .o("ca", op.cache() == triton::CacheModifier::CA)
                   .o("cg", op.cache() == triton::CacheModifier::CG)
                   .o("L1::evict_first",
                      op.evict() == triton::EvictionPolicy::EVICT_FIRST)
                   .o("L1::evict_last",
                      op.evict() == triton::EvictionPolicy::EVICT_LAST)
                   .o("L1::cache_hint", hasL2EvictPolicy)
                   .v(nWords)
                   .b(width);

    PTXBuilder::Operand *evictOpr{};

    // Here lack a mlir::Value to bind to this operation, so disabled.
    // if (has_l2_evict_policy)
    //   evictOpr = ptxBuilder.newOperand(l2Evict, "l");

    if (!evictOpr)
      ld(dstsOpr, addrOpr).predicate(pred, "b");
    else
      ld(dstsOpr, addrOpr, evictOpr).predicate(pred, "b");

    if (other) {
      for (size_t ii = 0; ii < nWords; ++ii) {
        // PTX doesn't support mov.u8, so we need to use mov.u16
        auto movWidth = width < 16 ? 16 : width;
        PTXInstr &mov =
            ptxBuilder.create<>("mov")->o("u" + std::to_string(movWidth));

        size_t size = width / valueElemNbits;

        auto vecTy = LLVM::getFixedVectorType(valueElemTy, size);
        Value v = undef(vecTy);
        for (size_t s = 0; s < size; ++s) {
          Value falseVal = otherElems[vecStart + ii * size + s];
          Value sVal = createIndexAttrConstant(
              rewriter, loc, this->getTypeConverter()->getIndexType(), s);
          v = insert_element(vecTy, v, falseVal, sVal);
        }
        v = bitcast(v, IntegerType::get(getContext(), width));

        PTXInstr::Operand *opr{};
        if (otherIsSplatConstInt)
          opr = ptxBuilder.newConstantOperand(splatVal);
        else
          opr = ptxBuilder.newOperand(v, readConstraint);

        mov(dstsOpr->listGet(ii), opr).predicateNot(pred, "b");
      }
    }

    // Create inline ASM signature
    SmallVector<Type> retTys(nWords, IntegerType::get(getContext(), width));
    Type retTy = retTys.size() > 1
                     ? LLVM::LLVMStructType::getLiteral(getContext(), retTys)
                     : retTys[0];

    // TODO: if (has_l2_evict_policy)
    // auto asmDialectAttr =
    // LLVM::AsmDialectAttr::get(rewriter.getContext(),
    //                                                 LLVM::AsmDialect::AD_ATT);
    Value ret = ptxBuilder.launch(rewriter, loc, retTy);

    // Extract and store return values
    SmallVector<Value> rets;
    for (unsigned int ii = 0; ii < nWords; ++ii) {
      Value curr;
      if (retTy.isa<LLVM::LLVMStructType>()) {
        curr = extract_val(IntegerType::get(getContext(), width), ret,
                           rewriter.getI64ArrayAttr(ii));
      } else {
        curr = ret;
      }
      curr = bitcast(curr, LLVM::getFixedVectorType(valueElemTy,
                                                    width / valueElemNbits));
      rets.push_back(curr);
    }
    int tmp = width / valueElemNbits;
    for (size_t ii = 0; ii < vec; ++ii) {
      Value vecIdx = createIndexAttrConstant(
          rewriter, loc, this->getTypeConverter()->getIndexType(), ii % tmp);
      Value loaded = extract_element(valueElemTy, rets[ii / tmp], vecIdx);
      loadedVals.push_back(loaded);
    }
  } // end vec

  Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
  Value resultStruct =
      getStructFromElements(loc, loadedVals, rewriter, llvmResultStructTy);
  rewriter.replaceOp(op, {resultStruct});
  return success();
}

LogicalResult StoreOpConversion::matchAndRewrite(
    triton::StoreOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value ptr = op.ptr();
  Value mask = op.mask();
  Value value = op.value();

  Value llPtr = adaptor.ptr();
  Value llMask = adaptor.mask();
  Value llValue = adaptor.value();

  auto loc = op->getLoc();
  MLIRContext *ctx = rewriter.getContext();

  auto valueTy = value.getType();
  Type valueElemTy =
      typeConverter->convertType(getElementTypeOrSelf(valueTy));

  unsigned vec = getVectorSize(ptr);
  unsigned numElems = getElemsPerThread(ptr.getType());

  auto ptrElems = getLLVMElems(ptr, llPtr, rewriter, loc);
  auto valueElems = getLLVMElems(value, llValue, rewriter, loc);
  assert(ptrElems.size() == valueElems.size());

  // Determine the vectorization size
  SmallVector<Value> maskElems;
  if (llMask) {
    maskElems = getLLVMElems(mask, llMask, rewriter, loc);
    assert(valueElems.size() == maskElems.size());

    unsigned maskAlign = getMaskAlignment(mask);
    vec = std::min(vec, maskAlign);
  }

  const size_t dtsize =
      std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
  const size_t valueElemNbits = dtsize * 8;

  const int numVecs = numElems / vec;
  for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
    // TODO: optimization when ptr is AddPtr with constant offset
    size_t in_off = 0;

    const size_t maxWordWidth = std::max<size_t>(32, valueElemNbits);
    const size_t totalWidth = valueElemNbits * vec;
    const size_t width = std::min(totalWidth, maxWordWidth);
    const size_t nWords = std::max<size_t>(1, totalWidth / width);
    const size_t wordNElems = width / valueElemNbits;
    assert(wordNElems * nWords * numVecs == numElems);

    // TODO(Superjomn) Add cache policy fields to StoreOp.
    // TODO(Superjomn) Deal with cache policy here.

    Type valArgTy = IntegerType::get(ctx, width);
    auto wordTy = vec_ty(valueElemTy, wordNElems);

    SmallVector<std::pair<Value, std::string>> asmArgs;
    for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
      // llWord is a width-len composition
      Value llWord = undef(wordTy);
      // Insert each value element to the composition
      for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
        const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
        assert(elemOffset < valueElems.size());
        Value elem = valueElems[elemOffset];
        if (elem.getType().isInteger(1))
          elem = rewriter.create<LLVM::SExtOp>(loc, type::i8Ty(ctx), elem);
        elem = bitcast(elem, valueElemTy);

        Type u32Ty = typeConverter->convertType(type::u32Ty(ctx));
        llWord = insert_element(wordTy, llWord, elem, i32_val(elemIdx));
      }
      llWord = bitcast(llWord, valArgTy);
      std::string constraint =
          (width == 64) ? "l" : ((width == 32) ? "r" : "c");
      asmArgs.emplace_back(llWord, constraint);
    }

    // Prepare the PTX inline asm.
    PTXBuilder ptxBuilder;
    auto *asmArgList = ptxBuilder.newListOperand(asmArgs);

    Value maskVal = llMask ? maskElems[vecStart] : int_val(1, 1);

    auto *asmAddr =
        ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

    auto &ptxStoreInstr =
        ptxBuilder.create<>("st")->global().v(nWords).b(width);
    ptxStoreInstr(asmAddr, asmArgList).predicate(maskVal, "b");

    Type boolTy = getTypeConverter()->convertType(rewriter.getIntegerType(1));
    llvm::SmallVector<Type> argTys({boolTy, ptr.getType()});
    argTys.insert(argTys.end(), nWords, valArgTy);

    auto asmReturnTy = void_ty(ctx);

    ptxBuilder.launch(rewriter, loc, asmReturnTy);
  }
  rewriter.eraseOp(op);
  return success();
}

LogicalResult AtomicCASOpConversion::matchAndRewrite(
    triton::AtomicCASOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  Value ptr = op.ptr();

  Value llPtr = adaptor.ptr();
  Value llCmp = adaptor.cmp();
  Value llVal = adaptor.val();

  auto ptrElements = getElementsFromStruct(loc, llPtr, rewriter);
  auto cmpElements = getElementsFromStruct(loc, llCmp, rewriter);
  auto valElements = getElementsFromStruct(loc, llVal, rewriter);

  auto valueTy = op.getResult().getType().dyn_cast<RankedTensorType>();
  Type valueElemTy =
      valueTy ? getTypeConverter()->convertType(valueTy.getElementType())
              : op.getResult().getType();
  auto tid = tid_val();
  Value pred = icmp_eq(tid, i32_val(0));
  PTXBuilder ptxBuilderMemfence;
  auto memfence = ptxBuilderMemfence.create<PTXInstr>("membar")->o("gl");
  memfence();
  auto ASMReturnTy = void_ty(ctx);
  ptxBuilderMemfence.launch(rewriter, loc, ASMReturnTy);

  Value atomPtr = getSharedMemoryBase(loc, rewriter, op.getOperation());
  atomPtr = bitcast(atomPtr, ptr_ty(valueElemTy, 3));

  Value casPtr = ptrElements[0];
  Value casCmp = cmpElements[0];
  Value casVal = valElements[0];

  PTXBuilder ptxBuilderAtomicCAS;
  auto *dstOpr = ptxBuilderAtomicCAS.newOperand("=r");
  auto *ptrOpr = ptxBuilderAtomicCAS.newAddrOperand(casPtr, "l");
  auto *cmpOpr = ptxBuilderAtomicCAS.newOperand(casCmp, "r");
  auto *valOpr = ptxBuilderAtomicCAS.newOperand(casVal, "r");
  auto &atom = *ptxBuilderAtomicCAS.create<PTXInstr>("atom");
  atom.global().o("cas").o("b32");
  atom(dstOpr, ptrOpr, cmpOpr, valOpr).predicate(pred);
  auto old = ptxBuilderAtomicCAS.launch(rewriter, loc, valueElemTy);
  barrier();

  PTXBuilder ptxBuilderStore;
  auto *dstOprStore = ptxBuilderStore.newAddrOperand(atomPtr, "l");
  auto *valOprStore = ptxBuilderStore.newOperand(old, "r");
  auto &st = *ptxBuilderStore.create<PTXInstr>("st");
  st.shared().o("b32");
  st(dstOprStore, valOprStore).predicate(pred);
  ptxBuilderStore.launch(rewriter, loc, ASMReturnTy);
  ptxBuilderMemfence.launch(rewriter, loc, ASMReturnTy);
  barrier();
  Value ret = load(atomPtr);
  barrier();
  rewriter.replaceOp(op, {ret});
  return success();
}

LogicalResult AtomicRMWOpConversion::matchAndRewrite(
    triton::AtomicRMWOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  auto atomicRmwAttr = op.atomic_rmw_op();
  Value ptr = op.ptr();
  Value val = op.val();

  Value llPtr = adaptor.ptr();
  Value llVal = adaptor.val();
  Value llMask = adaptor.mask();

  auto valElements = getElementsFromStruct(loc, llVal, rewriter);
  auto ptrElements = getElementsFromStruct(loc, llPtr, rewriter);
  auto maskElements = getElementsFromStruct(loc, llMask, rewriter);

  auto valueTy = op.getResult().getType().dyn_cast<RankedTensorType>();
  Type valueElemTy =
      valueTy ? getTypeConverter()->convertType(valueTy.getElementType())
              : op.getResult().getType();
  const size_t valueElemNbits = valueElemTy.getIntOrFloatBitWidth();
  auto elemsPerThread = getElemsPerThread(val.getType());
  // vec = 1 for scalar
  auto vec = getVectorSize(ptr);
  Value mask = int_val(1, 1);
  auto tid = tid_val();
  // tensor
  if (valueTy) {
    auto valTy = val.getType().cast<RankedTensorType>();
    vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
    // mask
    auto shape = valueTy.getShape();
    auto numElements = product(shape);
    mask = and_(mask, icmp_slt(mul(tid, i32_val(elemsPerThread)),
                               i32_val(numElements)));
  }

  auto vecTy = vec_ty(valueElemTy, vec);
  SmallVector<Value> resultVals(elemsPerThread);
  for (size_t i = 0; i < elemsPerThread; i += vec) {
    Value rmwVal = undef(vecTy);
    for (int ii = 0; ii < vec; ++ii) {
      Value iiVal = createIndexAttrConstant(
          rewriter, loc, getTypeConverter()->getIndexType(), ii);
      rmwVal = insert_element(vecTy, rmwVal, valElements[i + ii], iiVal);
    }

    Value rmwPtr = ptrElements[i];
    Value rmwMask = maskElements[i];
    rmwMask = and_(rmwMask, mask);
    std::string sTy;
    PTXBuilder ptxBuilderAtomicRMW;
    std::string tyId = valueElemNbits * vec == 64
                           ? "l"
                           : (valueElemNbits * vec == 32 ? "r" : "h");
    auto *dstOpr = ptxBuilderAtomicRMW.newOperand("=" + tyId);
    auto *ptrOpr = ptxBuilderAtomicRMW.newAddrOperand(rmwPtr, "l");
    auto *valOpr = ptxBuilderAtomicRMW.newOperand(rmwVal, tyId);

    auto &atom = ptxBuilderAtomicRMW.create<>("atom")->global().o("gpu");
    auto rmwOp = stringifyRMWOp(atomicRmwAttr).str();
    auto sBits = std::to_string(valueElemNbits);
    switch (atomicRmwAttr) {
    case RMWOp::AND:
      sTy = "b" + sBits;
      break;
    case RMWOp::OR:
      sTy = "b" + sBits;
      break;
    case RMWOp::XOR:
      sTy = "b" + sBits;
      break;
    case RMWOp::ADD:
      sTy = "s" + sBits;
      break;
    case RMWOp::FADD:
      rmwOp = "add";
      rmwOp += (valueElemNbits == 16 ? ".noftz" : "");
      sTy = "f" + sBits;
      sTy += (vec == 2 && valueElemNbits == 16) ? "x2" : "";
      break;
    case RMWOp::MAX:
      sTy = "s" + sBits;
      break;
    case RMWOp::MIN:
      sTy = "s" + sBits;
      break;
    case RMWOp::UMAX:
      rmwOp = "max";
      sTy = "u" + sBits;
      break;
    case RMWOp::UMIN:
      rmwOp = "min";
      sTy = "u" + sBits;
      break;
    case RMWOp::XCHG:
      sTy = "b" + sBits;
      break;
    default:
      return failure();
    }
    atom.o(rmwOp).o(sTy);
    if (valueTy) {
      atom(dstOpr, ptrOpr, valOpr).predicate(rmwMask);
      auto retType = vec == 1 ? valueElemTy : vecTy;
      auto ret = ptxBuilderAtomicRMW.launch(rewriter, loc, retType);
      for (int ii = 0; ii < vec; ++ii) {
        resultVals[i + ii] =
            vec == 1 ? ret : extract_element(valueElemTy, ret, idx_val(ii));
      }
    } else {
      PTXBuilder ptxBuilderMemfence;
      auto memfenc = ptxBuilderMemfence.create<PTXInstr>("membar")->o("gl");
      memfenc();
      auto ASMReturnTy = void_ty(ctx);
      ptxBuilderMemfence.launch(rewriter, loc, ASMReturnTy);
      rmwMask = and_(rmwMask, icmp_eq(tid, i32_val(0)));
      atom(dstOpr, ptrOpr, valOpr).predicate(rmwMask);
      auto old = ptxBuilderAtomicRMW.launch(rewriter, loc, valueElemTy);
      Value atomPtr = getSharedMemoryBase(loc, rewriter, op.getOperation());
      atomPtr = bitcast(atomPtr, ptr_ty(valueElemTy, 3));
      store(old, atomPtr);
      barrier();
      Value ret = load(atomPtr);
      barrier();
      rewriter.replaceOp(op, {ret});
    }
  }
  if (valueTy) {
    Type structTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct =
        getStructFromElements(loc, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, {resultStruct});
  }
  return success();
}

void populateLoadStoreOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, int numWarps,
    AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    PatternBenefit benefit) {
  patterns.add<LoadOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<StoreOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<AtomicCASOpConversion>(typeConverter, allocation, smem,
                                      axisInfoAnalysis, benefit);
  patterns.add<AtomicRMWOpConversion>(typeConverter, allocation, smem,
                                      axisInfoAnalysis, benefit);
}
