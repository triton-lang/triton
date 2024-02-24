// MMA encoding has a different order depending on the element's bit width;
// reorder if we're in this case.
static SmallVector<Value> reorderValues(const SmallVector<Value> &values,
                                        Type inType, Type ouType) {
  auto inTensorTy = inType.dyn_cast<RankedTensorType>();
  auto ouTensorTy = ouType.dyn_cast<RankedTensorType>();
  if (!inTensorTy || !ouTensorTy)
    return values;
  auto inEncoding = dyn_cast<DotOperandEncodingAttr>(inTensorTy.getEncoding());
  auto ouEncoding = dyn_cast<DotOperandEncodingAttr>(ouTensorTy.getEncoding());
  assert(inEncoding == ouEncoding);
  if (!inEncoding)
    return values;
  // If the parent of the dot operand is in block encoding, we don't need to
  // reorder elements
  auto parentEncoding = dyn_cast<NvidiaMmaEncodingAttr>(ouEncoding.getParent());
  if (!parentEncoding)
    return values;
  size_t inBitWidth = inTensorTy.getElementType().getIntOrFloatBitWidth();
  size_t ouBitWidth = ouTensorTy.getElementType().getIntOrFloatBitWidth();
  auto ouEltTy = ouTensorTy.getElementType();
  if (inBitWidth == ouBitWidth)
    return values;
  if (inBitWidth == 16 && ouBitWidth == 32) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 8) {
      ret.push_back(values[i]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
    }
    return ret;
  }
  if (inBitWidth == 8 && ouBitWidth == 16) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 16) {
      ret.push_back(values[i + 0]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 8]);
      ret.push_back(values[i + 9]);
      ret.push_back(values[i + 10]);
      ret.push_back(values[i + 11]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
      ret.push_back(values[i + 12]);
      ret.push_back(values[i + 13]);
      ret.push_back(values[i + 14]);
      ret.push_back(values[i + 15]);
    }
    return ret;
  }
  llvm_unreachable("unimplemented code path");
}

inline SmallVector<Value> unpackI32(const SmallVector<Value> &inValues,
                                    Type srcTy,
                                    ConversionPatternRewriter &rewriter,
                                    Location loc,
                                    const LLVMTypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<NvidiaMmaEncodingAttr>()))
    return inValues;
  SmallVector<Value> outValues;
  for (auto v : inValues) {
    // cast i32 to appropriate eltType vector and extract elements
    auto eltType = typeConverter->convertType(tensorTy.getElementType());
    auto vecType = vec_ty(eltType, 32 / eltType.getIntOrFloatBitWidth());
    auto vec = bitcast(v, vecType);
    for (int i = 0; i < 32 / eltType.getIntOrFloatBitWidth(); i++) {
      outValues.push_back(extract_element(vec, i32_val(i)));
    }
  }
  return outValues;
}

inline SmallVector<Value> packI32(const SmallVector<Value> &inValues,
                                  Type srcTy,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc,
                                  const LLVMTypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<NvidiaMmaEncodingAttr>()))
    return inValues;
  SmallVector<Value> outValues;
  auto eltType = typeConverter->convertType(tensorTy.getElementType());
  int vecWidth = 32 / eltType.getIntOrFloatBitWidth();
  auto vecType = vec_ty(eltType, vecWidth);
  for (int i = 0; i < inValues.size(); i += vecWidth) {
    Value vec = undef(vecType);
    for (int j = 0; j < vecWidth; j++) {
      vec = insert_element(vec, inValues[i + j], i32_val(j));
    }
    outValues.push_back(bitcast(vec, i32_ty));
  }
  return outValues;
}

template <typename SourceOp, typename ConcreteT>
SmallVector<Value>
ElementwiseOpConversionBase<SourceOp, ConcreteT>::maybeDeduplicate(
    SourceOp op, SmallVector<Value> resultVals) const {
  if (!isMemoryEffectFree(op))
    // the op has side effects: can't dedup
    return resultVals;
  SmallVector<Value> results = op->getResults();
  if (results.size() == 0 || results.size() > 1)
    // there must be exactly 1 result
    return resultVals;
  Value result = results[0];
  Type type = result.getType();
  if (!type)
    return resultVals;
  RankedTensorType rtType = type.dyn_cast<RankedTensorType>();
  if (!rtType)
    // the result must be a tensor
    return resultVals;
  Attribute encoding = rtType.getEncoding();
  if (!encoding)
    // encoding not available
    return resultVals;
  if (!encoding.dyn_cast<BlockedEncodingAttr>() &&
      !encoding.dyn_cast<SliceEncodingAttr>()) {
    // TODO: constraining the ecndoing type here is necessary for avoiding
    // crashes in the getElemsPerThread call below happening in the
    // test_core::test_fp8_dot_acc
    return resultVals;
  }

  SmallVector<unsigned> elemsPerThread = getElemsPerThread(rtType);
  int rank = elemsPerThread.size();
  if (product<unsigned>(elemsPerThread) != resultVals.size())
    return resultVals;
  AxisInfo *axisInfo = axisAnalysisPass.getAxisInfo(result);
  if (!axisInfo)
    // axis info (e.g., constancy) not available
    return resultVals;
  SmallVector<unsigned> sizePerThread = getSizePerThread(encoding);
  if (rank != sizePerThread.size())
    return resultVals;

  SmallVector<int64_t> constancy = axisInfo->getConstancy();
  if (rank != constancy.size())
    return resultVals;
  bool hasConstancy = false;
  for (int i = 0; i < rank; ++i) {
    if (constancy[i] > sizePerThread[i]) {
      if (constancy[i] % sizePerThread[i] != 0)
        // constancy is not evenly covered by sizePerThread
        return resultVals;
      // can't move the values across different
      // "sizePerThread"-sized blocks
      constancy[i] = sizePerThread[i];
    }
    if (elemsPerThread[i] < 1 || constancy[i] < 1)
      return resultVals;
    if (!(elemsPerThread[i] % constancy[i] == 0 ||
          constancy[i] % elemsPerThread[i] == 0))
      // either the constancy along each dimension must fit
      // into the elemsPerThread or the other way around
      return resultVals;
    if (constancy[i] > 1)
      hasConstancy = true;
  }
  if (!hasConstancy)
    // nothing to deduplicate
    return resultVals;

  if (rank > 1) {
    // reorder the shape and constancy vectors by the axis order:
    // from the fastest-changing to the smallest-changing axis
    SmallVector<unsigned> order = getOrder(encoding);
    if (rank != order.size())
      return resultVals;
    elemsPerThread = applyPermutation(elemsPerThread, order);
    constancy = applyPermutation(constancy, order);
  }

  SmallVector<unsigned> strides(rank, 1);
  for (int i = 1; i < rank; ++i) {
    strides[i] = strides[i - 1] * elemsPerThread[i - 1];
  }
  SmallVector<Value> dedupResultVals;
  dedupResultVals.reserve(resultVals.size());
  for (int i = 0; i < resultVals.size(); ++i) {
    // each coordinate of the orig_idx is "coarsened" using the
    // constancy along this dimension: the resulting dedup_idx
    // points to the reused value in the original resultsVal
    int orig_idx = i;
    int dedup_idx = 0;
    for (int j = 0; j < rank; ++j) {
      int coord_j = orig_idx % elemsPerThread[j];
      dedup_idx += (coord_j / constancy[j] * constancy[j]) * strides[j];
      orig_idx /= elemsPerThread[j];
    }
    dedupResultVals.push_back(resultVals[dedup_idx]);
  }

  return dedupResultVals;
}

template <typename SourceOp, typename ConcreteT>
LogicalResult ElementwiseOpConversionBase<SourceOp, ConcreteT>::matchAndRewrite(
    SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto resultTy = op.getType();
  Location loc = op->getLoc();
  // element type
  auto resultElementTy = getElementTypeOrSelf(resultTy);
  Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
  SmallVector<SmallVector<Value>> allOperands;
  for (auto operand : adaptor.getOperands()) {
    auto argTy = op->getOperand(0).getType();
    auto subOperands = unpackLLElements(loc, operand, rewriter);
    subOperands =
        unpackI32(subOperands, argTy, rewriter, loc, this->getTypeConverter());
    allOperands.resize(subOperands.size());
    for (auto v : llvm::enumerate(subOperands))
      allOperands[v.index()].push_back(v.value());
  }
  if (allOperands.size() == 0)
    allOperands.push_back({});

  SmallVector<Value> resultVals;
  for (auto it = allOperands.begin(), end = allOperands.end(); it != end;) {
    auto curr = static_cast<const ConcreteT *>(this)->createDestOps(
        op, adaptor, rewriter, elemTy, MultipleOperandsRange(it, end), loc);
    if (curr.size() == 0)
      return failure();
    for (auto v : curr) {
      if (!static_cast<bool>(v))
        return failure();
      resultVals.push_back(v);
    }
    it += curr.size();
  }
  if (op->getNumOperands() > 0) {
    auto argTy = op->getOperand(0).getType();
    resultVals = reorderValues(resultVals, argTy, resultTy);
  }
  resultVals = maybeDeduplicate(op, resultVals);
  resultVals =
      packI32(resultVals, resultTy, rewriter, loc, this->getTypeConverter());
  Value view = packLLElements(loc, this->getTypeConverter(), resultVals,
                              rewriter, resultTy);
  rewriter.replaceOp(op, view);

  return success();
}
