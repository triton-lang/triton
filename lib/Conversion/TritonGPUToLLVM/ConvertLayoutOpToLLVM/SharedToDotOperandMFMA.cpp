#ifdef USE_ROCM

#include "../ConvertLayoutOpToLLVM.h"
#include "../Utility.h"

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

Type getShemPtrTy(Type elemTy) {
  if (elemTy.isBF16()) {
    auto ctx = elemTy.getContext();
    return ptr_ty(type::i16Ty(ctx), 3);
  }
  return ptr_ty(elemTy, 3);
}

// Get a waveId for M axis.
Value getWaveM(ConversionPatternRewriter &rewriter, Location loc, Value wave,
               const ArrayRef<unsigned int> &wpt, int elemPerInstr, int M) {
  return urem(urem(wave, i32_val(wpt[0])), i32_val(M / elemPerInstr));
}
// Get a waveId for N axis.
Value getWaveN(ConversionPatternRewriter &rewriter, Location loc, Value wave,
               const ArrayRef<unsigned int> &wpt, int elemPerInstr, int N) {
  Value waveMN = udiv(wave, i32_val(wpt[0]));
  return urem(urem(waveMN, i32_val(wpt[1])), i32_val(N / elemPerInstr));
}

} // namespace

namespace SharedToDotOperandMFMA {

// Computes offsets for operand A or transposed operand B
// @param rewriter
// @param loc
// @param elemsPerThread operand tile shape consumed by one MFMA instruction
// @param waveM wave id for the "non K" axis
// @param laneId
// @param wptA
// @param numOfElems
// @param reps number of instructions repretition to fully cover dot operand
// @param cSwizzleOffset
llvm::SmallVector<Value>
computeOffsetsTy1(ConversionPatternRewriter &rewriter, Location loc,
                  const ArrayRef<int64_t> &elemsPerThread, Value waveM,
                  Value laneId, int wptA, int numOfElems,
                  ArrayRef<int64_t> reps, Value cSwizzleOffset) {
  auto numM = reps[0];
  auto numK = reps[1];
  SmallVector<Value> offsets(numM * numK * numOfElems);
  int lineSize = elemsPerThread[1] * numK;
  int blockSize = elemsPerThread[0] * numM * lineSize;
  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value waveHalf = udiv(laneId, _32);

  Value waveOffset = wptA > 1
                         ? mul(waveM, i32_val(elemsPerThread[0] * lineSize))
                         : mul(waveM, i32_val(blockSize));
  Value colOffset = select(icmp_uge(laneId, _32), i32_val(numOfElems), _0);

  for (int block = 0; block < numM; ++block) {
    Value blockOffset = wptA > 1
                            ? i32_val(block * blockSize)
                            : i32_val(block * elemsPerThread[0] * lineSize);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * elemsPerThread[1]);
      for (int elem = 0; elem < numOfElems; ++elem) {
        Value rowOffset =
            add(mul(urem(laneId, _32), i32_val(lineSize)), i32_val(elem));
        Value elemOffset = add(rowOffset, colOffset);
        Value offset =
            add(add(add(waveOffset, blockOffset), tileOffset), elemOffset);
        offsets[numK * numOfElems * block + numOfElems * tile + elem] = offset;
      }
    }
  }
  return offsets;
}

// Computes offsets for operand B or transposed operand A
// @param rewriter
// @param loc
// @param elemsPerThread operand tile shape consumed by one MFMA instruction
// @param warpsPerGroupN number of warps per horizontal axis
// @param waveM wave id for the "non K" axis
// @param laneId
// @param wptA
// @param numOfElems number of accessed elements per repetition
// @param reps number of instructions repretition to fully cover dot operand
// @param cSwizzleOffset
llvm::SmallVector<Value>
computeOffsetsTy2(ConversionPatternRewriter &rewriter, Location loc,
                  const ArrayRef<int64_t> &elemsPerThread, int warpsPerGroupN,
                  Value waveN, Value laneId, int wptB, int numOfElems,
                  ArrayRef<int64_t> reps, Value cSwizzleOffset) {
  auto numK = reps[0];
  auto numN = reps[1];
  SmallVector<Value> offsets(numK * numN * numOfElems);

  int lineSize = warpsPerGroupN * elemsPerThread[1] * numN;
  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value waveOffset = wptB > 1 ? mul(waveN, i32_val(elemsPerThread[1]))
                              : mul(waveN, i32_val(elemsPerThread[1] * numN));
  Value colOffset = urem(laneId, _32);

  for (int block = 0; block < numN; ++block) {
    Value blockOffset = wptB > 1 ? i32_val(block * elemsPerThread[1] * numN)
                                 : i32_val(block * elemsPerThread[1]);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * elemsPerThread[0] * lineSize);
      for (int elem = 0; elem < numOfElems; ++elem) {
        Value halfOffset =
            select(icmp_uge(laneId, _32), i32_val(numOfElems * lineSize), _0);
        Value rowOffset = add(i32_val(elem * lineSize), halfOffset);
        Value elemOffset = add(rowOffset, colOffset);
        Value offset =
            add(add(add(waveOffset, blockOffset), tileOffset), elemOffset);
        offsets[numK * numOfElems * block + numOfElems * tile + elem] = offset;
      }
    }
  }
  return offsets;
}

bool isTransposed(::llvm::ArrayRef<unsigned> order) {
  assert(order.size() == 2 && (order[0] & ~1ul) == 0 &&
         order[0] + order[1] == 1);
  return order[0] == 0;
}

Value loadA(ConversionPatternRewriter &rewriter, Location loc, Value thread,
            DotOperandEncodingAttr encoding,
            TritonGPUToLLVMTypeConverter *typeConverter, Value tensor,
            const SharedMemoryObject &smemObj) {
  auto mfmaLayout = encoding.getParent().cast<MfmaEncodingAttr>();
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto aElemTy = aTensorTy.getElementType();
  auto aElemsPerThread = encoding.getMFMAElemsPerThread(aElemTy);
  auto mfmaInstrM = aElemsPerThread[0];
  auto mfmaInstrK = aElemsPerThread[1];

  auto numReps = encoding.getMFMARep(shape, aElemTy);
  auto numRepM = numReps[0];
  auto numRepK = numReps[1];

  Value waveSize = i32_val(64);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveM =
      getWaveM(rewriter, loc, wave, warpsPerCTA, mfmaInstrM, shape[0]);
  int numOfElems = std::max<int>(mfmaInstrM * mfmaInstrK / 64 /*wave size*/, 1);
  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
  // TODO make macro tile size granilarity configurable
  int macroTileM =
      std::max<int>(shape[0] / (warpsPerCTA[0] * 32), 1);
  int wptM = std::min<int>(warpsPerCTA[0], macroTileM);
  int macroTileN =
      std::max<int>(shape[1] / (warpsPerCTA[1] * 32), 1);
  int wptN = std::min<int>(warpsPerCTA[1], macroTileN);
  int wpt = std::max<int>(wptM, wptN);

  SmallVector<Value> offsets;
  if (isTransposed(order)) {
    SmallVector<int64_t> elemsPerThread{aElemsPerThread[1], aElemsPerThread[0]};
    SmallVector<int64_t> reps{numReps[1], numReps[0]};
    unsigned int maxNumWarps = shape[0] / mfmaInstrM;
    int warpsPerGroupM = std::min(warpsPerCTA[0], maxNumWarps);
    offsets =
        computeOffsetsTy2(rewriter, loc, elemsPerThread, warpsPerGroupM, waveM,
                          lane, wpt, numOfElems, reps, cSwizzleOffset);
  } else {
    offsets = computeOffsetsTy1(rewriter, loc, aElemsPerThread, waveM, lane,
                                wpt, numOfElems, numReps, cSwizzleOffset);
  }

  Value smemBase = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);

  Type smemPtrTy = getShemPtrTy(aElemTy);

  SmallVector<Value> ha;
  for (int m = 0; m < numRepM; ++m) {
    for (int k = 0; k < numRepK; ++k) {
      auto vecTy = vec_ty(aElemTy, numOfElems);
      Value valVec = undef(vecTy);
      for (unsigned elem = 0; elem < numOfElems; ++elem) {
        Value elemOffset =
            offsets[m * numOfElems * numRepK + k * numOfElems + elem];
        Value elemValue = load(gep(smemPtrTy, smemBase, elemOffset));
        if (numOfElems > 1)
          valVec = insert_element(vecTy, valVec, elemValue, i32_val(elem));
        else
          valVec = elemValue;
      }
      if (aElemTy == i8_ty)
        valVec = bitcast(valVec, i32_ty);
      ha.push_back(valVec);
    }
  }

  MLIRContext *ctx = mfmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(ha.size(), ha[0].getType()));
  auto result = typeConverter->packLLElements(loc, ha, rewriter, structTy);
  return result;
}

Value loadB(ConversionPatternRewriter &rewriter, Location loc, Value thread,
            DotOperandEncodingAttr encoding,
            TritonGPUToLLVMTypeConverter *typeConverter, Value tensor,
            const SharedMemoryObject &smemObj) {
  auto mfmaLayout = encoding.getParent().cast<MfmaEncodingAttr>();
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto bTensorTy = tensor.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = bTensorTy.getShape();
  auto sharedLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto bElemTy = bTensorTy.getElementType();
  auto bElemsPerThread = encoding.getMFMAElemsPerThread(bElemTy);
  auto mfmaInstrK = bElemsPerThread[0];
  auto mfmaInstrN = bElemsPerThread[1];

  auto numReps = encoding.getMFMARep(shape, bElemTy);
  auto numRepK = numReps[0];
  auto numRepN = numReps[1];

  Value waveSize = i32_val(64);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveN = getWaveN(rewriter, loc, wave, warpsPerCTA,
                         mfmaInstrN, shape[1]);
  int numOfElems = std::max<int>(mfmaInstrK * mfmaInstrN / 64 /*wave size*/, 1);
  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);

  int macroTileM =
      std::max<int>(shape[0] / (warpsPerCTA[0] * 32), 1);
  int wptM = std::min<int>(warpsPerCTA[0], macroTileM);
  int macroTileN =
      std::max<int>(shape[1] / (warpsPerCTA[1] * 32), 1);
  int wptN = std::min<int>(warpsPerCTA[1], macroTileN);
  int wpt = std::max<int>(wptM, wptN);

  llvm::SmallVector<Value> offsets;
  if (isTransposed(order)) {
    SmallVector<int64_t> elemsPerThread{bElemsPerThread[1], bElemsPerThread[0]};
    SmallVector<int64_t> reps{numReps[1], numReps[0]};
    offsets = computeOffsetsTy1(rewriter, loc, elemsPerThread, waveN, lane, wpt,
                                numOfElems, reps, cSwizzleOffset);
  } else {
    unsigned int maxNumWarps = shape[1] / mfmaInstrN;
    int warpsPerGroupN = std::min(warpsPerCTA[1], maxNumWarps);
    offsets =
        computeOffsetsTy2(rewriter, loc, bElemsPerThread, warpsPerGroupN, waveN,
                          lane, wpt, numOfElems, numReps, cSwizzleOffset);
  }

  Value smemBase = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);

  Type smemPtrTy = getShemPtrTy(bElemTy);

  SmallVector<Value> hb;
  for (int n = 0; n < numRepN; ++n) {
    for (int k = 0; k < numRepK; ++k) {
      auto vecTy = vec_ty(bTensorTy.getElementType(), numOfElems);
      Value valVec = undef(vecTy);
      for (unsigned elem = 0; elem < numOfElems; ++elem) {
        Value elemOffset =
            offsets[n * numOfElems * numRepK + k * numOfElems + elem];
        Value elemValue = load(gep(smemPtrTy, smemBase, elemOffset));
        if (numOfElems > 1)
          valVec = insert_element(vecTy, valVec, elemValue, i32_val(elem));
        else
          valVec = elemValue;
      }
      if (bElemTy == i8_ty)
        valVec = bitcast(valVec, i32_ty);
      hb.push_back(valVec);
    }
  }

  MLIRContext *ctx = mfmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(hb.size(), hb[0].getType()));
  auto result = typeConverter->packLLElements(loc, hb, rewriter, structTy);
  return result;
}

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  switch (opIdx) {
  case 0:
    // operand $a
    return loadA(rewriter, loc, thread, encoding, typeConverter, tensor,
                 smemObj);
  case 1:
    // operand $b
    return loadB(rewriter, loc, thread, encoding, typeConverter, tensor,
                 smemObj);
  default:
    assert(false && "unexpected operand idx");
    return Value();
  }
}

} // namespace SharedToDotOperandMFMA

#endif // ifdef USE_ROCM
