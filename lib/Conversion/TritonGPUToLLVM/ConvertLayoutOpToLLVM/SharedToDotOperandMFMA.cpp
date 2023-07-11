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

/**
 * @brief This function maps particular load of mfma dot operand to element
 * indexes(row, col)
 *
 * Whole tensor is broken into "blocks" of waves along "non-K" axis.
 * One block could be processed by multiple waves.
 * One wave works on a piece of tensor size elemsPerInstr[0] x K.
 * Each of these pieces is broken into "tiles" of size elemsPerInstr[0] x
 * elemsPerInstr[1].
 *
 * Total offset of element is a sum of following values:
 * 1. Offset of wave block in tensor
 * 2. Offset of wave inside one wave block
 * 3. Offset of tile in one wave
 * 4. Offset of one lane data in a tile
 * 5. Offset of particular element of tensor processed by one lane
 *
 * This function computes these offsets for axies independently
 *
 * @param rewriter
 * @param loc
 * @param elemsPerInstr operand tile shape consumed by one MFMA instruction
 * @param waveId id component of 2d wave grid along nono-K axis
 * @param laneId lane id in warp [0..63]
 * @param warpsPerGroup number of warps in one block
 * @param numOfElems number of elements accessed by thread per repetition
 * @param reps number of instructions repretition to fully cover dot operand
 * @param smemStrides strides in LDS tensor
 * @return vector (i-th element corresponds to i-th load instruction) of
 * 2-element vectors(tensor row and col).
 */
llvm::SmallVector<llvm::SmallVector<Value>>
computeTensorElemMapping(ConversionPatternRewriter &rewriter, Location loc,
                         const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                         Value laneId, int warpsPerGroup, int numOfElems,
                         ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets) {
  auto numM = reps[0];
  auto numK = reps[1];
  SmallVector<llvm::SmallVector<Value>> mapping(numM * numK * numOfElems);

  Value _0 = i32_val(0);
  Value _32 = i32_val(32);

  for (int block = 0; block < numM; ++block) {
    Value blockVOffset = i32_val(block * elemsPerInstr[0] * warpsPerGroup);
    Value blockHOffset = _0;
    Value waveVOffset = mul(waveId, i32_val(elemsPerInstr[0]));
    Value waveHOffset = _0;
    for (int tile = 0; tile < numK; ++tile) {
      Value tileVOffset = _0;
      Value tileHOffset = i32_val(tile * elemsPerInstr[1]);

      Value laneVOffset = urem(laneId, _32);
      Value laneHOffset = mul(udiv(laneId, _32), i32_val(numOfElems));
      for (int elem = 0; elem < numOfElems; ++elem) {
        Value elemVOffset = _0;
        Value elemHOffset = i32_val(elem);

        Value sliceVOffset = add(
            add(add(add(blockVOffset, waveVOffset), tileVOffset), laneVOffset),
            elemVOffset);
        Value sliceHOffset = add(
            add(add(add(blockHOffset, waveHOffset), tileHOffset), laneHOffset),
            elemHOffset);

        Value row = add(sliceVOffset, smemOffsets[0]);
        Value col = add(sliceHOffset, smemOffsets[1]);

        mapping[numK * numOfElems * block + numOfElems * tile + elem] = {row,
                                                                         col};
      }
    }
  }
  return mapping;
}

Value computeOffset(ConversionPatternRewriter &rewriter, Location loc,
                    Value row, Value col, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout) {
  auto &strides = smemObj.strides;
  Value rowOffset = mul(row, strides[0]);
  Value colOffset = mul(col, strides[1]);
  return add(rowOffset, colOffset);
}

llvm::SmallVector<Value>
computeOffsetsAType(ConversionPatternRewriter &rewriter, Location loc,
                    const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                    Value laneId, int warpsPerGroup, int numOfElems,
                    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout) {
  SmallVector<Value> strides{smemObj.strides[0], smemObj.strides[1]};
  SmallVector<Value> offsets{smemObj.offsets[0], smemObj.offsets[1]};
  auto mapping =
      computeTensorElemMapping(rewriter, loc, elemsPerInstr, waveId, laneId,
                               warpsPerGroup, numOfElems, reps, offsets);
  llvm::SmallVector<Value> aOffsets(mapping.size());
  for (int i = 0; i < mapping.size(); ++i) {
    Value row = mapping[i][0];
    Value col = mapping[i][1];
    aOffsets[i] = computeOffset(rewriter, loc, row, col, smemObj, srcLayout);
  }
  return aOffsets;
}

llvm::SmallVector<Value>
computeOffsetsBType(ConversionPatternRewriter &rewriter, Location loc,
                    const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                    Value laneId, int warpsPerGroup, int numOfElems,
                    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout) {
  // transpose reps and offsets, because operand B has layout equal to
  // transposed operand A layout
  SmallVector<int64_t> tElemsPerInstr{elemsPerInstr[1], elemsPerInstr[0]};
  SmallVector<int64_t> tReps{reps[1], reps[0]};
  SmallVector<Value> toffsets{smemObj.offsets[1], smemObj.offsets[0]};
  auto mapping =
      computeTensorElemMapping(rewriter, loc, tElemsPerInstr, waveId, laneId,
                               warpsPerGroup, numOfElems, tReps, toffsets);
  llvm::SmallVector<Value> bOffsets(mapping.size());
  for (int i = 0; i < mapping.size(); ++i) {
    // swap row and col, because operand B layout is a transposed operand A
    // layout
    Value row = mapping[i][1];
    Value col = mapping[i][0];
    bOffsets[i] = computeOffset(rewriter, loc, row, col, smemObj, srcLayout);
  }
  return bOffsets;
}

Value computeBasePtr(ConversionPatternRewriter &rewriter, Location loc,
                     const SharedMemoryObject &smemObj) {
  Value base = smemObj.base;
  Type type = base.getType();
  for (int i = 0; i < smemObj.strides.size(); ++i) {
    Value offset = sub(i32_val(0), mul(smemObj.offsets[i], smemObj.strides[i]));
    base = gep(type, base, offset);
  }
  return base;
}

Value loadA(ConversionPatternRewriter &rewriter, Location loc, Value thread,
            DotOperandEncodingAttr encoding,
            TritonGPUToLLVMTypeConverter *typeConverter, Value tensor,
            const SharedMemoryObject &smemObj) {
  auto mfmaLayout = encoding.getParent().cast<MfmaEncodingAttr>();
  assert(mfmaLayout.getNonKDim() == 32);
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto aElemTy = aTensorTy.getElementType();
  auto aElemsPerInstr = encoding.getMFMAElemsPerThread(aElemTy);
  auto mfmaInstrM = aElemsPerInstr[0];
  auto mfmaInstrK = aElemsPerInstr[1];

  auto numReps = encoding.getMFMARep(shape, aElemTy);
  auto numRepM = numReps[0];
  auto numRepK = numReps[1];

  unsigned iWaveSize = triton::gpu::getWarpSize(mfmaLayout);
  Value waveSize = i32_val(iWaveSize);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveM =
      getWaveM(rewriter, loc, wave, warpsPerCTA, mfmaInstrM, shape[0]);
  int numOfElems =
      std::max<int>(mfmaInstrM * mfmaInstrK / iWaveSize /*wave size*/, 1);
  unsigned int maxNumWarps = shape[0] / mfmaInstrM;
  int warpsPerGroupM = std::min(warpsPerCTA[0], maxNumWarps);
  SmallVector<Value> offsets = computeOffsetsAType(
      rewriter, loc, aElemsPerInstr, waveM, lane, warpsPerGroupM, numOfElems,
      numReps, smemObj, sharedLayout);

  Value smemBase = computeBasePtr(rewriter, loc, smemObj);

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
  assert(mfmaLayout.getNonKDim() == 32);
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto bTensorTy = tensor.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = bTensorTy.getShape();
  auto sharedLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto bElemTy = bTensorTy.getElementType();
  auto bElemsPerInstr = encoding.getMFMAElemsPerThread(bElemTy);
  auto mfmaInstrK = bElemsPerInstr[0];
  auto mfmaInstrN = bElemsPerInstr[1];

  auto numReps = encoding.getMFMARep(shape, bElemTy);
  auto numRepK = numReps[0];
  auto numRepN = numReps[1];

  unsigned iWaveSize = triton::gpu::getWarpSize(mfmaLayout);
  Value waveSize = i32_val(iWaveSize);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveN =
      getWaveN(rewriter, loc, wave, warpsPerCTA, mfmaInstrN, shape[1]);
  int numOfElems =
      std::max<int>(mfmaInstrK * mfmaInstrN / iWaveSize /*wave size*/, 1);

  int macroTileM = std::max<int>(shape[0] / (warpsPerCTA[0] * 32), 1);
  int wptM = std::min<int>(warpsPerCTA[0], macroTileM);
  int macroTileN = std::max<int>(shape[1] / (warpsPerCTA[1] * 32), 1);
  int wptN = std::min<int>(warpsPerCTA[1], macroTileN);
  int wpt = std::max<int>(wptM, wptN);

  unsigned int maxNumWarps = shape[1] / mfmaInstrN;
  int warpsPerGroupN = std::min(warpsPerCTA[1], maxNumWarps);
  llvm::SmallVector<Value> offsets = computeOffsetsBType(
      rewriter, loc, bElemsPerInstr, waveN, lane, warpsPerGroupN, numOfElems,
      numReps, smemObj, sharedLayout);

  Value smemBase = computeBasePtr(rewriter, loc, smemObj);

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
