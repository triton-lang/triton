/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifdef USE_ROCM

#include "../ConvertLayoutOpToLLVM.h"
#include "../Utility.h"

using ::mlir::triton::gpu::MfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::AMD::TritonGPUToLLVMTypeConverter;

namespace {

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
 * @brief swizzling tensor element indexes according pattern encoded in
 * SharedEncodingAttr
 *
 * @param rewriter
 * @param loc
 * @param row row of target tensor element related to the start of smemObj
 * @param col col of target tensor element related to the start of smemObj
 * @param smemObj shared memory object, contains info about tensor in LDS
 * @param attr layout attribute, contains swizzling info
 * @return swizzled row, col indexes in tensor notation
 */
std::pair<mlir::Value, mlir::Value>
swizzleIndexes(ConversionPatternRewriter &rewriter, Location loc, Value row,
               Value col, SharedMemoryObject smemObj, SharedEncodingAttr attr) {
  (void)smemObj; // unused in current pattern
  bool transposed = (attr.getOrder()[0] != 1);
  if (transposed) {
    // tensor is column-wise, so swapping col and row in computations
    std::swap(row, col);
  }
  auto vec = i32_val(attr.getVec());
  auto perPhase = i32_val(attr.getPerPhase());
  auto maxPhase = i32_val(attr.getMaxPhase());

  // Original algorithm taken from getSwizzledSharedPtrs function
  // (TritonGPUToLLVMBase.h): Basic algorithm for row-major tensor is following:
  //
  // phase = (row // perPhase) % maxPhase
  // colOffSwizzled = ((col // vec) ^ phase) * vec
  // colOffOrdered = col % vec
  // colOff = colOffSwizzled + colOffOrdered
  auto phase = urem(udiv(row, perPhase), maxPhase);
  auto colOffSwizzled = mul(xor_(udiv(col, vec), phase), vec);
  auto colOffOrdered = urem(col, vec);
  auto colOff = add(colOffSwizzled, colOffOrdered);

  if (transposed)
    return {colOff, row};
  else
    return {row, colOff};
}

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
 * @param loadVecSize number of elements loaded by one operation
 * @param iNonKDim non-K dimension of dot operand
 * @return vector (i-th element corresponds to i-th load instruction) of
 * 2-element vectors(tensor row and col).
 */
llvm::SmallVector<llvm::SmallVector<Value>>
computeTensorElemMapping(ConversionPatternRewriter &rewriter, Location loc,
                         const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                         Value laneId, int warpsPerGroup, int numOfElems,
                         ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets,
                         int loadVecSize, unsigned iNonKDim) {
  auto numM = reps[0];
  auto numK = reps[1];
  const int loadsPerThread = numOfElems / loadVecSize;
  llvm::SmallVector<llvm::SmallVector<Value>> mapping(numM * numK *
                                                      loadsPerThread);

  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value nonKDim = i32_val(iNonKDim);

  for (int block = 0; block < numM; ++block) {
    Value blockVOffset = i32_val(block * elemsPerInstr[0] * warpsPerGroup);
    Value blockHOffset = _0;
    Value waveVOffset = mul(waveId, i32_val(elemsPerInstr[0]));
    Value waveHOffset = _0;
    for (int tile = 0; tile < numK; ++tile) {
      Value tileVOffset = _0;
      Value tileHOffset = i32_val(tile * elemsPerInstr[1]);

      Value laneVOffset = urem(laneId, nonKDim);
      Value laneHOffset;
      if (iNonKDim == 32)
        laneHOffset = select(icmp_uge(laneId, _32), i32_val(numOfElems), _0);
      else
        laneHOffset = mul(udiv(laneId, nonKDim), i32_val(numOfElems));

      for (int loadId = 0; loadId < loadsPerThread; ++loadId) {
        Value elemVOffset = _0;
        Value elemHOffset = i32_val(loadId * loadVecSize);

        Value sliceVOffset = add(
            add(add(add(blockVOffset, waveVOffset), tileVOffset), laneVOffset),
            elemVOffset);
        Value sliceHOffset = add(
            add(add(add(blockHOffset, waveHOffset), tileHOffset), laneHOffset),
            elemHOffset);

        Value row = add(sliceVOffset, smemOffsets[0]);
        Value col = add(sliceHOffset, smemOffsets[1]);

        mapping[numK * loadsPerThread * block + loadsPerThread * tile +
                loadId] = {row, col};
      }
    }
  }
  return mapping;
}

bool isSwizzled(SharedEncodingAttr layout) { return layout.getMaxPhase() != 1; }

Value computeOffset(ConversionPatternRewriter &rewriter, Location loc,
                    Value row, Value col, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout) {
  auto [swizzledRow, swizzledCol] =
      swizzleIndexes(rewriter, loc, row, col, smemObj, srcLayout);
  auto &strides = smemObj.strides;
  Value rowOffset = mul(swizzledRow, strides[0]);
  Value colOffset = mul(swizzledCol, strides[1]);
  return add(rowOffset, colOffset);
}

llvm::SmallVector<Value>
computeOffsetsAType(ConversionPatternRewriter &rewriter, Location loc,
                    const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                    Value laneId, int warpsPerGroup, int numOfElems,
                    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout, unsigned nonKDim) {
  SmallVector<Value> strides{smemObj.strides[0], smemObj.strides[1]};
  SmallVector<Value> offsets{smemObj.offsets[0], smemObj.offsets[1]};

  int vectorSize = 1;
  if (srcLayout.getOrder()[0] == 1) {
    if (isSwizzled(srcLayout))
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    else
      vectorSize = numOfElems;
  }

  auto mapping = computeTensorElemMapping(rewriter, loc, elemsPerInstr, waveId,
                                          laneId, warpsPerGroup, numOfElems,
                                          reps, offsets, vectorSize, nonKDim);
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
                    SharedEncodingAttr srcLayout, unsigned nonKDim) {
  // transpose reps and offsets, because operand B has layout equal to
  // transposed operand A layout
  SmallVector<int64_t> tElemsPerInstr{elemsPerInstr[1], elemsPerInstr[0]};
  SmallVector<int64_t> tReps{reps[1], reps[0]};
  SmallVector<Value> toffsets{smemObj.offsets[1], smemObj.offsets[0]};

  int vectorSize = 1;
  if (srcLayout.getOrder()[0] == 0) {
    if (isSwizzled(srcLayout))
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    else
      vectorSize = numOfElems;
  }

  auto mapping = computeTensorElemMapping(rewriter, loc, tElemsPerInstr, waveId,
                                          laneId, warpsPerGroup, numOfElems,
                                          tReps, toffsets, vectorSize, nonKDim);
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
    base = gep(type, smemObj.baseElemType, base, offset);
  }
  return base;
}

/**
 * @brief try find if value is an integer constant
 *
 * Trace def-use chain and return integer in case we can proof it is constant.
 * Current implementation can trace chains of insertValue->extractValue
 * operations.
 *
 * @param val Value for that we want to get constant
 * @return std::optional on found integer value or empty std::optional
 */
std::optional<int> findConstValue(Value val) {
  while (val && !val.getDefiningOp<LLVM::ConstantOp>()) {
    LLVM::ExtractValueOp extractValOp =
        val.getDefiningOp<LLVM::ExtractValueOp>();
    if (!extractValOp)
      return std::optional<int>();
    auto extractPosArr = extractValOp.getPosition();
    if (extractPosArr.size() > 1)
      return std::optional<int>();
    int extractPos = extractPosArr[0];

    int insertPos = -1;
    LLVM::InsertValueOp insertValOp;
    Value container = extractValOp.getOperand();
    do {
      insertValOp = container.getDefiningOp<LLVM::InsertValueOp>();
      if (!insertValOp)
        return std::optional<int>();
      auto insertPosArr = insertValOp.getPosition();
      if (insertPosArr.size() > 1)
        return std::optional<int>();
      insertPos = insertPosArr[0];
      container = insertValOp.getContainer();
    } while (insertPos != extractPos);
    val = insertValOp.getValue();
  }
  if (!val)
    return std::optional<int>();
  auto cOp = val.getDefiningOp<LLVM::ConstantOp>();
  assert(cOp);
  auto valAttr = cOp.getValueAttr();
  auto intAttr = dyn_cast<mlir::IntegerAttr>(valAttr);
  assert(intAttr);
  return intAttr.getInt();
}

bool fastPathAvailable(const SharedMemoryObject &smemObj,
                       const SharedEncodingAttr &srcEncoding,
                       const MfmaEncodingAttr &dstEncoding) {
  if (dstEncoding.getNonKDim() != 32)
    return false;
  if (srcEncoding.getMaxPhase() > 1)
    return false;
  auto stride0 = findConstValue(smemObj.strides[0]);
  auto stride1 = findConstValue(smemObj.strides[1]);
  auto offset0 = findConstValue(smemObj.offsets[0]);
  auto offset1 = findConstValue(smemObj.offsets[1]);
  bool allValuesDefined = stride0.has_value() && stride1.has_value() &&
                          offset0.has_value() && offset1.has_value();
  if (!allValuesDefined)
    return false;
  if (offset0.value() != 0 || offset1.value() != 0)
    return false;
  return true;
}

// Computes offsets for operand A or transposed operand B
// @param rewriter
// @param loc
// @param elemsPerInstr operand tile shape consumed by one MFMA instruction
// @param waveM wave id for the "non K" axis
// @param laneId lane id in warp [0..63]
// @param warpsPerGroup number of warps in one block
// @param numOfElems number of elements accessed by thread per repetition
// @param reps number of instructions repretition to fully cover dot operand
// @param cSwizzleOffset
llvm::SmallVector<Value>
fastPathComputeOffsetsTy1(ConversionPatternRewriter &rewriter, Location loc,
                  const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                  Value laneId, int warpsPerGroup, int numOfElems,
                  ArrayRef<int64_t> reps, Value cSwizzleOffset) {
  const int loadVecSize = numOfElems;
  const int loadsPerThread = 1; // 1 is just in case if we decide to use different loadVecSize
  auto numM = reps[0];
  auto numK = reps[1];
  SmallVector<Value> offsets(numM * numK * loadsPerThread);
  int lineSize = elemsPerInstr[1] * numK;
  int blockSize = elemsPerInstr[0] * warpsPerGroup * lineSize;
  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value waveHalf = udiv(laneId, _32);

  Value waveOffset = mul(waveId, i32_val(elemsPerInstr[0] * lineSize));
  Value colOffset = select(icmp_uge(laneId, _32), i32_val(numOfElems), _0);

  for (int block = 0; block < numM; ++block) {
    Value blockOffset = i32_val(block * blockSize);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * elemsPerInstr[1]);
      for (int loadId = 0; loadId < loadsPerThread; ++loadId) {
        Value rowOffset =
            add(mul(urem(laneId, _32), i32_val(lineSize)), i32_val(loadId * loadVecSize));
        Value elemOffset = add(rowOffset, colOffset);
        Value offset =
            add(add(add(waveOffset, blockOffset), tileOffset), elemOffset);
        offsets[numK * loadsPerThread * block + loadsPerThread * tile + loadId] = offset;
      }
    }
  }
  return offsets;
}

// Computes offsets for operand B or transposed operand A
// @param rewriter
// @param loc
// @param elemsPerInstr operand tile shape consumed by one MFMA instruction
// @param waveId wave id for the "non K" axis
// @param laneId lane id in warp [0..63]
// @param warpsPerGroup number of warps per horizontal axis
// @param numOfElems number of elements accessed by threads per repetition
// @param reps number of instructions repretition to fully cover dot operand
// @param cSwizzleOffset
llvm::SmallVector<Value>
fastPathComputeOffsetsTy2(ConversionPatternRewriter &rewriter, Location loc,
                          const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                          Value laneId, int warpsPerGroup, int numOfElems,
                          ArrayRef<int64_t> reps, Value cSwizzleOffset) {
  auto numK = reps[0];
  auto numN = reps[1];
  SmallVector<Value> offsets(numK * numN * numOfElems);

  int lineSize = warpsPerGroup * elemsPerInstr[1] * numN;
  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value waveOffset = mul(waveId, i32_val(elemsPerInstr[1]));
  Value colOffset = urem(laneId, _32);

  for (int block = 0; block < numN; ++block) {
    Value blockOffset = i32_val(block * elemsPerInstr[1] * warpsPerGroup);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * elemsPerInstr[0] * lineSize);
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
  auto nonKDim = mfmaLayout.getNonKDim();
  assert(nonKDim == 32 || nonKDim == 16 || nonKDim == 4);
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto aElemTy = aTensorTy.getElementType();
  int kWidth = encoding.getKWidth();
  auto aElemsPerInstr = mfmaLayout.getMFMAElemsPerInstrForOperands(kWidth, 0);
  auto mfmaInstrM = aElemsPerInstr[0];
  auto mfmaInstrK = aElemsPerInstr[1];

  auto numReps = mfmaLayout.getMFMARepForOperands(shape, aElemTy, kWidth, 0);
  auto numRepM = numReps[0];
  auto numRepK = numReps[1];

  unsigned iWaveSize = triton::gpu::getWarpSize(mfmaLayout);
  assert(iWaveSize == 64);
  Value waveSize = i32_val(iWaveSize);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveM =
      getWaveM(rewriter, loc, wave, warpsPerCTA, mfmaInstrM, shape[0]);
  int numOfElems = mfmaInstrM * mfmaInstrK / iWaveSize;
  assert(numOfElems >= 1);
  unsigned int maxNumWarps = shape[0] / mfmaInstrM;
  int warpsPerGroupM = std::min(warpsPerCTA[0], maxNumWarps);
  aElemTy = typeConverter->convertType(aElemTy);
  Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);
  Type smemElemTy = aElemTy;

  SmallVector<Value> ha;
  if (fastPathAvailable(smemObj, sharedLayout, mfmaLayout)) {
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    SmallVector<Value> offsets;
    if (isTransposed(order)) {
      SmallVector<int64_t> elemsPerInstr{mfmaInstrK, mfmaInstrM};
      SmallVector<int64_t> reps{numReps[1], numReps[0]};
      offsets = fastPathComputeOffsetsTy2(rewriter, loc, elemsPerInstr, waveM,
                                          lane, warpsPerGroupM, numOfElems,
                                          reps, cSwizzleOffset);
    } else {
      offsets = fastPathComputeOffsetsTy1(rewriter, loc, aElemsPerInstr, waveM,
                                          lane, warpsPerGroupM, numOfElems,
                                          numReps, cSwizzleOffset);
    }
    Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);

    Type resElemTy = typeConverter->convertType(aElemTy);

    int loadsPerThread = offsets.size() / (numRepM * numRepK);
    const int elemsPerLoad = numOfElems / loadsPerThread;
    assert(numOfElems % loadsPerThread == 0);

    for (int m = 0; m < numRepM; ++m) {
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(resElemTy, numOfElems);
        Value valVec = undef(vecTy);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          auto loadVecTy = vec_ty(aElemTy, elemsPerLoad);
          Value loadOffset =
              offsets[m * loadsPerThread * numRepK + k * loadsPerThread + loadId];
          Value loadAddress = gep(smemPtrTy, smemElemTy, smemBase, loadOffset);
          Value vectorValue = load(loadVecTy, loadAddress);
          if (numOfElems > 1) {
            for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
              Value elemVal =
                  extract_element(aElemTy, vectorValue, i32_val(elemId));
              elemVal = bitcast(elemVal, resElemTy);
              valVec = insert_element(vecTy, valVec, elemVal,
                                      i32_val(loadId * elemsPerLoad + elemId));
            }
          } else {
            valVec = extract_element(aElemTy, vectorValue, i32_val(0));
            valVec = bitcast(valVec, resElemTy);
          }
        }
        if (aElemTy == i8_ty && numOfElems == 4)
          valVec = bitcast(valVec, i32_ty);
        if (aElemTy == i8_ty && numOfElems == 8)
          valVec = bitcast(valVec, i64_ty);
        ha.push_back(valVec);
      }
    }
  } else { // normal path
    SmallVector<Value> offsets = computeOffsetsAType(
        rewriter, loc, aElemsPerInstr, waveM, lane, warpsPerGroupM, numOfElems,
        numReps, smemObj, sharedLayout, nonKDim);

    Value smemBase = computeBasePtr(rewriter, loc, smemObj);
    Type resElemTy = typeConverter->convertType(aElemTy);


    int loadsPerThread = offsets.size() / (numReps[0] * numReps[1]);
    int elemsPerLoad = numOfElems / loadsPerThread;

    for (int m = 0; m < numRepM; ++m) {
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(resElemTy, numOfElems);
        Value valVec = undef(vecTy);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          auto loadVecTy = vec_ty(aElemTy, elemsPerLoad);
          Value loadOffset = offsets[m * loadsPerThread * numRepK +
                                     k * loadsPerThread + loadId];
          Value loadAddress = gep(smemPtrTy, smemElemTy, smemBase, loadOffset);
          Value vectorValue = load(loadVecTy, loadAddress);
          if (numOfElems > 1) {
            for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
              Value elemVal =
                  extract_element(aElemTy, vectorValue, i32_val(elemId));
              elemVal = bitcast(elemVal, resElemTy);
              valVec = insert_element(vecTy, valVec, elemVal,
                                      i32_val(loadId * elemsPerLoad + elemId));
            }
          } else {
            valVec = extract_element(aElemTy, vectorValue, i32_val(0));
            valVec = bitcast(valVec, resElemTy);
          }
        }
        if (aElemTy == i8_ty && numOfElems == 4)
          valVec = bitcast(valVec, i32_ty);
        if (aElemTy == i8_ty && numOfElems == 8)
          valVec = bitcast(valVec, i64_ty);
        ha.push_back(valVec);
      }
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
  auto nonKDim = mfmaLayout.getNonKDim();
  assert(nonKDim == 32 || nonKDim == 16 || nonKDim == 4);
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto bTensorTy = tensor.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = bTensorTy.getShape();
  auto sharedLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto bElemTy = bTensorTy.getElementType();
  int kWidth = encoding.getKWidth();
  auto bElemsPerInstr = mfmaLayout.getMFMAElemsPerInstrForOperands(kWidth, 1);
  auto mfmaInstrK = bElemsPerInstr[0];
  auto mfmaInstrN = bElemsPerInstr[1];

  auto numReps = mfmaLayout.getMFMARepForOperands(shape, bElemTy, kWidth, 1);
  auto numRepK = numReps[0];
  auto numRepN = numReps[1];

  unsigned iWaveSize = triton::gpu::getWarpSize(mfmaLayout);
  assert(iWaveSize == 64);
  Value waveSize = i32_val(iWaveSize);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveN =
      getWaveN(rewriter, loc, wave, warpsPerCTA, mfmaInstrN, shape[1]);
  int numOfElems = mfmaInstrK * mfmaInstrN / iWaveSize;
  assert(numOfElems >= 1);

  unsigned int maxNumWarps = shape[1] / mfmaInstrN;
  int warpsPerGroupN = std::min(warpsPerCTA[1], maxNumWarps);
  bElemTy = typeConverter->convertType(bElemTy);
  Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);
  Type smemElemTy = bElemTy;

  SmallVector<Value> hb;
  if (fastPathAvailable(smemObj, sharedLayout, mfmaLayout)) {
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);

    llvm::SmallVector<Value> offsets;
    unsigned int maxNumWarps = shape[1] / mfmaInstrN;
    int warpsPerGroupN = std::min(warpsPerCTA[1], maxNumWarps);
    if (isTransposed(order)) {
      SmallVector<int64_t> elemsPerInstr{mfmaInstrN, mfmaInstrK};
      SmallVector<int64_t> reps{numReps[1], numReps[0]};
      offsets = fastPathComputeOffsetsTy1(rewriter, loc, elemsPerInstr, waveN,
                                          lane, warpsPerGroupN, numOfElems,
                                          reps, cSwizzleOffset);
    } else {
      offsets = fastPathComputeOffsetsTy2(rewriter, loc, bElemsPerInstr, waveN,
                                          lane, warpsPerGroupN, numOfElems,
                                          numReps, cSwizzleOffset);
    }

    Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);

    Type resElemTy = typeConverter->convertType(bElemTy);

    const int loadsPerThread = offsets.size() / (numRepN * numRepK);
    const int elemsPerLoad = numOfElems / loadsPerThread;
    assert(numOfElems % loadsPerThread == 0);

    for (int n = 0; n < numRepN; ++n) {
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(resElemTy, numOfElems);
        Value valVec = undef(vecTy);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          auto loadVecTy = vec_ty(bElemTy, elemsPerLoad);
          Value loadOffset =
              offsets[n * loadsPerThread * numRepK + k * loadsPerThread + loadId];
          Value loadAddress = gep(smemPtrTy, smemElemTy, smemBase, loadOffset);
          Value vectorValue = load(loadVecTy, loadAddress);
          if (numOfElems > 1) {
            for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
              Value elemVal =
                  extract_element(bElemTy, vectorValue, i32_val(elemId));
              elemVal = bitcast(elemVal, resElemTy);
              valVec = insert_element(vecTy, valVec, elemVal,
                                      i32_val(loadId * elemsPerLoad + elemId));
            }
          } else {
            valVec = extract_element(bElemTy, vectorValue, i32_val(0));
            valVec = bitcast(valVec, resElemTy);
          }
        }
        if (bElemTy == i8_ty && numOfElems == 4)
          valVec = bitcast(valVec, i32_ty);
        if (bElemTy == i8_ty && numOfElems == 8)
          valVec = bitcast(valVec, i64_ty);
        hb.push_back(valVec);
      }
    }
  } else { // normal path
    llvm::SmallVector<Value> offsets = computeOffsetsBType(
        rewriter, loc, bElemsPerInstr, waveN, lane, warpsPerGroupN, numOfElems,
        numReps, smemObj, sharedLayout, nonKDim);

    Value smemBase = computeBasePtr(rewriter, loc, smemObj);
    Type resElemTy = typeConverter->convertType(bElemTy);

    int loadsPerThread = offsets.size() / (numReps[0] * numReps[1]);
    int elemsPerLoad = numOfElems / loadsPerThread;
    for (int n = 0; n < numRepN; ++n) {
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(resElemTy, numOfElems);
        Value valVec = undef(vecTy);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          auto loadVecTy = vec_ty(bElemTy, elemsPerLoad);
          Value loadOffset = offsets[n * loadsPerThread * numRepK +
                                     k * loadsPerThread + loadId];
          Value loadAddress = gep(smemPtrTy, smemElemTy, smemBase, loadOffset);
          Value vectorValue = load(loadVecTy, loadAddress);
          if (numOfElems > 1) {
            for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
              Value elemVal =
                  extract_element(bElemTy, vectorValue, i32_val(elemId));
              elemVal = bitcast(elemVal, resElemTy);
              valVec = insert_element(vecTy, valVec, elemVal,
                                      i32_val(loadId * elemsPerLoad + elemId));
            }
          } else {
            valVec = extract_element(bElemTy, vectorValue, i32_val(0));
            valVec = bitcast(valVec, resElemTy);
          }
        }
        if (bElemTy == i8_ty && numOfElems == 4)
          valVec = bitcast(valVec, i32_ty);
        if (bElemTy == i8_ty && numOfElems == 8)
          valVec = bitcast(valVec, i64_ty);
        hb.push_back(valVec);
      }
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
