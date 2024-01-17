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

// Get waveId inside block of waves.
Value getWaveIdInBlock(ConversionPatternRewriter &rewriter, Location loc,
                       Value waveId, const ArrayRef<unsigned int> &wpt,
                       int elemPerInstrNonK, int tensorSizeNonK, int nonKIdx) {
  if (nonKIdx == 1)
    waveId = udiv(waveId, i32_val(wpt[0]));
  return urem(urem(waveId, i32_val(wpt[nonKIdx])),
              i32_val(tensorSizeNonK / elemPerInstrNonK));
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
 * 1. Offset of wave-block in tensor
 * 2. Offset of wave inside one wave-block
 * 3. Offset of tile in one wave
 * 4. Offset of one lane data in a tile
 * 5. Offset of particular element of tensor processed by one lane
 *
 * This function computes these offsets for axies independently
 * Note that this function returns the offsets of elements in the first
 * wave-block. The offsets of elements in later wave-blocks can be computed
 * by adding a constant stride to the xor-ed offsets of elements in the
 * first wave-block.
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
  llvm::SmallVector<llvm::SmallVector<Value>> mapping(numK * loadsPerThread);

  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value nonKDim = i32_val(iNonKDim);
  Value waveVOffset = mul(waveId, i32_val(elemsPerInstr[0]));

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

      Value sliceVOffset =
          add(add(add(tileVOffset, laneVOffset), elemVOffset), waveVOffset);
      Value sliceHOffset = add(add(tileHOffset, laneHOffset), elemHOffset);

      Value row = add(sliceVOffset, smemOffsets[0]);
      Value col = add(sliceHOffset, smemOffsets[1]);

      mapping[loadsPerThread * tile + loadId] = {row, col};
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
    base = gep(type, base, offset);
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
fastPathComputeOffsets(ConversionPatternRewriter &rewriter, Location loc,
                       const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                       Value laneId, int warpsPerGroup, int numOfElems,
                       ArrayRef<int64_t> reps, Value cSwizzleOffset) {
  auto numK = reps[0];
  auto numN = reps[1];
  SmallVector<Value> offsets(numK * numN * numOfElems);

  int lineSize = warpsPerGroup * elemsPerInstr[1] * numN;
  Value _nonKDim = i32_val(elemsPerInstr[1]);
  Value waveOffset = mul(waveId, i32_val(elemsPerInstr[1]));
  Value colOffset = urem(laneId, _nonKDim);

  for (int block = 0; block < numN; ++block) {
    Value blockOffset = i32_val(block * elemsPerInstr[1] * warpsPerGroup);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * elemsPerInstr[0] * lineSize);
      for (int elem = 0; elem < numOfElems; ++elem) {
        Value halfOffset =
            mul(udiv(laneId, _nonKDim), i32_val(numOfElems * lineSize));
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

bool isColMajor(::llvm::ArrayRef<unsigned> order) {
  assert(order.size() == 2 && (order[0] & ~1ul) == 0 &&
         order[0] + order[1] == 1);
  return order[0] == 0;
}

bool isKMajor(::llvm::ArrayRef<unsigned> order, int opIdx) {
  if (order[0] + opIdx == 1)
    return true;
  else
    return false;
}

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");

  int kDimIdx = opIdx == 0 ? 1 : 0;
  int nonKDimIdx = opIdx == 0 ? 0 : 1;

  auto mfmaLayout = encoding.getParent().cast<MfmaEncodingAttr>();
  int nonKDim = mfmaLayout.getMDim();
  assert(nonKDim == 32 || nonKDim == 16 || nonKDim == 4);
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = aTensorTy.getShape();
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto elemTy = aTensorTy.getElementType();
  auto elemsPerInstr = encoding.getMFMAElemsPerInstr();
  auto mfmaInstrNonK = elemsPerInstr[nonKDimIdx];
  auto mfmaInstrK = elemsPerInstr[kDimIdx];

  auto numReps = encoding.getMFMARep(shape);
  auto numRepNonK = numReps[nonKDimIdx];
  auto numRepK = numReps[kDimIdx];

  unsigned iWaveSize = triton::gpu::getWarpSize(mfmaLayout);
  assert(iWaveSize == 64);
  Value waveSize = i32_val(iWaveSize);
  Value linearWaveId = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value spatialWaveId =
      getWaveIdInBlock(rewriter, loc, linearWaveId, warpsPerCTA, mfmaInstrNonK,
                       shape[nonKDimIdx], nonKDimIdx);
  int numOfElems = mfmaInstrNonK * mfmaInstrK / iWaveSize;
  assert(numOfElems >= 1);

  unsigned int maxNumWarps = shape[nonKDimIdx] / mfmaInstrNonK;
  int warpsPerGroupNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
  elemTy = typeConverter->convertType(elemTy);

  SmallVector<Value> loadedValues;
  SmallVector<Value> offsets;
  Value smemBase;
  bool isFastPath = fastPathAvailable(smemObj, sharedLayout, mfmaLayout);
  if (!isKMajor(order, opIdx) && isFastPath) {
    // fast path handles tensors that are not k-major, in which case swizzling
    // is disabled and offsets computation can be simplified
    // TODO (zhanglx): later when we enable vector access to LDS for non k-major
    // tensors, we'll refactor the scope of fast and normal path
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    if (opIdx == 0) {
      if (isColMajor(order)) {
        SmallVector<int64_t> elemsPerInstr{mfmaInstrK, mfmaInstrNonK};
        SmallVector<int64_t> reps{numReps[1], numReps[0]};
        offsets = fastPathComputeOffsets(rewriter, loc, elemsPerInstr,
                                         spatialWaveId, lane, warpsPerGroupNonK,
                                         numOfElems, reps, cSwizzleOffset);
      } else {
        llvm_unreachable(
            "row major operand A should be handled in the normal path");
      }
    } else {
      if (isColMajor(order)) {
        llvm_unreachable(
            "col major operand B should be handled in the normal path");
      } else {
        offsets = fastPathComputeOffsets(rewriter, loc, elemsPerInstr,
                                         spatialWaveId, lane, warpsPerGroupNonK,
                                         numOfElems, numReps, cSwizzleOffset);
      }
    }
    smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
  } else { // normal path
    // Normal path handles tensors that are k-major, in which case swizzling
    // is enabled and it requires a 2-step method to compute the offsets.
    if (opIdx == 0) {
      offsets = computeOffsetsAType(rewriter, loc, elemsPerInstr, spatialWaveId,
                                    lane, warpsPerGroupNonK, numOfElems,
                                    numReps, smemObj, sharedLayout, nonKDim);
    } else {
      assert(opIdx == 1);
      offsets = computeOffsetsBType(rewriter, loc, elemsPerInstr, spatialWaveId,
                                    lane, warpsPerGroupNonK, numOfElems,
                                    numReps, smemObj, sharedLayout, nonKDim);
    }
    smemBase = computeBasePtr(rewriter, loc, smemObj);
  }

  Type resElemTy = typeConverter->convertType(elemTy);
  Type smemPtrTy = getShemPtrTy(elemTy);

  int loadsPerThread = offsets.size() / numRepK / (isFastPath ? numRepNonK : 1);
  int elemsPerLoad = numOfElems / loadsPerThread;
  assert(numOfElems % loadsPerThread == 0);

  for (int nonK = 0; nonK < numRepNonK; ++nonK) {
    Value blockVOffset = i32_val(nonK * mfmaInstrNonK * warpsPerGroupNonK);
    Value offAdjust = mul(blockVOffset, i32_val(shape[order[0]]));
    for (int k = 0; k < numRepK; ++k) {
      auto vecTy = vec_ty(resElemTy, numOfElems);
      Value valVec = undef(vecTy);
      for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
        auto loadVecTy = vec_ty(elemTy, elemsPerLoad);
        Value loadOffset;
        if (isFastPath)
          loadOffset = offsets[nonK * loadsPerThread * numRepK +
                               k * loadsPerThread + loadId];
        else
          // In the normal path, we only computed the offsets of elements
          // in the first wave-block. Therefore, we update the offsets
          // of elements in later wave-blocks by adding a constant stride
          loadOffset = add(offAdjust, offsets[k * loadsPerThread + loadId]);
        Value loadAddress = bitcast(gep(smemPtrTy, smemBase, loadOffset),
                                    getShemPtrTy(loadVecTy));
        Value loadedValue = load(loadAddress);
        if (loadsPerThread > 1) {
          for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
            Value elemVal =
                extract_element(elemTy, loadedValue, i32_val(elemId));
            elemVal = bitcast(elemVal, resElemTy);
            valVec = insert_element(vecTy, valVec, elemVal,
                                    i32_val(loadId * elemsPerLoad + elemId));
          }
        } else {
          valVec = loadedValue;
        }
      }
      loadedValues.push_back(valVec);
    }
  }

  MLIRContext *ctx = mfmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
  auto result =
      typeConverter->packLLElements(loc, loadedValues, rewriter, structTy);
  return result;
}

} // namespace SharedToDotOperandMFMA

#endif // ifdef USE_ROCM
