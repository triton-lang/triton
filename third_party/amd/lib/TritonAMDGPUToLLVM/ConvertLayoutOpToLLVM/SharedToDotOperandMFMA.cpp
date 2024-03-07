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

#include "../PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

/* Get waveId inside block of waves.
 * For operand A or B, each wave will load elements of shape
 * elemsPerInstrNonK x K (for opA) or K x elemsPerInstrNonK (for opB).
 * All waves in the workgroup will load multiple of the above shape
 * based on warpsPerCTA. The block that is loaded by the workgroup
 * is referred to as a wave-block.
 * For some tile sizes, each wave may need to load more than one wave-blocks.
 * This function computes the id of the first wave-block in opA or opB
 * of the current wave (identified by waveId)
 */
Value getWaveIdInBlock(ConversionPatternRewriter &rewriter, Location loc,
                       Value waveId, const ArrayRef<unsigned int> &wpt,
                       int elemPerInstrNonK, int tensorSizeNonK, int nonKIdx) {
  auto rank = wpt.size();
  if (rank == 3)
    waveId = udiv(waveId, i32_val(wpt[0]));
  if (nonKIdx == rank - 1)
    waveId = udiv(waveId, i32_val(wpt[rank - 2]));
  // The urem( ,i32_val(tensorSizeNonK / elemPerInstrNonK)) part is to make sure
  // warps wrap to the begining when they go beyond M or N dim
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
  auto rank = attr.getOrder().size();
  bool transposed = (attr.getOrder()[0] != rank - 1);
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
 * @param waveId id component of 2d wave grid along non-K axis
 * @param laneId lane id in warp [0..63]
 * @param warpsPerGroup number of warps in one block
 * @param numOfElems number of elements accessed by thread per repetition
 * @param reps number of instructions repretition to fully cover dot operand
 * @param smemStrides strides in LDS tensor
 * @param loadVecSize number of elements loaded by one operation
 * @param iNonKDim non-K dimension size of one MFMA instruction
 * @param iKDim K dimension size of one MFMA instruction
 * @return vector (i-th element corresponds to i-th load instruction) of
 * 2-element vectors(tensor row and col).
 */
llvm::SmallVector<llvm::SmallVector<Value>>
computeTensorElemMapping(ConversionPatternRewriter &rewriter, Location loc,
                         const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                         Value laneId, int warpsPerGroup, int numOfElems,
                         ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets,
                         int loadVecSize, unsigned iNonKDim, unsigned iKDim) {
  auto numM = reps[1];
  auto numK = reps[2];
  const int loadsPerThread = numOfElems / loadVecSize;
  llvm::SmallVector<llvm::SmallVector<Value>> mapping(numK * loadsPerThread);

  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value nonKDim = i32_val(iNonKDim);
  Value waveVOffset = mul(waveId, i32_val(elemsPerInstr[0]));

  auto rank = smemOffsets.size();

  for (int tile = 0; tile < numK; ++tile) {
    Value tileVOffset = _0;
    Value tileHOffset = i32_val(tile * elemsPerInstr[1]);

    Value laneVOffset = urem(laneId, nonKDim);
    Value laneHOffset;
    if (iNonKDim == 32)
      laneHOffset = select(icmp_uge(laneId, _32), i32_val(numOfElems), _0);
    else {
      // In this configuration wave contains 16 copies of same data
      if ((iKDim == 1 || iKDim == 4) && iNonKDim == 4) {
        laneHOffset = i32_val(0);
      } else {
        assert(iKDim * iNonKDim / numOfElems == 64 &&
               "seems no all threads in wave contain unique elements");
        laneHOffset = mul(udiv(laneId, nonKDim), i32_val(numOfElems));
      }
    }

    for (int loadId = 0; loadId < loadsPerThread; ++loadId) {
      Value elemVOffset = _0;
      Value elemHOffset = i32_val(loadId * loadVecSize);

      Value sliceVOffset =
          add(add(add(tileVOffset, laneVOffset), elemVOffset), waveVOffset);
      Value sliceHOffset = add(add(tileHOffset, laneHOffset), elemHOffset);

      Value row = add(sliceVOffset, smemOffsets[rank - 2]);
      Value col = add(sliceHOffset, smemOffsets[rank - 1]);

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
  auto strides = smemObj.getStrides();
  auto rank = strides.size();
  Value rowOffset = mul(swizzledRow, strides[rank - 2]);
  Value colOffset = mul(swizzledCol, strides[rank - 1]);
  return add(rowOffset, colOffset);
}

llvm::SmallVector<Value>
computeOffsetsAType(ConversionPatternRewriter &rewriter, Location loc,
                    const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                    Value laneId, int warpsPerGroup, int numOfElems,
                    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout, unsigned nonKDim,
                    unsigned kDim) {
  // SmallVector<Value> offsets{smemObj.offsets[0], smemObj.offsets[1]};
  auto offsets = smemObj.getOffsets();
  auto rank = offsets.size();

  llvm::outs() << "offsets.size() = " << offsets.size() << "\n";
  llvm::outs() << "srcLayout.order: ";
  for (int i = 0; i < rank; i++)
    llvm::outs() << srcLayout.getOrder()[i] << ", ";
  llvm::outs() << "\n";

  int vectorSize = 1;
  if (srcLayout.getOrder()[0] == rank - 1) {
    if (isSwizzled(srcLayout))
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    else
      vectorSize = numOfElems;
  }

  llvm::outs() << "vectorSize = " << vectorSize << "\n";

  auto mapping = computeTensorElemMapping(
      rewriter, loc, elemsPerInstr, waveId, laneId, warpsPerGroup, numOfElems,
      reps, offsets, vectorSize, nonKDim, kDim);
  llvm::SmallVector<Value> aOffsets(mapping.size());
  llvm::outs() << "mapping.size() = " << mapping.size() << "\n";
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
                    SharedEncodingAttr srcLayout, unsigned nonKDim,
                    unsigned kDim) {
  // transpose reps and offsets, because operand B has layout equal to
  // transposed operand A layout
  SmallVector<int64_t> tElemsPerInstr{elemsPerInstr[1], elemsPerInstr[0]};
  SmallVector<int64_t> tReps{reps[0], reps[2], reps[1]};
  auto _offsets = smemObj.getOffsets();
  auto rank = _offsets.size();
  SmallVector<Value> toffsets(rank, _offsets[0]);
  toffsets[rank - 2] = _offsets[rank - 1];
  toffsets[rank - 1] = _offsets[rank - 2];

  int vectorSize = 1;
  if (srcLayout.getOrder()[0] == rank - 2) {
    if (isSwizzled(srcLayout))
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    else
      vectorSize = numOfElems;
  }

  auto mapping = computeTensorElemMapping(
      rewriter, loc, tElemsPerInstr, waveId, laneId, warpsPerGroup, numOfElems,
      tReps, toffsets, vectorSize, nonKDim, kDim);
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
    base = gep(ptr_ty(rewriter.getContext(), 3), type, base, offset);
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

bool hasSwizzleEnabled(const SharedEncodingAttr &srcEncoding) {
  return srcEncoding.getMaxPhase() > 1;
}

// Computes offsets for operand B or transposed operand A
// @param rewriter
// @param loc
// @param elemsPerInstr operand tile shape [K, nonK] consumed by one MFMA
// instruction
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
  auto numK = reps[1];
  auto numN = reps[2];
  SmallVector<Value> offsets(numK * numN * numOfElems);

  auto iKDim = elemsPerInstr[0];
  auto iNonKDim = elemsPerInstr[1];

  llvm::outs() << "  In fast path\n";
  llvm::outs() << "  iKDim, iNonKDim: " << iKDim << ", " << iNonKDim << "\n";

  int lineSize = warpsPerGroup * iNonKDim * numN;
  Value _nonKDim = i32_val(iNonKDim);
  Value waveOffset = mul(waveId, i32_val(iNonKDim));
  Value colOffset = urem(laneId, _nonKDim);

  for (int block = 0; block < numN; ++block) {
    Value blockOffset = i32_val(block * iNonKDim * warpsPerGroup);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * iKDim * lineSize);
      for (int elem = 0; elem < numOfElems; ++elem) {
        // halfOffset is an offset related to wrapping of wave in the tile.
        // for example, mfma 32 case (mapping of tensor elements to lane ids in
        // wave):
        //
        //  0  1  2  3 ... 31
        //  0  1  2  3 ... 31
        //  0  1  2  3 ... 31
        //  0  1  2  3 ... 31
        // 32 33 34 35 ... 63  <- at this point wave is wrapping
        // 32 33 34 35 ... 63
        // 32 33 34 35 ... 63
        // 32 33 34 35 ... 63
        Value halfOffset;
        if ((iKDim == 1 || iKDim == 4) && iNonKDim == 4)
          halfOffset = i32_val(0);
        else
          halfOffset =
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
  auto rank = order.size();
  return order[0] == (rank - 2);
}

bool isKMajor(::llvm::ArrayRef<unsigned> order, int opIdx) {
  auto rank = order.size();
  if ((rank == 2) && (order[0] + opIdx == 1))
    return true;
  if ((rank == 3) && (order[0] + opIdx == 2))
    return true;
  return false;
}

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread) {
  assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
  llvm::outs() << ">>> SharedToDotOperand with opIdx = " << opIdx << " <<<\n";
  auto aTensorTy = tensor.getType().cast<MemDescType>();
  ArrayRef<int64_t> shape = aTensorTy.getShape();
  auto rank = shape.size();
  int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
  int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;

  auto mfmaLayout = encoding.getParent().cast<AMDMfmaEncodingAttr>();
  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto elemTy = aTensorTy.getElementType();
  auto kWidth = encoding.getKWidth();
  auto elemsPerInstr = mfmaLayout.getMFMAInstrShapeForOperands(kWidth, opIdx);

  int64_t mfmaInstrNonK;
  int64_t mfmaInstrK;
  // TODO(Lixun): make it simpler
  // getMFMAInstrShapeForOperands always returns a 2D vector
  if (rank == 3) {
    mfmaInstrNonK = elemsPerInstr[nonKDimIdx - 1];
    mfmaInstrK = elemsPerInstr[kDimIdx - 1];
  } else {
    mfmaInstrNonK = elemsPerInstr[nonKDimIdx];
    mfmaInstrK = elemsPerInstr[kDimIdx];
  }

  auto numReps = mfmaLayout.getMFMARepForOperands(shape, kWidth, opIdx);
  auto numRepNonK = numReps[nonKDimIdx];
  auto numRepK = numReps[kDimIdx];
  // TODO(Lixun): make it simpler
  // getMFMARepForOperands always returns a 3D vector
  if (rank == 2) {
    numRepNonK = numReps[nonKDimIdx + 1];
    numRepK = numReps[kDimIdx + 1];
  }

  unsigned iWaveSize = triton::gpu::getWarpSize(mfmaLayout);
  assert(iWaveSize == 64);
  Value waveSize = i32_val(iWaveSize);
  Value linearWaveId = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  llvm::outs() << "kDimIdx, nonKDimIdx = " << kDimIdx << ", " << nonKDimIdx
               << "\n";
  llvm::outs() << "warpsPerCTA: ";
  for (int i = 0; i < warpsPerCTA.size(); i++)
    llvm::outs() << warpsPerCTA[i] << ",";
  llvm::outs() << "\n";
  llvm::outs() << "kWidth = " << kWidth << "\n";
  llvm::outs() << "mfmaInstrShape: " << mfmaInstrNonK << ", " << mfmaInstrK
               << "\n";
  llvm::outs() << "repNonK, repK: " << numRepNonK << ", " << numRepK << "\n";

  Value waveIdInBlock =
      getWaveIdInBlock(rewriter, loc, linearWaveId, warpsPerCTA, mfmaInstrNonK,
                       shape[nonKDimIdx], nonKDimIdx);
  // number of duplicates of elements in wave
  // In case of 64x4 x 4x4 multiplication, 4x4 B operand is duplicated 16 times
  int numSubBlocks = 1;
  if ((mfmaInstrK == 4 || mfmaInstrK == 1) && mfmaInstrNonK == 4)
    numSubBlocks = 16;
  // numOfElemsPerThreadPerMfmaInstr
  int numOfElems = mfmaInstrNonK * mfmaInstrK * numSubBlocks / iWaveSize;
  assert(numOfElems >= 1);

  unsigned int maxNumWarps = shape[nonKDimIdx] / mfmaInstrNonK;
  int warpsPerGroupNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
  elemTy = typeConverter->convertType(elemTy);

  llvm::outs() << "numOfElems, warpsPerGroupNonK: " << numOfElems << ", "
               << warpsPerGroupNonK << "\n";

  SmallVector<Value> loadedValues;
  SmallVector<Value> offsets;
  Value smemBase;
  bool isFastPath = !isKMajor(order, opIdx) && !hasSwizzleEnabled(sharedLayout);
  if (isFastPath) {
    // fast path handles tensors that are not k-major and have swizzling
    // disabled, in which case offsets computation can be simplified
    // TODO (zhanglx): later when we enable vector access to LDS for non k-major
    // tensors, we'll refactor the scope of fast and normal path

    llvm::outs() << "go to fast path\n";
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    if (opIdx == 0) {
      if (isColMajor(order)) {
        SmallVector<int64_t> elemsPerInstr{mfmaInstrK, mfmaInstrNonK};
        SmallVector<int64_t> reps{numReps[0], numReps[2], numReps[1]};
        offsets = fastPathComputeOffsets(rewriter, loc, elemsPerInstr,
                                         waveIdInBlock, lane, warpsPerGroupNonK,
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
                                         waveIdInBlock, lane, warpsPerGroupNonK,
                                         numOfElems, numReps, cSwizzleOffset);
      }
    }
    smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
  } else { // normal path
    // Normal path handles tensors that fall into either of the following three
    // cases:
    //   1. k-major + swizzling is enabled <-- this should be the most
    //   performant case
    //   2. k-major + swizzling is disabled <-- for testing purpose only
    //   3. non k-major + swizzling is enabled <-- for testing purpose only
    //
    // In this path, it requires a 2-step method to compute the offsets.
    llvm::outs() << "go to normal path\n";
    if (opIdx == 0) {
      offsets = computeOffsetsAType(
          rewriter, loc, elemsPerInstr, waveIdInBlock, lane, warpsPerGroupNonK,
          numOfElems, numReps, smemObj, sharedLayout, mDim, mfmaInstrK);
    } else {
      assert(opIdx == 1);
      offsets = computeOffsetsBType(
          rewriter, loc, elemsPerInstr, waveIdInBlock, lane, warpsPerGroupNonK,
          numOfElems, numReps, smemObj, sharedLayout, nDim, mfmaInstrK);
    }
    smemBase = computeBasePtr(rewriter, loc, smemObj);
  }

  Type resElemTy = typeConverter->convertType(elemTy);
  Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

  int loadsPerThread = offsets.size() / numRepK / (isFastPath ? numRepNonK : 1);
  int elemsPerLoad = numOfElems / loadsPerThread;
  assert(numOfElems % loadsPerThread == 0);

  for (int nonK = 0; nonK < numRepNonK; ++nonK) {
    int blockNonKOffset = nonK * mfmaInstrNonK * warpsPerGroupNonK;
    Value offAdjust = i32_val(blockNonKOffset * shape[order[0]]);
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
        Value loadAddress = gep(smemPtrTy, elemTy, smemBase, loadOffset);
        Value loadedValue = load(loadVecTy, loadAddress);
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
      packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
  llvm::outs() << ">>> END SharedToDotOperand with opIdx = " << opIdx
               << ", loadedValues.size() = " << loadedValues.size() << " <<<\n";
  return result;
}

} // namespace SharedToDotOperandMFMA

#endif // ifdef USE_ROCM
