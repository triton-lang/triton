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
#include "../PatternTritonGPUOpToLLVM.h"
#include "SharedToDotOperandHelper.h"
#include "Utility.h"

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::SharedEncodingAttr;

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
 * @param numOfElems number of elements accessed by thread per repetition
 * @param reps number of instructions repetition to fully cover dot operand
 * @param smemStrides strides in LDS tensor
 * @param loadVecSize number of elements loaded by one operation
 * @param iNonKDim non-K dimension size of one MFMA instruction
 * @param iKDim K dimension size of one MFMA instruction
 * @return vector (i-th element corresponds to i-th load instruction) of
 * 2-element vectors(tensor row and col).
 */
llvm::SmallVector<llvm::SmallVector<Value>> computeTensorElemMappingInBlock(
    ConversionPatternRewriter &rewriter, Location loc,
    const ArrayRef<int64_t> &elemsPerInstr, Value waveId, Value laneId,
    int numOfElems, ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets,
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
// @param warpsPerBlock number of warps per horizontal axis
// @param numOfElems number of elements accessed by threads per repetition
// @param reps number of instructions repretition to fully cover dot operand
// @param cSwizzleOffset
llvm::SmallVector<Value>
fastPathComputeOffsets(ConversionPatternRewriter &rewriter, Location loc,
                       const ArrayRef<int64_t> &elemsPerInstr, Value waveId,
                       Value laneId, int warpsPerBlock, int numOfElems,
                       ArrayRef<int64_t> reps, Value cSwizzleOffset) {
  auto numK = reps[1];
  auto numN = reps[2];
  SmallVector<Value> offsets(numK * numN * numOfElems);

  auto iKDim = elemsPerInstr[0];
  auto iNonKDim = elemsPerInstr[1];
  int lineSize = warpsPerBlock * iNonKDim * numN;
  Value _nonKDim = i32_val(iNonKDim);
  Value waveOffset = mul(waveId, i32_val(iNonKDim));
  Value colOffset = urem(laneId, _nonKDim);

  for (int block = 0; block < numN; ++block) {
    Value blockOffset = i32_val(block * iNonKDim * warpsPerBlock);
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

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread) {
  assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
  auto aTensorTy = cast<MemDescType>(tensor.getType());
  ArrayRef<int64_t> shape = aTensorTy.getShape();
  auto rank = shape.size();
  int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
  int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;

  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(encoding.getParent());
  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  auto sharedLayout = cast<SharedEncodingAttr>(aTensorTy.getEncoding());
  auto order = sharedLayout.getOrder();
  assert((rank == 2 || order[2] == 0) &&
         "expect batch to be the slowest dimension");

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
  auto repB = numReps[0];
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

  Value spatialWaveId = AMD::getWarpIdInBlock(
      rewriter, loc, linearWaveId, warpsPerCTA, mfmaInstrNonK,
      shape[nonKDimIdx], nonKDimIdx, triton::gpu::getOrder(mfmaLayout));

  // number of duplicates of elements in wave
  // In case of 64x4 x 4x4 multiplication, 4x4 B operand is duplicated 16 times
  int numSubBlocks = 1;
  if ((mfmaInstrK == 4 || mfmaInstrK == 1) && mfmaInstrNonK == 4)
    numSubBlocks = 16;
  // numOfElemsPerThreadPerMfmaInstr
  int numOfElems = mfmaInstrNonK * mfmaInstrK * numSubBlocks / iWaveSize;
  assert(numOfElems >= 1);

  unsigned int maxNumWarps = shape[nonKDimIdx] / mfmaInstrNonK;
  int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
  int warpsPerBatch =
      rank == 3 ? std::min<unsigned>(shape[0], warpsPerCTA[0]) : 1;
  Value waveIdInBatch = urem(linearWaveId, i32_val(warpsPerBatch));
  elemTy = typeConverter->convertType(elemTy);

  SmallVector<Value> loadedValues;
  SmallVector<Value> offsets;
  Value smemBase;
  bool isFastPath =
      !AMD::isKMajor(order, opIdx) && !hasSwizzleEnabled(sharedLayout);
  if (isFastPath) {
    // fast path handles tensors that are not k-major and have swizzling
    // disabled, in which case offsets computation can be simplified
    // TODO (zhanglx): later when we enable vector access to LDS for non k-major
    // tensors, we'll refactor the scope of fast and normal path
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    if (opIdx == 0) {
      if (isColMajor(order)) {
        SmallVector<int64_t> elemsPerInstr{mfmaInstrK, mfmaInstrNonK};
        SmallVector<int64_t> reps{numReps[0], numReps[2], numReps[1]};
        offsets = fastPathComputeOffsets(rewriter, loc, elemsPerInstr,
                                         spatialWaveId, lane, warpsPerBlockNonK,
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
                                         spatialWaveId, lane, warpsPerBlockNonK,
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
    if (opIdx == 0) {
      offsets = AMD::computeOffsetsAType(
          rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
          spatialWaveId, lane, warpsPerBlockNonK, numOfElems, numReps, smemObj,
          sharedLayout, mDim, mfmaInstrK);
    } else {
      assert(opIdx == 1);
      offsets = AMD::computeOffsetsBType(
          rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
          spatialWaveId, lane, warpsPerBlockNonK, numOfElems, numReps, smemObj,
          sharedLayout, nDim, mfmaInstrK);
    }
    smemBase = AMD::computeBasePtr(rewriter, loc, smemObj);
  }

  Type resElemTy = typeConverter->convertType(elemTy);
  Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

  int loadsPerThread = offsets.size() / numRepK / numRepNonK;
  int elemsPerLoad = numOfElems / loadsPerThread;
  assert(numOfElems % loadsPerThread == 0);

  for (int b = 0; b < repB; ++b) {
    int operandSize = shape[rank - 1] * shape[rank - 2];
    Value batchOffset = mul(i32_val(operandSize),
                            add(waveIdInBatch, i32_val(b * warpsPerBatch)));
    for (int nonK = 0; nonK < numRepNonK; ++nonK) {
      int blockNonKOffset = nonK * mfmaInstrNonK * warpsPerBlockNonK;
      Value waveBlockOffAdjust = i32_val(blockNonKOffset * shape[order[0]]);
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(resElemTy, numOfElems);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          auto loadVecTy = vec_ty(elemTy, elemsPerLoad);
          Value loadOffset;
          loadOffset = offsets[nonK * loadsPerThread * numRepK +
                               k * loadsPerThread + loadId];
          loadOffset = add(loadOffset, batchOffset);
          Value loadAddress = gep(smemPtrTy, elemTy, smemBase, loadOffset);
          Value loadedValue = load(loadVecTy, loadAddress);
          for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
            Value elemVal =
                extract_element(elemTy, loadedValue, i32_val(elemId));
            loadedValues.push_back(elemVal);
          }
        }
      }
    }
  }

  MLIRContext *ctx = mfmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
  auto result =
      packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
  return result;
}

} // namespace SharedToDotOperandMFMA
