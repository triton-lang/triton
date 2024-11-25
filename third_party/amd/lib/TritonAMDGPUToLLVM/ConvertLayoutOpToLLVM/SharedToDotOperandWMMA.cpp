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
#include "../TritonAMDGPUToLLVM/SchedInstructions.h"
#include "SharedToDotOperandHelper.h"
#include "Utility.h"

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace SharedToDotOperandWMMA {

/**
 * @brief Following functions maps particular load of wmma dot operand to
 * element indexes(row, col). For each WMMA generation separate function is
 * used.
 *
 * Whole tensor is broken into "blocks" of warps along "non-K" axis.
 * One block could be processed by multiple warps.
 * One warp works on a piece of tensor size elemsPerInstr[0] x K.
 * Each of these pieces is broken into "tiles" of size elemsPerInstr[0] x
 * elemsPerInstr[1].
 *
 * Total offset of element is a sum of following values:
 * 1. Offset of warp block in tensor
 * 2. Offset of warp inside one warp block
 * 3. Offset of tile in one warp
 * 4. Offset of one lane data in a tile
 * 5. Offset of particular element of tensor processed by one lane
 *
 * This function computes these offsets for axes independently
 *
 * @param rewriter
 * @param loc
 * @param elemsPerInstr operand tile shape consumed by one WMMA instruction
 * @param warpId id component of 2d warp grid along non-K axis
 * @param laneId lane id in warp [0..63]
 * @param numOfElems number of elements accessed by thread per repetition
 * @param reps number of instructions repetition to fully cover dot operand
 * @param smemStrides strides in LDS tensor
 * @param loadVecSize number of elements loaded by one operation
 * @param iNonKDim non-K dimension of dot operand
 * @return vector (i-th element corresponds to i-th load instruction) of
 * 2-element vectors(tensor row and col).
 */
llvm::SmallVector<llvm::SmallVector<Value>>
computeTensorElemMappingInBlockWmma1(
    ConversionPatternRewriter &rewriter, Location loc,
    const ArrayRef<int64_t> &elemsPerInstr, Value warpId, Value laneId,
    int numOfElems, ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets,
    int loadVecSize, unsigned iNonKDim, [[maybe_unused]] unsigned iKDim) {
  assert(reps.size() == 3);
  assert(elemsPerInstr.size() == 2);
  auto numK = reps[2];
  const int loadsPerThread = numOfElems / loadVecSize;
  llvm::SmallVector<llvm::SmallVector<Value>> mapping(numK * loadsPerThread);

  Value elemsPerInstrV = i32_val(elemsPerInstr[0]);
  Value warpVOffset = mul(warpId, elemsPerInstrV);
  Value sliceVOffset = add(urem(laneId, elemsPerInstrV), warpVOffset);
  auto rank = smemOffsets.size();
  Value row = add(sliceVOffset, smemOffsets[rank - 2]);

  for (int tile = 0; tile < numK; ++tile) {
    Value tileHOffset = i32_val(tile * elemsPerInstr[1]);

    for (int loadId = 0; loadId < loadsPerThread; ++loadId) {
      Value elemHOffset = i32_val(loadId * loadVecSize);
      Value sliceHOffset = add(tileHOffset, elemHOffset);

      Value col = add(sliceHOffset, smemOffsets[rank - 1]);
      mapping[loadsPerThread * tile + loadId] = {row, col};
    }
  }

  return mapping;
}

llvm::SmallVector<llvm::SmallVector<Value>>
computeTensorElemMappingInBlockWmma2(
    ConversionPatternRewriter &rewriter, Location loc,
    const ArrayRef<int64_t> &elemsPerInstr, Value warpId, Value laneId,
    int numOfElems, ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets,
    int loadVecSize, unsigned iNonKDim, [[maybe_unused]] unsigned iKDim) {
  assert(reps.size() == 3);
  assert(elemsPerInstr.size() == 2);
  auto numK = reps[2];
  const int loadsPerThread = numOfElems / loadVecSize;
  llvm::SmallVector<llvm::SmallVector<Value>> mapping(numK * loadsPerThread);

  Value rowsPerInstr = i32_val(elemsPerInstr[0]);
  Value colsPerInstr = i32_val(elemsPerInstr[1]);
  Value elemsPerThread = i32_val(elemsPerInstr[1] / 2);
  Value warpVOffset = mul(warpId, rowsPerInstr);
  Value sliceVOffset = add(urem(laneId, rowsPerInstr), warpVOffset);

  auto rank = smemOffsets.size();
  Value row = add(sliceVOffset, smemOffsets[rank - 2]);
  Value laneHOffset = mul(udiv(laneId, colsPerInstr), elemsPerThread);

  for (int tile = 0; tile < numK; ++tile) {
    Value tileHOffset = add(laneHOffset, i32_val(tile * elemsPerInstr[1]));
    for (int loadId = 0; loadId < loadsPerThread; ++loadId) {
      Value elemHOffset = i32_val(loadId * loadVecSize);
      Value sliceHOffset = add(tileHOffset, elemHOffset);

      Value col = add(sliceHOffset, smemOffsets[rank - 1]);

      mapping[loadsPerThread * tile + loadId] = {row, col};
    }
  }

  return mapping;
}

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread) {
  assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
  auto rank = smemObj.getStrides().size();
  int kDimIdx = opIdx == 0 ? rank - 1 : rank - 2;
  int nonKDimIdx = opIdx == 0 ? rank - 2 : rank - 1;

  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(encoding.getParent());
  auto computeTensorElemMappingInBlock =
      wmmaLayout.getVersion() == 1 ? computeTensorElemMappingInBlockWmma1
                                   : computeTensorElemMappingInBlockWmma2;
  assert(wmmaLayout.getMNKDimPerInstr()[nonKDimIdx] == 16);
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();

  auto aTensorTy = cast<triton::gpu::MemDescType>(tensor.getType());
  ArrayRef<int64_t> shape = aTensorTy.getShape();
  auto sharedLayout = cast<SharedEncodingAttr>(aTensorTy.getEncoding());
  auto order = sharedLayout.getOrder();
  assert((rank == 2 || order[2] == 0) &&
         "expect batch to be the slowest dimension");

  auto elemTy = aTensorTy.getElementType();
  int kWidth = encoding.getKWidth();
  auto elemsPerInstr = wmmaLayout.getElemsPerInstrForOperands();
  auto wmmaInstrK = elemsPerInstr[opIdx == 0 ? 1 : 0];
  auto wmmaInstrNonK = elemsPerInstr[opIdx == 0 ? 0 : 1];
  assert(wmmaInstrNonK == 16);

  auto numReps = wmmaLayout.getRepForOperand(shape, elemTy, kWidth, opIdx);
  auto numRepNonK = numReps[opIdx == 0 ? 1 : 2];
  auto numRepK = numReps[opIdx == 0 ? 2 : 1];
  auto repB = numReps[0];

  unsigned iWaveSize = triton::gpu::getWarpSize(wmmaLayout);
  assert(iWaveSize == 32);
  Value waveSize = i32_val(iWaveSize);
  Value linearWaveId = udiv(thread, waveSize);

  unsigned numElemsPerThreadPerRep =
      wmmaLayout.getSizePerThreadForOperand(kWidth, opIdx)[kDimIdx];

  Value lane = urem(thread, waveSize);
  unsigned int maxNumWarps = shape[nonKDimIdx] / wmmaInstrNonK;
  int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
  int warpsPerBatch =
      rank == 3 ? std::min<unsigned>(shape[0], warpsPerCTA[0]) : 1;
  Value waveIdInBatch = urem(linearWaveId, i32_val(warpsPerBatch));
  elemTy = typeConverter->convertType(elemTy);

  SmallVector<Value> loadedValues;
  SmallVector<Value> offsets;
  Value smemBase;
  Value spatialWarpId = AMD::getWarpIdInBlock(
      rewriter, loc, linearWaveId, warpsPerCTA, elemsPerInstr[0],
      shape[nonKDimIdx], nonKDimIdx, triton::gpu::getOrder(wmmaLayout));
  if (opIdx == 0) {
    offsets = AMD::computeOffsetsAType(
        rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
        spatialWarpId, lane, warpsPerBlockNonK, numElemsPerThreadPerRep,
        numReps, smemObj, sharedLayout, wmmaInstrNonK, wmmaInstrK);
  } else {
    assert(opIdx == 1);
    offsets = AMD::computeOffsetsBType(
        rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
        spatialWarpId, lane, warpsPerBlockNonK, numElemsPerThreadPerRep,
        numReps, smemObj, sharedLayout, wmmaInstrNonK, wmmaInstrK);
  }
  smemBase = AMD::computeBasePtr(rewriter, loc, smemObj);

  Type resElemTy = typeConverter->convertType(elemTy);
  Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

  int loadsPerThread = offsets.size() / (numRepNonK * numRepK);
  int elemsPerLoad = numElemsPerThreadPerRep / loadsPerThread;
  assert(numElemsPerThreadPerRep % loadsPerThread == 0);
  auto loadVecTy = vec_ty(elemTy, elemsPerLoad);
  for (int b = 0; b < repB; ++b) {
    int operandSize = shape[rank - 1] * shape[rank - 2];
    Value batchOffset = mul(i32_val(operandSize),
                            add(waveIdInBatch, i32_val(b * warpsPerBatch)));
    for (int nonK = 0; nonK < numRepNonK; ++nonK) {
      for (int k = 0; k < numRepK; ++k) {
        auto vecTy = vec_ty(resElemTy, numElemsPerThreadPerRep);
        Value valVec = undef(vecTy);
        for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
          Value loadOffset = offsets[nonK * loadsPerThread * numRepK +
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

  for (auto op : tensor.getUsers()) {
    if (auto localLoadOp = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op)) {
      const size_t numDsReadsCount =
          repB * numRepNonK * numRepK * loadsPerThread;
      setNumGeneratedDsReads(localLoadOp, numDsReadsCount, loadVecTy);
    }
  }

  MLIRContext *ctx = wmmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
  auto result =
      packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
  return result;
}

} // namespace SharedToDotOperandWMMA
