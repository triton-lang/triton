#include "SharedToDotOperandHelper.h"

using ::mlir::triton::gpu::SharedEncodingAttr;

namespace mlir::triton::AMD {

// Get warpId inside block of warps.
Value getWarpIdInBlock(ConversionPatternRewriter &rewriter, Location loc,
                       Value warpId, const ArrayRef<unsigned int> &wpt,
                       int elemPerInstrNonK, int tensorSizeNonK, int nonKIdx,
                       const ArrayRef<unsigned int> &order) {
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, wpt, order);

  return urem(multiDimWarpId[nonKIdx],
              i32_val(tensorSizeNonK / elemPerInstrNonK));
}

bool isSwizzled(SharedEncodingAttr layout) { return layout.getMaxPhase() != 1; }

std::pair<mlir::Value, mlir::Value>
swizzleIndexes(ConversionPatternRewriter &rewriter, Location loc, Value row,
               Value col, SharedMemoryObject smemObj, SharedEncodingAttr attr) {
  (void)smemObj; // unused in current pattern
  const auto &order = attr.getOrder();
  auto rank = order.size();
  bool transposed = (order[rank - 2] != 1);
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

Value computeOffset(ConversionPatternRewriter &rewriter, Location loc,
                    Value row, Value col, SharedMemoryObject smemObj,
                    SharedEncodingAttr srcLayout) {
  auto [swizzledRow, swizzledCol] =
      swizzleIndexes(rewriter, loc, row, col, smemObj, srcLayout);
  const auto &strides = smemObj.getStrides();
  auto rank = strides.size();
  assert(rank == 2 || rank == 3);
  Value rowOffset = mul(swizzledRow, strides[rank - 2]);
  Value colOffset = mul(swizzledCol, strides[rank - 1]);
  return add(rowOffset, colOffset);
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

llvm::SmallVector<Value> computeOffsetsAType(
    ConversionPatternRewriter &rewriter, Location loc,
    computeTensorElemMappingInBlockT fn, const ArrayRef<int64_t> &elemsPerInstr,
    Value warpId, Value laneId, int warpsPerBlock, int numOfElems,
    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
    SharedEncodingAttr srcLayout, unsigned nonKDim, unsigned kDim) {
  SmallVector<Value> strides = smemObj.getStrides();
  SmallVector<Value> offsets = smemObj.getOffsets();
  auto rank = offsets.size();

  int vectorSize = 1;
  if (srcLayout.getOrder()[0] == rank - 1) {
    if (isSwizzled(srcLayout))
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    else
      vectorSize = numOfElems;
  }

  auto mapping = fn(rewriter, loc, elemsPerInstr, warpId, laneId, numOfElems,
                    reps, offsets, vectorSize, nonKDim, kDim);
  const auto numBlocks = reps[reps.size() - 2];
  const auto blockSize = mapping.size();
  auto order = srcLayout.getOrder();
  llvm::SmallVector<Value> aOffsets(blockSize * numBlocks);

  for (int block = 0; block < numBlocks; ++block) {
    int blockNonKOffset = block * nonKDim * warpsPerBlock;
    Value offAdjust = mul(i32_val(blockNonKOffset), strides[rank - 2]);
    for (int i = 0; i < blockSize; ++i) {
      Value row = mapping[i][0];
      Value col = mapping[i][1];
      aOffsets[block * blockSize + i] =
          add(offAdjust,
              computeOffset(rewriter, loc, row, col, smemObj, srcLayout));
    }
  }
  return aOffsets;
}

template <typename Container>
static SmallVector<typename Container::value_type>
transposeSpatialDims(const Container &vec) {
  auto rank = vec.size();
  assert(rank == 2 || rank == 3);
  SmallVector<typename Container::value_type> res(rank, vec[0]);
  res[rank - 2] = vec[rank - 1];
  res[rank - 1] = vec[rank - 2];
  return res;
}

llvm::SmallVector<Value> computeOffsetsBType(
    ConversionPatternRewriter &rewriter, Location loc,
    computeTensorElemMappingInBlockT fn, const ArrayRef<int64_t> &elemsPerInstr,
    Value warpId, Value laneId, int warpsPerBlock, int numOfElems,
    ArrayRef<int64_t> reps, SharedMemoryObject smemObj,
    SharedEncodingAttr srcLayout, unsigned nonKDim, unsigned kDim) {
  // transpose reps and offsets, because operand B has layout equal to
  // transposed operand A layout
  // this unifies axis order, so non-K dim is 0, k dim is 1
  auto rank = smemObj.getOffsets().size();
  SmallVector<int64_t> tElemsPerInstr{elemsPerInstr[1], elemsPerInstr[0]};
  SmallVector<int64_t> tReps = transposeSpatialDims(reps);
  SmallVector<Value> tOffsets = transposeSpatialDims(smemObj.getOffsets());
  SmallVector<Value> tStrides = transposeSpatialDims(smemObj.getStrides());

  int vectorSize = 1;
  if (srcLayout.getOrder()[0] == rank - 2) {
    if (isSwizzled(srcLayout))
      vectorSize = std::min(static_cast<int>(srcLayout.getVec()), numOfElems);
    else
      vectorSize = numOfElems;
  }

  auto mapping = fn(rewriter, loc, tElemsPerInstr, warpId, laneId, numOfElems,
                    tReps, tOffsets, vectorSize, nonKDim, kDim);
  const auto numBlocks = tReps[tReps.size() - 2];
  const auto blockSize = mapping.size();
  llvm::SmallVector<Value> bOffsets(blockSize * numBlocks);

  for (int block = 0; block < numBlocks; ++block) {
    int blockNonKOffset = block * nonKDim * warpsPerBlock;
    Value offAdjust = mul(i32_val(blockNonKOffset), tStrides[rank - 2]);
    for (int i = 0; i < mapping.size(); ++i) {
      // swap row and col, because operand B layout is a transposed operand A
      // layout
      Value row = mapping[i][1];
      Value col = mapping[i][0];
      bOffsets[block * blockSize + i] =
          add(offAdjust,
              computeOffset(rewriter, loc, row, col, smemObj, srcLayout));
    }
  }
  return bOffsets;
}

} // namespace mlir::triton::AMD
