#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using ValueTable = std::map<std::pair<int, int>, Value>;
using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::expandMatrixOrderWithBatch;
using ::mlir::triton::gpu::expandMatrixShapeWithBatch;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::triton::gpu::SharedEncodingAttr;

Value getStructFromValueTable(ArrayRef<Value> vals,
                              ConversionPatternRewriter &rewriter, Location loc,
                              const LLVMTypeConverter *typeConverter,
                              Type elemTy) {
  SmallVector<Type> elemTypes(vals.size(), elemTy);
  SmallVector<Value> elems;
  elems.reserve(vals.size());
  for (auto &val : vals) {
    elems.push_back(val);
  }
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = struct_ty(elemTypes);
  return packLLElements(loc, typeConverter, elems, rewriter, structTy);
}

bool isSwizzled(SharedEncodingAttr layout) { return layout.getMaxPhase() != 1; }

SmallVector<Value> swizzleIndices(ConversionPatternRewriter &rewriter,
                                  Location loc, SmallVector<Value> rawIndices,
                                  SharedEncodingAttr layout) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  const auto &order = layout.getOrder();
  auto rank = order.size();

  if (!isSwizzled(layout))
    return rawIndices;

  auto vec = b.i32_val(layout.getVec());
  auto perPhase = b.i32_val(layout.getPerPhase());
  auto maxPhase = b.i32_val(layout.getMaxPhase());

  auto fastIdx = rawIndices[order[0]];
  auto secondIdx = rawIndices[order[1]];
  // phase = (secondIdx // perPhase) % maxPhase
  // swizzledGroup = ((fastIdx // vec) ^ phase) * vec
  // groupRemainder = fastIdx % vec
  // colOff = swizzledGroup + groupRemainder
  auto phase = b.urem(b.udiv(secondIdx, perPhase), maxPhase);
  auto swizzledGroup = b.mul(b.xor_(b.udiv(fastIdx, vec), phase), vec);
  auto groupRemainder = b.urem(fastIdx, vec);
  auto colOff = b.add(swizzledGroup, groupRemainder);

  SmallVector<Value> swizzledIndices = rawIndices;
  swizzledIndices[order[0]] = colOff;

  return swizzledIndices;
}

struct DimIdx {
  unsigned batch;
  unsigned k;
  unsigned nonK;
};

/// Put elements from Value vec to appropriate indexes in opValues array.
///
/// This function maps elements of 3d sub-tensor in linear array.
/// Axes are arranged in an order provided "opOrder" argument
void storeValuesInLinearVector(PatternRewriter &rewriter, Location loc,
                               SmallVector<Value> &opValues, Value vec,
                               ArrayRef<unsigned> perThreadTileShape,
                               unsigned kIdx, unsigned nonKIdx, unsigned bIdx,
                               const DimIdx &dim, int vecDim,
                               ArrayRef<unsigned> opOrder) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto vecTy = cast<VectorType>(vec.getType());
  auto vectorSize = vecTy.getNumElements();
  auto elemTy = vecTy.getElementType();
  for (int elem = 0; elem < vectorSize; ++elem) {
    unsigned spatialIdx[3] = {};
    spatialIdx[dim.batch] = bIdx;
    spatialIdx[dim.k] = kIdx;
    spatialIdx[dim.nonK] = nonKIdx;
    spatialIdx[vecDim] += elem;

    unsigned linearIdx = linearize(spatialIdx, perThreadTileShape, opOrder);
    opValues[linearIdx] = b.extract_element(elemTy, vec, b.i32_val(elem));
  }
}

bool verifyCTALayout(CTALayoutAttr ctaLayout) {
  auto ctaSplit = ctaLayout.getCTASplitNum();
  for (auto split : ctaSplit) {
    if (split != 1)
      return false;
  }
  return true;
}

/// Get a linear offset of first element loaded by thread.
///
/// In unswizzled case offset of any element computed with formula:
/// smem.base + first_element_offset + constant_offset.
///
/// first_element_offset depends on lane Id and warp Id
/// constant_offset depends on value number, which is same for all threads.
/// \returns first_element_offset
Value getUnswizzledFirstElemOffset(ConversionPatternRewriter &rewriter,
                                   Location loc, unsigned B, unsigned NonK,
                                   Value bTileOffset, Value nonKTileOffset,
                                   Value bStride, Value nonKStride) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto bOffset = b.mul(b.urem(bTileOffset, b.i32_val(B)), bStride);
  auto nonKOffset = b.mul(b.urem(nonKTileOffset, b.i32_val(NonK)), nonKStride);
  Value threadIdDependantOffset = b.add(bOffset, nonKOffset);
  return threadIdDependantOffset;
}

/// \returns number of elements stored by one thread across each dimension
SmallVector<unsigned> getElemsPerThreadInOp(ArrayRef<int64_t> opTensorShape,
                                            ArrayRef<unsigned> shapePerCTATile,
                                            ArrayRef<unsigned> sizePerThread) {
  int rank = opTensorShape.size();
  SmallVector<unsigned> elemsPerThread(rank);
  for (int d = 0; d < rank; ++d) {
    auto numReps =
        ceil(static_cast<unsigned>(opTensorShape[d]), shapePerCTATile[d]);
    elemsPerThread[d] = numReps * sizePerThread[d];
  }
  return elemsPerThread;
}

struct Indexes {
  unsigned bTile;
  unsigned b;
  unsigned k;
  unsigned nonKTile;
  unsigned nonK;
};

/// Computes a linear memory offset of a given element relative to
/// beginning of shared memory object.
Value computeSwizzledOffset(ConversionPatternRewriter &rewriter, Location loc,
                            const Indexes &i, const DimIdx &dim,
                            Value bTileOffset, Value nonKTileOffset,
                            unsigned shapePerCTABTile,
                            unsigned shapePerCTANonKTile,
                            SharedEncodingAttr sharedLayout,
                            ArrayRef<int64_t> opTensorShape,
                            ArrayRef<Value> strides) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value offset = b.i32_val(0);
  // Compute unswizzled multi dim coordinates in shared memory object
  SmallVector<Value> elemMultiDimIndices(3);
  elemMultiDimIndices[dim.batch] =
      b.add(bTileOffset, b.i32_val(i.bTile * shapePerCTABTile + i.b));
  elemMultiDimIndices[dim.nonK] = b.add(
      nonKTileOffset, b.i32_val(i.nonKTile * shapePerCTANonKTile + i.nonK));
  elemMultiDimIndices[dim.k] = b.i32_val(i.k);

  // Apply swizzling pattern to fastest dimension
  SmallVector<Value> swizzledIndices =
      swizzleIndices(rewriter, loc, elemMultiDimIndices, sharedLayout);

  // Linearize shared mem object dimensions into flat offset
  for (int d = 0; d < 3; ++d) {
    // wrap index if it is larger than tensor
    auto wrappedDimIndex =
        b.urem(swizzledIndices[d], b.i32_val(opTensorShape[d]));
    auto dimOffset = b.mul(wrappedDimIndex, strides[d]);
    offset = b.add(offset, dimOffset);
  }
  return offset;
}

/// Computes memory offset of a given element relative to the
/// first element loaded by a thread.
Value computeNonSwizzledOffset(ConversionPatternRewriter &rewriter,
                               Location loc, const Indexes &i,
                               const DimIdx &dim, ArrayRef<int64_t> tensorShape,
                               unsigned shapePerCTABTile,
                               unsigned shapePerCTANonKTile,
                               ArrayRef<Value> strides) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> offsetIndices(3);
  offsetIndices[dim.batch] =
      b.i32_val((i.bTile * shapePerCTABTile + i.b) % tensorShape[dim.batch]);
  offsetIndices[dim.nonK] = b.i32_val(
      (i.nonKTile * shapePerCTANonKTile + i.nonK) % tensorShape[dim.nonK]);
  offsetIndices[dim.k] = b.i32_val(i.k);

  Value offset = b.i32_val(0);
  for (int d = 0; d < 3; ++d)
    offset = b.add(offset, b.mul(offsetIndices[d], strides[d]));
  return offset;
}

/// Generates llvm IR for loading FMA dot operand from shared memory.
///
/// \param srcVal triton_gpu MemDescType value
/// \param llVal llvm IR values corresponding to srcVal
/// \param dLayout parent dot operand layout
/// \param thread thread id
/// \param loc
/// \param typeConverter
/// \param rewriter
/// \param dotOpNo
/// \returns llvm value with loaded elements
Value loadFMAOp(Value srcVal, Value llVal, BlockedEncodingAttr dLayout,
                Value thread, Location loc,
                const LLVMTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, const int dotOpNo) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  if (!verifyCTALayout(dLayout.getCTALayout()))
    return Value();

  DimIdx dim;
  dim.batch = 0;
  dim.k = dotOpNo == 0 ? 2 : 1;
  dim.nonK = dotOpNo == 0 ? 1 : 2;
  auto opTensorTy = cast<MemDescType>(srcVal.getType());
  auto opTensorShape = expandMatrixShapeWithBatch(opTensorTy.getShape());
  auto sharedLayout = cast<SharedEncodingAttr>(opTensorTy.getEncoding());

  auto opOrder = expandMatrixOrderWithBatch(dLayout.getOrder());

  auto origSmem = getSharedMemoryObjectFromStruct(
      loc, llVal, typeConverter->convertType(opTensorTy.getElementType()),
      rewriter);
  auto smem = getExpandedSharedMemoryObject(rewriter, loc, origSmem,
                                            opTensorTy.getShape());
  auto smemStrides = origSmem.getStrides(opTensorTy, loc, rewriter);
  int B = opTensorShape[dim.batch];
  int K = opTensorShape[dim.k];
  int NonK = opTensorShape[dim.nonK];

  auto shapePerCTATile =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTATile(dLayout)));
  shapePerCTATile[dim.k] = K;
  auto sizePerThread =
      expandMatrixShapeWithBatch(ArrayRef(getSizePerThread(dLayout)));
  sizePerThread[dim.k] = K;
  auto threadsPerWarp =
      expandMatrixShapeWithBatch(ArrayRef(dLayout.getThreadsPerWarp()));
  auto warpsPerCTA =
      expandMatrixShapeWithBatch(ArrayRef(dLayout.getWarpsPerCTA()));

  auto warpSize = tb.i32_val(triton::gpu::getWarpSize(dLayout));
  auto laneId = tb.urem(thread, warpSize);
  auto warpId = tb.udiv(thread, warpSize);
  auto laneIds =
      mlir::LLVM::delinearize(rewriter, loc, laneId, threadsPerWarp, opOrder);
  auto warpIds =
      mlir::LLVM::delinearize(rewriter, loc, warpId, warpsPerCTA, opOrder);
  auto sizePerWarpB = sizePerThread[dim.batch] * threadsPerWarp[dim.batch];
  auto sizePerWarpNonK = sizePerThread[dim.nonK] * threadsPerWarp[dim.nonK];

  Value bTileOffset =
      tb.mul(laneIds[dim.batch], tb.i32_val(sizePerThread[dim.batch]));
  bTileOffset =
      tb.add(bTileOffset, tb.mul(warpIds[dim.batch], tb.i32_val(sizePerWarpB)));
  Value nonKTileOffset =
      tb.mul(laneIds[dim.nonK], tb.i32_val(sizePerThread[dim.nonK]));
  nonKTileOffset = tb.add(
      nonKTileOffset, tb.mul(warpIds[dim.nonK], tb.i32_val(sizePerWarpNonK)));

  auto elemTy = typeConverter->convertType(opTensorTy.getElementType());
  Type ptrTy = smem.getBase().getType();

  auto sharedOrder = expandMatrixOrderWithBatch(sharedLayout.getOrder());
  // compute contiguity of fastest dimension in shared layout.
  unsigned vectorSize = sizePerThread[sharedOrder[0]];
  vectorSize = std::min(vectorSize, 128 / elemTy.getIntOrFloatBitWidth());

  bool swizzlePath = isSwizzled(sharedLayout);

  if (swizzlePath)
    vectorSize = std::min(vectorSize, sharedLayout.getVec());
  auto vecTy = vec_ty(elemTy, vectorSize);
  // loop increments depend on fastest dim
  unsigned dimStep[3] = {1, 1, 1};
  dimStep[sharedOrder[0]] = vectorSize;

  auto shapePerCTABTile = shapePerCTATile[dim.batch];
  auto shapePerCTANonKTile = shapePerCTATile[dim.nonK];
  auto sizeBPerThread = sizePerThread[dim.batch];
  auto sizeNonKPerThread = sizePerThread[dim.nonK];
  auto numBTiles = std::max(1u, B / shapePerCTABTile);
  auto numNonKTiles = std::max(1u, NonK / shapePerCTANonKTile);

  // Found discrepancy in this case,
  // use linear layout based converter for this case
  // TODO: break batch and non-k dimension iterations in
  // "repeat" and "inside-repeate" parts, pack them in llvm structure
  // according repeat and register order.
  // See FMA.cpp:getValueTableFromStructFMA for reference
  if (numBTiles != 1 || numNonKTiles != 1)
    return Value();

  auto perThreadShape =
      getElemsPerThreadInOp(opTensorShape, shapePerCTATile, sizePerThread);

  SmallVector<Value> opValues(numBTiles * sizeBPerThread * K * numNonKTiles *
                              sizeNonKPerThread);

  // In swizzled memory case basePtr stores pointer to the beginning of shared
  // memory object.
  //
  // If memory is not swizzled, algorithm breaks element offset pointer into
  // constant and non-constant part. Non-constant (depends on thread id) part is
  // the offset of the first element of the thread, which is same for all
  // elements of the thread. It is computed only once. basePtr stores this
  // non-constant part
  Value basePtr;
  if (swizzlePath) {
    basePtr = smem.getBase();
  } else {
    auto laneOffset = getUnswizzledFirstElemOffset(
        rewriter, loc, B, NonK, bTileOffset, nonKTileOffset,
        smemStrides[dim.batch], smemStrides[dim.nonK]);
    basePtr = tb.gep(ptrTy, elemTy, smem.getBase(), laneOffset);
  }

  // This loop nest iterates over all values loaded in one thread across batch,
  // k and nonK dimensions. Blocked dot operand layout allocates data in tiles
  // of size <sizePerThread>*<threadsPerWarp>*<numberWarps> for batch and nonK
  // dimensions. If tensor shape is larger than tile, pattern repeats. To take
  // these repeats into account iterations for batch and nonK are split into
  // "intra tile" + "inter tile" indexes: b + bTile, nonK + nonKTile
  for (unsigned bTile = 0; bTile < numBTiles; ++bTile)
    for (unsigned b = 0; b < sizeBPerThread; b += dimStep[dim.batch])
      for (unsigned k = 0; k < K; k += dimStep[dim.k])
        for (unsigned nonKTile = 0; nonKTile < numNonKTiles; ++nonKTile)
          for (unsigned nonK = 0; nonK < sizeNonKPerThread;
               nonK += dimStep[dim.nonK]) {
            Value offset = tb.i32_val(0);
            Indexes idx = {bTile, b, k, nonKTile, nonK};

            // swizzled variant is more general, but it limits optimization of
            // address computation,
            if (swizzlePath) {
              offset = computeSwizzledOffset(
                  rewriter, loc, idx, dim, bTileOffset, nonKTileOffset,
                  shapePerCTABTile, shapePerCTANonKTile, sharedLayout,
                  opTensorShape, smemStrides);
            } else {
              offset = computeNonSwizzledOffset(
                  rewriter, loc, idx, dim, opTensorShape, shapePerCTABTile,
                  shapePerCTANonKTile, smemStrides);
            }

            Value elemAddr = tb.gep(ptrTy, elemTy, basePtr, offset);
            Value vec = tb.load(vecTy, elemAddr);
            storeValuesInLinearVector(
                rewriter, loc, opValues, vec, perThreadShape, /*kIdx*/ k,
                /*nonKIdx*/ nonKTile * sizeNonKPerThread + nonK,
                /*bIdx*/ bTile * sizeBPerThread + b, dim, sharedOrder[0],
                opOrder);
          }

  return getStructFromValueTable(opValues, rewriter, loc, typeConverter,
                                 elemTy);
}

namespace SharedToDotOperandFMA {
Value convertLayout(int opIdx, Value val, Value llVal,
                    BlockedEncodingAttr dLayout, Value thread, Location loc,
                    const LLVMTypeConverter *typeConverter,
                    ConversionPatternRewriter &rewriter) {
  return loadFMAOp(val, llVal, dLayout, thread, loc, typeConverter, rewriter,
                   opIdx);
}
} // namespace SharedToDotOperandFMA
