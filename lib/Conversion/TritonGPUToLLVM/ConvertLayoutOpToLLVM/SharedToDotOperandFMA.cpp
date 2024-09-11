#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using ValueTable = std::map<std::pair<int, int>, Value>;
using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::expandMatrixOrderWithBatch;
using ::mlir::triton::gpu::expandMatrixShapeWithBatch;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
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
  const auto &order = layout.getOrder();
  auto rank = order.size();

  if (!isSwizzled(layout))
    return rawIndices;

  auto vec = i32_val(layout.getVec());
  auto perPhase = i32_val(layout.getPerPhase());
  auto maxPhase = i32_val(layout.getMaxPhase());

  auto fastIdx = rawIndices[order[0]];
  auto secondIdx = rawIndices[order[1]];
  // Original algorithm taken from getSwizzledSharedPtrs function
  // (TritonGPUToLLVMBase.h)
  //
  // phase = (secondIdx // perPhase) % maxPhase
  // swizzledGroup = ((fastIdx // vec) ^ phase) * vec
  // groupRemainder = fastIdx % vec
  // colOff = swizzledGroup + groupRemainder
  auto phase = urem(udiv(secondIdx, perPhase), maxPhase);
  auto swizzledGroup = mul(xor_(udiv(fastIdx, vec), phase), vec);
  auto groupRemainder = urem(fastIdx, vec);
  auto colOff = add(swizzledGroup, groupRemainder);

  SmallVector<Value> swizzledIndices = rawIndices;
  swizzledIndices[order[0]] = colOff;

  return swizzledIndices;
}

/**
 * @brief put elements from Value vec to appropriate indexes in opValues array
 *
 * This function maps elements of 3d sub-tensor in linear array.
 * Axes are arranged in following order from fastest to slowest: [nonKdim, kDim,
 * bDim]
 */
void storeValuesInLinearVector(PatternRewriter &rewriter, Location loc,
                               SmallVector<Value> &opValues, Value vec,
                               unsigned kElems, unsigned nonKElems,
                               unsigned kIdx, unsigned nonKIdx, unsigned bIdx,
                               int kDim, int nonKDim, int bDim, int fastDim) {
  auto vecTy = cast<VectorType>(vec.getType());
  auto vectorSize = vecTy.getNumElements();
  auto elemTy = vecTy.getElementType();
  for (int elem = 0; elem < vectorSize; ++elem) {
    unsigned outIdx[3] = {};
    outIdx[bDim] = bIdx;
    outIdx[kDim] = kIdx;
    outIdx[nonKDim] = nonKIdx;

    outIdx[fastDim] += elem;
    auto idx = outIdx[bDim] * kElems * nonKElems + outIdx[kDim] * nonKElems +
               outIdx[nonKDim];
    opValues[idx] = extract_element(elemTy, vec, i32_val(elem));
  }
}

Value loadFMAOp(Value dotOp, Value llA, BlockedEncodingAttr dLayout,
                Value thread, Location loc,
                const LLVMTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, const int dotOpNo) {
  auto ctaSplit = dLayout.getCTALayout().getCTASplitNum();
  for (auto split : ctaSplit) {
    if (split != 1)
      llvm::report_fatal_error("tensors splited in CGA(thread group clusters) "
                               "are not supported in FMA dot yet.");
  }

  auto ctx = dotOp.getContext();
  const int bDim = 0;
  const int kDim = dotOpNo == 0 ? 2 : 1;
  const int nonKDim = dotOpNo == 0 ? 1 : 2;
  auto opTensorTy = cast<MemDescType>(dotOp.getType());
  auto opTensorShape = expandMatrixShapeWithBatch(opTensorTy.getShape());
  auto sharedLayout = cast<SharedEncodingAttr>(opTensorTy.getEncoding());
  auto opShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(opTensorTy)));

  auto order = expandMatrixOrderWithBatch(dLayout.getOrder());

  auto origSmem = getSharedMemoryObjectFromStruct(
      loc, llA, typeConverter->convertType(opTensorTy.getElementType()),
      rewriter);
  auto smem = getExpandedSharedMemoryObject(rewriter, loc, origSmem,
                                            opTensorTy.getShape());
  auto strides = smem.strides;
  int B = opTensorShape[bDim];
  int K = opTensorShape[kDim];
  int NonK = opTensorShape[nonKDim];

  auto shapePerCTATile =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTATile(dLayout)));
  auto sizePerThread =
      expandMatrixShapeWithBatch(ArrayRef(getSizePerThread(dLayout)));
  auto threadsPerWarp =
      expandMatrixShapeWithBatch(ArrayRef(dLayout.getThreadsPerWarp()));
  auto warpsPerCTA =
      expandMatrixShapeWithBatch(ArrayRef(dLayout.getWarpsPerCTA()));

  auto warpSize = i32_val(triton::gpu::getWarpSize(dLayout));
  auto laneId = urem(thread, warpSize);
  auto warpId = udiv(thread, warpSize);
  auto laneIds =
      mlir::LLVM::delinearize(rewriter, loc, laneId, threadsPerWarp, order);
  auto warpIds =
      mlir::LLVM::delinearize(rewriter, loc, warpId, warpsPerCTA, order);
  auto sizePerWarpB = sizePerThread[bDim] * threadsPerWarp[bDim];
  auto sizePerWarpNonK = sizePerThread[nonKDim] * threadsPerWarp[nonKDim];

  Value bTileOffset = mul(laneIds[bDim], i32_val(sizePerThread[bDim]));
  bTileOffset = add(bTileOffset, mul(warpIds[bDim], i32_val(sizePerWarpB)));
  Value nonKTileOffset = mul(laneIds[nonKDim], i32_val(sizePerThread[nonKDim]));
  nonKTileOffset =
      add(nonKTileOffset, mul(warpIds[nonKDim], i32_val(sizePerWarpNonK)));

  auto elemTy = typeConverter->convertType(opTensorTy.getElementType());
  Type ptrTy = ptr_ty(ctx, 3);

  auto sharedOrder = expandMatrixOrderWithBatch(sharedLayout.getOrder());
  unsigned vectorSize =
      sharedOrder[0] == kDim ? K : sizePerThread[sharedOrder[0]];
  if (sharedLayout.getMaxPhase() > 1)
    vectorSize = std::min(vectorSize, sharedLayout.getVec());
  auto vecTy = vec_ty(elemTy, vectorSize);

  unsigned dimStep[3] = {1, 1, 1};
  dimStep[sharedOrder[0]] = vectorSize;

  auto shapePerCTABTile = shapePerCTATile[bDim];
  auto shapePerCTANonKTile = shapePerCTATile[nonKDim];
  auto sizeBPerThread = sizePerThread[bDim];
  auto sizeNonKPerThread = sizePerThread[nonKDim];
  auto numBTiles = std::max(1u, B / shapePerCTABTile);
  auto numNonKTiles = std::max(1u, NonK / shapePerCTANonKTile);

  SmallVector<Value> opValues(numBTiles * sizeBPerThread * K * numNonKTiles *
                              sizeNonKPerThread);
  if (isSwizzled(sharedLayout)) {
    for (unsigned bTile = 0; bTile < numBTiles; ++bTile)
      for (unsigned b = 0; b < sizeBPerThread; b += dimStep[bDim])
        for (unsigned k = 0; k < K; k += dimStep[kDim])
          for (unsigned nonKTile = 0; nonKTile < numNonKTiles; ++nonKTile)
            for (unsigned nonK = 0; nonK < sizeNonKPerThread;
                 nonK += dimStep[nonKDim]) {
              SmallVector<Value> rawIndices(3);
              rawIndices[bDim] =
                  add(bTileOffset, i32_val(bTile * shapePerCTABTile + b));
              rawIndices[nonKDim] =
                  add(nonKTileOffset,
                      i32_val(nonKTile * shapePerCTANonKTile + nonK));
              rawIndices[kDim] = i32_val(k);

              SmallVector<Value> swizzledIndices =
                  swizzleIndices(rewriter, loc, rawIndices, sharedLayout);

              Value offset = i32_val(0);
              for (int dim = 0; dim < order.size(); ++dim) {
                auto wrappedDimIndex =
                    urem(swizzledIndices[dim], i32_val(opTensorShape[dim]));
                auto dimOffset = mul(wrappedDimIndex, strides[dim]);
                offset = add(offset, dimOffset);
              }

              Value elemAddr = gep(ptrTy, elemTy, smem.base, offset);
              Value vec = load(vecTy, elemAddr);
              storeValuesInLinearVector(
                  rewriter, loc, opValues, vec, /*kElems*/ K,
                  /*nonKElems*/ numNonKTiles * sizeNonKPerThread, /*kIdx*/ k,
                  /*nonKIdx*/ nonKTile * sizeNonKPerThread + nonK,
                  /*bIdx*/ bTile * sizeBPerThread + b, kDim, nonKDim, bDim,
                  sharedOrder[0]);
            }
  } else {
    auto bOffset = mul(urem(bTileOffset, i32_val(B)), strides[bDim]);
    auto nonKOffset =
        mul(urem(nonKTileOffset, i32_val(NonK)), strides[nonKDim]);
    Value threadDependantOffset = add(bOffset, nonKOffset);

    auto basePtr = gep(ptrTy, elemTy, smem.base, threadDependantOffset);

    for (unsigned bTile = 0; bTile < numBTiles; ++bTile)
      for (unsigned b = 0; b < sizeBPerThread; b += dimStep[bDim])
        for (unsigned k = 0; k < K; k += dimStep[kDim])
          for (unsigned nonKTile = 0; nonKTile < numNonKTiles; ++nonKTile)
            for (unsigned nonK = 0; nonK < sizeNonKPerThread;
                 nonK += dimStep[nonKDim]) {
              SmallVector<Value> offsetIndices(3);
              offsetIndices[bDim] = i32_val((bTile * shapePerCTABTile + b) % B);
              offsetIndices[nonKDim] =
                  i32_val((nonKTile * shapePerCTANonKTile + nonK) % NonK);
              offsetIndices[kDim] = i32_val(k);

              Value offset = i32_val(0);
              for (int dim = 0; dim < order.size(); ++dim)
                offset = add(offset, mul(offsetIndices[dim], strides[dim]));

              Value elemAddr = gep(ptrTy, elemTy, basePtr, offset);
              Value vec = load(vecTy, elemAddr);
              storeValuesInLinearVector(
                  rewriter, loc, opValues, vec, /*kElems*/ K,
                  /*nonKElems*/ numNonKTiles * sizeNonKPerThread, /*kIdx*/ k,
                  /*nonKIdx*/ nonKTile * sizeNonKPerThread + nonK,
                  /*bIdx*/ bTile * sizeBPerThread + b, kDim, nonKDim, bDim,
                  sharedOrder[0]);
            }
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
