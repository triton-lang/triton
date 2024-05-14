#include "Utility.h"

using CoordTy = SmallVector<Value>;
using ValueTable = std::map<std::pair<int, int>, std::pair<Value, Value>>;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Compute the offset of the matrix to load.
// Returns offsetAM, offsetAK, offsetBN, offsetBK.
// NOTE, the information M(from $a) and N(from $b) couldn't be retrieved at
// the same time in the usage in convert_layout[shared->dot_op], we leave
// the noexist info to be 0 and only use the desired argument from the
// composed result. In this way we want to retain the original code
// structure in convert_mma884 method for easier debugging.
static std::tuple<Value, Value, Value, Value>
computeOffsets(Value threadId, bool isARow, bool isBRow, ArrayRef<int> fpw,
               ArrayRef<int> spw, ArrayRef<int> rep,
               ConversionPatternRewriter &rewriter, Location loc,
               Type resultTy) {
  auto *ctx = rewriter.getContext();
  auto wpt = cast<NvidiaMmaEncodingAttr>(
                 cast<DotOperandEncodingAttr>(
                     cast<RankedTensorType>(resultTy).getEncoding())
                     .getParent())
                 .getWarpsPerCTA();

  Value _1 = i32_val(1);
  Value _3 = i32_val(3);
  Value _4 = i32_val(4);
  Value _16 = i32_val(16);
  Value _32 = i32_val(32);

  Value lane = urem(threadId, _32);
  Value warp = udiv(threadId, _32);

  // warp offset
  Value warp0 = urem(warp, i32_val(wpt[0]));
  Value warp12 = udiv(warp, i32_val(wpt[0]));
  Value warp1 = urem(warp12, i32_val(wpt[1]));
  Value warpMOff = mul(warp0, i32_val(spw[0]));
  Value warpNOff = mul(warp1, i32_val(spw[1]));
  // Quad offset
  Value quadMOff = mul(udiv(and_(lane, _16), _4), i32_val(fpw[0]));
  Value quadNOff = mul(udiv(and_(lane, _16), _4), i32_val(fpw[1]));
  // Pair offset
  Value pairMOff = udiv(urem(lane, _16), _4);
  pairMOff = urem(pairMOff, i32_val(fpw[0]));
  pairMOff = mul(pairMOff, _4);
  Value pairNOff = udiv(urem(lane, _16), _4);
  pairNOff = udiv(pairNOff, i32_val(fpw[0]));
  pairNOff = urem(pairNOff, i32_val(fpw[1]));
  pairNOff = mul(pairNOff, _4);
  // scale
  pairMOff = mul(pairMOff, i32_val(rep[0] / 2));
  quadMOff = mul(quadMOff, i32_val(rep[0] / 2));
  pairNOff = mul(pairNOff, i32_val(rep[1] / 2));
  quadNOff = mul(quadNOff, i32_val(rep[1] / 2));
  // Quad pair offset
  Value laneMOff = add(pairMOff, quadMOff);
  Value laneNOff = add(pairNOff, quadNOff);
  // A offset
  Value offsetAM = add(warpMOff, laneMOff);
  Value offsetAK = and_(lane, _3);
  // B offset
  Value offsetBN = add(warpNOff, laneNOff);
  Value offsetBK = and_(lane, _3);
  // i indices
  Value offsetCM = add(and_(lane, _1), offsetAM);
  if (isARow) {
    offsetAM = add(offsetAM, urem(threadId, _4));
    offsetAK = i32_val(0);
  }
  if (!isBRow) {
    offsetBN = add(offsetBN, urem(threadId, _4));
    offsetBK = i32_val(0);
  }

  return std::make_tuple(offsetAM, offsetAK, offsetBN, offsetBK);
}

static Value loadA(Value tensor, const SharedMemoryObject &smemObj,
                   Value thread, Location loc,
                   const LLVMTypeConverter *typeConverter,
                   ConversionPatternRewriter &rewriter, Type resultTy) {
  static constexpr std::array<int, 3> fpw{{2, 2, 1}};
  auto mmaEncoding = cast<NvidiaMmaEncodingAttr>(
      cast<DotOperandEncodingAttr>(
          cast<RankedTensorType>(resultTy).getEncoding())
          .getParent());
  auto wpt = mmaEncoding.getWarpsPerCTA();

  auto *ctx = rewriter.getContext();
  auto tensorTy = cast<MemDescType>(tensor.getType());
  auto sharedLayout = cast<SharedEncodingAttr>(tensorTy.getEncoding());
  auto shape = tensorTy.getShape();
  auto order = sharedLayout.getOrder();

  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
  Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);

  bool isARow = order[0] != 0;
  auto resultEncoding = cast<DotOperandEncodingAttr>(
      cast<RankedTensorType>(resultTy).getEncoding());
  auto [offsetAM, offsetAK, _3, _4] = computeOffsets(
      thread, isARow, false, fpw,
      mmaEncoding.getMMAv1ShapePerWarp(resultEncoding.getOpIdx()),
      mmaEncoding.getMMAv1Rep(resultEncoding.getOpIdx()), rewriter, loc,
      resultTy);

  int vecA = sharedLayout.getVec();

  auto strides = smemObj.strides;
  Value strideAM = isARow ? strides[0] : i32_val(1);
  Value strideAK = isARow ? i32_val(1) : strides[1];
  Value strideA0 = isARow ? strideAK : strideAM;
  Value strideA1 = isARow ? strideAM : strideAK;

  int strideRepM = wpt[0] * fpw[0] * 8;
  int strideRepK = 1;

  // swizzling
  int perPhaseA = sharedLayout.getPerPhase();
  int maxPhaseA = sharedLayout.getMaxPhase();
  int stepA0 = isARow ? strideRepK : strideRepM;
  int numPtrA = std::max(2 * perPhaseA * maxPhaseA / stepA0, 1);
  int NK = shape[1];

  // pre-compute pointer lanes
  Value offA0 = isARow ? offsetAK : offsetAM;
  Value offA1 = isARow ? offsetAM : offsetAK;
  Value phaseA = urem(udiv(offA1, i32_val(perPhaseA)), i32_val(maxPhaseA));
  offA0 = add(offA0, cSwizzleOffset);
  SmallVector<Value> offA(numPtrA);
  for (int i = 0; i < numPtrA; i++) {
    Value offA0I = add(offA0, i32_val(i * (isARow ? 4 : strideRepM)));
    offA0I = udiv(offA0I, i32_val(vecA));
    offA0I = xor_(offA0I, phaseA);
    offA0I = mul(offA0I, i32_val(vecA));
    offA[i] = add(mul(offA0I, strideA0), mul(offA1, strideA1));
  }

  Type elemX2Ty = vec_ty(f16_ty, 2);
  Type elemTy = f16_ty;
  if (tensorTy.getElementType().isBF16()) {
    elemX2Ty = vec_ty(i16_ty, 2);
    elemTy = i16_ty;
  }

  // prepare arguments
  SmallVector<Value> ptrA(numPtrA);

  std::map<std::pair<int, int>, std::pair<Value, Value>> has;
  for (int i = 0; i < numPtrA; i++)
    ptrA[i] = gep(ptr_ty(ctx, 3), f16_ty, smemBase, offA[i]);

  auto ld = [&](decltype(has) &vals, int m, int k, Value val0, Value val1) {
    vals[{m, k}] = {val0, val1};
  };
  auto loadA = [&](int m, int k) {
    int offidx = (isARow ? k / 4 : m) % numPtrA;
    Value thePtrA = gep(ptr_ty(ctx, 3), elemTy, smemBase, offA[offidx]);

    int stepAM = isARow ? m : m / numPtrA * numPtrA;
    int stepAK = isARow ? k / (numPtrA * vecA) * (numPtrA * vecA) : k;
    Value offset = add(mul(i32_val(stepAM * strideRepM), strideAM),
                       mul(i32_val(stepAK), strideAK));
    Value pa = gep(ptr_ty(ctx, 3), elemTy, thePtrA, offset);
    Type vecTy = vec_ty(i32_ty, std::max<int>(vecA / 2, 1));
    Type aPtrTy = ptr_ty(ctx, 3);
    Value ha = load(vecTy, bitcast(pa, aPtrTy));
    // record lds that needs to be moved
    Value ha00 = bitcast(extract_element(ha, i32_val(0)), elemX2Ty);
    Value ha01 = bitcast(extract_element(ha, i32_val(1)), elemX2Ty);
    ld(has, m, k, ha00, ha01);

    if (vecA > 4) {
      Value ha10 = bitcast(extract_element(ha, i32_val(2)), elemX2Ty);
      Value ha11 = bitcast(extract_element(ha, i32_val(3)), elemX2Ty);
      if (isARow)
        ld(has, m, k + 4, ha10, ha11);
      else
        ld(has, m + 1, k, ha10, ha11);
    }
  };

  bool isARow_ = mmaEncoding.getMMAv1IsRow(resultEncoding.getOpIdx());
  bool isAVec4 = mmaEncoding.getMMAv1IsVec4(resultEncoding.getOpIdx());
  unsigned numM =
      mmaEncoding.getMMAv1NumOuter(shape, resultEncoding.getOpIdx());
  for (unsigned k = 0; k < NK; k += 4)
    for (unsigned m = 0; m < numM / 2; ++m)
      if (!has.count({m, k}))
        loadA(m, k);

  SmallVector<Value> elems;
  elems.reserve(has.size() * 2);
  for (auto item : has) { // has is a map, the key should be ordered.
    elems.push_back(bitcast(item.second.first, i32_ty));
    elems.push_back(bitcast(item.second.second, i32_ty));
  }

  Value res = packLLElements(loc, typeConverter, elems, rewriter, resultTy);
  return res;
}

static Value loadB(Value tensor, const SharedMemoryObject &smemObj,
                   Value thread, Location loc,
                   const LLVMTypeConverter *typeConverter,
                   ConversionPatternRewriter &rewriter, Type resultTy) {
  static constexpr std::array<int, 3> fpw{{2, 2, 1}};
  auto mmaEncoding = cast<NvidiaMmaEncodingAttr>(
      cast<DotOperandEncodingAttr>(
          cast<RankedTensorType>(resultTy).getEncoding())
          .getParent());
  auto wpt = mmaEncoding.getWarpsPerCTA();
  // smem
  auto strides = smemObj.strides;

  auto *ctx = rewriter.getContext();
  auto tensorTy = cast<MemDescType>(tensor.getType());
  auto sharedLayout = cast<SharedEncodingAttr>(tensorTy.getEncoding());

  auto shape = tensorTy.getShape();
  auto order = sharedLayout.getOrder();

  Value smem = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
  bool isBRow = order[0] != 0; // is row-major in shared memory layout
  // isBRow_ indicates whether B is row-major in DotOperand layout
  auto resultEncoding = cast<DotOperandEncodingAttr>(
      cast<RankedTensorType>(resultTy).getEncoding());

  int vecB = sharedLayout.getVec();
  Value strideBN = isBRow ? i32_val(1) : strides[1];
  Value strideBK = isBRow ? strides[0] : i32_val(1);
  Value strideB0 = isBRow ? strideBN : strideBK;
  Value strideB1 = isBRow ? strideBK : strideBN;
  int strideRepN = wpt[1] * fpw[1] * 8;
  int strideRepK = 1;

  auto [_3, _4, offsetBN, offsetBK] = computeOffsets(
      thread, false, isBRow, fpw,
      mmaEncoding.getMMAv1ShapePerWarp(resultEncoding.getOpIdx()),
      mmaEncoding.getMMAv1Rep(resultEncoding.getOpIdx()), rewriter, loc,
      resultTy);

  // swizzling
  int perPhaseB = sharedLayout.getPerPhase();
  int maxPhaseB = sharedLayout.getMaxPhase();
  int stepB0 = isBRow ? strideRepN : strideRepK;
  int numPtrB = std::max(2 * perPhaseB * maxPhaseB / stepB0, 1);
  int NK = shape[0];

  Value offB0 = isBRow ? offsetBN : offsetBK;
  Value offB1 = isBRow ? offsetBK : offsetBN;
  Value phaseB = urem(udiv(offB1, i32_val(perPhaseB)), i32_val(maxPhaseB));
  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);

  offB0 = add(offB0, cSwizzleOffset);
  SmallVector<Value> offB(numPtrB);
  for (int i = 0; i < numPtrB; ++i) {
    Value offB0I = add(offB0, i32_val(i * (isBRow ? strideRepN : 4)));
    offB0I = udiv(offB0I, i32_val(vecB));
    offB0I = xor_(offB0I, phaseB);
    offB0I = mul(offB0I, i32_val(vecB));
    offB[i] = add(mul(offB0I, strideB0), mul(offB1, strideB1));
  }

  Type elemTy = f16_ty;
  Type elemX2Ty = vec_ty(f16_ty, 2);
  if (tensorTy.getElementType().isBF16()) {
    elemTy = i16_ty;
    elemX2Ty = vec_ty(i16_ty, 2);
  }

  SmallVector<Value> ptrB(numPtrB);
  ValueTable hbs;
  for (int i = 0; i < numPtrB; ++i)
    ptrB[i] = gep(ptr_ty(ctx, 3), f16_ty, smem, offB[i]);

  auto ld = [&](decltype(hbs) &vals, int m, int k, Value val0, Value val1) {
    vals[{m, k}] = {val0, val1};
  };

  auto loadB = [&](int n, int K) {
    int offidx = (isBRow ? n : K / 4) % numPtrB;
    Value thePtrB = ptrB[offidx];

    int stepBN = isBRow ? n / numPtrB * numPtrB : n;
    int stepBK = isBRow ? K : K / (numPtrB * vecB) * (numPtrB * vecB);
    Value offset = add(mul(i32_val(stepBN * strideRepN), strideBN),
                       mul(i32_val(stepBK), strideBK));
    Value pb = gep(ptr_ty(ctx, 3), elemTy, thePtrB, offset);

    Type vecTy = vec_ty(i32_ty, std::max(vecB / 2, 1));
    Value hb = load(vecTy, bitcast(pb, ptr_ty(ctx, 3)));
    // record lds that needs to be moved
    Value hb00 = bitcast(extract_element(hb, i32_val(0)), elemX2Ty);
    Value hb01 = bitcast(extract_element(hb, i32_val(1)), elemX2Ty);
    ld(hbs, n, K, hb00, hb01);
    if (vecB > 4) {
      Value hb10 = bitcast(extract_element(hb, i32_val(2)), elemX2Ty);
      Value hb11 = bitcast(extract_element(hb, i32_val(3)), elemX2Ty);
      if (isBRow)
        ld(hbs, n + 1, K, hb10, hb11);
      else
        ld(hbs, n, K + 4, hb10, hb11);
    }
  };

  bool isBRow_ = mmaEncoding.getMMAv1IsRow(resultEncoding.getOpIdx());
  assert(isBRow == isBRow_ && "B need smem isRow");
  bool isBVec4 = mmaEncoding.getMMAv1IsVec4(resultEncoding.getOpIdx());
  unsigned numN =
      mmaEncoding.getMMAv1NumOuter(shape, resultEncoding.getOpIdx());
  for (unsigned k = 0; k < NK; k += 4)
    for (unsigned n = 0; n < numN / 2; ++n) {
      if (!hbs.count({n, k}))
        loadB(n, k);
    }

  SmallVector<Value> elems;
  for (auto &item : hbs) { // has is a map, the key should be ordered.
    elems.push_back(bitcast(item.second.first, i32_ty));
    elems.push_back(bitcast(item.second.second, i32_ty));
  }

  Value res = packLLElements(loc, typeConverter, elems, rewriter, resultTy);
  return res;
}

namespace SharedToDotOperandMMAv1 {

Value convertLayout(int opIdx, Value tensor, const SharedMemoryObject &smemObj,
                    Value thread, Location loc,
                    const LLVMTypeConverter *typeConverter,
                    ConversionPatternRewriter &rewriter, Type resultTy) {
  if (opIdx == 0)
    return loadA(tensor, smemObj, thread, loc, typeConverter, rewriter,
                 resultTy);
  else {
    assert(opIdx == 1);
    return loadB(tensor, smemObj, thread, loc, typeConverter, rewriter,
                 resultTy);
  }
}

} // namespace SharedToDotOperandMMAv1
