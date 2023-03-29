#include "../ConvertLayoutOpToLLVM.h"
#include "../Utility.h"

using CoordTy = SmallVector<Value>;
using ValueTable = std::map<std::pair<int, int>, std::pair<Value, Value>>;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
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
  auto wpt = resultTy.cast<RankedTensorType>()
                 .getEncoding()
                 .cast<DotOperandEncodingAttr>()
                 .getParent()
                 .cast<MmaEncodingAttr>()
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
                   TritonGPUToLLVMTypeConverter *typeConverter,
                   ConversionPatternRewriter &rewriter, Type resultTy) {
  static constexpr std::array<int, 3> fpw{{2, 2, 1}};
  auto wpt = resultTy.cast<RankedTensorType>()
                 .getEncoding()
                 .cast<DotOperandEncodingAttr>()
                 .getParent()
                 .cast<MmaEncodingAttr>()
                 .getWarpsPerCTA();

  auto *ctx = rewriter.getContext();
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  auto sharedLayout = tensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto shape = tensorTy.getShape();
  auto order = sharedLayout.getOrder();

  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
  Value smemBase = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);

  bool isARow = order[0] != 0;
  auto resultEncoding = resultTy.cast<RankedTensorType>()
                            .getEncoding()
                            .cast<DotOperandEncodingAttr>();
  auto [offsetAM, offsetAK, _3, _4] = computeOffsets(
      thread, isARow, false, fpw, resultEncoding.getMMAv1ShapePerWarp(),
      resultEncoding.getMMAv1Rep(), rewriter, loc, resultTy);

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
  Type elemPtrTy = ptr_ty(f16_ty, 3);
  if (tensorTy.getElementType().isBF16()) {
    elemX2Ty = vec_ty(i16_ty, 2);
    elemPtrTy = ptr_ty(i16_ty, 3);
  }

  // prepare arguments
  SmallVector<Value> ptrA(numPtrA);

  std::map<std::pair<int, int>, std::pair<Value, Value>> has;
  for (int i = 0; i < numPtrA; i++)
    ptrA[i] = gep(ptr_ty(f16_ty, 3), smemBase, offA[i]);

  auto ld = [&](decltype(has) &vals, int m, int k, Value val0, Value val1) {
    vals[{m, k}] = {val0, val1};
  };
  auto loadA = [&](int m, int k) {
    int offidx = (isARow ? k / 4 : m) % numPtrA;
    Value thePtrA = gep(elemPtrTy, smemBase, offA[offidx]);

    int stepAM = isARow ? m : m / numPtrA * numPtrA;
    int stepAK = isARow ? k / (numPtrA * vecA) * (numPtrA * vecA) : k;
    Value offset = add(mul(i32_val(stepAM * strideRepM), strideAM),
                       mul(i32_val(stepAK), strideAK));
    Value pa = gep(elemPtrTy, thePtrA, offset);
    Type aPtrTy = ptr_ty(vec_ty(i32_ty, std::max<int>(vecA / 2, 1)), 3);
    Value ha = load(bitcast(pa, aPtrTy));
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

  bool isARow_ = resultEncoding.getMMAv1IsRow();
  bool isAVec4 = resultEncoding.getMMAv1IsVec4();
  unsigned numM = resultEncoding.getMMAv1NumOuter(shape);
  for (unsigned k = 0; k < NK; k += 4)
    for (unsigned m = 0; m < numM / 2; ++m)
      if (!has.count({m, k}))
        loadA(m, k);

  SmallVector<Value> elems;
  elems.reserve(has.size() * 2);
  for (auto item : has) { // has is a map, the key should be ordered.
    elems.push_back(item.second.first);
    elems.push_back(item.second.second);
  }

  Value res = typeConverter->packLLElements(loc, elems, rewriter, resultTy);
  return res;
}

static Value loadB(Value tensor, const SharedMemoryObject &smemObj,
                   Value thread, Location loc,
                   TritonGPUToLLVMTypeConverter *typeConverter,
                   ConversionPatternRewriter &rewriter, Type resultTy) {
  static constexpr std::array<int, 3> fpw{{2, 2, 1}};
  auto wpt = resultTy.cast<RankedTensorType>()
                 .getEncoding()
                 .cast<DotOperandEncodingAttr>()
                 .getParent()
                 .cast<MmaEncodingAttr>()
                 .getWarpsPerCTA();
  // smem
  auto strides = smemObj.strides;

  auto *ctx = rewriter.getContext();
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  auto sharedLayout = tensorTy.getEncoding().cast<SharedEncodingAttr>();

  auto shape = tensorTy.getShape();
  auto order = sharedLayout.getOrder();

  Value smem = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);
  bool isBRow = order[0] != 0; // is row-major in shared memory layout
  // isBRow_ indicates whether B is row-major in DotOperand layout
  auto resultEncoding = resultTy.cast<RankedTensorType>()
                            .getEncoding()
                            .cast<DotOperandEncodingAttr>();

  int vecB = sharedLayout.getVec();
  Value strideBN = isBRow ? i32_val(1) : strides[1];
  Value strideBK = isBRow ? strides[0] : i32_val(1);
  Value strideB0 = isBRow ? strideBN : strideBK;
  Value strideB1 = isBRow ? strideBK : strideBN;
  int strideRepN = wpt[1] * fpw[1] * 8;
  int strideRepK = 1;

  auto [_3, _4, offsetBN, offsetBK] = computeOffsets(
      thread, false, isBRow, fpw, resultEncoding.getMMAv1ShapePerWarp(),
      resultEncoding.getMMAv1Rep(), rewriter, loc, resultTy);

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

  Type elemPtrTy = ptr_ty(f16_ty, 3);
  Type elemX2Ty = vec_ty(f16_ty, 2);
  if (tensorTy.getElementType().isBF16()) {
    elemPtrTy = ptr_ty(i16_ty, 3);
    elemX2Ty = vec_ty(i16_ty, 2);
  }

  SmallVector<Value> ptrB(numPtrB);
  ValueTable hbs;
  for (int i = 0; i < numPtrB; ++i)
    ptrB[i] = gep(ptr_ty(f16_ty, 3), smem, offB[i]);

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
    Value pb = gep(elemPtrTy, thePtrB, offset);

    Value hb =
        load(bitcast(pb, ptr_ty(vec_ty(i32_ty, std::max(vecB / 2, 1)), 3)));
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

  bool isBRow_ = resultEncoding.getMMAv1IsRow();
  assert(isBRow == isBRow_ && "B need smem isRow");
  bool isBVec4 = resultEncoding.getMMAv1IsVec4();
  unsigned numN = resultEncoding.getMMAv1NumOuter(shape);
  for (unsigned k = 0; k < NK; k += 4)
    for (unsigned n = 0; n < numN / 2; ++n) {
      if (!hbs.count({n, k}))
        loadB(n, k);
    }

  SmallVector<Value> elems;
  for (auto &item : hbs) { // has is a map, the key should be ordered.
    elems.push_back(item.second.first);
    elems.push_back(item.second.second);
  }

  Value res = typeConverter->packLLElements(loc, elems, rewriter, resultTy);
  return res;
}

namespace SharedToDotOperandMMAv1 {
using CoordTy = SmallVector<Value>;
using ValueTable = std::map<std::pair<int, int>, std::pair<Value, Value>>;

SmallVector<CoordTy> getMNCoords(Value thread,
                                 ConversionPatternRewriter &rewriter,
                                 ArrayRef<unsigned int> wpt,
                                 const MmaEncodingAttr &mmaLayout,
                                 ArrayRef<int64_t> shape, bool isARow,
                                 bool isBRow, bool isAVec4, bool isBVec4) {
  static constexpr std::array<int, 3> fpw{{2, 2, 1}};

  auto *ctx = thread.getContext();
  auto loc = UnknownLoc::get(ctx);
  Value _1 = i32_val(1);
  Value _2 = i32_val(2);
  Value _4 = i32_val(4);
  Value _16 = i32_val(16);
  Value _32 = i32_val(32);
  Value _fpw0 = i32_val(fpw[0]);
  Value _fpw1 = i32_val(fpw[1]);

  // A info
  auto aEncoding = DotOperandEncodingAttr::get(ctx, 0, mmaLayout);
  auto aRep = aEncoding.getMMAv1Rep();
  auto aSpw = aEncoding.getMMAv1ShapePerWarp();
  // B info
  auto bEncoding = DotOperandEncodingAttr::get(ctx, 1, mmaLayout);
  auto bSpw = bEncoding.getMMAv1ShapePerWarp();
  auto bRep = bEncoding.getMMAv1Rep();

  SmallVector<int, 2> rep({aRep[0], bRep[1]});
  SmallVector<int, 2> spw({aSpw[0], bSpw[1]});
  SmallVector<unsigned, 2> shapePerCTA({spw[0] * wpt[0], spw[1] * wpt[1]});

  Value lane = urem(thread, _32);
  Value warp = udiv(thread, _32);

  Value warp0 = urem(warp, i32_val(wpt[0]));
  Value warp12 = udiv(warp, i32_val(wpt[0]));
  Value warp1 = urem(warp12, i32_val(wpt[1]));

  // warp offset
  Value offWarpM = mul(warp0, i32_val(spw[0]));
  Value offWarpN = mul(warp1, i32_val(spw[1]));
  // quad offset
  Value offQuadM = mul(udiv(and_(lane, _16), _4), _fpw0);
  Value offQuadN = mul(udiv(and_(lane, _16), _4), _fpw1);
  // pair offset
  Value offPairM = udiv(urem(lane, _16), _4);
  offPairM = urem(offPairM, _fpw0);
  offPairM = mul(offPairM, _4);
  Value offPairN = udiv(urem(lane, _16), _4);
  offPairN = udiv(offPairN, _fpw0);
  offPairN = urem(offPairN, _fpw1);
  offPairN = mul(offPairN, _4);

  // sclare
  offPairM = mul(offPairM, i32_val(rep[0] / 2));
  offQuadM = mul(offQuadM, i32_val(rep[0] / 2));
  offPairN = mul(offPairN, i32_val(rep[1] / 2));
  offQuadN = mul(offQuadN, i32_val(rep[1] / 2));

  // quad pair offset
  Value offLaneM = add(offPairM, offQuadM);
  Value offLaneN = add(offPairN, offQuadN);
  // a, b offset
  Value offsetAM = add(offWarpM, offLaneM);
  Value offsetBN = add(offWarpN, offLaneN);
  // m indices
  Value offsetCM = add(and_(lane, _1), offsetAM);
  SmallVector<Value> idxM;
  for (unsigned m = 0; m < shape[0]; m += shapePerCTA[0])
    for (unsigned mm = 0; mm < rep[0]; ++mm)
      idxM.push_back(add(offsetCM, i32_val(m + mm * 2)));

  // n indices
  Value offsetCN = add((and_(lane, _2)), (add(offWarpN, offPairN)));
  SmallVector<Value> idxN;
  for (int n = 0; n < shape[1]; n += shapePerCTA[1]) {
    for (int nn = 0; nn < rep[1]; ++nn) {
      idxN.push_back(add(
          offsetCN, i32_val(n + nn / 2 * 4 + (nn % 2) * 2 * fpw[1] * rep[1])));
      idxN.push_back(
          add(offsetCN,
              i32_val(n + nn / 2 * 4 + (nn % 2) * 2 * fpw[1] * rep[1] + 1)));
    }
  }

  SmallVector<SmallVector<Value>> axes({idxM, idxN});

  // product the axis M and axis N to get coords, ported from
  // generator::init_idx method from triton2.0

  // TODO[Superjomn]: check the order.
  SmallVector<CoordTy> coords;
  for (Value x1 : axes[1]) {   // N
    for (Value x0 : axes[0]) { // M
      SmallVector<Value, 2> idx(2);
      idx[0] = x0; // M
      idx[1] = x1; // N
      coords.push_back(std::move(idx));
    }
  }

  return coords; // {M,N} in row-major
}

Value convertLayout(int opIdx, Value tensor, const SharedMemoryObject &smemObj,
                    Value thread, Location loc,
                    TritonGPUToLLVMTypeConverter *typeConverter,
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
