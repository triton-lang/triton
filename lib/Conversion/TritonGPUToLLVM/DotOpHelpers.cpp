#include "DotOpHelpers.h"
#include "TypeConverter.h"

namespace mlir {
namespace LLVM {

Value DotOpMmaV1ConversionHelper::loadA(
    Value tensor, const SharedMemoryObject &smemObj, Value thread, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, Type resultTy) const {
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
      resultEncoding.getMMAv1Rep(), rewriter, loc);

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

Value DotOpMmaV1ConversionHelper::loadB(
    Value tensor, const SharedMemoryObject &smemObj, Value thread, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, Type resultTy) const {
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
      resultEncoding.getMMAv1Rep(), rewriter, loc);

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

std::tuple<Value, Value, Value, Value>
DotOpMmaV1ConversionHelper::computeOffsets(Value threadId, bool isARow,
                                           bool isBRow, ArrayRef<int> fpw,
                                           ArrayRef<int> spw, ArrayRef<int> rep,
                                           ConversionPatternRewriter &rewriter,
                                           Location loc) const {
  auto *ctx = rewriter.getContext();
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

DotOpMmaV1ConversionHelper::ValueTable
DotOpMmaV1ConversionHelper::extractLoadedOperand(
    Value llStruct, int NK, ConversionPatternRewriter &rewriter,
    TritonGPUToLLVMTypeConverter *typeConverter, Type type) const {
  ValueTable rcds;
  SmallVector<Value> elems = typeConverter->unpackLLElements(
      llStruct.getLoc(), llStruct, rewriter, type);

  int offset = 0;
  for (int i = 0; offset < elems.size(); ++i) {
    for (int k = 0; k < NK; k += 4) {
      rcds[{i, k}] = std::make_pair(elems[offset], elems[offset + 1]);
      offset += 2;
    }
  }

  return rcds;
}

// TODO: Mostly a duplicate of TritonGPUToLLVMBase::emitBaseIndexforMMaLayoutV1
SmallVector<DotOpMmaV1ConversionHelper::CoordTy>
DotOpMmaV1ConversionHelper::getMNCoords(Value thread,
                                        ConversionPatternRewriter &rewriter,
                                        ArrayRef<unsigned int> wpt,
                                        const MmaEncodingAttr &mmaLayout,
                                        ArrayRef<int64_t> shape, bool isARow,
                                        bool isBRow, bool isAVec4,
                                        bool isBVec4) {

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

std::tuple<int, int>
DotOpMmaV2ConversionHelper::getRepMN(const RankedTensorType &tensorTy) {
  auto mmaLayout = tensorTy.getEncoding().cast<MmaEncodingAttr>();
  auto wpt = mmaLayout.getWarpsPerCTA();

  int M = tensorTy.getShape()[0];
  int N = tensorTy.getShape()[1];
  auto [instrM, instrN] = getInstrShapeMN();
  int repM = std::max<int>(M / (wpt[0] * instrM), 1);
  int repN = std::max<int>(N / (wpt[1] * instrN), 1);
  return {repM, repN};
}

Type DotOpMmaV2ConversionHelper::getShemPtrTy() const {
  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return ptr_ty(type::f16Ty(ctx), 3);
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return ptr_ty(type::i16Ty(ctx), 3);
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return ptr_ty(type::f32Ty(ctx), 3);
  case TensorCoreType::FP16_FP16_FP16_FP16:
    return ptr_ty(type::f16Ty(ctx), 3);
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return ptr_ty(type::i8Ty(ctx), 3);
  default:
    llvm::report_fatal_error("mma16816 data type not supported");
  }
  return Type{};
}

Type DotOpMmaV2ConversionHelper::getMatType() const {
  // floating point types
  Type fp32x1Ty = vec_ty(type::f32Ty(ctx), 1);
  Type fp16x2Ty = vec_ty(type::f16Ty(ctx), 2);
  Type i16x2Ty = vec_ty(type::i16Ty(ctx), 2);
  Type fp16x2Pack4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp16x2Ty));
  // LLVM 14.0 does not support bf16 type, so we use i16 instead.
  Type bf16x2Pack4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, i16x2Ty));
  Type fp32Pack4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp32x1Ty));
  // integer types
  Type i8x4Ty = vec_ty(type::i8Ty(ctx), 4);
  Type i8x4Pack4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, i8x4Ty));

  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return fp16x2Pack4Ty;
  case TensorCoreType::FP16_FP16_FP16_FP16:
    return fp16x2Pack4Ty;
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return bf16x2Pack4Ty;
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return fp32Pack4Ty;
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return i8x4Pack4Ty;
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }

  return Type{};
}

Type DotOpMmaV2ConversionHelper::getLoadElemTy() {
  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return vec_ty(type::f16Ty(ctx), 2);
  case TensorCoreType::FP16_FP16_FP16_FP16:
    return vec_ty(type::f16Ty(ctx), 2);
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return vec_ty(type::bf16Ty(ctx), 2);
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return type::f32Ty(ctx);
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return type::i32Ty(ctx);
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }

  return Type{};
}

Type DotOpMmaV2ConversionHelper::getMmaRetType() const {
  Type fp32Ty = type::f32Ty(ctx);
  Type fp16Ty = type::f16Ty(ctx);
  Type i32Ty = type::i32Ty(ctx);
  Type fp32x4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, fp32Ty));
  Type i32x4Ty =
      LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, i32Ty));
  Type fp16x2Pack2Ty = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(2, vec_ty(fp16Ty, 2)));
  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP16_FP16_FP16_FP16:
    return fp16x2Pack2Ty;
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return fp32x4Ty;
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return i32x4Ty;
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }

  return Type{};
}

int DotOpMmaV2ConversionHelper::getMmaRetSize() const {
  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return 4;
  case TensorCoreType::FP16_FP16_FP16_FP16:
    return 2;
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return 4;
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return 4;
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return 4;
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }

  return 4;
}

DotOpMmaV2ConversionHelper::TensorCoreType
DotOpMmaV2ConversionHelper::getTensorCoreTypeFromOperand(Type operandTy) {
  auto tensorTy = operandTy.cast<RankedTensorType>();
  auto elemTy = tensorTy.getElementType();
  if (elemTy.isF16())
    return TensorCoreType::FP32_FP16_FP16_FP32;
  if (elemTy.isF32())
    return TensorCoreType::FP32_TF32_TF32_FP32;
  if (elemTy.isBF16())
    return TensorCoreType::FP32_BF16_BF16_FP32;
  if (elemTy.isInteger(8))
    return TensorCoreType::INT32_INT8_INT8_INT32;
  return TensorCoreType::NOT_APPLICABLE;
}

DotOpMmaV2ConversionHelper::TensorCoreType
DotOpMmaV2ConversionHelper::getMmaType(triton::DotOp op) {
  Value A = op.getA();
  Value B = op.getB();
  auto aTy = A.getType().cast<RankedTensorType>();
  auto bTy = B.getType().cast<RankedTensorType>();
  // d = a*b + c
  auto dTy = op.getD().getType().cast<RankedTensorType>();

  if (dTy.getElementType().isF32()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP32_FP16_FP16_FP32;
    if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
      return TensorCoreType::FP32_BF16_BF16_FP32;
    if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
        op.getAllowTF32())
      return TensorCoreType::FP32_TF32_TF32_FP32;
  } else if (dTy.getElementType().isInteger(32)) {
    if (aTy.getElementType().isInteger(8) && bTy.getElementType().isInteger(8))
      return TensorCoreType::INT32_INT8_INT8_INT32;
  } else if (dTy.getElementType().isF16()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP16_FP16_FP16_FP16;
  }

  return TensorCoreType::NOT_APPLICABLE;
}

SmallVector<Value>
MMA16816SmemLoader::computeLdmatrixMatOffs(Value warpId, Value lane,
                                           Value cSwizzleOffset) {
  // 4x4 matrices
  Value c = urem(lane, i32_val(8));
  Value s = udiv(lane, i32_val(8)); // sub-warp-id

  // Decompose s => s_0, s_1, that is the coordinate in 2x2 matrices in a
  // warp
  Value s0 = urem(s, i32_val(2));
  Value s1 = udiv(s, i32_val(2));

  // We use different orders for a and b for better performance.
  Value kMatArr = kOrder == 1 ? s1 : s0;
  Value nkMatArr = kOrder == 1 ? s0 : s1;

  // matrix coordinate inside a CTA, the matrix layout is [2x2wpt] for A and
  // [2wptx2] for B. e.g. Setting wpt=3, The data layout for A(kOrder=1) is
  //   |0 0 1 1 2 2| -> 0,1,2 are the warpids
  //   |0 0 1 1 2 2|
  //
  // for B(kOrder=0) is
  //   |0 0|  -> 0,1,2 are the warpids
  //   |1 1|
  //   |2 2|
  //   |0 0|
  //   |1 1|
  //   |2 2|
  // Note, for each warp, it handles a 2x2 matrices, that is the coordinate
  // address (s0,s1) annotates.

  Value matOff[2];
  matOff[kOrder ^ 1] =
      add(mul(warpId, i32_val(warpOffStride)),   // warp offset
          mul(nkMatArr, i32_val(matArrStride))); // matrix offset inside a warp
  matOff[kOrder] = kMatArr;

  // Physical offset (before swizzling)
  Value cMatOff = matOff[order[0]];
  Value sMatOff = matOff[order[1]];
  Value cSwizzleMatOff = udiv(cSwizzleOffset, i32_val(cMatShape));
  cMatOff = add(cMatOff, cSwizzleMatOff);

  // row offset inside a matrix, each matrix has 8 rows.
  Value sOffInMat = c;

  SmallVector<Value> offs(numPtrs);
  Value phase = urem(udiv(sOffInMat, i32_val(perPhase)), i32_val(maxPhase));
  Value sOff = add(sOffInMat, mul(sMatOff, i32_val(sMatShape)));
  for (int i = 0; i < numPtrs; ++i) {
    Value cMatOffI = add(cMatOff, i32_val(i * pLoadStrideInMat));
    cMatOffI = xor_(cMatOffI, phase);
    offs[i] = add(mul(cMatOffI, i32_val(cMatShape)), mul(sOff, sStride));
  }

  return offs;
}

SmallVector<Value> MMA16816SmemLoader::computeB32MatOffs(Value warpOff,
                                                         Value lane,
                                                         Value cSwizzleOffset) {
  assert(needTrans && "Only used in transpose mode.");
  // Load tf32 matrices with lds32
  Value cOffInMat = udiv(lane, i32_val(4));
  Value sOffInMat = urem(lane, i32_val(4));

  Value phase = urem(udiv(sOffInMat, i32_val(perPhase)), i32_val(maxPhase));
  SmallVector<Value> offs(numPtrs);

  for (int mat = 0; mat < 4; ++mat) { // Load 4 mats each time
    int kMatArrInt = kOrder == 1 ? mat / 2 : mat % 2;
    int nkMatArrInt = kOrder == 1 ? mat % 2 : mat / 2;
    if (kMatArrInt > 0) // we don't need pointers for k
      continue;
    Value kMatArr = i32_val(kMatArrInt);
    Value nkMatArr = i32_val(nkMatArrInt);

    Value cMatOff = add(mul(warpOff, i32_val(warpOffStride)),
                        mul(nkMatArr, i32_val(matArrStride)));
    Value cSwizzleMatOff = udiv(cSwizzleOffset, i32_val(cMatShape));
    cMatOff = add(cMatOff, cSwizzleMatOff);

    Value sMatOff = kMatArr;
    Value sOff = add(sOffInMat, mul(sMatOff, i32_val(sMatShape)));
    // FIXME: (kOrder == 1?) is really dirty hack
    for (int i = 0; i < numPtrs / 2; ++i) {
      Value cMatOffI =
          add(cMatOff, i32_val(i * pLoadStrideInMat * (kOrder == 1 ? 1 : 2)));
      cMatOffI = xor_(cMatOffI, phase);
      Value cOff = add(cOffInMat, mul(cMatOffI, i32_val(cMatShape)));
      cOff = urem(cOff, i32_val(tileShape[order[0]]));
      sOff = urem(sOff, i32_val(tileShape[order[1]]));
      offs[2 * i + nkMatArrInt] = add(cOff, mul(sOff, sStride));
    }
  }
  return offs;
}

SmallVector<Value> MMA16816SmemLoader::computeB8MatOffs(Value warpOff,
                                                        Value lane,
                                                        Value cSwizzleOffset) {
  assert(needTrans && "Only used in transpose mode.");
  Value cOffInMat = udiv(lane, i32_val(4));
  Value sOffInMat =
      mul(urem(lane, i32_val(4)), i32_val(4)); // each thread load 4 cols

  SmallVector<Value> offs(numPtrs);
  for (int mat = 0; mat < 4; ++mat) {
    int kMatArrInt = kOrder == 1 ? mat / 2 : mat % 2;
    int nkMatArrInt = kOrder == 1 ? mat % 2 : mat / 2;
    if (kMatArrInt > 0) // we don't need pointers for k
      continue;
    Value kMatArr = i32_val(kMatArrInt);
    Value nkMatArr = i32_val(nkMatArrInt);

    Value cMatOff = add(mul(warpOff, i32_val(warpOffStride)),
                        mul(nkMatArr, i32_val(matArrStride)));
    Value sMatOff = kMatArr;

    for (int loadx4Off = 0; loadx4Off < numPtrs / 8; ++loadx4Off) {
      for (int elemOff = 0; elemOff < 4; ++elemOff) {
        int ptrOff = loadx4Off * 8 + nkMatArrInt * 4 + elemOff;
        Value cMatOffI = add(cMatOff, i32_val(loadx4Off * pLoadStrideInMat *
                                              (kOrder == 1 ? 1 : 2)));
        Value sOffInMatElem = add(sOffInMat, i32_val(elemOff));

        // disable swizzling ...

        Value cOff = add(cOffInMat, mul(cMatOffI, i32_val(cMatShape)));
        Value sOff = add(sOffInMatElem, mul(sMatOff, i32_val(sMatShape)));
        // To prevent out-of-bound access when tile is too small.
        cOff = urem(cOff, i32_val(tileShape[order[0]]));
        sOff = urem(sOff, i32_val(tileShape[order[1]]));
        offs[ptrOff] = add(cOff, mul(sOff, sStride));
      }
    }
  }
  return offs;
}

std::tuple<Value, Value, Value, Value>
MMA16816SmemLoader::loadX4(int mat0, int mat1, ArrayRef<Value> offs,
                           ArrayRef<Value> ptrs, Type matTy,
                           Type shemPtrTy) const {
  assert(mat0 % 2 == 0 && mat1 % 2 == 0 && "smem matrix load must be aligned");
  int matIdx[2] = {mat0, mat1};

  int ptrIdx{-1};

  if (canUseLdmatrix)
    ptrIdx = matIdx[order[0]] / (instrShape[order[0]] / matShape[order[0]]);
  else if (elemBytes == 4 && needTrans)
    ptrIdx = matIdx[order[0]];
  else if (elemBytes == 1 && needTrans)
    ptrIdx = matIdx[order[0]] * 4;
  else
    llvm::report_fatal_error("unsupported mma type found");

  // The main difference with the original triton code is we removed the
  // prefetch-related logic here for the upstream optimizer phase should
  // take care with it, and that is transparent in dot conversion.
  auto getPtr = [&](int idx) { return ptrs[idx]; };

  Value ptr = getPtr(ptrIdx);

  // The struct should have exactly the same element types.
  auto resTy = matTy.cast<LLVM::LLVMStructType>();
  Type elemTy = matTy.cast<LLVM::LLVMStructType>().getBody()[0];

  // For some reasons, LLVM's NVPTX backend inserts unnecessary (?) integer
  // instructions to pack & unpack sub-word integers. A workaround is to
  // store the results of ldmatrix in i32
  if (auto vecElemTy = elemTy.dyn_cast<VectorType>()) {
    Type elemElemTy = vecElemTy.getElementType();
    if (auto intTy = elemElemTy.dyn_cast<IntegerType>()) {
      if (intTy.getWidth() <= 16) {
        elemTy = rewriter.getI32Type();
        resTy =
            LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, elemTy));
      }
    }
  }

  if (canUseLdmatrix) {
    Value sOffset =
        mul(i32_val(matIdx[order[1]] * sMatStride * sMatShape), sStride);
    Value sOffsetPtr = gep(shemPtrTy, ptr, sOffset);

    PTXBuilder builder;
    // ldmatrix.m8n8.x4 returns 4x2xfp16(that is 4xb32) elements for a
    // thread.
    auto resArgs = builder.newListOperand(4, "=r");
    auto addrArg = builder.newAddrOperand(sOffsetPtr, "r");

    auto ldmatrix = builder.create("ldmatrix.sync.aligned.m8n8.x4")
                        ->o("trans", needTrans /*predicate*/)
                        .o("shared.b16");
    ldmatrix(resArgs, addrArg);

    // The result type is 4xi32, each i32 is composed of 2xf16
    // elements (adjacent two columns in a row) or a single f32 element.
    Value resV4 = builder.launch(rewriter, loc, resTy);
    return {extract_val(elemTy, resV4, 0), extract_val(elemTy, resV4, 1),
            extract_val(elemTy, resV4, 2), extract_val(elemTy, resV4, 3)};
  } else if (elemBytes == 4 && needTrans) { // Use lds.32 to load tf32 matrices
    Value ptr2 = getPtr(ptrIdx + 1);
    assert(sMatStride == 1);
    int sOffsetElem = matIdx[order[1]] * (sMatStride * sMatShape);
    Value sOffsetElemVal = mul(i32_val(sOffsetElem), sStride);
    int sOffsetArrElem = sMatStride * sMatShape;
    Value sOffsetArrElemVal =
        add(sOffsetElemVal, mul(i32_val(sOffsetArrElem), sStride));

    Value elems[4];
    if (kOrder == 1) {
      elems[0] = load(gep(shemPtrTy, ptr, sOffsetElemVal));
      elems[1] = load(gep(shemPtrTy, ptr2, sOffsetElemVal));
      elems[2] = load(gep(shemPtrTy, ptr, sOffsetArrElemVal));
      elems[3] = load(gep(shemPtrTy, ptr2, sOffsetArrElemVal));
    } else {
      elems[0] = load(gep(shemPtrTy, ptr, sOffsetElemVal));
      elems[2] = load(gep(shemPtrTy, ptr2, sOffsetElemVal));
      elems[1] = load(gep(shemPtrTy, ptr, sOffsetArrElemVal));
      elems[3] = load(gep(shemPtrTy, ptr2, sOffsetArrElemVal));
    }
    std::array<Value, 4> retElems;
    retElems.fill(undef(elemTy));
    for (auto i = 0; i < 4; ++i) {
      retElems[i] = insert_element(elemTy, retElems[i], elems[i], i32_val(0));
    }
    return {retElems[0], retElems[1], retElems[2], retElems[3]};
  } else if (elemBytes == 1 && needTrans) { // work with int8
    // Can't use i32 here. Use LLVM's VectorType
    elemTy = matTy.cast<LLVM::LLVMStructType>().getBody()[0];
    std::array<std::array<Value, 4>, 2> ptrs;
    ptrs[0] = {
        getPtr(ptrIdx),
        getPtr(ptrIdx + 1),
        getPtr(ptrIdx + 2),
        getPtr(ptrIdx + 3),
    };

    ptrs[1] = {
        getPtr(ptrIdx + 4),
        getPtr(ptrIdx + 5),
        getPtr(ptrIdx + 6),
        getPtr(ptrIdx + 7),
    };

    assert(sMatStride == 1);
    int sOffsetElem = matIdx[order[1]] * (sMatStride * sMatShape);
    Value sOffsetElemVal = mul(i32_val(sOffsetElem), sStride);
    int sOffsetArrElem = 1 * (sMatStride * sMatShape);
    Value sOffsetArrElemVal =
        add(sOffsetElemVal, mul(i32_val(sOffsetArrElem), sStride));

    std::array<Value, 4> i8v4Elems;
    i8v4Elems.fill(undef(elemTy));

    Value i8Elems[4][4];
    if (kOrder == 1) {
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
          i8Elems[i][j] = load(gep(shemPtrTy, ptrs[i][j], sOffsetElemVal));

      for (int i = 2; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          i8Elems[i][j] =
              load(gep(shemPtrTy, ptrs[i - 2][j], sOffsetArrElemVal));

      for (int m = 0; m < 4; ++m) {
        for (int e = 0; e < 4; ++e)
          i8v4Elems[m] = insert_element(i8v4Elems[m].getType(), i8v4Elems[m],
                                        i8Elems[m][e], i32_val(e));
      }
    } else { // k first
      for (int j = 0; j < 4; ++j)
        i8Elems[0][j] = load(gep(shemPtrTy, ptrs[0][j], sOffsetElemVal));
      for (int j = 0; j < 4; ++j)
        i8Elems[2][j] = load(gep(shemPtrTy, ptrs[1][j], sOffsetElemVal));
      for (int j = 0; j < 4; ++j)
        i8Elems[1][j] = load(gep(shemPtrTy, ptrs[0][j], sOffsetArrElemVal));
      for (int j = 0; j < 4; ++j)
        i8Elems[3][j] = load(gep(shemPtrTy, ptrs[1][j], sOffsetArrElemVal));

      for (int m = 0; m < 4; ++m) {
        for (int e = 0; e < 4; ++e)
          i8v4Elems[m] = insert_element(i8v4Elems[m].getType(), i8v4Elems[m],
                                        i8Elems[m][e], i32_val(e));
      }
    }

    return {bitcast(i8v4Elems[0], i32_ty), bitcast(i8v4Elems[1], i32_ty),
            bitcast(i8v4Elems[2], i32_ty), bitcast(i8v4Elems[3], i32_ty)};
  }

  assert(false && "Invalid smem load");
  return {Value{}, Value{}, Value{}, Value{}};
}

MMA16816SmemLoader::MMA16816SmemLoader(
    int wpt, ArrayRef<uint32_t> order, uint32_t kOrder,
    ArrayRef<Value> smemStrides, ArrayRef<int64_t> tileShape,
    ArrayRef<int> instrShape, ArrayRef<int> matShape, int perPhase,
    int maxPhase, int elemBytes, ConversionPatternRewriter &rewriter,
    TypeConverter *typeConverter, const Location &loc)
    : order(order.begin(), order.end()), kOrder(kOrder),
      tileShape(tileShape.begin(), tileShape.end()),
      instrShape(instrShape.begin(), instrShape.end()),
      matShape(matShape.begin(), matShape.end()), perPhase(perPhase),
      maxPhase(maxPhase), elemBytes(elemBytes), rewriter(rewriter), loc(loc),
      ctx(rewriter.getContext()) {
  cMatShape = matShape[order[0]];
  sMatShape = matShape[order[1]];

  sStride = smemStrides[order[1]];

  // rule: k must be the fast-changing axis.
  needTrans = kOrder != order[0];
  canUseLdmatrix = elemBytes == 2 || (!needTrans); // b16

  if (canUseLdmatrix) {
    // Each CTA, the warps is arranged as [1xwpt] if not transposed,
    // otherwise [wptx1], and each warp will perform a mma.
    numPtrs =
        tileShape[order[0]] / (needTrans ? wpt : 1) / instrShape[order[0]];
  } else {
    numPtrs = tileShape[order[0]] / wpt / matShape[order[0]];
  }
  numPtrs = std::max<int>(numPtrs, 2);

  // Special rule for i8/u8, 4 ptrs for each matrix
  if (!canUseLdmatrix && elemBytes == 1)
    numPtrs *= 4;

  int loadStrideInMat[2];
  loadStrideInMat[kOrder] =
      2; // instrShape[kOrder] / matShape[kOrder], always 2
  loadStrideInMat[kOrder ^ 1] =
      wpt * (instrShape[kOrder ^ 1] / matShape[kOrder ^ 1]);

  pLoadStrideInMat = loadStrideInMat[order[0]];
  sMatStride =
      loadStrideInMat[order[1]] / (instrShape[order[1]] / matShape[order[1]]);

  // Each matArr contains warpOffStride matrices.
  matArrStride = kOrder == 1 ? 1 : wpt;
  warpOffStride = instrShape[kOrder ^ 1] / matShape[kOrder ^ 1];
}
Value MMA16816ConversionHelper::loadA(Value tensor,
                                      const SharedMemoryObject &smemObj) const {
  auto aTensorTy = tensor.getType().cast<RankedTensorType>();

  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());

  ValueTable ha;
  std::function<void(int, int)> loadFn;
  auto [matShapeM, matShapeN, matShapeK] = getMmaMatShape(aTensorTy);
  auto [mmaInstrM, mmaInstrN, mmaInstrK] = getMmaInstrShape(aTensorTy);

  int numRepM = getNumRepM(aTensorTy, shape[0]);
  int numRepK = getNumRepK(aTensorTy, shape[1]);

  if (aTensorTy.getEncoding().isa<SharedEncodingAttr>()) {
    Value warpM = getWarpM(shape[0]);
    // load from smem
    // we use ldmatrix.x4 so each warp processes 16x16 elements.
    int wpt = std::min<int>(mmaLayout.getWarpsPerCTA()[0], shape[0] / 16);
    loadFn =
        getLoadMatrixFn(tensor, smemObj, mmaLayout, wpt /*wpt*/, 1 /*kOrder*/,
                        {mmaInstrM, mmaInstrK} /*instrShape*/,
                        {matShapeM, matShapeK} /*matShape*/, warpM /*warpId*/,
                        ha /*vals*/, true /*isA*/);
  } else if (aTensorTy.getEncoding().isa<BlockedEncodingAttr>()) {
    // load from registers, used in gemm fuse
    // TODO(Superjomn) Port the logic.
    assert(false && "Loading A from register is not supported yet.");
  } else {
    assert(false && "A's layout is not supported.");
  }

  // step1. Perform loading.
  for (int m = 0; m < numRepM; ++m)
    for (int k = 0; k < numRepK; ++k)
      loadFn(2 * m, 2 * k);

  // step2. Format the values to LLVM::Struct to passing to mma codegen.
  return composeValuesToDotOperandLayoutStruct(ha, numRepM, numRepK);
}
Value MMA16816ConversionHelper::loadB(Value tensor,
                                      const SharedMemoryObject &smemObj) {
  ValueTable hb;
  auto tensorTy = tensor.getType().cast<RankedTensorType>();

  SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                             tensorTy.getShape().end());

  // TODO[Superjomn]: transB cannot be accessed in ConvertLayoutOp.
  bool transB = false;
  if (transB) {
    std::swap(shape[0], shape[1]);
  }

  auto [matShapeM, matShapeN, matShapeK] = getMmaMatShape(tensorTy);
  auto [mmaInstrM, mmaInstrN, mmaInstrK] = getMmaInstrShape(tensorTy);
  int numRepK = getNumRepK(tensorTy, shape[0]);
  int numRepN = getNumRepN(tensorTy, shape[1]);

  Value warpN = getWarpN(shape[1]);
  // we use ldmatrix.x4 so each warp processes 16x16 elements.
  int wpt = std::min<int>(mmaLayout.getWarpsPerCTA()[1], shape[1] / 16);
  auto loadFn =
      getLoadMatrixFn(tensor, smemObj, mmaLayout, wpt /*wpt*/, 0 /*kOrder*/,
                      {mmaInstrK, mmaInstrN} /*instrShape*/,
                      {matShapeK, matShapeN} /*matShape*/, warpN /*warpId*/,
                      hb /*vals*/, false /*isA*/);

  for (int n = 0; n < std::max(numRepN / 2, 1); ++n) {
    for (int k = 0; k < numRepK; ++k)
      loadFn(2 * n, 2 * k);
  }

  Value result = composeValuesToDotOperandLayoutStruct(
      hb, std::max(numRepN / 2, 1), numRepK);
  return result;
}
Value MMA16816ConversionHelper::loadC(Value tensor, Value llTensor) const {
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  auto [repM, repN] = DotOpMmaV2ConversionHelper::getRepMN(tensorTy);
  size_t fcSize = 4 * repM * repN;

  assert(tensorTy.getEncoding().isa<MmaEncodingAttr>() &&
         "Currently, we only support $c with a mma layout.");
  // Load a normal C tensor with mma layout, that should be a
  // LLVM::struct with fcSize elements.
  auto structTy = llTensor.getType().cast<LLVM::LLVMStructType>();
  assert(structTy.getBody().size() == fcSize &&
         "DotOp's $c operand should pass the same number of values as $d in "
         "mma layout.");

  auto numMmaRets = helper.getMmaRetSize();
  assert(numMmaRets == 4 || numMmaRets == 2);
  if (numMmaRets == 4) {
    return llTensor;
  } else if (numMmaRets == 2) {
    auto cPack = SmallVector<Value>();
    auto cElemTy = tensorTy.getElementType();
    int numCPackedElem = 4 / numMmaRets;
    Type cPackTy = vec_ty(cElemTy, numCPackedElem);
    for (int i = 0; i < fcSize; i += numCPackedElem) {
      Value pack = rewriter.create<LLVM::UndefOp>(loc, cPackTy);
      for (int j = 0; j < numCPackedElem; ++j) {
        pack = insert_element(
            cPackTy, pack, extract_val(cElemTy, llTensor, i + j), i32_val(j));
      }
      cPack.push_back(pack);
    }

    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(cPack.size(), cPackTy));
    Value result =
        typeConverter->packLLElements(loc, cPack, rewriter, structTy);
    return result;
  }
  return llTensor;
}
LogicalResult MMA16816ConversionHelper::convertDot(Value a, Value b, Value c,
                                                   Value d, Value loadedA,
                                                   Value loadedB, Value loadedC,
                                                   DotOp op,
                                                   DotOpAdaptor adaptor) const {
  helper.deduceMmaType(op);

  auto aTensorTy = a.getType().cast<RankedTensorType>();
  auto bTensorTy = b.getType().cast<RankedTensorType>();
  auto dTensorTy = d.getType().cast<RankedTensorType>();

  SmallVector<int64_t> aShape(aTensorTy.getShape().begin(),
                              aTensorTy.getShape().end());

  auto dShape = dTensorTy.getShape();

  // shape / shape_per_cta
  int numRepM = getNumRepM(aTensorTy, dShape[0]);
  int numRepN = getNumRepN(aTensorTy, dShape[1]);
  int numRepK = getNumRepK(aTensorTy, aShape[1]);

  ValueTable ha =
      getValuesFromDotOperandLayoutStruct(loadedA, numRepM, numRepK, aTensorTy);
  ValueTable hb = getValuesFromDotOperandLayoutStruct(
      loadedB, std::max(numRepN / 2, 1), numRepK, bTensorTy);
  auto fc = typeConverter->unpackLLElements(loc, loadedC, rewriter, dTensorTy);
  auto numMmaRets = helper.getMmaRetSize();
  int numCPackedElem = 4 / numMmaRets;

  auto callMma = [&](unsigned m, unsigned n, unsigned k) {
    unsigned colsPerThread = numRepN * 2;
    PTXBuilder builder;
    auto &mma = *builder.create(helper.getMmaInstr().str());
    // using =r for float32 works but leads to less readable ptx.
    bool isIntMMA = dTensorTy.getElementType().isInteger(32);
    bool isAccF16 = dTensorTy.getElementType().isF16();
    auto retArgs =
        builder.newListOperand(numMmaRets, isIntMMA || isAccF16 ? "=r" : "=f");
    auto aArgs = builder.newListOperand({
        {ha[{m, k}], "r"},
        {ha[{m + 1, k}], "r"},
        {ha[{m, k + 1}], "r"},
        {ha[{m + 1, k + 1}], "r"},
    });
    auto bArgs =
        builder.newListOperand({{hb[{n, k}], "r"}, {hb[{n, k + 1}], "r"}});
    auto cArgs = builder.newListOperand();
    for (int i = 0; i < numMmaRets; ++i) {
      cArgs->listAppend(builder.newOperand(
          fc[(m * colsPerThread + 4 * n) / numCPackedElem + i],
          std::to_string(i)));
      // reuse the output registers
    }

    mma(retArgs, aArgs, bArgs, cArgs);
    Value mmaOut = builder.launch(rewriter, loc, helper.getMmaRetType());

    Type elemTy = mmaOut.getType().cast<LLVM::LLVMStructType>().getBody()[0];
    for (int i = 0; i < numMmaRets; ++i) {
      fc[(m * colsPerThread + 4 * n) / numCPackedElem + i] =
          extract_val(elemTy, mmaOut, i);
    }
  };

  for (int k = 0; k < numRepK; ++k)
    for (int m = 0; m < numRepM; ++m)
      for (int n = 0; n < numRepN; ++n)
        callMma(2 * m, n, 2 * k);

  Type resElemTy = dTensorTy.getElementType();

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(fc.size() * numCPackedElem, resElemTy));
  SmallVector<Value> results(fc.size() * numCPackedElem);
  for (int i = 0; i < fc.size(); ++i) {
    for (int j = 0; j < numCPackedElem; ++j) {
      results[i * numCPackedElem + j] =
          numCPackedElem > 1
              ? bitcast(extract_element(fc[i], i32_val(j)), resElemTy)
              : bitcast(fc[i], resElemTy);
    }
  }
  Value res = typeConverter->packLLElements(loc, results, rewriter, structTy);
  rewriter.replaceOp(op, res);

  return success();
}
std::function<void(int, int)> MMA16816ConversionHelper::getLoadMatrixFn(
    Value tensor, const SharedMemoryObject &smemObj, MmaEncodingAttr mmaLayout,
    int wpt, uint32_t kOrder, SmallVector<int> instrShape,
    SmallVector<int> matShape, Value warpId,
    MMA16816ConversionHelper::ValueTable &vals, bool isA) const {
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  // We assumes that the input operand of Dot should be from shared layout.
  // TODO(Superjomn) Consider other layouts if needed later.
  auto sharedLayout = tensorTy.getEncoding().cast<SharedEncodingAttr>();
  const int perPhase = sharedLayout.getPerPhase();
  const int maxPhase = sharedLayout.getMaxPhase();
  const int elemBytes = tensorTy.getElementTypeBitWidth() / 8;
  auto order = sharedLayout.getOrder();

  // the original register_lds2, but discard the prefetch logic.
  auto ld2 = [](ValueTable &vals, int mn, int k, Value val) {
    vals[{mn, k}] = val;
  };

  // (a, b) is the coordinate.
  auto load = [=, &vals, &ld2](int a, int b) {
    MMA16816SmemLoader loader(
        wpt, sharedLayout.getOrder(), kOrder, smemObj.strides,
        tensorTy.getShape() /*tileShape*/, instrShape, matShape, perPhase,
        maxPhase, elemBytes, rewriter, typeConverter, loc);
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    SmallVector<Value> offs =
        loader.computeOffsets(warpId, lane, cSwizzleOffset);
    const int numPtrs = loader.getNumPtrs();
    SmallVector<Value> ptrs(numPtrs);

    Value smemBase = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);

    Type smemPtrTy = helper.getShemPtrTy();
    for (int i = 0; i < numPtrs; ++i) {
      ptrs[i] =
          bitcast(gep(smemPtrTy, smemBase, ValueRange({offs[i]})), smemPtrTy);
    }

    auto [ha0, ha1, ha2, ha3] = loader.loadX4(
        (kOrder == 1) ? a : b /*mat0*/, (kOrder == 1) ? b : a /*mat1*/, offs,
        ptrs, helper.getMatType(), helper.getShemPtrTy());

    if (isA) {
      ld2(vals, a, b, ha0);
      ld2(vals, a + 1, b, ha1);
      ld2(vals, a, b + 1, ha2);
      ld2(vals, a + 1, b + 1, ha3);
    } else {
      ld2(vals, a, b, ha0);
      ld2(vals, a + 1, b, ha2);
      ld2(vals, a, b + 1, ha1);
      ld2(vals, a + 1, b + 1, ha3);
    }
  };

  return load;
}
Value MMA16816ConversionHelper::composeValuesToDotOperandLayoutStruct(
    const MMA16816ConversionHelper::ValueTable &vals, int n0, int n1) const {
  std::vector<Value> elems;
  for (int m = 0; m < n0; ++m)
    for (int k = 0; k < n1; ++k) {
      elems.push_back(vals.at({2 * m, 2 * k}));
      elems.push_back(vals.at({2 * m, 2 * k + 1}));
      elems.push_back(vals.at({2 * m + 1, 2 * k}));
      elems.push_back(vals.at({2 * m + 1, 2 * k + 1}));
    }

  assert(!elems.empty());

  Type elemTy = elems[0].getType();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(elems.size(), elemTy));
  auto result = typeConverter->packLLElements(loc, elems, rewriter, structTy);
  return result;
}
MMA16816ConversionHelper::ValueTable
MMA16816ConversionHelper::getValuesFromDotOperandLayoutStruct(Value value,
                                                              int n0, int n1,
                                                              Type type) const {
  auto elems = typeConverter->unpackLLElements(loc, value, rewriter, type);

  int offset{};
  ValueTable vals;
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; j++) {
      vals[{2 * i, 2 * j}] = elems[offset++];
      vals[{2 * i, 2 * j + 1}] = elems[offset++];
      vals[{2 * i + 1, 2 * j}] = elems[offset++];
      vals[{2 * i + 1, 2 * j + 1}] = elems[offset++];
    }
  }
  return vals;
}
SmallVector<Value> DotOpFMAConversionHelper::getThreadIds(
    Value threadId, ArrayRef<unsigned int> shapePerCTA,
    ArrayRef<unsigned int> sizePerThread, ArrayRef<unsigned int> order,
    ConversionPatternRewriter &rewriter, Location loc) const {
  int dim = order.size();
  SmallVector<Value> threadIds(dim);
  for (unsigned k = 0; k < dim - 1; k++) {
    Value dimK = i32_val(shapePerCTA[order[k]] / sizePerThread[order[k]]);
    Value rem = urem(threadId, dimK);
    threadId = udiv(threadId, dimK);
    threadIds[order[k]] = rem;
  }
  Value dimK = i32_val(shapePerCTA[order[dim - 1]]);
  threadIds[order[dim - 1]] = urem(threadId, dimK);
  return threadIds;
}
Value DotOpFMAConversionHelper::loadA(
    Value A, Value llA, BlockedEncodingAttr dLayout, Value thread, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter) const {
  auto aTensorTy = A.getType().cast<RankedTensorType>();
  auto aLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto aShape = aTensorTy.getShape();

  auto aOrder = aLayout.getOrder();
  auto order = dLayout.getOrder();

  bool isARow = aOrder[0] == 1;

  auto aSmem = getSharedMemoryObjectFromStruct(loc, llA, rewriter);
  Value strideAM = aSmem.strides[0];
  Value strideAK = aSmem.strides[1];
  Value strideA0 = isARow ? strideAK : strideAM;
  Value strideA1 = isARow ? strideAM : strideAK;
  int aNumPtr = 8;
  int K = aShape[1];
  int M = aShape[0];

  auto shapePerCTA = getShapePerCTA(dLayout);
  auto sizePerThread = getSizePerThread(dLayout);

  Value _0 = i32_val(0);

  Value mContig = i32_val(sizePerThread[order[1]]);

  // threadId in blocked layout
  auto threadIds =
      getThreadIds(thread, shapePerCTA, sizePerThread, order, rewriter, loc);
  Value threadIdM = threadIds[0];

  Value offA0 = isARow ? _0 : mul(threadIdM, mContig);
  Value offA1 = isARow ? mul(threadIdM, mContig) : _0;
  SmallVector<Value> aOff(aNumPtr);
  for (int i = 0; i < aNumPtr; ++i) {
    aOff[i] = add(mul(offA0, strideA0), mul(offA1, strideA1));
  }
  auto elemTy = A.getType().cast<RankedTensorType>().getElementType();

  Type ptrTy = ptr_ty(elemTy, 3);
  SmallVector<Value> aPtrs(aNumPtr);
  for (int i = 0; i < aNumPtr; ++i)
    aPtrs[i] = gep(ptrTy, aSmem.base, aOff[i]);

  SmallVector<Value> vas;

  int mShapePerCTA = getShapePerCTAForMN(dLayout, true /*isM*/);
  int mSizePerThread = getSizePerThreadForMN(dLayout, true /*isM*/);

  for (unsigned k = 0; k < K; ++k)
    for (unsigned m = 0; m < M; m += mShapePerCTA)
      for (unsigned mm = 0; mm < mSizePerThread; ++mm) {
        Value offset =
            add(mul(i32_val(m + mm), strideAM), mul(i32_val(k), strideAK));
        Value pa = gep(ptrTy, aPtrs[0], offset);
        Value va = load(pa);
        vas.emplace_back(va);
      }

  return getStructFromValueTable(vas, rewriter, loc, typeConverter, elemTy);
}
Value DotOpFMAConversionHelper::loadB(
    Value B, Value llB, BlockedEncodingAttr dLayout, Value thread, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter) const {
  auto bTensorTy = B.getType().cast<RankedTensorType>();
  auto bLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto bShape = bTensorTy.getShape();

  auto bOrder = bLayout.getOrder();
  auto order = dLayout.getOrder();

  bool isBRow = bOrder[0] == 1;

  auto bSmem = getSharedMemoryObjectFromStruct(loc, llB, rewriter);
  Value strideBN = bSmem.strides[1];
  Value strideBK = bSmem.strides[0];
  Value strideB0 = isBRow ? strideBN : strideBK;
  Value strideB1 = isBRow ? strideBK : strideBN;
  int bNumPtr = 8;
  int K = bShape[0];
  int N = bShape[1];

  auto shapePerCTA = getShapePerCTA(dLayout);
  auto sizePerThread = getSizePerThread(dLayout);

  Value _0 = i32_val(0);

  Value nContig = i32_val(sizePerThread[order[0]]);

  // threadId in blocked layout
  auto threadIds =
      getThreadIds(thread, shapePerCTA, sizePerThread, order, rewriter, loc);
  Value threadIdN = threadIds[1];

  Value offB0 = isBRow ? mul(threadIdN, nContig) : _0;
  Value offB1 = isBRow ? _0 : mul(threadIdN, nContig);
  SmallVector<Value> bOff(bNumPtr);
  for (int i = 0; i < bNumPtr; ++i) {
    bOff[i] = add(mul(offB0, strideB0), mul(offB1, strideB1));
  }
  auto elemTy = B.getType().cast<RankedTensorType>().getElementType();

  Type ptrTy = ptr_ty(elemTy, 3);
  SmallVector<Value> bPtrs(bNumPtr);
  for (int i = 0; i < bNumPtr; ++i)
    bPtrs[i] = gep(ptrTy, bSmem.base, bOff[i]);

  SmallVector<Value> vbs;

  int nShapePerCTA = getShapePerCTAForMN(dLayout, false /*isM*/);
  int nSizePerThread = getSizePerThreadForMN(dLayout, false /*isM*/);

  for (unsigned k = 0; k < K; ++k)
    for (unsigned n = 0; n < N; n += nShapePerCTA)
      for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
        Value offset =
            add(mul(i32_val(n + nn), strideBN), mul(i32_val(k), strideBK));
        Value pb = gep(ptrTy, bPtrs[0], offset);
        Value vb = load(pb);
        vbs.emplace_back(vb);
      }

  return getStructFromValueTable(vbs, rewriter, loc, typeConverter, elemTy);
}
DotOpFMAConversionHelper::ValueTable
DotOpFMAConversionHelper::getValueTableFromStruct(
    Value val, int K, int n0, int shapePerCTA, int sizePerThread,
    ConversionPatternRewriter &rewriter, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter, Type type) const {
  ValueTable res;
  auto elems = typeConverter->unpackLLElements(loc, val, rewriter, type);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTA)
      for (unsigned mm = 0; mm < sizePerThread; ++mm) {
        res[{m + mm, k}] = elems[index++];
      }
  }
  return res;
}
Value DotOpFMAConversionHelper::getStructFromValueTable(
    ArrayRef<Value> vals, ConversionPatternRewriter &rewriter, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter, Type elemTy) const {
  SmallVector<Type> elemTypes(vals.size(), elemTy);
  SmallVector<Value> elems;
  elems.reserve(vals.size());
  for (auto &val : vals) {
    elems.push_back(val);
  }

  Type structTy = struct_ty(elemTypes);
  return typeConverter->packLLElements(loc, elems, rewriter, structTy);
}
int DotOpFMAConversionHelper::getNumElemsPerThread(
    ArrayRef<int64_t> shape, DotOperandEncodingAttr dotOpLayout) {
  auto blockedLayout = dotOpLayout.getParent().cast<BlockedEncodingAttr>();
  auto shapePerCTA = getShapePerCTA(blockedLayout);
  auto sizePerThread = getSizePerThread(blockedLayout);

  // TODO[Superjomn]: we assume the k axis is fixed for $a and $b here, fix it
  // if not.
  int K = dotOpLayout.getOpIdx() == 0 ? shape[1] : shape[0];
  int otherDim = dotOpLayout.getOpIdx() == 1 ? shape[1] : shape[0];

  bool isM = dotOpLayout.getOpIdx() == 0;
  int shapePerCTAMN = getShapePerCTAForMN(blockedLayout, isM);
  int sizePerThreadMN = getSizePerThreadForMN(blockedLayout, isM);
  return K * std::max<int>(otherDim / shapePerCTAMN, 1) * sizePerThreadMN;
}
} // namespace LLVM
} // namespace mlir
