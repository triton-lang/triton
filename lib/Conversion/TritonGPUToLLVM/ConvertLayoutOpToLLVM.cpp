#include "ConvertLayoutOpToLLVM.h"
#include "DotOpHelpers.h"
#include "Utility.h"

using ::mlir::LLVM::DotOpFMAConversionHelper;
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

// --- v1 --- //
using CoordTy = SmallVector<Value>;
using ValueTable = std::map<std::pair<int, int>, std::pair<Value, Value>>;

static SmallVector<CoordTy>
getMNCoords(Value thread, ConversionPatternRewriter &rewriter,
            ArrayRef<unsigned int> wpt, const MmaEncodingAttr &mmaLayout,
            ArrayRef<int64_t> shape, bool isARow, bool isBRow, bool isAVec4,
            bool isBVec4) {
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

// --- v2 ----

using ValueTableV2 = std::map<std::pair<unsigned, unsigned>, Value>;

// Data loader for mma.16816 instruction.
class MMA16816SmemLoader {
public:
  MMA16816SmemLoader(int wpt, ArrayRef<uint32_t> order, uint32_t kOrder,
                     ArrayRef<Value> smemStrides, ArrayRef<int64_t> tileShape,
                     ArrayRef<int> instrShape, ArrayRef<int> matShape,
                     int perPhase, int maxPhase, int elemBytes,
                     ConversionPatternRewriter &rewriter,
                     TritonGPUToLLVMTypeConverter *typeConverter,
                     const Location &loc);

  // lane = thread % 32
  // warpOff = (thread/32) % wpt(0)
  llvm::SmallVector<Value> computeOffsets(Value warpOff, Value lane,
                                          Value cSwizzleOffset) {
    if (canUseLdmatrix)
      return computeLdmatrixMatOffs(warpOff, lane, cSwizzleOffset);
    else if (elemBytes == 4 && needTrans)
      return computeB32MatOffs(warpOff, lane, cSwizzleOffset);
    else if (elemBytes == 1 && needTrans)
      return computeB8MatOffs(warpOff, lane, cSwizzleOffset);
    else
      llvm::report_fatal_error("Invalid smem load config");

    return {};
  }

  int getNumPtrs() const { return numPtrs; }

  // Compute the offset to the matrix this thread(indexed by warpOff and lane)
  // mapped to.
  SmallVector<Value> computeLdmatrixMatOffs(Value warpId, Value lane,
                                            Value cSwizzleOffset);

  // Compute 32-bit matrix offsets.
  SmallVector<Value> computeB32MatOffs(Value warpOff, Value lane,
                                       Value cSwizzleOffset);

  // compute 8-bit matrix offset.
  SmallVector<Value> computeB8MatOffs(Value warpOff, Value lane,
                                      Value cSwizzleOffset);

  // Load 4 matrices and returns 4 vec<2> elements.
  std::tuple<Value, Value, Value, Value>
  loadX4(int mat0, int mat1, ArrayRef<Value> offs, ArrayRef<Value> ptrs,
         Type matTy, Type shemPtrTy) const;

private:
  SmallVector<uint32_t> order;
  int kOrder;
  SmallVector<int64_t> tileShape;
  SmallVector<int> instrShape;
  SmallVector<int> matShape;
  int perPhase;
  int maxPhase;
  int elemBytes;
  ConversionPatternRewriter &rewriter;
  const Location &loc;
  MLIRContext *ctx{};

  int cMatShape;
  int sMatShape;

  Value sStride;

  bool needTrans;
  bool canUseLdmatrix;

  int numPtrs;

  int pLoadStrideInMat;
  int sMatStride;

  int matArrStride;
  int warpOffStride;
};

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
    TritonGPUToLLVMTypeConverter *typeConverter, const Location &loc)
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

Type getShemPtrTy(Type argType) {
  MLIRContext *ctx = argType.getContext();
  if (argType.isF16())
    return ptr_ty(type::f16Ty(ctx), 3);
  else if (argType.isBF16())
    return ptr_ty(type::i16Ty(ctx), 3);
  else if (argType.isF32())
    return ptr_ty(type::f32Ty(ctx), 3);
  else if (argType.isInteger(8))
    return ptr_ty(type::i8Ty(ctx), 3);
  else
    llvm::report_fatal_error("mma16816 data type not supported");
}

Type getMatType(Type argType) {
  MLIRContext *ctx = argType.getContext();
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

  if (argType.isF16())
    return fp16x2Pack4Ty;
  else if (argType.isBF16())
    return bf16x2Pack4Ty;
  else if (argType.isF32())
    return fp32Pack4Ty;
  else if (argType.isInteger(8))
    return i8x4Pack4Ty;
  else
    llvm::report_fatal_error("mma16816 data type not supported");
}

Value composeValuesToDotOperandLayoutStruct(
    const ValueTableV2 &vals, int n0, int n1,
    TritonGPUToLLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter) {
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
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(elems.size(), elemTy));
  auto result = typeConverter->packLLElements(loc, elems, rewriter, structTy);
  return result;
}

std::function<void(int, int)>
getLoadMatrixFn(Value tensor, const SharedMemoryObject &smemObj,
                MmaEncodingAttr mmaLayout, int wpt, uint32_t kOrder,
                SmallVector<int> instrShape, SmallVector<int> matShape,
                Value warpId, Value lane, ValueTableV2 &vals, bool isA,
                TritonGPUToLLVMTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, Location loc) {
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  Type eltTy = tensorTy.getElementType();
  // We assumes that the input operand of Dot should be from shared layout.
  // TODO(Superjomn) Consider other layouts if needed later.
  auto sharedLayout = tensorTy.getEncoding().cast<SharedEncodingAttr>();
  const int perPhase = sharedLayout.getPerPhase();
  const int maxPhase = sharedLayout.getMaxPhase();
  const int elemBytes = tensorTy.getElementTypeBitWidth() / 8;
  auto order = sharedLayout.getOrder();

  // the original register_lds2, but discard the prefetch logic.
  auto ld2 = [](ValueTableV2 &vals, int mn, int k, Value val) {
    vals[{mn, k}] = val;
  };

  // (a, b) is the coordinate.
  auto load = [=, &rewriter, &vals, &ld2](int a, int b) {
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

    Type smemPtrTy = getShemPtrTy(eltTy);
    for (int i = 0; i < numPtrs; ++i) {
      ptrs[i] =
          bitcast(gep(smemPtrTy, smemBase, ValueRange({offs[i]})), smemPtrTy);
    }

    auto [ha0, ha1, ha2, ha3] = loader.loadX4(
        (kOrder == 1) ? a : b /*mat0*/, (kOrder == 1) ? b : a /*mat1*/, offs,
        ptrs, getMatType(eltTy), getShemPtrTy(eltTy));

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

Value loadAv2(ConversionPatternRewriter &rewriter, Location loc, Value tensor,
              DotOperandEncodingAttr aEncoding,
              const SharedMemoryObject &smemObj,
              TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  int bitwidth = aTensorTy.getElementTypeBitWidth();
  auto mmaLayout = aEncoding.getParent().cast<MmaEncodingAttr>();

  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());

  ValueTableV2 ha;
  std::function<void(int, int)> loadFn;
  int mmaInstrM = 16, mmaInstrN = 8, mmaInstrK = 4 * 64 / bitwidth;
  int matShapeM = 8, matShapeN = 8, matShapeK = 2 * 64 / bitwidth;

  auto numRep = aEncoding.getMMAv2Rep(aTensorTy.getShape(), bitwidth);
  int numRepM = numRep[0], numRepK = numRep[1];

  if (aTensorTy.getEncoding().isa<SharedEncodingAttr>()) {
    int wpt0 = mmaLayout.getWarpsPerCTA()[0];
    Value warp = udiv(thread, i32_val(32));
    Value lane = urem(thread, i32_val(32));
    Value warpM = urem(urem(warp, i32_val(wpt0)), i32_val(shape[0] / 16));
    // load from smem
    // we use ldmatrix.x4 so each warp processes 16x16 elements.
    int wpt = std::min<int>(wpt0, shape[0] / 16);
    loadFn = getLoadMatrixFn(
        tensor, smemObj, mmaLayout, wpt /*wpt*/, 1 /*kOrder*/,
        {mmaInstrM, mmaInstrK} /*instrShape*/,
        {matShapeM, matShapeK} /*matShape*/, warpM /*warpId*/, lane /*laneId*/,
        ha /*vals*/, true /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/);
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
  return composeValuesToDotOperandLayoutStruct(ha, numRepM, numRepK,
                                               typeConverter, loc, rewriter);
}

Value loadBv2(ConversionPatternRewriter &rewriter, Location loc, Value tensor,
              DotOperandEncodingAttr bEncoding,
              const SharedMemoryObject &smemObj,
              TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  ValueTableV2 hb;
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  int bitwidth = tensorTy.getElementTypeBitWidth();
  auto mmaLayout = bEncoding.getParent().cast<MmaEncodingAttr>();

  SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                             tensorTy.getShape().end());

  // TODO[Superjomn]: transB cannot be accessed in ConvertLayoutOp.
  bool transB = false;
  if (transB) {
    std::swap(shape[0], shape[1]);
  }

  int mmaInstrM = 16, mmaInstrN = 8, mmaInstrK = 4 * 64 / bitwidth;
  int matShapeM = 8, matShapeN = 8, matShapeK = 2 * 64 / bitwidth;

  auto numRep = bEncoding.getMMAv2Rep(tensorTy.getShape(), bitwidth);
  int numRepK = numRep[0];
  int numRepN = numRep[1];

  int wpt0 = mmaLayout.getWarpsPerCTA()[0];
  int wpt1 = mmaLayout.getWarpsPerCTA()[1];
  Value warp = udiv(thread, i32_val(32));
  Value lane = urem(thread, i32_val(32));
  Value warpMN = udiv(warp, i32_val(wpt0));
  Value warpN = urem(urem(warpMN, i32_val(wpt1)), i32_val(shape[1] / 8));
  // we use ldmatrix.x4 so each warp processes 16x16 elements.
  int wpt = std::min<int>(wpt1, shape[1] / 16);
  auto loadFn = getLoadMatrixFn(
      tensor, smemObj, mmaLayout, wpt /*wpt*/, 0 /*kOrder*/,
      {mmaInstrK, mmaInstrN} /*instrShape*/,
      {matShapeK, matShapeN} /*matShape*/, warpN /*warpId*/, lane /*laneId*/,
      hb /*vals*/, false /*isA*/, typeConverter /* typeConverter */,
      rewriter /*rewriter*/, loc /*loc*/);

  for (int n = 0; n < std::max(numRepN / 2, 1); ++n) {
    for (int k = 0; k < numRepK; ++k)
      loadFn(2 * n, 2 * k);
  }

  Value result = composeValuesToDotOperandLayoutStruct(
      hb, std::max(numRepN / 2, 1), numRepK, typeConverter, loc, rewriter);
  return result;
}

struct ConvertLayoutOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::ConvertLayoutOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isaDistributedLayout(srcLayout) &&
        dstLayout.isa<SharedEncodingAttr>()) {
      return lowerDistributedToShared(op, adaptor, rewriter);
    }
    if (srcLayout.isa<SharedEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerSharedToDotOperand(op, adaptor, rewriter);
    }
    if (isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    if (srcLayout.isa<MmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }
    // TODO: to be implemented
    llvm_unreachable("unsupported layout conversion");
    return failure();
  }

private:
  SmallVector<Value> getMultiDimOffset(Attribute layout, Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       unsigned elemId, RankedTensorType type,
                                       ArrayRef<unsigned> multiDimCTAInRepId,
                                       ArrayRef<unsigned> shapePerCTA) const {
    auto shape = type.getShape();
    unsigned rank = shape.size();
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
      auto multiDimOffsetFirstElem =
          emitBaseIndexForLayout(loc, rewriter, blockedLayout, type);
      SmallVector<Value> multiDimOffset(rank);
      SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, getSizePerThread(layout), getOrder(layout));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] = add(multiDimOffsetFirstElem[d],
                                i32_val(multiDimCTAInRepId[d] * shapePerCTA[d] +
                                        multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
      unsigned dim = sliceLayout.getDim();
      auto parentEncoding = sliceLayout.getParent();
      auto parentShape = sliceLayout.paddedShape(shape);
      auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                            parentEncoding);
      auto multiDimOffsetParent =
          getMultiDimOffset(parentEncoding, loc, rewriter, elemId, parentTy,
                            sliceLayout.paddedShape(multiDimCTAInRepId),
                            sliceLayout.paddedShape(shapePerCTA));
      SmallVector<Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim)
          continue;
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }
    if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
      SmallVector<Value> mmaColIdx(4);
      SmallVector<Value> mmaRowIdx(2);
      Value threadId = getThreadId(rewriter, loc);
      Value warpSize = i32_val(32);
      Value laneId = urem(threadId, warpSize);
      Value warpId = udiv(threadId, warpSize);
      // TODO: fix the bug in MMAEncodingAttr document
      SmallVector<Value> multiDimWarpId(2);
      multiDimWarpId[0] = urem(warpId, i32_val(mmaLayout.getWarpsPerCTA()[0]));
      multiDimWarpId[1] = udiv(warpId, i32_val(mmaLayout.getWarpsPerCTA()[0]));
      Value _1 = i32_val(1);
      Value _2 = i32_val(2);
      Value _4 = i32_val(4);
      Value _8 = i32_val(8);
      Value _16 = i32_val(16);
      if (mmaLayout.isAmpere()) {
        multiDimWarpId[0] = urem(multiDimWarpId[0], i32_val(shape[0] / 16));
        multiDimWarpId[1] = urem(multiDimWarpId[1], i32_val(shape[1] / 8));
        Value mmaGrpId = udiv(laneId, _4);
        Value mmaGrpIdP8 = add(mmaGrpId, _8);
        Value mmaThreadIdInGrp = urem(laneId, _4);
        Value mmaThreadIdInGrpM2 = mul(mmaThreadIdInGrp, _2);
        Value mmaThreadIdInGrpM2P1 = add(mmaThreadIdInGrpM2, _1);
        Value rowWarpOffset = mul(multiDimWarpId[0], _16);
        mmaRowIdx[0] = add(mmaGrpId, rowWarpOffset);
        mmaRowIdx[1] = add(mmaGrpIdP8, rowWarpOffset);
        Value colWarpOffset = mul(multiDimWarpId[1], _8);
        mmaColIdx[0] = add(mmaThreadIdInGrpM2, colWarpOffset);
        mmaColIdx[1] = add(mmaThreadIdInGrpM2P1, colWarpOffset);
      } else if (mmaLayout.isVolta()) {
        // Volta doesn't follow the pattern here."
      } else {
        llvm_unreachable("Unexpected MMALayout version");
      }

      assert(rank == 2);
      SmallVector<Value> multiDimOffset(rank);
      if (mmaLayout.isAmpere()) {
        multiDimOffset[0] = elemId < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
        multiDimOffset[1] = elemId % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
        multiDimOffset[0] = add(
            multiDimOffset[0], i32_val(multiDimCTAInRepId[0] * shapePerCTA[0]));
        multiDimOffset[1] = add(
            multiDimOffset[1], i32_val(multiDimCTAInRepId[1] * shapePerCTA[1]));
      } else if (mmaLayout.isVolta()) {
        auto [isARow, isBRow, isAVec4, isBVec4, _] =
            mmaLayout.decodeVoltaLayoutStates();
        auto coords =
            getMNCoords(threadId, rewriter, mmaLayout.getWarpsPerCTA(),
                        mmaLayout, shape, isARow, isBRow, isAVec4, isBVec4);
        return coords[elemId];
      } else {
        llvm_unreachable("Unexpected MMALayout version");
      }
      return multiDimOffset;
    }
    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }

  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const {
    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto rank = type.getRank();
    auto sizePerThread = getSizePerThread(layout);
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<unsigned> numCTAs(rank);
    auto shapePerCTA = getShapePerCTA(layout, type.getShape());
    auto order = getOrder(layout);
    for (unsigned d = 0; d < rank; ++d) {
      numCTAs[d] = ceil<unsigned>(type.getShape()[d], shapePerCTA[d]);
    }
    auto elemTy = type.getElementType();
    bool isInt1 = elemTy.isInteger(1);
    bool isPtr = elemTy.isa<triton::PointerType>();
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isInt1)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);

    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
      auto multiDimCTAInRepId =
          getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
      SmallVector<unsigned> multiDimCTAId(rank);
      for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
        auto d = it.index();
        multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
      }

      auto linearCTAId =
          getLinearIndex<unsigned>(multiDimCTAId, numCTAs, order);
      // TODO: This is actually redundant index calculation, we should
      //       consider of caching the index calculation result in case
      //       of performance issue observed.
      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                              multiDimCTAInRepId, shapePerCTA);
        Value offset =
            linearize(rewriter, loc, multiDimOffset, paddedRepShape, outOrd);

        auto elemPtrTy = ptr_ty(llvmElemTy, 3);
        Value ptr = gep(elemPtrTy, smemBase, offset);
        auto vecTy = vec_ty(llvmElemTy, vec);
        ptr = bitcast(ptr, ptr_ty(vecTy, 3));
        if (stNotRd) {
          Value valVec = undef(vecTy);
          for (unsigned v = 0; v < vec; ++v) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread + v];
            if (isInt1)
              currVal = zext(llvmElemTy, currVal);
            else if (isPtr)
              currVal = ptrtoint(llvmElemTy, currVal);
            valVec = insert_element(vecTy, valVec, currVal, i32_val(v));
          }
          store(valVec, ptr);
        } else {
          Value valVec = load(ptr);
          for (unsigned v = 0; v < vec; ++v) {
            Value currVal = extract_element(llvmElemTy, valVec, i32_val(v));
            if (isInt1)
              currVal = icmp_ne(currVal,
                                rewriter.create<LLVM::ConstantOp>(
                                    loc, i8_ty, rewriter.getI8IntegerAttr(0)));
            else if (isPtr)
              currVal = inttoptr(llvmElemTyOrig, currVal);
            vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
          }
        }
      }
    }
  }

  // The MMAV1's result is quite different from the exising "Replica" structure,
  // add a new simple but clear implementation for it to avoid modificating the
  // logic of the exising one.
  void processReplicaForMMAV1(Location loc, ConversionPatternRewriter &rewriter,
                              bool stNotRd, RankedTensorType type,
                              ArrayRef<unsigned> multiDimRepId, unsigned vec,
                              ArrayRef<unsigned> paddedRepShape,
                              ArrayRef<unsigned> outOrd,
                              SmallVector<Value> &vals, Value smemBase,
                              ArrayRef<int64_t> shape,
                              bool isDestMma = false) const {
    unsigned accumNumCTAsEachRep = 1;
    auto layout = type.getEncoding();
    MmaEncodingAttr mma = layout.dyn_cast<MmaEncodingAttr>();
    auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>();
    if (sliceLayout)
      mma = sliceLayout.getParent().cast<MmaEncodingAttr>();

    auto order = getOrder(layout);
    auto rank = type.getRank();
    int accumSizePerThread = vals.size();

    SmallVector<unsigned> numCTAs(rank, 1);
    SmallVector<unsigned> numCTAsEachRep(rank, 1);
    SmallVector<unsigned> shapePerCTA = getShapePerCTA(layout, shape);
    auto elemTy = type.getElementType();

    int ctaId = 0;

    auto multiDimCTAInRepId =
        getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
    SmallVector<unsigned> multiDimCTAId(rank);
    for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
      auto d = it.index();
      multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
    }

    std::vector<std::pair<SmallVector<Value>, Value>> coord2valT(
        accumSizePerThread);
    bool needTrans = outOrd[0] != 0;
    if (sliceLayout || isDestMma)
      needTrans = false;

    vec = needTrans ? 2 : 1;
    {
      // We need to transpose the coordinates and values here to enable vec=2
      // when store to smem.
      std::vector<std::pair<SmallVector<Value>, Value>> coord2val(
          accumSizePerThread);
      for (unsigned elemId = 0; elemId < accumSizePerThread; ++elemId) {
        // TODO[Superjomn]: Move the coordinate computation out of loop, it is
        // duplicate in Volta.
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                              multiDimCTAInRepId, shapePerCTA);
        coord2val[elemId] = std::make_pair(multiDimOffset, vals[elemId]);
      }

      if (needTrans) {
        // do transpose
        auto aEncoding = DotOperandEncodingAttr::get(mma.getContext(), 0, mma);
        int numM = aEncoding.getMMAv1NumOuter(shape);
        int numN = accumSizePerThread / numM;

        for (int r = 0; r < numM; r++) {
          for (int c = 0; c < numN; c++) {
            coord2valT[r * numN + c] = std::move(coord2val[c * numM + r]);
          }
        }
      } else {
        coord2valT = std::move(coord2val);
      }
    }

    // Now the coord2valT has the transposed and contiguous elements(with
    // vec=2), the original vals is not needed.
    for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
      auto coord = coord2valT[elemId].first;
      Value offset = linearize(rewriter, loc, coord, paddedRepShape, outOrd);
      auto elemPtrTy = ptr_ty(elemTy, 3);
      Value ptr = gep(elemPtrTy, smemBase, offset);
      auto vecTy = vec_ty(elemTy, vec);
      ptr = bitcast(ptr, ptr_ty(vecTy, 3));
      if (stNotRd) {
        Value valVec = undef(vecTy);
        for (unsigned v = 0; v < vec; ++v) {
          auto currVal = coord2valT[elemId + v].second;
          valVec = insert_element(vecTy, valVec, currVal, i32_val(v));
        }
        store(valVec, ptr);
      } else {
        Value valVec = load(ptr);
        for (unsigned v = 0; v < vec; ++v) {
          Value currVal = extract_element(elemTy, valVec, i32_val(v));
          vals[elemId + v] = currVal;
        }
      }
    }
  }

  // blocked/mma -> blocked/mma.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto elemPtrTy = ptr_ty(llvmElemTy, 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    auto srcShapePerCTA = getShapePerCTA(srcLayout, srcTy.getShape());
    auto dstShapePerCTA = getShapePerCTA(dstLayout, shape);

    // For Volta, all the coords for a CTA are calculated.
    bool isSrcMmaV1{}, isDstMmaV1{};
    if (auto mmaLayout = srcLayout.dyn_cast<MmaEncodingAttr>()) {
      isSrcMmaV1 = mmaLayout.isVolta();
    }
    if (auto sliceLayout = srcLayout.dyn_cast<SliceEncodingAttr>()) {
      isSrcMmaV1 = sliceLayout.getParent().isa<MmaEncodingAttr>() &&
                   sliceLayout.getParent().cast<MmaEncodingAttr>().isVolta();
    }
    if (auto mmaLayout = dstLayout.dyn_cast<MmaEncodingAttr>()) {
      isDstMmaV1 = mmaLayout.isVolta();
    }
    if (auto sliceLayout = dstLayout.dyn_cast<SliceEncodingAttr>()) {
      isDstMmaV1 = sliceLayout.getParent().isa<MmaEncodingAttr>() &&
                   sliceLayout.getParent().cast<MmaEncodingAttr>().isVolta();
    }

    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA = std::min<unsigned>(shape[d], srcShapePerCTA[d]);
      unsigned outPerCTA = std::min<unsigned>(shape[d], dstShapePerCTA[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      numReplicates[d] = ceil<unsigned>(shape[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shape[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shape[d], outPerCTA);
    }
    // Potentially we need to store for multiple CTAs in this replication
    auto accumNumReplicates = product<unsigned>(numReplicates);
    // unsigned elems = getElemsPerThread(srcTy);
    auto vals = getTypeConverter()->unpackLLElements(loc, adaptor.getSrc(),
                                                     rewriter, srcTy);
    unsigned inVec = 0;
    unsigned outVec = 0;
    auto paddedRepShape = getScratchConfigForCvtLayout(op, inVec, outVec);

    unsigned outElems = getElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0)
        barrier();
      if (srcLayout.isa<BlockedEncodingAttr>() ||
          srcLayout.isa<SliceEncodingAttr>() ||
          srcLayout.isa<MmaEncodingAttr>()) {
        if (isSrcMmaV1)
          processReplicaForMMAV1(loc, rewriter, /*stNotRd*/ true, srcTy,
                                 multiDimRepId, inVec, paddedRepShape, outOrd,
                                 vals, smemBase, shape);
        else
          processReplica(loc, rewriter, /*stNotRd*/ true, srcTy,
                         inNumCTAsEachRep, multiDimRepId, inVec, paddedRepShape,
                         outOrd, vals, smemBase);
      } else {
        assert(0 && "ConvertLayout with input layout not implemented");
        return failure();
      }

      barrier();
      if (dstLayout.isa<BlockedEncodingAttr>() ||
          dstLayout.isa<SliceEncodingAttr>() ||
          dstLayout.isa<MmaEncodingAttr>()) {
        if (isDstMmaV1)
          processReplicaForMMAV1(loc, rewriter, /*stNotRd*/ false, dstTy,
                                 multiDimRepId, outVec, paddedRepShape, outOrd,
                                 outVals, smemBase, shape, /*isDestMma=*/true);
        else
          processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
                         outNumCTAsEachRep, multiDimRepId, outVec,
                         paddedRepShape, outOrd, outVals, smemBase);
      } else {
        assert(0 && "ConvertLayout with output layout not implemented");
        return failure();
      }
    }

    SmallVector<Type> types(outElems, llvmElemTy);
    auto *ctx = llvmElemTy.getContext();
    Type structTy = struct_ty(types);
    Value result =
        getTypeConverter()->packLLElements(loc, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  // blocked -> shared.
  // Swizzling in shared memory to avoid bank conflict. Normally used for
  // A/B operands of dots.
  LogicalResult
  lowerDistributedToShared(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto srcShape = srcTy.getShape();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    assert(srcShape.size() == 2 &&
           "Unexpected rank of ConvertLayout(blocked->shared)");
    auto srcLayout = srcTy.getEncoding();
    auto dstSharedLayout = dstTy.getEncoding().cast<SharedEncodingAttr>();
    auto inOrd = getOrder(srcLayout);
    auto outOrd = dstSharedLayout.getOrder();
    Value smemBase = getSharedMemoryBase(loc, rewriter, dst);
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto elemPtrTy = ptr_ty(getTypeConverter()->convertType(elemTy), 3);
    smemBase = bitcast(smemBase, elemPtrTy);

    auto dstStrides =
        getStridesFromShapeAndOrder(dstShape, outOrd, loc, rewriter);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy);
    storeDistributedToShared(src, adaptor.getSrc(), dstStrides, srcIndices, dst,
                             smemBase, elemTy, loc, rewriter);
    auto smemObj =
        SharedMemoryObject(smemBase, dstShape, outOrd, loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

  // shared -> mma_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto dstTensorTy = dst.getType().cast<RankedTensorType>();
    auto srcTensorTy = src.getType().cast<RankedTensorType>();
    auto dotOperandLayout =
        dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto sharedLayout = srcTensorTy.getEncoding().cast<SharedEncodingAttr>();

    bool isOuter{};
    int K{};
    if (dotOperandLayout.getOpIdx() == 0) // $a
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[1]];
    isOuter = K == 1;

    Value res;
    if (auto mmaLayout =
            dotOperandLayout.getParent().dyn_cast_or_null<MmaEncodingAttr>()) {
      res = lowerSharedToDotOperandMMA(op, adaptor, rewriter, mmaLayout,
                                       dotOperandLayout, isOuter);
    } else if (auto blockedLayout =
                   dotOperandLayout.getParent()
                       .dyn_cast_or_null<BlockedEncodingAttr>()) {
      auto dotOpLayout =
          dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
      DotOpFMAConversionHelper helper(blockedLayout);
      auto thread = getThreadId(rewriter, loc);
      if (dotOpLayout.getOpIdx() == 0) { // $a
        res = helper.loadA(src, adaptor.getSrc(), blockedLayout, thread, loc,
                           getTypeConverter(), rewriter);
      } else { // $b
        res = helper.loadB(src, adaptor.getSrc(), blockedLayout, thread, loc,
                           getTypeConverter(), rewriter);
      }
    } else {
      assert(false && "Unsupported dot operand layout found");
    }

    rewriter.replaceOp(op, res);
    return success();
  }

  // mma -> dot_operand
  LogicalResult
  lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto dstTy = op.getResult().getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();
    auto srcMmaLayout = srcLayout.cast<MmaEncodingAttr>();
    auto dstDotLayout = dstLayout.cast<DotOperandEncodingAttr>();
    if (isMmaToDotShortcut(srcMmaLayout, dstDotLayout)) {
      // get source values
      auto vals = getTypeConverter()->unpackLLElements(loc, adaptor.getSrc(),
                                                       rewriter, srcTy);
      unsigned elems = getElemsPerThread(srcTy);
      Type elemTy =
          this->getTypeConverter()->convertType(srcTy.getElementType());
      // for the destination type, we need to pack values together
      // so they can be consumed by tensor core operations
      SmallVector<Value> vecVals;
      SmallVector<Type> types;
      // For some reasons, LLVM's NVPTX backend inserts unnecessary (?) integer
      // instructions to pack & unpack sub-word integers. A workaround is to
      // store the results of ldmatrix in i32
      auto elemSize = elemTy.getIntOrFloatBitWidth();
      if (auto intTy = elemTy.dyn_cast<IntegerType>() && elemSize <= 16) {
        auto fold = 32 / elemSize;
        for (unsigned i = 0; i < elems; i += fold) {
          Value val = i32_val(0);
          for (unsigned j = 0; j < fold; j++) {
            auto ext =
                shl(i32_ty, zext(i32_ty, vals[i + j]), i32_val(elemSize * j));
            val = or_(i32_ty, val, ext);
          }
          vecVals.push_back(val);
        }
        elems = elems / (32 / elemSize);
        types = SmallVector<Type>(elems, i32_ty);
      } else {
        unsigned vecSize = std::max<unsigned>(32 / elemSize, 1);
        Type vecTy = vec_ty(elemTy, vecSize);
        types = SmallVector<Type>(elems / vecSize, vecTy);
        for (unsigned i = 0; i < elems; i += vecSize) {
          Value packed = rewriter.create<LLVM::UndefOp>(loc, vecTy);
          for (unsigned j = 0; j < vecSize; j++)
            packed = insert_element(vecTy, packed, vals[i + j], i32_val(j));
          vecVals.push_back(packed);
        }
      }

      // This needs to be ordered the same way that
      // ldmatrix.x4 would order it
      // TODO: this needs to be refactor so we don't
      // implicitly depends on how emitOffsetsForMMAV2
      // is implemented
      SmallVector<Value> reorderedVals;
      for (unsigned i = 0; i < vecVals.size(); i += 4) {
        reorderedVals.push_back(vecVals[i]);
        reorderedVals.push_back(vecVals[i + 2]);
        reorderedVals.push_back(vecVals[i + 1]);
        reorderedVals.push_back(vecVals[i + 3]);
      }

      Value view = getTypeConverter()->packLLElements(loc, reorderedVals,
                                                      rewriter, dstTy);
      rewriter.replaceOp(op, view);
      return success();
    }
    return failure();
  }

  // shared -> dot_operand if the result layout is mma
  Value lowerSharedToDotOperandMMA(
      triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, const MmaEncodingAttr &mmaLayout,
      const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    bool isHMMA = supportMMA(dst, mmaLayout.getVersionMajor());

    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), rewriter);
    Value res;

    if (!isOuter && mmaLayout.isAmpere() && isHMMA) { // tensor core v2
      // MMA16816ConversionHelper mmaHelper(src.getType(), mmaLayout,
      //                                    getThreadId(rewriter, loc),
      //                                    rewriter, getTypeConverter(),
      //                                    op.getTritonGPUToLLVMLoc());

      if (dotOperandLayout.getOpIdx() == 0) {
        // operand $a
        res = loadAv2(rewriter, loc, src, dotOperandLayout, smemObj,
                      getTypeConverter(), tid_val());
      } else if (dotOperandLayout.getOpIdx() == 1) {
        // operand $b
        res = loadBv2(rewriter, loc, src, dotOperandLayout, smemObj,
                      getTypeConverter(), tid_val());
      }
    } else if (!isOuter && mmaLayout.isVolta() && isHMMA) { // tensor core v1
      bool isMMAv1Row = dotOperandLayout.getMMAv1IsRow();
      auto srcSharedLayout = src.getType()
                                 .cast<RankedTensorType>()
                                 .getEncoding()
                                 .cast<SharedEncodingAttr>();

      // Can only convert [1, 0] to row or [0, 1] to col for now
      if ((srcSharedLayout.getOrder()[0] == 1 && !isMMAv1Row) ||
          (srcSharedLayout.getOrder()[0] == 0 && isMMAv1Row)) {
        llvm::errs() << "Unsupported Shared -> DotOperand[MMAv1] conversion\n";
        return Value();
      }

      if (dotOperandLayout.getOpIdx() == 0) { // operand $a
        res = loadA(src, smemObj, getThreadId(rewriter, loc), loc,
                    getTypeConverter(), rewriter, dst.getType());
      } else if (dotOperandLayout.getOpIdx() == 1) { // operand $b
        res = loadB(src, smemObj, getThreadId(rewriter, loc), loc,
                    getTypeConverter(), rewriter, dst.getType());
      }
    } else {
      assert(false && "Unsupported mma layout found");
    }
    return res;
  }
};

void populateConvertLayoutOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, allocation, smem,
                                          indexCacheInfo, benefit);
}
