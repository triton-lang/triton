#include "../ConvertLayoutOpToLLVM.h"
#include "../Utility.h"

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;
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

  // Matrix coordinates inside a CTA,
  // the matrix layout is [2wpt[0], 2] for A and [2, 2wpt[1]] for B.
  // e.g., Setting wpt=4, the data layout for A(kOrder=1) is
  //   |0 0|  -> 0,1,2,3 are the warpids
  //   |0 0|
  //   |1 1|
  //   |1 1|
  //   |2 2|
  //   |2 2|
  //   |3 3|
  //   |3 3|
  //
  // for B(kOrder=0) is
  //   |0 1 2 3 0 1 2 3| -> 0,1,2,3 are the warpids
  //   |0 1 2 3 0 1 2 3|
  // Note, for each warp, it handles a 2x2 matrices, that is the coordinate
  // address (s0,s1) annotates.

  Value matOff[2];
  matOff[kOrder ^ 1] =
      add(mul(warpId, i32_val(warpOffStride)), // warp offset (kOrder=1)
          mul(nkMatArr,
              i32_val(matArrStride))); // matrix offset inside a warp (kOrder=1)
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
  // To prevent out-of-bound access of B when wpt * 16 > tile_size.
  // In such a case, we need to wrap around the offset of B.
  // |0 1 2 3 0 1 2 3| -> | 0(0) 1(1) 2(2) 3(3) |
  // |0 1 2 3 0 1 2 3|    | 0(0) 1(1) 2(2) 3(3) |
  //          ~~~~~~~ out-of-bound access
  Value sOff = urem(add(sOffInMat, mul(sMatOff, i32_val(sMatShape))),
                    i32_val(tileShape[order[1]]));
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
    const ValueTable &vals, int n0, int n1,
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
                Value warpId, Value lane, ValueTable &vals, bool isA,
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
  auto ld2 = [](ValueTable &vals, int mn, int k, Value val) {
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

Value loadA(ConversionPatternRewriter &rewriter, Location loc, Value tensor,
            DotOperandEncodingAttr aEncoding, const SharedMemoryObject &smemObj,
            TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  int bitwidth = aTensorTy.getElementTypeBitWidth();
  auto mmaLayout = aEncoding.getParent().cast<MmaEncodingAttr>();

  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());

  ValueTable ha;
  std::function<void(int, int)> loadFn;
  int mmaInstrM = 16, mmaInstrN = 8, mmaInstrK = 4 * 64 / bitwidth;
  int matShapeM = 8, matShapeN = 8, matShapeK = 2 * 64 / bitwidth;

  auto numRep = aEncoding.getMMAv2Rep(aTensorTy.getShape(), bitwidth);
  int numRepM = numRep[0];
  int numRepK = numRep[1];

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

Value loadB(ConversionPatternRewriter &rewriter, Location loc, Value tensor,
            DotOperandEncodingAttr bEncoding, const SharedMemoryObject &smemObj,
            TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  ValueTable hb;
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  int bitwidth = tensorTy.getElementTypeBitWidth();
  auto mmaLayout = bEncoding.getParent().cast<MmaEncodingAttr>();

  SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                             tensorTy.getShape().end());

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

namespace SharedToDotOperandMMAv2 {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  if (opIdx == 0)
    return loadA(rewriter, loc, tensor, encoding, smemObj, typeConverter,
                 thread);
  else {
    assert(opIdx == 1);
    return loadB(rewriter, loc, tensor, encoding, smemObj, typeConverter,
                 thread);
  }
}
} // namespace SharedToDotOperandMMAv2
