#include "../ConvertLayoutOpToLLVM.h"
#include "../Utility.h"

using namespace mlir;

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;
using ::mlir::LLVM::delinearize;
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

// Data loader for mma.16816 instruction.
class MMA16816SmemLoader {
public:
  MMA16816SmemLoader(int nPerWarp, int warpsPerTile, ArrayRef<uint32_t> order,
                     ArrayRef<uint32_t> warpsPerCTA, uint32_t kOrder,
                     int kWidth, ArrayRef<Value> smemStrides,
                     ArrayRef<int64_t> tileShape, ArrayRef<int> instrShape,
                     ArrayRef<int> matShape, int perPhase, int maxPhase,
                     int elemBytes, ConversionPatternRewriter &rewriter,
                     TritonGPUToLLVMTypeConverter *typeConverter,
                     const Location &loc);

  // lane = thread % 32
  // warpOff = (thread/32) % warpsPerTile(0)
  llvm::SmallVector<Value> computeOffsets(Value warpOff, Value lane,
                                          Value cSwizzleOffset) {
    if (canUseLdmatrix)
      return computeLdmatrixMatOffs(warpOff, lane, cSwizzleOffset);
    else
      return computeLdsMatOffs(warpOff, lane, cSwizzleOffset);
    return {};
  }

  int getNumPtrs() const { return numPtrs; }

  // Compute the offset to the matrix this thread(indexed by warpOff and lane)
  // mapped to.
  SmallVector<Value> computeLdmatrixMatOffs(Value warpId, Value lane,
                                            Value cSwizzleOffset);
  // compute 8-bit matrix offset.
  SmallVector<Value> computeLdsMatOffs(Value warpOff, Value lane,
                                       Value cSwizzleOffset);

  // Load 4 matrices and returns 4 vec<2> elements.
  std::tuple<Value, Value, Value, Value> loadX4(int mat0, int mat1,
                                                ArrayRef<Value> ptrs,
                                                Type matTy,
                                                Type shemPtrTy) const;

private:
  SmallVector<uint32_t> order;
  SmallVector<uint32_t> warpsPerCTA;
  int kOrder;
  int kWidth;
  int vecWidth;
  SmallVector<int64_t> tileShape;
  SmallVector<int> instrShape;
  SmallVector<int> matShape;
  int perPhase;
  int maxPhase;
  int elemBytes;
  ConversionPatternRewriter &rewriter;
  const Location &loc;
  MLIRContext *ctx{};

  // ldmatrix loads a matrix of size stridedMatShape x contiguousMatShape
  int contiguousMatShape;
  int stridedMatShape;

  // Offset in shared memory to increment on the strided axis
  // This would be different than the tile shape in the case of a sliced tensor
  Value stridedSmemOffset;

  bool needTrans;
  bool canUseLdmatrix;

  int numPtrs;

  // Load operations offset in number of Matrices on contiguous and strided axes
  int contiguousLoadMatOffset;
  int stridedLoadMatOffset;

  // Offset in number of matrices to increment on non-k dim within a warp's 2x2
  // matrices
  int inWarpMatOffset;
  // Offset in number of matrices to increment on non-k dim across warps
  int warpMatOffset;

  int nPerWarp;
};

SmallVector<Value>
MMA16816SmemLoader::computeLdmatrixMatOffs(Value warpId, Value lane,
                                           Value cSwizzleOffset) {
  // 4x4 matrices
  Value rowInMat = urem(lane, i32_val(8)); // row in the 8x8 matrix
  Value matIndex =
      udiv(lane, i32_val(8)); // linear index of the matrix in the 2x2 matrices

  // Decompose matIndex => s_0, s_1, that is the coordinate in 2x2 matrices in a
  // warp
  Value s0 = urem(matIndex, i32_val(2));
  Value s1 = udiv(matIndex, i32_val(2));

  // We use different orders for a and b for better performance.
  Value kMatArr = kOrder == 1 ? s1 : s0;  // index of matrix on the k dim
  Value nkMatArr = kOrder == 1 ? s0 : s1; // index of matrix on the non-k dim

  // Matrix coordinates inside a CTA,
  // the matrix layout is [2warpsPerTile[0], 2] for A and [2, 2warpsPerTile[1]]
  // for B. e.g., Setting warpsPerTile=4, the data layout for A(kOrder=1) is
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
  // When B's shape(k, n) is (16, 8) and ldmatrix.x4 is used, the shared memory
  // access will be out of bound. In the future we should change this case to
  // ldmatrix.x2
  if (kOrder == 0 && nPerWarp == 8) {
    matOff[kOrder ^ 1] = mul(warpId, i32_val(warpMatOffset));
  } else {
    matOff[kOrder ^ 1] = add(
        mul(warpId, i32_val(warpMatOffset)), // warp offset (kOrder=1)
        mul(nkMatArr,
            i32_val(
                inWarpMatOffset))); // matrix offset inside a warp (kOrder=1)
  }
  matOff[kOrder] = kMatArr;

  // Physical offset (before swizzling)
  Value contiguousMatIndex = matOff[order[0]];
  Value stridedMatIndex = matOff[order[1]];
  // Add the offset of the slice
  Value contiguousSliceMatOffset =
      udiv(cSwizzleOffset, i32_val(contiguousMatShape));

  SmallVector<Value> offs(numPtrs);
  Value phase = urem(udiv(rowInMat, i32_val(perPhase)), i32_val(maxPhase));
  // To prevent out-of-bound access of B when warpsPerTile * 16 > tile_size.
  // In such a case, we need to wrap around the offset of B.
  // |0 1 2 3 0 1 2 3| -> | 0(0) 1(1) 2(2) 3(3) |
  // |0 1 2 3 0 1 2 3|    | 0(0) 1(1) 2(2) 3(3) |
  //          ~~~~~~~ out-of-bound access

  Value rowOffset =
      urem(add(rowInMat, mul(stridedMatIndex, i32_val(stridedMatShape))),
           i32_val(tileShape[order[1]]));
  auto contiguousTileNumMats = tileShape[order[0]] / matShape[order[0]];

  for (int i = 0; i < numPtrs; ++i) {
    Value contiguousIndex =
        add(contiguousMatIndex, i32_val(i * contiguousLoadMatOffset));
    if (warpsPerCTA[order[0]] > contiguousTileNumMats ||
        contiguousTileNumMats % warpsPerCTA[order[0]] != 0)
      contiguousIndex = urem(contiguousIndex, i32_val(contiguousTileNumMats));
    contiguousIndex = add(contiguousIndex, contiguousSliceMatOffset);
    Value contiguousIndexSwizzled = xor_(contiguousIndex, phase);
    offs[i] = add(mul(contiguousIndexSwizzled, i32_val(contiguousMatShape)),
                  mul(rowOffset, stridedSmemOffset));
  }

  return offs;
}

// clang-format off
// Each `ldmatrix.x4` loads data as follows when `needTrans == False`:
//
//               quad width
// <----------------------------------------->
// vecWidth
// <------->
//  *#t0 ... *#t0  t1 ... t1  t2 ... t2  t3 ... t3   ||  *t0 ... *t0  t1 ... t1  t2 ... t2  t3 ... t3  /|\
//  t4 ... t4  t5 ... t5  t6 ... t6  t7 ... t7   ||  t4 ... t4  t5 ... t5  t6 ... t6  t7 ... t7   |
//  t8 ... t8  t9 ... t9 t10 .. t10 t11 .. t11   ||  t8 ... t8  t9 ... t9 t10 .. t10 t11 .. t11   | quad height
// ...                                                                                            |
// t28 .. t28 t29 .. t29 t30 .. t30 t31 .. t31   || t28 .. t28 t29 .. t29 t30 .. t30 t31 .. t31  \|/
// --------------------------------------------- || --------------------------------------------
//  *#t0 ... *#t0  t1 ... t1  t2 ... t2  t3 ... t3   ||  t0 ... t0  t1 ... t1  t2 ... t2  t3 ... t3
//  t4 ... t4  t5 ... t5  t6 ... t6  t7 ... t7   ||  t4 ... t4  t5 ... t5  t6 ... t6  t7 ... t7
//  t8 ... t8  t9 ... t9 t10 .. t10 t11 .. t11   ||  t8 ... t8  t9 ... t9 t10 .. t10 t11 .. t11
// ...
// t28 .. t28 t29 .. t29 t30 .. t30 t31 .. t31   || t28 .. t28 t29 .. t29 t30 .. t30 t31 .. t31
//
// we assume that the phase is < 8 so we don't need to maintain a separate pointer for the two
// lower quadrants. This pattern repeats every warpsPerTile[0] (resp. warpsPerTile[1]) blocks
// along the row (resp. col) dimension.
// clang-format on

SmallVector<Value> MMA16816SmemLoader::computeLdsMatOffs(Value warpOff,
                                                         Value lane,
                                                         Value cSwizzleOffset) {
  int cTileShape = tileShape[order[0]];
  int sTileShape = tileShape[order[1]];
  if (!needTrans) {
    std::swap(cTileShape, sTileShape);
  }

  SmallVector<Value> offs(numPtrs);

  int threadsPerQuad[2] = {8, 4};
  int laneWidth = 4;
  int laneHeight = 8;
  int quadWidth = laneWidth * kWidth;
  int quadHeight = laneHeight;
  int numQuadI = 2;

  // outer index base
  Value iBase = udiv(lane, i32_val(laneWidth));

  for (int rep = 0; rep < numPtrs / (2 * kWidth); ++rep)
    for (int quadId = 0; quadId < 2; ++quadId)
      for (int elemId = 0; elemId < kWidth; ++elemId) {
        // inner index base
        Value jBase = mul(urem(lane, i32_val(laneWidth)), i32_val(kWidth));
        jBase = add(jBase, i32_val(elemId));
        // inner index offset
        Value jOff = i32_val(0);
        if (!needTrans) {
          jOff = add(jOff, i32_val(quadId));
          jOff = add(jOff, i32_val(rep * contiguousLoadMatOffset));
        }
        // outer index offset
        Value iOff = mul(warpOff, i32_val(warpMatOffset));
        if (needTrans) {
          int pStride = kOrder == 1 ? 1 : 2;
          iOff = add(iOff, i32_val(quadId * inWarpMatOffset));
          iOff = add(iOff, i32_val(rep * contiguousLoadMatOffset * pStride));
        }
        // swizzle
        if (!needTrans) {
          Value phase = urem(udiv(iBase, i32_val(perPhase)), i32_val(maxPhase));
          jOff = add(jOff, udiv(cSwizzleOffset, i32_val(quadWidth)));
          jOff = xor_(jOff, phase);
        } else {
          Value phase = urem(udiv(jBase, i32_val(perPhase)), i32_val(maxPhase));
          iOff = add(iOff, udiv(cSwizzleOffset, i32_val(quadHeight)));
          iOff = xor_(iOff, phase);
        }
        // To prevent out-of-bound access when tile is too small.
        Value i = add(iBase, mul(iOff, i32_val(quadHeight)));
        Value j = add(jBase, mul(jOff, i32_val(quadWidth)));
        // Compute id of this ptr
        int idx = rep * 2 * kWidth;
        if (needTrans) {
          idx += quadId * vecWidth;
          idx += elemId % vecWidth;
          idx += elemId / vecWidth * kWidth;
        } else {
          idx += quadId * kWidth;
          idx += elemId;
        }

        if (needTrans) {
          offs[idx] = add(i, mul(j, stridedSmemOffset));
        } else {
          offs[idx] = add(mul(i, stridedSmemOffset), j);
        }
      }

  return offs;
}

std::tuple<Value, Value, Value, Value>
MMA16816SmemLoader::loadX4(int mat0, int mat1, ArrayRef<Value> ptrs, Type matTy,
                           Type shemPtrTy) const {
  assert(mat0 % 2 == 0 && mat1 % 2 == 0 && "smem matrix load must be aligned");
  int matIdx[2] = {mat0, mat1};

  int ptrIdx{-1};

  if (canUseLdmatrix)
    ptrIdx = matIdx[order[0]] / (instrShape[order[0]] / matShape[order[0]]);
  else
    ptrIdx = matIdx[order[0]] * (needTrans ? kWidth : vecWidth);

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
    Value stridedOffset =
        mul(i32_val(matIdx[order[1]] * stridedLoadMatOffset * stridedMatShape),
            stridedSmemOffset);
    Value readPtr = gep(shemPtrTy, ptr, stridedOffset);

    PTXBuilder builder;
    // ldmatrix.m8n8.x4 returns 4x2xfp16(that is 4xb32) elements for a
    // thread.
    auto resArgs = builder.newListOperand(4, "=r");
    auto addrArg = builder.newAddrOperand(readPtr, "r");

    auto ldmatrix = builder.create("ldmatrix.sync.aligned.m8n8.x4")
                        ->o("trans", needTrans /*predicate*/)
                        .o("shared.b16");
    ldmatrix(resArgs, addrArg);

    // The result type is 4xi32, each i32 is composed of 2xf16
    // elements (adjacent two columns in a row) or a single f32 element.
    Value resV4 = builder.launch(rewriter, loc, resTy);
    return {extract_val(elemTy, resV4, 0), extract_val(elemTy, resV4, 1),
            extract_val(elemTy, resV4, 2), extract_val(elemTy, resV4, 3)};
  } else {
    // base pointers
    std::array<std::array<Value, 4>, 2> ptrs;
    for (int i = 0; i < vecWidth; i++)
      ptrs[0][i] = getPtr(ptrIdx + i);
    for (int i = 0; i < vecWidth; i++)
      ptrs[1][i] = getPtr(ptrIdx + i + vecWidth);
    // static offsets along outer dimension
    int _i0 = matIdx[order[1]] * (stridedLoadMatOffset * stridedMatShape);
    int _i1 = _i0;
    if (needTrans)
      _i1 += (kWidth != vecWidth) ? vecWidth
                                  : stridedLoadMatOffset * stridedMatShape;
    else
      _i1 += (kOrder == 1 ? 1 : stridedLoadMatOffset) * stridedMatShape;
    Value i0 = mul(i32_val(_i0), stridedSmemOffset);
    Value i1 = mul(i32_val(_i1), stridedSmemOffset);
    std::array<Value, 2> ii = {i0, i1};
    // load 4 32-bit values from shared memory
    // (equivalent to ldmatrix.x4)
    SmallVector<SmallVector<Value>> vptrs(4, SmallVector<Value>(vecWidth));

    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < vecWidth; ++j) {
        vptrs[i][j] = gep(shemPtrTy, ptrs[i / 2][j], ii[i % 2]);
      }
    // row + trans and col + no-trans are equivalent
    bool isActualTrans =
        (needTrans && kOrder == 1) || (!needTrans && kOrder == 0);
    // pack loaded vectors into 4 32-bit values
    int inc = needTrans ? 1 : kWidth;
    VectorType packedTy = vec_ty(int_ty(8 * elemBytes), inc);
    int canonBits = std::min(32, 8 * elemBytes * inc);
    int canonWidth = (8 * elemBytes * inc) / canonBits;
    Type canonInt = int_ty(canonBits);
    std::array<Value, 4> retElems;
    retElems.fill(undef(vec_ty(canonInt, 32 / canonBits)));
    for (int r = 0; r < 2; ++r) {
      for (int em = 0; em < 2 * vecWidth; em += inc) {
        int e = em % vecWidth;
        int m = em / vecWidth;
        int idx = m * 2 + r;
        Value ptr = bitcast(vptrs[idx][e], ptr_ty(packedTy, 3));
        Value val = load(ptr);
        Value canonval = bitcast(val, vec_ty(canonInt, canonWidth));
        for (int w = 0; w < canonWidth; ++w) {
          int ridx = idx + w * kWidth / vecWidth;
          retElems[ridx] =
              insert_element(retElems[ridx],
                             extract_element(canonval, i32_val(w)), i32_val(e));
        }
      }
    }
    if (isActualTrans)
      std::swap(retElems[1], retElems[2]);
    return {bitcast(retElems[0], i32_ty), bitcast(retElems[1], i32_ty),
            bitcast(retElems[2], i32_ty), bitcast(retElems[3], i32_ty)};
  }
}

MMA16816SmemLoader::MMA16816SmemLoader(
    int nPerWarp, int warpsPerTile, ArrayRef<uint32_t> order,
    ArrayRef<uint32_t> warpsPerCTA, uint32_t kOrder, int kWidth,
    ArrayRef<Value> smemStrides, ArrayRef<int64_t> tileShape,
    ArrayRef<int> instrShape, ArrayRef<int> matShape, int perPhase,
    int maxPhase, int elemBytes, ConversionPatternRewriter &rewriter,
    TritonGPUToLLVMTypeConverter *typeConverter, const Location &loc)
    : nPerWarp(nPerWarp), order(order.begin(), order.end()),
      warpsPerCTA(warpsPerCTA.begin(), warpsPerCTA.end()), kOrder(kOrder),
      kWidth(kWidth), tileShape(tileShape.begin(), tileShape.end()),
      instrShape(instrShape.begin(), instrShape.end()),
      matShape(matShape.begin(), matShape.end()), perPhase(perPhase),
      maxPhase(maxPhase), elemBytes(elemBytes), rewriter(rewriter), loc(loc),
      ctx(rewriter.getContext()) {
  contiguousMatShape = matShape[order[0]];
  stridedMatShape = matShape[order[1]];
  stridedSmemOffset = smemStrides[order[1]];
  vecWidth = 4 / elemBytes;

  // rule: k must be the fast-changing axis.
  needTrans = kOrder != order[0];
  canUseLdmatrix = elemBytes == 2 || (!needTrans);
  canUseLdmatrix = canUseLdmatrix && (kWidth == vecWidth);

  if (canUseLdmatrix) {
    // Each CTA, the warps is arranged as [1xwarpsPerTile] if not transposed,
    // otherwise [warpsPerTilex1], and each warp will perform a mma.
    numPtrs = tileShape[order[0]] / (needTrans ? warpsPerTile : 1) /
              instrShape[order[0]];
  } else {
    numPtrs = tileShape[order[0]] / (needTrans ? warpsPerTile : 1) /
              matShape[order[0]];
    numPtrs *= kWidth;
  }
  numPtrs = std::max<int>(numPtrs, 2);

  // Special rule for i8/u8, 4 ptrs for each matrix
  // if (!canUseLdmatrix && elemBytes == 1)

  int loadOffsetInMat[2];
  loadOffsetInMat[kOrder] =
      2; // instrShape[kOrder] / matShape[kOrder], always 2
  loadOffsetInMat[kOrder ^ 1] =
      warpsPerTile * (instrShape[kOrder ^ 1] / matShape[kOrder ^ 1]);

  contiguousLoadMatOffset = loadOffsetInMat[order[0]];

  stridedLoadMatOffset =
      loadOffsetInMat[order[1]] / (instrShape[order[1]] / matShape[order[1]]);

  // The stride (in number of matrices) within warp
  inWarpMatOffset = kOrder == 1 ? 1 : warpsPerTile;
  // The stride (in number of matrices) of each warp
  warpMatOffset = instrShape[kOrder ^ 1] / matShape[kOrder ^ 1];
}

Type getSharedMemPtrTy(Type argType) {
  MLIRContext *ctx = argType.getContext();
  if (argType.isF16())
    return ptr_ty(type::f16Ty(ctx), 3);
  else if (argType.isBF16())
    return ptr_ty(type::i16Ty(ctx), 3);
  else if (argType.isF32())
    return ptr_ty(type::f32Ty(ctx), 3);
  else if (argType.getIntOrFloatBitWidth() == 8)
    return ptr_ty(type::i8Ty(ctx), 3);
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

std::function<void(int, int)> getLoadMatrixFn(
    Value tensor, const SharedMemoryObject &smemObj, MmaEncodingAttr mmaLayout,
    int warpsPerTile, uint32_t kOrder, int kWidth, SmallVector<int> instrShape,
    SmallVector<int> matShape, Value warpId, Value lane, ValueTable &vals,
    bool isA, TritonGPUToLLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, Location loc) {
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  auto shapePerCTA = getShapePerCTA(tensorTy);
  Type eltTy = tensorTy.getElementType();
  // We assumes that the input operand of Dot should be from shared layout.
  // TODO(Superjomn) Consider other layouts if needed later.
  auto sharedLayout = tensorTy.getEncoding().cast<SharedEncodingAttr>();
  const int perPhase = sharedLayout.getPerPhase();
  const int maxPhase = sharedLayout.getMaxPhase();
  const int vecPhase = sharedLayout.getVec();
  const int elemBytes = tensorTy.getElementTypeBitWidth() / 8;
  auto order = sharedLayout.getOrder();

  if (kWidth != (4 / elemBytes))
    assert(vecPhase == 1 || vecPhase == 4 * kWidth);

  int nPerWarp =
      std::max<int>(shapePerCTA[1] / mmaLayout.getWarpsPerCTA()[1], 8);

  // (a, b) is the coordinate.
  auto load = [=, &rewriter, &vals](int a, int b) {
    MMA16816SmemLoader loader(nPerWarp, warpsPerTile, sharedLayout.getOrder(),
                              mmaLayout.getWarpsPerCTA(), kOrder, kWidth,
                              smemObj.strides, shapePerCTA /*tileShape*/,
                              instrShape, matShape, perPhase, maxPhase,
                              elemBytes, rewriter, typeConverter, loc);
    // Offset of a slice within the original tensor in shared memory
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    SmallVector<Value> offs =
        loader.computeOffsets(warpId, lane, cSwizzleOffset);
    // initialize pointers
    const int numPtrs = loader.getNumPtrs();
    SmallVector<Value> ptrs(numPtrs);
    Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
    Type smemPtrTy = getSharedMemPtrTy(eltTy);
    for (int i = 0; i < numPtrs; ++i)
      ptrs[i] = bitcast(gep(smemPtrTy, smemBase, offs[i]), smemPtrTy);
    // actually load from shared memory
    auto matTy = LLVM::LLVMStructType::getLiteral(eltTy.getContext(),
                                                  SmallVector<Type>(4, i32_ty));
    auto [ha0, ha1, ha2, ha3] = loader.loadX4(
        (kOrder == 1) ? a : b /*mat0*/, (kOrder == 1) ? b : a /*mat1*/, ptrs,
        matTy, getSharedMemPtrTy(eltTy));
    if (!isA)
      std::swap(ha1, ha2);
    // the following is incorrect
    // but causes dramatically better performance in ptxas
    // although it only changes the order of operands in
    // `mma.sync`
    // if(isA)
    //   std::swap(ha1, ha2);
    // update user-provided values in-place
    vals[{a, b}] = ha0;
    vals[{a + 1, b}] = ha1;
    vals[{a, b + 1}] = ha2;
    vals[{a + 1, b + 1}] = ha3;
  };

  return load;
}

Value loadArg(ConversionPatternRewriter &rewriter, Location loc, Value tensor,
              DotOperandEncodingAttr encoding,
              const SharedMemoryObject &smemObj,
              TritonGPUToLLVMTypeConverter *typeConverter, Value thread,
              bool isA) {
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  auto shapePerCTA = getShapePerCTA(tensorTy);
  int bitwidth = tensorTy.getElementTypeBitWidth();
  auto mmaLayout = encoding.getParent().cast<MmaEncodingAttr>();

  ValueTable vals;
  int mmaInstrM = 16, mmaInstrN = 8, mmaInstrK = 4 * 64 / bitwidth;
  int matShapeM = 8, matShapeN = 8, matShapeK = 2 * 64 / bitwidth;

  auto numRep =
      mmaLayout.getMMAv2Rep(shapePerCTA, bitwidth, encoding.getOpIdx());
  int kWidth = encoding.getKWidth();

  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto order = triton::gpu::getOrder(mmaLayout);
  Value warp = udiv(thread, i32_val(32));
  Value lane = urem(thread, i32_val(32));

  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warp, warpsPerCTA, order);
  Value warpM = urem(multiDimWarpId[0], i32_val(shapePerCTA[0] / 16));
  Value warpN = urem(multiDimWarpId[1], i32_val(shapePerCTA[1] / 8));

  int warpsPerTile;
  if (isA)
    warpsPerTile = std::min<int>(warpsPerCTA[0], shapePerCTA[0] / 16);
  else
    warpsPerTile = std::min<int>(warpsPerCTA[1], shapePerCTA[1] / 16);

  std::function<void(int, int)> loadFn;
  if (isA)
    loadFn = getLoadMatrixFn(
        tensor, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/, 1 /*kOrder*/,
        kWidth, {mmaInstrM, mmaInstrK} /*instrShape*/,
        {matShapeM, matShapeK} /*matShape*/, warpM /*warpId*/, lane /*laneId*/,
        vals /*vals*/, isA /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/);
  else
    loadFn = getLoadMatrixFn(
        tensor, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/, 0 /*kOrder*/,
        kWidth, {mmaInstrK, mmaInstrN} /*instrShape*/,
        {matShapeK, matShapeN} /*matShape*/, warpN /*warpId*/, lane /*laneId*/,
        vals /*vals*/, isA /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/);

  // Perform loading.
  int numRepOuter = isA ? numRep[0] : std::max<int>(numRep[1] / 2, 1);
  int numRepK = isA ? numRep[1] : numRep[0];
  for (int m = 0; m < numRepOuter; ++m)
    for (int k = 0; k < numRepK; ++k)
      loadFn(2 * m, 2 * k);

  // Format the values to LLVM::Struct to passing to mma codegen.
  return composeValuesToDotOperandLayoutStruct(vals, numRepOuter, numRepK,
                                               typeConverter, loc, rewriter);
}

namespace SharedToDotOperandMMAv2 {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  if (opIdx == 0)
    return loadArg(rewriter, loc, tensor, encoding, smemObj, typeConverter,
                   thread, true);
  else {
    assert(opIdx == 1);
    return loadArg(rewriter, loc, tensor, encoding, smemObj, typeConverter,
                   thread, false);
  }
}
} // namespace SharedToDotOperandMMAv2
