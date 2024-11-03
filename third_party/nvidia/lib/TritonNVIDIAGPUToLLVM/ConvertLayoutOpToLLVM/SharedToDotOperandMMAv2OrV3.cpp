#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

using ValueTable = std::map<std::array<int, 3>, Value>;
using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Data loader for mma.16816 instruction.
class MMA16816SmemLoader {
public:
  MMA16816SmemLoader(int nPerWarp, int warpsPerTile, ArrayRef<uint32_t> order,
                     ArrayRef<uint32_t> warpsPerCTA, uint32_t kOrder,
                     int kWidth, ArrayRef<Value> smemStrides,
                     ArrayRef<int64_t> tileShape, ArrayRef<int> instrShape,
                     ArrayRef<int> matShape, SmallVector<Value> multiDimWarpId,
                     int perPhase, int maxPhase, int elemBytes,
                     int mmaElemBytes, bool isHopper,
                     ConversionPatternRewriter &rewriter,
                     const LLVMTypeConverter *typeConverter,
                     const Location &loc);

  // lane = thread % 32
  // warpOff = (thread/32) % warpsPerTile(0)
  llvm::SmallVector<Value> computeOffsets(Value lane, Value cSwizzleOffset) {
    if (canUseLdmatrix)
      return computeLdmatrixMatOffs(lane, cSwizzleOffset);
    else
      return computeLdsMatOffs(lane, cSwizzleOffset);
    return {};
  }

  int getNumPtrs() const { return numPtrs; }

  // Compute the offset to the matrix this thread(indexed by warpOff and lane)
  // mapped to.
  SmallVector<Value> computeLdmatrixMatOffs(Value lane, Value cSwizzleOffset);
  // compute 8-bit matrix offset.
  SmallVector<Value> computeLdsMatOffs(Value lane, Value cSwizzleOffset);

  // Load 4 matrices and returns 4 vec<2> elements.
  std::tuple<Value, Value, Value, Value> loadX4(int batch, int mat0, int mat1,
                                                ArrayRef<Value> ptrs,
                                                Type matTy,
                                                Type shemPtrTy) const;

private:
  SmallVector<uint32_t> order;
  SmallVector<uint32_t> warpsPerCTA;
  int kOrder;
  int nonKOrder;
  int kWidth;
  int vecWidth;
  SmallVector<int64_t> tileShape;
  SmallVector<int> instrShape;
  SmallVector<int> matShape;
  SmallVector<Value> multiDimWarpId;
  int perPhase;
  int maxPhase;
  int elemBytes;
  int mmaElemBytes;
  bool isHopper;
  ConversionPatternRewriter &rewriter;
  const Location &loc;
  MLIRContext *ctx{};

  // ldmatrix loads a matrix of size stridedMatShape x contiguousMatShape
  int contiguousMatShape;
  int stridedMatShape;

  // Offset in shared memory to increment on the strided axis
  // This would be different than the tile shape in the case of a sliced tensor
  Value stridedSmemOffset;
  Value smemBatchOffset;

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
MMA16816SmemLoader::computeLdmatrixMatOffs(Value lane, Value cSwizzleOffset) {
  Value warpB = multiDimWarpId[0];
  Value warpId = kOrder == 2 ? multiDimWarpId[1] : multiDimWarpId[2];
  // 4x4 matrices
  Value rowInMat = urem(lane, i32_val(8)); // row in the 8x8 matrix
  Value matIndex =
      udiv(lane, i32_val(8)); // linear index of the matrix in the 2x2 matrices

  // Decompose matIndex => s_0, s_1, that is the coordinate in 2x2 matrices in a
  // warp
  Value matIndexY = urem(matIndex, i32_val(2));
  Value matIndexX = udiv(matIndex, i32_val(2));

  // We use different orders for a and b for better performance.
  Value kMatArr =
      kOrder == 2 ? matIndexX : matIndexY; // index of matrix on the k dim
  Value nkMatArr =
      kOrder == 2 ? matIndexY : matIndexX; // index of matrix on the non-k dim

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

  Value matOff[3];
  // When B's shape(k, n) is (16, 8) and ldmatrix.x4 is used, the shared memory
  // access will be out of bound. In the future we should change this case to
  // ldmatrix.x2
  if (kOrder == 1 && nPerWarp == 8) {
    matOff[nonKOrder] = mul(warpId, i32_val(warpMatOffset));
  } else {
    matOff[nonKOrder] = add(
        mul(warpId, i32_val(warpMatOffset)), // warp offset (kOrder=2)
        mul(nkMatArr,
            i32_val(
                inWarpMatOffset))); // matrix offset inside a warp (kOrder=2)
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
    if (tileShape[0] != 1) {
      Value batchOffset =
          mul(warpB, i32_val(tileShape[order[0]] * tileShape[order[1]]));
      offs[i] =
          add(batchOffset,
              add(mul(contiguousIndexSwizzled, i32_val(contiguousMatShape)),
                  mul(rowOffset, stridedSmemOffset)));
    } else {
      offs[i] = add(mul(contiguousIndexSwizzled, i32_val(contiguousMatShape)),
                    mul(rowOffset, stridedSmemOffset));
    }
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
//   t4 ...    t4  t5 ... t5  t6 ... t6  t7 ... t7   ||   t4 ...  t4  t5 ... t5  t6 ... t6  t7 ... t7   |
//   t8 ...    t8  t9 ... t9 t10 .. t10 t11 .. t11   ||   t8 ...  t8  t9 ... t9 t10 .. t10 t11 .. t11   | quad height
//      ...                                                                                             |
//   t28 ...  t28 t29 .. t29 t30 .. t30 t31 .. t31   ||   t28 .. t28 t29 .. t29 t30 .. t30 t31 .. t31  \|/
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

SmallVector<Value> MMA16816SmemLoader::computeLdsMatOffs(Value lane,
                                                         Value cSwizzleOffset) {
  Value warpB = multiDimWarpId[0];
  Value warpOff = kOrder == 2 ? multiDimWarpId[1] : multiDimWarpId[2];
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
          int pStride = kOrder == 2 ? 1 : 2;
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
        if (tileShape[0] != 1) {
          Value batchOffset =
              mul(warpB, i32_val(tileShape[order[0]] * tileShape[order[1]]));
          offs[idx] = add(batchOffset, offs[idx]);
        }
      }

  return offs;
}

std::tuple<Value, Value, Value, Value>
MMA16816SmemLoader::loadX4(int batch, int mat0, int mat1, ArrayRef<Value> ptrs,
                           Type matTy, Type shemTy) const {
  assert(mat0 % 2 == 0 && mat1 % 2 == 0 && "smem matrix load must be aligned");
  int matIdx[3] = {0, mat0, mat1};

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
  auto resTy = cast<LLVM::LLVMStructType>(matTy);
  Type elemTy = cast<LLVM::LLVMStructType>(matTy).getBody()[0];

  // For some reasons, LLVM's NVPTX backend inserts unnecessary (?) integer
  // instructions to pack & unpack sub-word integers. A workaround is to
  // store the results of ldmatrix in i32
  if (auto vecElemTy = dyn_cast<VectorType>(elemTy)) {
    Type elemElemTy = vecElemTy.getElementType();
    if (auto intTy = dyn_cast<IntegerType>(elemElemTy)) {
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
    if (batch != 0)
      stridedOffset = add(
          stridedOffset, mul(i32_val(batch * warpsPerCTA[0]), smemBatchOffset));

    Value readPtr = gep(ptr_ty(ctx, 3), shemTy, ptr, stridedOffset);

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
    // ptrs[k][...] holds `vec` pointers each for (quadK == k)
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
      _i1 += (kOrder == 2 ? 1 : stridedLoadMatOffset) * stridedMatShape;
    Value i0 = mul(i32_val(_i0), stridedSmemOffset);
    Value i1 = mul(i32_val(_i1), stridedSmemOffset);
    if (batch != 0) {
      i0 = add(i0, mul(i32_val(batch * warpsPerCTA[0]), smemBatchOffset));
      i1 = add(i1, mul(i32_val(batch * warpsPerCTA[0]), smemBatchOffset));
    }
    // ii[m] holds the offset for (quadM == m)
    std::array<Value, 2> ii = {i0, i1};
    // load 4 32-bit values from shared memory
    // (equivalent to ldmatrix.x4)
    SmallVector<SmallVector<Value>> vptrs(4, SmallVector<Value>(vecWidth));

    // i iterates the 2x2 quads, m-first
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < vecWidth; ++j) {
        vptrs[i][j] = gep(ptr_ty(ctx, 3), shemTy, ptrs[i / 2][j], ii[i % 2]);
      }
    // row + trans and col + no-trans are equivalent
    bool isActualTrans =
        (needTrans && kOrder == 2) || (!needTrans && kOrder == 1);
    // pack loaded vectors into 4 32-bit values
    int inc = needTrans ? 1 : kWidth;
    VectorType packedTy = vec_ty(int_ty(8 * elemBytes), inc);
    int canonBits = std::min(32, 8 * elemBytes * inc);
    int canonWidth = (8 * elemBytes * inc) / canonBits;
    Type canonInt = int_ty(canonBits);
    std::array<Value, 4> retElems;
    // don't pack to 32b for Hopper
    int vecSize = isHopper ? 1 : 32 / canonBits;
    retElems.fill(undef(vec_ty(canonInt, vecSize)));
    for (int r = 0; r < 2; ++r) {
      for (int em = 0; em < 2 * vecWidth; em += inc) {
        int e = em % vecWidth;
        int m = em / vecWidth;
        int idx = m * 2 + r;
        Value ptr = bitcast(vptrs[idx][e], ptr_ty(ctx, 3));
        Value val = load(packedTy, ptr);
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

    auto iTy = isHopper ? int_ty(8 * elemBytes * inc) : i32_ty;

    return {bitcast(retElems[0], iTy), bitcast(retElems[1], iTy),
            bitcast(retElems[2], iTy), bitcast(retElems[3], iTy)};
  }
}

MMA16816SmemLoader::MMA16816SmemLoader(
    int nPerWarp, int warpsPerTile, ArrayRef<uint32_t> order,
    ArrayRef<uint32_t> warpsPerCTA, uint32_t kOrder, int kWidth,
    ArrayRef<Value> smemStrides, ArrayRef<int64_t> tileShape,
    ArrayRef<int> instrShape, ArrayRef<int> matShape,
    SmallVector<Value> multiDimWarpId, int perPhase, int maxPhase,
    int elemBytes, int mmaElemBytes, bool isHopper,
    ConversionPatternRewriter &rewriter, const LLVMTypeConverter *typeConverter,
    const Location &loc)
    : nPerWarp(nPerWarp), order(order.begin(), order.end()),
      warpsPerCTA(warpsPerCTA.begin(), warpsPerCTA.end()), kOrder(kOrder),
      kWidth(kWidth), tileShape(tileShape.begin(), tileShape.end()),
      instrShape(instrShape.begin(), instrShape.end()),
      matShape(matShape.begin(), matShape.end()),
      multiDimWarpId(multiDimWarpId.begin(), multiDimWarpId.end()),
      perPhase(perPhase), maxPhase(maxPhase), elemBytes(elemBytes),
      mmaElemBytes(mmaElemBytes), isHopper(isHopper), rewriter(rewriter),
      loc(loc), ctx(rewriter.getContext()) {
  // If the current elemType width is different from the MMA elemType width,
  // i.e. width-changing casting is done later in DotOp Layout... then, in the
  // case of Hopper, the number of bytes held by each thread after loading will
  // no longer be 32B. Hence this flag is required to stipulate different logic.
  bool isHopperWidthChange = isHopper && (mmaElemBytes != elemBytes);

  contiguousMatShape = matShape[order[0]];
  stridedMatShape = matShape[order[1]];
  stridedSmemOffset = smemStrides[order[1]];
  smemBatchOffset = smemStrides[order[2]];
  if (isHopperWidthChange) {
    vecWidth = 4 / mmaElemBytes;
  } else {
    vecWidth = 4 / elemBytes;
  }
  // rule: k must be the fast-changing axis.
  needTrans = kOrder != order[0];
  nonKOrder = (kOrder == 2) ? 1 : 2;
  canUseLdmatrix = elemBytes == 2 || (!needTrans);
  canUseLdmatrix = canUseLdmatrix && (kWidth == vecWidth);
  canUseLdmatrix = canUseLdmatrix && !isHopperWidthChange;

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

  int loadOffsetInMat[3];
  loadOffsetInMat[kOrder] =
      2; // instrShape[kOrder] / matShape[kOrder], always 2
  loadOffsetInMat[nonKOrder] =
      warpsPerTile * (instrShape[nonKOrder] / matShape[nonKOrder]);

  contiguousLoadMatOffset = loadOffsetInMat[order[0]];

  stridedLoadMatOffset =
      loadOffsetInMat[order[1]] / (instrShape[order[1]] / matShape[order[1]]);

  // The stride (in number of matrices) within warp
  inWarpMatOffset = kOrder == 2 ? 1 : warpsPerTile;
  // The stride (in number of matrices) of each warp
  warpMatOffset = instrShape[nonKOrder] / matShape[nonKOrder];
}

Type getSharedMemTy(Type argType) {
  MLIRContext *ctx = argType.getContext();
  if (argType.isF16())
    return type::f16Ty(ctx);
  else if (argType.isBF16())
    return type::bf16Ty(ctx);
  else if (argType.isF32())
    return type::f32Ty(ctx);
  else if (argType.getIntOrFloatBitWidth() == 8)
    return type::i8Ty(ctx);
  else if (argType.isInteger(16) || argType.isInteger(32)) {
    auto bitwidth = argType.getIntOrFloatBitWidth();
    auto signed_type =
        argType.isSignedInteger() ? IntegerType::Signed : IntegerType::Unsigned;
    return IntegerType::get(ctx, bitwidth, signed_type);
  } else
    llvm::report_fatal_error("mma16816 data type not supported");
}

Value composeValuesToDotOperandLayoutStruct(
    const ValueTable &vals, int batch, int repOuter, int repK,
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter, Type eltTy, bool isHopper, bool isA) {
  auto bitwidth = eltTy.getIntOrFloatBitWidth();
  assert(32 >= bitwidth && "only support 32-bit or less");
  auto numElemsPerVec = 32 / bitwidth;

  std::vector<Value> elems;
  auto unpackVec = [&](Value val) {
    // The following code unfortunately does not work, not sure why
    // auto vec = bitcast(val, vecTy);
    // for (auto i = 0; i < numElemsPerVec; ++i)
    //  elems.push_back(bitcast(extract_element(vec, i32_val(i)), eltTy));
    Value valI32 = bitcast(val, i32_ty);
    if (bitwidth == 32) {
      elems.push_back(bitcast(valI32, eltTy));
    } else if (bitwidth == 16) {
      auto val0 = and_(valI32, i32_val(0xffff));
      auto val1 = and_(lshr(valI32, i32_val(16)), i32_val(0xffff));
      auto val0I16 = bitcast(trunc(i16_ty, val0), eltTy);
      auto val1I16 = bitcast(trunc(i16_ty, val1), eltTy);
      elems.push_back(val0I16);
      elems.push_back(val1I16);
    } else if (bitwidth == 8) {
      auto val0 = and_(valI32, i32_val(0xff));
      auto val1 = and_(lshr(valI32, i32_val(8)), i32_val(0xff));
      auto val2 = and_(lshr(valI32, i32_val(16)), i32_val(0xff));
      auto val3 = and_(lshr(valI32, i32_val(24)), i32_val(0xff));
      auto val0I8 = bitcast(trunc(i8_ty, val0), eltTy);
      auto val1I8 = bitcast(trunc(i8_ty, val1), eltTy);
      auto val2I8 = bitcast(trunc(i8_ty, val2), eltTy);
      auto val3I8 = bitcast(trunc(i8_ty, val3), eltTy);
      elems.push_back(val0I8);
      elems.push_back(val1I8);
      elems.push_back(val2I8);
      elems.push_back(val3I8);
    } else {
      llvm_unreachable("unsupported bitwidth");
    }
  };

  // Loading A tile is different from loading B tile since each tile of A is
  // 16x16 while B is 16x8.
  if (isA) {
    for (int b = 0; b < batch; ++b)
      for (int m = 0; m < repOuter; ++m)
        for (int k = 0; k < repK; ++k) {
          unpackVec(vals.at({b, 2 * m, 2 * k}));
          unpackVec(vals.at({b, 2 * m + 1, 2 * k}));
          unpackVec(vals.at({b, 2 * m, 2 * k + 1}));
          unpackVec(vals.at({b, 2 * m + 1, 2 * k + 1}));
        }
  } else {
    for (int b = 0; b < batch; ++b)
      for (int n = 0; n < repOuter; ++n)
        for (int k = 0; k < repK; ++k) {
          unpackVec(vals.at({b, n, 2 * k}));
          unpackVec(vals.at({b, n, 2 * k + 1}));
        }
  }
  assert(!elems.empty());

  MLIRContext *ctx = eltTy.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(elems.size(), eltTy));
  auto result = packLLElements(loc, typeConverter, elems, rewriter, structTy);
  return result;
}

std::function<void(int, int, int)>
getLoadMatrixFn(MemDescType descTy, const SharedMemoryObject &smemObj,
                NvidiaMmaEncodingAttr mmaLayout, int warpsPerTile,
                uint32_t kOrder, int kWidth, SmallVector<int> instrShape,
                SmallVector<int> matShape, SmallVector<Value> multiDimWarpId,
                Value lane, ValueTable &vals, bool isA,
                const LLVMTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, Location loc) {
  auto shapePerCTA = getShapePerCTA(descTy);
  Type eltTy = descTy.getElementType();
  // We assumes that the input operand of Dot should be from shared layout.
  // TODO(Superjomn) Consider other layouts if needed later.
  auto sharedLayout = mlir::cast<SharedEncodingAttr>(descTy.getEncoding());
  const int perPhase = sharedLayout.getPerPhase();
  const int maxPhase = sharedLayout.getMaxPhase();
  const int vecPhase = sharedLayout.getVec();
  const int elemBytes = descTy.getElementTypeBitWidth() / 8;
  const int mmaElemBytes = 4 / kWidth;
  const bool isHopper = mmaLayout.getVersionMajor() == 3;
  auto order = sharedLayout.getOrder();

  int nPerWarp =
      std::max<int>(shapePerCTA[2] / mmaLayout.getWarpsPerCTA()[2], 8);
  // (a, b) is the coordinate.
  auto load = [=, &rewriter, &vals](int batch, int a, int b) {
    MMA16816SmemLoader loader(nPerWarp, warpsPerTile, sharedLayout.getOrder(),
                              mmaLayout.getWarpsPerCTA(), kOrder, kWidth,
                              smemObj.strides, shapePerCTA /*tileShape*/,
                              instrShape, matShape, multiDimWarpId, perPhase,
                              maxPhase, elemBytes, mmaElemBytes, isHopper,
                              rewriter, typeConverter, loc);
    // Offset of a slice within the original tensor in shared memory
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    SmallVector<Value> offs = loader.computeOffsets(lane, cSwizzleOffset);
    // initialize pointers
    const int numPtrs = loader.getNumPtrs();
    SmallVector<Value> ptrs(numPtrs);
    Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
    Type smemTy = getSharedMemTy(eltTy);
    for (int i = 0; i < numPtrs; ++i)
      ptrs[i] =
          gep(ptr_ty(rewriter.getContext(), 3), smemTy, smemBase, offs[i]);
    // actually load from shared memory
    auto matTy = LLVM::LLVMStructType::getLiteral(eltTy.getContext(),
                                                  SmallVector<Type>(4, i32_ty));
    auto [ha0, ha1, ha2, ha3] = loader.loadX4(
        batch, (kOrder == 2) ? a : b /*mat0*/, (kOrder == 2) ? b : a /*mat1*/,
        ptrs, matTy, getSharedMemTy(eltTy));

    if (!isA)
      std::swap(ha1, ha2);
    // the following is incorrect
    // but causes dramatically better performance in ptxas
    // although it only changes the order of operands in
    // `mma.sync`
    // if(isA)
    //   std::swap(ha1, ha2);
    // update user-provided values in-place
    vals[{batch, a, b}] = ha0;
    vals[{batch, a + 1, b}] = ha1;
    vals[{batch, a, b + 1}] = ha2;
    vals[{batch, a + 1, b + 1}] = ha3;
  };

  return load;
}

Value loadArg(ConversionPatternRewriter &rewriter, Location loc,
              MemDescType descTy, DotOperandEncodingAttr encoding,
              const SharedMemoryObject &smemObj,
              const LLVMTypeConverter *typeConverter, Value thread, bool isA) {
  auto mmaLayout = mlir::cast<NvidiaMmaEncodingAttr>(encoding.getParent());
  bool isHopper = mmaLayout.getVersionMajor() == 3;
  auto shapePerCTA = getShapePerCTA(descTy);
  int bitwidth = descTy.getElementTypeBitWidth();
  // For Hopper WGMMA, the sum of bitwidth of the elements in each quad should
  // add up to 32. We use kWidth to compute the element bitwidth of the input to
  // WGMMA, which could be different from `bitwidth` due to later casting.
  int mmaBitwidth = isHopper ? (32 / encoding.getKWidth()) : bitwidth;

  ValueTable vals;
  int mmaInstrM = 16, mmaInstrN = 8, mmaInstrK = 4 * 64 / mmaBitwidth;
  int matShapeM = 8, matShapeN = 8, matShapeK = 2 * 64 / mmaBitwidth;

  int kWidth = encoding.getKWidth();
  auto numRep =
      mmaLayout.getRepForOperand(shapePerCTA, bitwidth, encoding.getOpIdx());

  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto order = triton::gpu::getOrder(mmaLayout);
  Value warp = udiv(thread, i32_val(32));
  Value lane = urem(thread, i32_val(32));

  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warp, warpsPerCTA, order);
  Value warpB = urem(multiDimWarpId[0], i32_val(shapePerCTA[0]));
  int warpsPerTile;
  Value warpM = urem(multiDimWarpId[1], i32_val(shapePerCTA[1] / 16));
  Value warpN = urem(multiDimWarpId[2], i32_val(shapePerCTA[2] / 8));
  if (isA)
    warpsPerTile = std::min<int>(warpsPerCTA[1], shapePerCTA[1] / 16);
  else
    warpsPerTile = std::min<int>(warpsPerCTA[2], shapePerCTA[2] / 16);
  std::function<void(int, int, int)> loadFn;
  if (isA)
    loadFn = getLoadMatrixFn(
        descTy, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/, 2 /*kOrder*/,
        kWidth, {1, mmaInstrM, mmaInstrK} /*instrShape*/,
        {1, matShapeM, matShapeK} /*matShape*/,
        {warpB, warpM, warpN} /*multiDimWarpId*/, lane /*laneId*/,
        vals /*vals*/, isA /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/);
  else
    loadFn = getLoadMatrixFn(
        descTy, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/, 1 /*kOrder*/,
        kWidth, {1, mmaInstrK, mmaInstrN} /*instrShape*/,
        {1, matShapeK, matShapeN} /*matShape*/,
        {warpB, warpM, warpN} /*multiDimWarpId*/, lane /*laneId*/,
        vals /*vals*/, isA /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/);

  // Perform loading.
  int numRepBatch = numRep[0];
  int numRepOuter = isA ? numRep[1] : std::max<int>(numRep[2] / 2, 1);
  int numRepK = isA ? numRep[2] : numRep[1];
  for (int b = 0; b < numRepBatch; ++b)
    for (int m = 0; m < numRepOuter; ++m)
      for (int k = 0; k < numRepK; ++k)
        loadFn(b, 2 * m, 2 * k);

  // Format the values to LLVM::Struct to passing to mma codegen.
  Type eltTy = typeConverter->convertType(descTy.getElementType());
  return composeValuesToDotOperandLayoutStruct(
      vals, numRepBatch, isA ? numRep[1] : numRep[2], numRepK, typeConverter,
      loc, rewriter, eltTy, isHopper, isA);
}

template <typename T>
SmallVector<T> insertValue(ArrayRef<T> vec, unsigned index, T value) {
  SmallVector<T> res(vec.begin(), vec.end());
  res.insert(res.begin() + index, value);
  return res;
}
template <typename T>
SmallVector<T> insertValue(const SmallVector<T> &vec, unsigned index, T value) {
  SmallVector<T> res(vec.begin(), vec.end());
  res.insert(res.begin() + index, value);
  return res;
}

CTALayoutAttr getExpandedCTALayout(MLIRContext *ctx,
                                   CTALayoutAttr ctaLayoutAttr) {
  auto rank = ctaLayoutAttr.getCTAsPerCGA().size();
  auto ctasPerCGA3d =
      insertValue<unsigned>(ctaLayoutAttr.getCTAsPerCGA(), rank, 1);
  auto ctasSplitNum3d =
      insertValue<unsigned>(ctaLayoutAttr.getCTASplitNum(), rank, 1);
  auto ctaOrder3d =
      insertValue<unsigned>(ctaLayoutAttr.getCTAOrder(), rank, rank);
  auto expandedCTALayout = triton::gpu::CTALayoutAttr::get(
      ctx, ctasPerCGA3d, ctasSplitNum3d, ctaOrder3d);
  return expandedCTALayout;
}

Attribute getExpandedEncoding(Attribute encoding) {
  auto ctx = encoding.getContext();
  if (auto sharedEncoding = mlir::dyn_cast<SharedEncodingAttr>(encoding)) {
    auto order = sharedEncoding.getOrder();
    auto rank = order.size();
    if (rank == 3) {
      return encoding;
    }
    auto expandedOrder = SmallVector<unsigned>(3, 0);
    expandedOrder[0] = order[0] + 1;
    expandedOrder[1] = order[1] + 1;
    ArrayRef<unsigned> expandedOrderArr(expandedOrder);
    auto expandedEncoding = SharedEncodingAttr::get(
        ctx, sharedEncoding.getVec(), sharedEncoding.getPerPhase(),
        sharedEncoding.getMaxPhase(), expandedOrderArr,
        getExpandedCTALayout(ctx, sharedEncoding.getCTALayout()),
        sharedEncoding.getHasLeadingOffset());
    return expandedEncoding;
  } else if (auto mmaEncoding =
                 mlir::dyn_cast<NvidiaMmaEncodingAttr>(encoding)) {
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(mmaEncoding);
    auto rank = warpsPerCTA.size();
    if (rank == 3) {
      return encoding;
    }
    auto expandedWarpsPerCTA = insertValue<unsigned>(warpsPerCTA, 0, 1);
    auto instrShape = mmaEncoding.getInstrShape();
    auto expandedInstrShape = insertValue<unsigned>(instrShape, 0, 1);
    auto expandedMmaEncoding = NvidiaMmaEncodingAttr::get(
        ctx, mmaEncoding.getVersionMajor(), mmaEncoding.getVersionMinor(),
        expandedWarpsPerCTA,
        getExpandedCTALayout(ctx, mmaEncoding.getCTALayout()),
        expandedInstrShape);
    return expandedMmaEncoding;
  } else if (auto dotOperandEncoding =
                 mlir::dyn_cast<DotOperandEncodingAttr>(encoding)) {
    auto mmaEncoding =
        mlir::cast<NvidiaMmaEncodingAttr>(dotOperandEncoding.getParent());
    auto expandedMMAEncoding = getExpandedEncoding(mmaEncoding);
    auto expandedEncoding = DotOperandEncodingAttr::get(
        ctx, dotOperandEncoding.getOpIdx(), expandedMMAEncoding,
        dotOperandEncoding.getKWidth());
    return expandedEncoding;
  } else
    llvm_unreachable("unsupported encoding");
}

MemDescType getExpandedDesc(MemDescType descTy) {
  auto shapePerCTA = getShapePerCTA(descTy);
  auto rank = shapePerCTA.size();
  if (rank == 3)
    return descTy;

  auto elTy = descTy.getElementType();
  auto shape = descTy.getShape();
  auto expandedShape = SmallVector<int64_t>(3, 1);
  expandedShape[1] = shape[0];
  expandedShape[2] = shape[1];
  auto encoding = descTy.getEncoding();
  auto expandedEncoding = getExpandedEncoding(encoding);
  auto expandedDesc = MemDescType::get(expandedShape, elTy, expandedEncoding,
                                       descTy.getMemorySpace());
  return expandedDesc;
}

SharedMemoryObject
getExpandedSharedMemoryObject(ConversionPatternRewriter &rewriter, Location loc,
                              SharedMemoryObject smemObj,
                              ArrayRef<int64_t> shape) {
  auto strides = smemObj.getStrides();
  auto offsets = smemObj.getOffsets();
  auto rank = strides.size();
  if (rank == 3)
    return smemObj;
  auto expandedStrides = insertValue(strides, 0, i32_val(shape[0] * shape[1]));
  auto expandedOffsets = insertValue(offsets, 0, i32_val(0));
  auto expandedSmemObj =
      SharedMemoryObject(smemObj.getBase(), smemObj.getBaseElemType(),
                         expandedStrides, expandedOffsets);
  return expandedSmemObj;
}

namespace SharedToDotOperandMMAv2OrV3 {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread) {
  // Expand shared/dotOp to 3D before calling loadArg.
  auto descTy = cast<MemDescType>(tensor.getType());
  auto expandedDescTy = getExpandedDesc(descTy);
  auto expandedEncoding =
      cast<DotOperandEncodingAttr>(getExpandedEncoding(encoding));
  auto expandedSmemObj =
      getExpandedSharedMemoryObject(rewriter, loc, smemObj, descTy.getShape());
  if (opIdx == 0)
    return loadArg(rewriter, loc, expandedDescTy, expandedEncoding,
                   expandedSmemObj, typeConverter, thread, true);
  else {
    assert(opIdx == 1);
    return loadArg(rewriter, loc, expandedDescTy, expandedEncoding,
                   expandedSmemObj, typeConverter, thread, false);
  }
}
} // namespace SharedToDotOperandMMAv2OrV3
