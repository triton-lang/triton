#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"

namespace SharedToDotOperandMMAv1 {
using CoordTy = SmallVector<Value>;
using ValueTable = std::map<std::pair<int, int>, std::pair<Value, Value>>;

static SmallVector<CoordTy>
getMNCoords(Value thread, Location loc, ConversionPatternRewriter &rewriter,
            ArrayRef<unsigned int> wpt, const NvidiaMmaEncodingAttr &mmaLayout,
            ArrayRef<int64_t> shape, bool isARow, bool isBRow, bool isAVec4,
            bool isBVec4) {
  static constexpr std::array<int, 3> fpw{{2, 2, 1}};

  auto *ctx = thread.getContext();
  Value _1 = i32_val(1);
  Value _2 = i32_val(2);
  Value _4 = i32_val(4);
  Value _16 = i32_val(16);
  Value _32 = i32_val(32);
  Value _fpw0 = i32_val(fpw[0]);
  Value _fpw1 = i32_val(fpw[1]);

  // A info
  auto aRep = mmaLayout.getMMAv1Rep(0);
  auto aSpw = mmaLayout.getMMAv1ShapePerWarp(0);
  // B info
  auto bSpw = mmaLayout.getMMAv1ShapePerWarp(1);
  auto bRep = mmaLayout.getMMAv1Rep(1);

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
} // namespace SharedToDotOperandMMAv1
namespace mlir {

namespace LLVM {
using namespace mlir::triton;
using mlir::triton::gpu::getOrder;
using mlir::triton::gpu::getSizePerThread;

Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<LLVM::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

Value createConstantF32(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f32Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF32FloatAttr(v));
}

Value createConstantF64(Location loc, OpBuilder &rewriter, double v) {
  auto type = type::f64Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF64FloatAttr(v));
}

Value createNaNConstant(Location loc, OpBuilder &rewriter, Type type) {
  if (!type.isa<FloatType>()) {
    llvm::report_fatal_error("Creating NaN constant for non-float type!");
  }
  return rewriter.create<LLVM::ConstantOp>(
      loc, type, APFloat::getNaN(type.cast<FloatType>().getFloatSemantics()));
}

// Create an index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,
                          TypeConverter *converter, int64_t value) {
  Type ty = converter->convertType(builder.getIndexType());
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

// Create an integer constant of \param width bits.
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value) {
  Type ty = builder.getIntegerType(width);
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

// A wrapper of LoadDSmemOp when vec = 1
// (1) Get bitwidth from elemTy
// (2) Create LoadDSmemOp
// (3) Bitcast result from dataTy (u16/u32/u64) back to elemTy
Value createLoadDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Type elemTy) {
  assert(addr.getType().isa<LLVMPointerType>() &&
         "addr must be a pointer type");
  auto ptrTy = addr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  Value ret =
      rewriter.create<triton::nvgpu::LoadDSmemOp>(loc, addr, ctaId, bitwidth);
  return bitcast(ret, elemTy);
}

// A wrapper of LoadDSmemOp when vec > 1
// (1) Get bitwidth from elemTy
// (2) Create LoadDSmemOp and extract results from retStruct
// (3) Bitcast results from dataTy (u16/u32/u64) back to elemTy
SmallVector<Value> createLoadDSmem(Location loc, PatternRewriter &rewriter,
                                   Value addr, Value ctaId, unsigned vec,
                                   Type elemTy) {
  assert(addr.getType().isa<LLVMPointerType>() &&
         "addr must be a pointer type");
  auto ptrTy = addr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  Value retStruct = rewriter.create<triton::nvgpu::LoadDSmemOp>(
      loc, addr, ctaId, bitwidth, vec);
  SmallVector<Value> retVals;
  for (unsigned i = 0; i < vec; ++i) {
    auto dataTy = rewriter.getIntegerType(bitwidth);
    Value data = extract_val(dataTy, retStruct, i);
    retVals.push_back(bitcast(data, elemTy));
  }
  return retVals;
}

// A wrapper of StoreDSmemOp when vec = 1
// (1) Get bitwidth from elemTy
// (2) Bitcast value from elemTy to dataTy (u16/u32/u64)
// (3) Create StoreDSmemOp
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value, Value pred) {
  assert(addr.getType().isa<LLVMPointerType>() &&
         "addr must be a pointer type");
  auto ptrTy = addr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = value.getType().getIntOrFloatBitWidth();
  auto dataTy = rewriter.getIntegerType(bitwidth);
  Value data = bitcast(value, dataTy);
  rewriter.create<triton::nvgpu::StoreDSmemOp>(loc, addr, ctaId, data, pred);
}

// A wrapper of StoreDSmemOp when vec = 1 and pred = 1
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value) {
  Value pred = int_val(/*width=*/1, 1);
  createStoreDSmem(loc, rewriter, addr, ctaId, value, pred);
}

// A wrapper of StoreDSmemOp when vec > 1
// (1) Get bitwidth from elemTy
// (2) Bitcast values from elemTy to dataTy (u16/u32/u64)
// (3) Create StoreDSmemOp
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values, Value pred) {
  assert(addr.getType().isa<LLVMPointerType>() &&
         "addr must be a pointer type");
  auto ptrTy = addr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = 0;
  if (!values.empty()) {
    bitwidth = values.back().getType().getIntOrFloatBitWidth();
  }
  auto dataTy = rewriter.getIntegerType(bitwidth);
  SmallVector<Value> data;
  for (unsigned i = 0; i < values.size(); ++i)
    data.push_back(bitcast(values[i], dataTy));
  rewriter.create<triton::nvgpu::StoreDSmemOp>(loc, addr, ctaId, data, pred);
}

// A wrapper of StoreDSmemOp when vec > 1 and pred = 1
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values) {
  Value pred = int_val(/*width=*/1, 1);
  createStoreDSmem(loc, rewriter, addr, ctaId, values, pred);
}

SharedMemoryObject
getSharedMemoryObjectFromStruct(Location loc, Value llvmStruct, Type elemTy,
                                ConversionPatternRewriter &rewriter) {
  ArrayRef<Type> types =
      llvmStruct.getType().cast<LLVM::LLVMStructType>().getBody();
  SmallVector<Value> elems(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    elems[i] = extract_val(type, llvmStruct, i);
  }

  auto rank = (elems.size() - 1) / 2;
  return {/*base=*/elems[0],
          /*baseElemType=*/elemTy,
          /*strides=*/{elems.begin() + 1, elems.begin() + 1 + rank},
          /*offsets=*/{elems.begin() + 1 + rank, elems.end()}};
}

SmallVector<Value> getStridesFromShapeAndOrder(ArrayRef<int64_t> shape,
                                               ArrayRef<unsigned> order,
                                               Location loc,
                                               RewriterBase &rewriter) {
  auto rank = shape.size();
  SmallVector<Value> strides(rank);
  int64_t stride = 1;
  for (auto idx : order) {
    strides[idx] = i32_val(stride);
    stride *= shape[idx];
  }
  return strides;
}

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order) {
  unsigned rank = shape.size();
  assert(rank == order.size());
  auto reordered = applyPermutation(shape, order);
  SmallVector<Value> reorderedMultiDim(rank);
  if (auto constantOp = linear.getDefiningOp<arith::ConstantOp>()) {
    unsigned intVal =
        constantOp.getValue().cast<IntegerAttr>().getValue().getSExtValue();
    reorderedMultiDim = delinearize(rewriter, loc, intVal, reordered);
  } else {
    reorderedMultiDim = delinearize(rewriter, loc, linear, reordered);
  }
  SmallVector<Value> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               unsigned linear, ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  unsigned remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    unsigned dimSize = en.value();
    multiDim[en.index()] = i32_val(remained % dimSize);
    remained = remained / dimSize;
  }
  return multiDim;
}

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  Value remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    Value dimSize = i32_val(en.value());
    multiDim[en.index()] = urem(remained, dimSize);
    remained = udiv(remained, dimSize);
  }
  return multiDim;
}

Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<Value> multiDim, ArrayRef<unsigned> shape,
                ArrayRef<unsigned> order) {
  return linearize(rewriter, loc, applyPermutation(multiDim, order),
                   applyPermutation(shape, order));
}

Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<Value> multiDim, ArrayRef<unsigned> shape) {
  auto rank = multiDim.size();
  Value linear = i32_val(0);
  if (rank > 0) {
    linear = multiDim.back();
    for (auto [dim, dimShape] :
         llvm::reverse(llvm::zip(multiDim.drop_back(), shape.drop_back()))) {
      Value dimSize = i32_val(dimShape);
      linear = add(mul(linear, dimSize), dim);
    }
  }
  return linear;
}

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(ctx), globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
        rewriter.getStringAttr(contentStr));
  }

  Value zero = i32_val(0);
  Type globalPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value stringStart =
      gep(ptr_ty(ctx), i8_ty, globalPtr, SmallVector<Value>({zero}));
  return stringStart;
}

SmallVector<Value> getMultiDimOffset(Attribute layout, Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     unsigned elemId, RankedTensorType type,
                                     ArrayRef<unsigned> multiDimCTAInRepId,
                                     ArrayRef<unsigned> shapePerCTATile) {
  auto shape = type.getShape();
  unsigned rank = shape.size();
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
    auto multiDimOffsetFirstElem =
        emitBaseIndexForLayout(loc, rewriter, blockedLayout, type, false);
    SmallVector<Value> multiDimOffset(rank);
    SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
        elemId, getSizePerThread(layout), getOrder(layout));
    for (unsigned d = 0; d < rank; ++d) {
      multiDimOffset[d] =
          add(multiDimOffsetFirstElem[d],
              i32_val(multiDimCTAInRepId[d] * shapePerCTATile[d] +
                      multiDimElemId[d]));
    }
    return multiDimOffset;
  }
  if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    unsigned dim = sliceLayout.getDim();
    auto parentEncoding = sliceLayout.getParent();
    auto parentSizePerThread = getSizePerThread(parentEncoding);
    auto parentShape = sliceLayout.paddedShape(shape);
    auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                          parentEncoding);
    auto offsets = emitOffsetForLayout(layout, type);
    auto parentOffset = emitOffsetForLayout(parentEncoding, parentTy);
    SmallVector<int> idxs;
    for (SmallVector<unsigned> off : offsets) {
      off.insert(off.begin() + dim, 0);
      auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
      idxs.push_back(std::distance(parentOffset.begin(), it));
    }
    auto multiDimOffsetParent =
        getMultiDimOffset(parentEncoding, loc, rewriter, idxs[elemId], parentTy,
                          sliceLayout.paddedShape(multiDimCTAInRepId),
                          sliceLayout.paddedShape(shapePerCTATile));
    SmallVector<Value> multiDimOffset(rank);
    for (unsigned d = 0; d < rank + 1; ++d) {
      if (d == dim)
        continue;
      unsigned slicedD = d < dim ? d : (d - 1);
      multiDimOffset[slicedD] = multiDimOffsetParent[d];
    }
    return multiDimOffset;
  }
  if (auto mmaLayout = layout.dyn_cast<NvidiaMmaEncodingAttr>()) {
    assert(rank == 2 ||
           (rank == 3 && mmaLayout.isAmpere()) && "Unexpected rank");
    auto shapePerCTA = getShapePerCTA(mmaLayout, shape);
    auto instrShape = mmaLayout.getInstrShape();
    SmallVector<Value> mmaColIdx(2);
    SmallVector<Value> mmaRowIdx(2);
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    // TODO: fix the bug in MMAEncodingAttr document
    SmallVector<Value> multiDimWarpId(2);
    auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
    if (mmaLayout.isHopper()) {
      multiDimWarpId[0] = urem(warpId, i32_val(warpsPerCTA[0]));
      multiDimWarpId[1] = udiv(warpId, i32_val(warpsPerCTA[0]));
    } else {
      auto order = triton::gpu::getOrder(mmaLayout);
      multiDimWarpId = delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    }
    Value _1 = i32_val(1);
    Value _2 = i32_val(2);
    Value _4 = i32_val(4);
    Value _8 = i32_val(8);
    Value _16 = i32_val(16);
    if (mmaLayout.isAmpere() || mmaLayout.isHopper()) {
      multiDimWarpId[rank - 1] = urem(
          multiDimWarpId[rank - 1],
          i32_val(ceil<unsigned>(shapePerCTA[rank - 1], instrShape[rank - 1])));
      multiDimWarpId[rank - 2] = urem(
          multiDimWarpId[rank - 2],
          i32_val(ceil<unsigned>(shapePerCTA[rank - 2], instrShape[rank - 2])));

      Value mmaGrpId = udiv(laneId, _4);
      Value mmaGrpIdP8 = add(mmaGrpId, _8);
      Value mmaThreadIdInGrp = urem(laneId, _4);
      Value mmaThreadIdInGrpM2 = mul(mmaThreadIdInGrp, _2);
      Value mmaThreadIdInGrpM2P1 = add(mmaThreadIdInGrpM2, _1);
      Value rowWarpOffset =
          mul(multiDimWarpId[rank - 2], i32_val(instrShape[rank - 2]));
      mmaRowIdx[0] = add(mmaGrpId, rowWarpOffset);
      mmaRowIdx[1] = add(mmaGrpIdP8, rowWarpOffset);
      Value colWarpOffset =
          mul(multiDimWarpId[rank - 1], i32_val(instrShape[rank - 1]));
      mmaColIdx[0] = add(mmaThreadIdInGrpM2, colWarpOffset);
      mmaColIdx[1] = add(mmaThreadIdInGrpM2P1, colWarpOffset);
    } else if (mmaLayout.isVolta()) {
      // Volta doesn't follow the pattern here."
    } else {
      llvm_unreachable("Unexpected MMALayout version");
    }

    SmallVector<Value> multiDimOffset(rank);
    if (mmaLayout.isHopper()) {
      unsigned elemIdRem4 = elemId % 4;
      unsigned nGrpId = elemId / 4;
      multiDimOffset[0] = elemIdRem4 < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
      multiDimOffset[1] = elemIdRem4 % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
      multiDimOffset[1] = add(multiDimOffset[1], i32_val(8 * nGrpId));
      multiDimOffset[0] = add(multiDimOffset[0], i32_val(multiDimCTAInRepId[0] *
                                                         shapePerCTATile[0]));
      multiDimOffset[1] = add(multiDimOffset[1], i32_val(multiDimCTAInRepId[1] *
                                                         shapePerCTATile[1]));
    } else if (mmaLayout.isAmpere()) {
      if (rank == 3)
        multiDimOffset[0] =
            add(multiDimWarpId[0],
                i32_val(multiDimCTAInRepId[0] * shapePerCTATile[0]));
      multiDimOffset[rank - 2] = elemId < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
      multiDimOffset[rank - 1] = elemId % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
      multiDimOffset[rank - 2] =
          add(multiDimOffset[rank - 2], i32_val(multiDimCTAInRepId[rank - 2] *
                                                shapePerCTATile[rank - 2]));
      multiDimOffset[rank - 1] =
          add(multiDimOffset[rank - 1], i32_val(multiDimCTAInRepId[rank - 1] *
                                                shapePerCTATile[rank - 1]));
    } else if (mmaLayout.isVolta()) {
      auto [isARow, isBRow, isAVec4, isBVec4, _] =
          mmaLayout.decodeVoltaLayoutStates();
      auto coords = SharedToDotOperandMMAv1::getMNCoords(
          threadId, loc, rewriter, mmaLayout.getWarpsPerCTA(), mmaLayout, shape,
          isARow, isBRow, isAVec4, isBVec4);
      return coords[elemId];
    } else {
      llvm_unreachable("Unexpected MMALayout version");
    }
    return multiDimOffset;
  }
  if (auto mfmaLayout = layout.dyn_cast<AMDMfmaEncodingAttr>()) {
    auto multiDimBase =
        emitBaseIndexForLayout(loc, rewriter, layout, type, false);
    SmallVector<SmallVector<unsigned>> offsets;
    assert(rank == 2);
    SmallVector<Value> multiDimOffset(rank);
    emitMfmaOffsetForCTA(mfmaLayout, offsets, multiDimCTAInRepId[0],
                         multiDimCTAInRepId[1]);
    multiDimOffset[0] = add(multiDimBase[0], i32_val(offsets[elemId][0]));
    multiDimOffset[1] = add(multiDimBase[1], i32_val(offsets[elemId][1]));
    return multiDimOffset;
  }
  llvm_unreachable("unexpected layout in getMultiDimOffset");
}

SmallVector<Value> getWrappedMultiDimOffset(
    ConversionPatternRewriter &rewriter, Location loc,
    ArrayRef<Value> multiDimOffset, ArrayRef<unsigned> shape,
    SmallVector<unsigned> shapePerCTATile, SmallVector<int64_t> shapePerCTA) {
  unsigned rank = shape.size();
  SmallVector<Value> multiDimOffsetWrapped(rank);
  for (unsigned d = 0; d < rank; ++d) {
    if (shapePerCTATile[d] > shapePerCTA[d])
      multiDimOffsetWrapped[d] = urem(multiDimOffset[d], i32_val(shape[d]));
    else
      multiDimOffsetWrapped[d] = multiDimOffset[d];
  }
  return multiDimOffsetWrapped;
}

} // namespace LLVM
} // namespace mlir
