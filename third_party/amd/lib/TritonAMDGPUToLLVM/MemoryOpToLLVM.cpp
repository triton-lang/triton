#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

using mlir::triton::amdgpu::ISAFamily;
using ::mlir::triton::gpu::MemDescType;

namespace {
class TransLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  TransLocalLoadOpConversion(const LLVMTypeConverter &converter,
                             const AMD::TargetInfo &targetInfo,
                             PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename triton::gpu::LocalLoadOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();

    auto typeConverter = this->getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    unsigned bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    // FP4 is represented as i8 and, when packed along K, can be
    // transposed using ds_read_tr8 which doesn't change packing.
    if (bitWidth != 16 && bitWidth != 8) {
      return failure();
    }
    auto ldsParamsVec = targetInfo.queryLDSTransLoadParams(bitWidth);
    if (ldsParamsVec.empty())
      return failure();

    LinearLayout sharedLL;
    if (triton::gpu::isPaddedEncoding(srcTy.getEncoding())) {
      sharedLL = triton::gpu::paddedLinearLayout(srcTy);
    } else {
      sharedLL = triton::gpu::toLinearLayout(srcTy);
    }
    LinearLayout cvtDstLL =
        triton::gpu::toLinearLayout(dstTy).invertAndCompose(sharedLL);
    auto kBlock = StringAttr::get(ctx, "block");
    auto maybeSublayout = cvtDstLL.quotient({kBlock});
    if (!maybeSublayout) {
      return failure();
    }
    cvtDstLL = maybeSublayout.value();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    SmallVector<Value> smemBases = llvm::to_vector(smemObj.getBases());
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
    auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);
    auto paddingShifts = getPaddedSharedShifts(srcTy.getEncoding(),
                                               srcTy.getElementTypeBitWidth(),
                                               /*offsetInBytes=*/true);

    for (const auto &ldsParams : ldsParamsVec) {
      if (triton::gpu::isPaddedEncoding(srcTy.getEncoding()) &&
          triton::gpu::getMinInterval(srcTy.getEncoding()) <
              ldsParams.tileSize) {
        continue;
      }

      llvm::SmallVector<Value> values;
      auto result =
          lowerDsReadTr(op, ldsParams, loc, cvtDstLL, values, smemBases,
                        affineOffset, maskSpanAffineOffset, paddingShifts,
                        llvmElemTy, rewriter, targetInfo);
      if (failed(result))
        continue;

      auto structTy = LLVM::LLVMStructType::getLiteral(
          ctx, SmallVector<Type>(values.size(), llvmElemTy));
      auto value =
          packLLElements(loc, typeConverter, values, rewriter, structTy);

      rewriter.replaceOp(op, value);
      return success();
    }
    return failure();
  }

private:
  LogicalResult
  lowerDsReadTr(triton::gpu::LocalLoadOp op,
                ::triton::AMD::TargetInfo::LDSTransLoadParams ldsParams,
                Location loc, LinearLayout cvt, SmallVector<Value> &vals,
                ArrayRef<Value> smemBases, Value affineOffset,
                uint64_t maskSpanAffineOffset,
                ArrayRef<std::pair<unsigned, unsigned>> paddingShifts,
                Type llvmElemTy, ConversionPatternRewriter &rewriter,
                const ::triton::AMD::TargetInfo &targetInfo) const {

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();

    auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
    auto kReg = S("register");
    auto kLane = S("lane");
    auto kWarp = S("warp");
    auto kOffset = S("offset");
    auto kAddr = S("addr");
    auto kPartition = S("partition");
    auto smemPtrTy = ptr_ty(ctx, 3);
    auto bitWidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);

    assert(!smemBases.empty() && "expected at least one smem base");
    LinearLayout cvtLayout = cvt;
    LinearLayout partitionLayout;
    Value basesVec;
    const bool isPartitioned = smemBases.size() > 1;

    if (isPartitioned) {
      assert(cvtLayout.hasOutDim(kPartition) &&
             cvtLayout.getOutDimSize(kPartition) ==
                 static_cast<int32_t>(smemBases.size()) &&
             "smemBases size must match partition dimension size");
      auto inDimNames = llvm::to_vector(cvtLayout.getInDimNames());
      partitionLayout = cvtLayout.sublayout(inDimNames, {kPartition});
      SmallVector<StringAttr> outDims =
          llvm::to_vector(cvtLayout.getOutDimNames());
      llvm::erase(outDims, kPartition);
      cvtLayout = cvtLayout.sublayout(inDimNames, outDims);
      basesVec = LLVM::buildBasePtrVector(loc, rewriter, smemBases);
    }

    // Map onto offsets (contiguous part) and addr (non-contiguous part)
    LinearLayout fullTile;
    // Contiguous tile
    LinearLayout tile;
    // ds_read_tr*_b64 performs a cooperative transposed load across 16
    // threads. The instruction processes an Nx16 tile (N=4 for 16-bit, N=8 for
    // 8-bit). The loaded tile is re-packed/transposed where lane i will
    // receive the i-th column.
    //
    // Loaded tile layout (input):     Register layout (output after transpose):
    //     K0  K1  ... K15               R0  R1  R2  R3
    // M0[ ............... ]    =>  T0 [ .   .   .   . ]
    // M1[ ............... ]        T1 [ .   .   .   . ]
    // M2[ ............... ]        ...
    // M3[ ............... ]        T15[ .   .   .   . ]
    //
    // Each lane loads 64 contiguous bits from LDS. After the transpose,
    // lane i receives column i from the input (elements strided by 16
    // the loaded tile).
    //
    // For example with N=4 (16-bit):
    // - Lane 0 receives elements from column 0: originally at [t0,t4,t8,t12]
    // - Lane 1 receives elements from column 1: originally at [t0,t4,t8,t12]
    //   These are the second 16 bits loaded by the same lanes before repacking
    // - Lane 4 receives elements from column 4: originally at [t1,t5,t9,t13]
    //
    // Note that there is no restriction on where elements are loaded
    // from, only that each lane needs to load 64 contiguous bits from shared
    // memory. We require N number of lanes to be contiguous since they read
    // consecutive 64 bits loaded from the same lanes.
    tile = LinearLayout::identity1D(ldsParams.tileSize, kLane, kOffset);
    const auto isaFamily = targetInfo.getISAFamily();
    const unsigned missingLanes =
        targetInfo.getWarpSize() / tile.getInDimSize(kLane);
    unsigned otherLanes = 1;
    if (isaFamily == ISAFamily::CDNA4) {
      otherLanes = (bitWidth == 8) ? 2 : 4;
    } else if (ldsParams.tileKind ==
               AMD::TargetInfo::TileKind::DoubleContiguity) {
      otherLanes = 2;
    }

    switch (ldsParams.tileKind) {
    case AMD::TargetInfo::TileKind::DoubleContiguity:
      fullTile =
          tile * LinearLayout::identity1D(ldsParams.tileSize / 2, kReg, kAddr) *
          LinearLayout::identity1D(otherLanes, kLane, kAddr) *
          LinearLayout::identity1D(2, kReg, kAddr) *
          LinearLayout::identity1D(missingLanes / otherLanes, kLane, kAddr);
      break;
    case AMD::TargetInfo::TileKind::Standard:
      fullTile =
          tile * LinearLayout::identity1D(otherLanes, kLane, kAddr) *
          LinearLayout::identity1D(ldsParams.tileSize, kReg, kAddr) *
          LinearLayout::identity1D(missingLanes / otherLanes, kLane, kAddr);
      break;
    }
    // Add warp dimension so we can invert and compose with reps later
    fullTile *= LinearLayout::identity1D(1, kWarp, kAddr);

    if (cvtLayout.getInDimSize(kReg) < fullTile.getInDimSize(kReg)) {
      return failure();
    }

    auto maybeQuot = divideLeft(cvtLayout, tile);
    if (!maybeQuot.has_value()) {
      return failure();
    }

    // From here on we perform the lowering
    auto reps = zerosLike(tile) * maybeQuot.value();

    // Sanity check
    assert(fullTile.getInDimSize(kReg) * bitWidth == ldsParams.instBitWidth);

    // If we are lowering a subslice, the subslice offsets shall not touch the
    // contiguous part of the tile
    if (maskSpanAffineOffset & (tile.getOutDimSize(kOffset) - 1)) {
      return failure();
    }

    // fullTile.invert() is a map from kOffset, kAddr into kReg, kLane, kWarp
    // addrToOffset gives us a map from kAddr into kOffset, which is the map of
    // the addresses each lane should hold
    auto addrToOffset = fullTile.invert().compose(reps);
    // sanity check
    assert(addrToOffset.getInDimSizeLog2(kAddr) >= 3 &&
           addrToOffset.getInDimSizeLog2(kAddr) <= 6);

    LinearLayout addrLayout =
        LinearLayout({{kLane, addrToOffset.getBases().lookup(kAddr)},
                      {kWarp, reps.getBases().lookup(kWarp)}},
                     {{kOffset, reps.getOutDimSize(kOffset)}}, false);

    // Compute the bits that are moved by one instruction
    // Compute elements for which we can swap the xor by an add
    auto [nAdditive, permStrides] = actionAdditiveStrides(
        reps, addrLayout, maskSpanAffineOffset, fullTile.getInDimSize(kReg));
    reps = permStrides.apply(reps);
    if (isPartitioned) {
      partitionLayout = permStrides.apply(partitionLayout);

      // One ds_read_tr* instruction produces `fullTile.getInDimSize(kReg)`
      // consecutive register values from a single LDS base pointer. We only
      // select a partition once per instruction, so all of those register
      // positions must map to the same partition. For a LinearLayout that holds
      // iff the low log2(elemsPerInstr) register bases contribute 0 to
      // kPartition. Bail out if not, so a generic lowering can take over.
      const unsigned numInstrRegBits =
          llvm::Log2_32(fullTile.getInDimSize(kReg));
      for (unsigned pos = 0; pos < numInstrRegBits; ++pos) {
        if (partitionLayout.getBasis(kReg, pos, kPartition) != 0)
          return failure();
      }
    }

    // Perform computation in bytes, LLVM optimises this better
    assert(bitWidth >= 8);
    auto i8Tile =
        zerosLike(LinearLayout::identity1D(bitWidth / 8, kReg, kOffset));
    auto i8AddrLayout = i8Tile * addrLayout;

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto regBase =
        applyLinearLayout(
            loc, rewriter, i8AddrLayout,
            {{kReg, b.i32_val(0)}, {kLane, laneId}, {kWarp, warpId}})[0]
            .second;

    // It's fine that we don't compute the offset in bytes as affineOffset
    // will be folded into a constant
    auto affineOffsetI8 = b.mul(affineOffset, b.i32_val(bitWidth / 8));
    bool hasPadding = !paddingShifts.empty();
    Value paddedAffineOffsetI8 = b.i32_val(0);
    if (hasPadding && maskSpanAffineOffset != 0) {
      // `maskSpanAffineOffset != 0` indicates the affine offsets come from
      // MemDescSubsliceOp, whose verifier guarantees that the affine offsets
      // are bitwise disjoint from other offset contributors. Padding can thus
      // be applied separately. This helps LLVM reuse base pointers.
      paddedAffineOffsetI8 =
          applyPadding(loc, rewriter, affineOffsetI8, paddingShifts);
    } else {
      regBase = b.xor_(regBase, affineOffsetI8);
    }

    // Elements per op
    auto elemsPerInstr = fullTile.getInDimSize(kReg);
    auto elemsPerVec = ldsParams.instBitWidth / bitWidth;
    auto vecTy = vec_ty(llvmElemTy, elemsPerVec);
    for (int i = 0; i < cvtLayout.getInDimSize(kReg); i += nAdditive) {
      auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
      auto regIdxI8 = regIdx * (bitWidth / 8);
      Value offset = b.xor_(regBase, b.i32_val(regIdxI8));

      if (hasPadding) {
        offset = applyPadding(loc, rewriter, offset, paddingShifts);
        if (maskSpanAffineOffset != 0)
          offset = b.add(offset, paddedAffineOffsetI8);
      }

      for (int i2 = 0; i2 < nAdditive; i2 += elemsPerInstr) {
        // all these constants will go as immediate values to ds_read_tr
        auto regIdxAdd =
            reps.apply({{kReg, i2}, {kLane, 0}, {kWarp, 0}})[0].second;
        auto regIdxAddI8 = regIdxAdd * (bitWidth / 8);
        // `actionAdditiveStrides` forces `regIdxAddI8` and `offset` to be
        // bitwise disjoint, so we can calculate their padding contributions
        // separately.
        regIdxAddI8 = applyPadding(regIdxAddI8, paddingShifts);
        Value innerOffset = b.add(offset, b.i32_val(regIdxAddI8));
        Value smemBaseVal = smemBases[0];
        if (isPartitioned) {
          auto partOut = applyLinearLayout(
              loc, rewriter, partitionLayout,
              {{kReg, b.i32_val(i + i2)}, {kLane, laneId}, {kWarp, warpId}});
          smemBaseVal = b.extract_element(basesVec, partOut[0].second);
        }
        auto vecAddr = b.gep(smemPtrTy, i8_ty, smemBaseVal, innerOffset,
                             LLVM::GEPNoWrapFlags::inbounds);
        llvm::append_range(vals,
                           emitDsReadTr(op, loc, vecAddr, vecTy, llvmElemTy,
                                        rewriter, targetInfo));
      }
    }
    // apply all the inverse permutations in the reverse order
    assert(vals.size() == cvtLayout.getInDimSize(kReg));
    vals = permStrides.inverse().apply(vals);

    return success();
  }

  // Emits a single ds_read_tr* operation at `vecAddr` and unpacks the loaded
  // vector into individual element Values. Returns an empty vector if the ISA
  // family does not support a ds_read_tr* instruction.
  SmallVector<Value>
  emitDsReadTr(triton::gpu::LocalLoadOp op, Location loc, Value vecAddr,
               VectorType vTy, Type llvmElemTy,
               ConversionPatternRewriter &rewriter,
               const ::triton::AMD::TargetInfo &targetInfo) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    const auto bitWidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
    assert(bitWidth == 16 || bitWidth == 8);

    Value dsReadTr = createDsReadTr(op, rewriter, loc, vecAddr, vTy,
                                    targetInfo.getISAFamily(), bitWidth);
    if (!dsReadTr)
      return {};

    Value vecVal = b.bitcast(dsReadTr, vTy);
    SmallVector<Value> loadedVals;
    for (int v = 0; v < vTy.getNumElements(); v++)
      loadedVals.push_back(b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
    return loadedVals;
  }

  // Creates and returns the result Value of a single ds_read_tr* op for the
  // given (isaFamily, bitWidth).
  static Value createDsReadTr(triton::gpu::LocalLoadOp op,
                              RewriterBase &rewriter, Location loc,
                              Value vecAddr, VectorType vTy,
                              ISAFamily isaFamily, unsigned bitWidth) {
    // tr16 instructions return vectors of bf16/f16 while "tr8" instructions
    // return vectors of i32. Generate the corresponding i32 vector type.
    const auto numElemsI32 = (vTy.getNumElements() * bitWidth / 32);
    const auto vTyI32 = VectorType::get(numElemsI32, i32_ty);

    // GFX1250 uses opaque LLVM intrinsic calls; their results cannot be cast to
    // AliasAnalysisOpInterface, so no no-alias scope is attached.
    auto callIntrinsic = [&](StringRef name, VectorType retTy) -> Value {
      return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, {retTy},
                                             {vecAddr})
          .getResult(0);
    };

    switch (isaFamily) {
    case ISAFamily::GFX1250:
      if (bitWidth == 16)
        return callIntrinsic("llvm.amdgcn.ds.load.tr16.b128", vTy);
      return callIntrinsic("llvm.amdgcn.ds.load.tr8.b64", vTyI32);
    case ISAFamily::CDNA4: {
      Value dsReadTr;
      if (bitWidth == 16)
        dsReadTr = ROCDL::ds_read_tr16_b64::create(rewriter, loc, vTy, vecAddr);
      else
        dsReadTr =
            ROCDL::ds_read_tr8_b64::create(rewriter, loc, vTyI32, vecAddr);
      AMD::addLocalLoadNoAliasScope(
          op, cast<LLVM::AliasAnalysisOpInterface>(dsReadTr.getDefiningOp()));
      return dsReadTr;
    }
    default:
      return {};
    }
  }

private:
  const AMD::TargetInfo &targetInfo;
};

class LocalLoadPackedTransposedOpConversion
    : public ConvertOpToLLVMPattern<
          triton::amdgpu::LocalLoadPackedTransposedOp> {
public:
  LocalLoadPackedTransposedOpConversion(const LLVMTypeConverter &converter,
                                        const AMD::TargetInfo &targetInfo,
                                        PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<triton::amdgpu::LocalLoadPackedTransposedOp>(
            converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor =
      typename triton::amdgpu::LocalLoadPackedTransposedOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::amdgpu::LocalLoadPackedTransposedOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType dstTy = op.getType();
    auto typeConverter = this->getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    unsigned bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    // FP4 is represented as i8 and
    if (bitWidth != 8) {
      return failure();
    }
    // FP4 packed along M/N are not supported yet on GFX1250
    if (targetInfo.getISAFamily() == ISAFamily::GFX1250) {
      return failure();
    }

    return lowerSharedToDotOperandTransLL(op, adaptor, typeConverter, rewriter);
  }

private:
  LogicalResult
  lowerSharedToDotOperandTransLL(triton::amdgpu::LocalLoadPackedTransposedOp op,
                                 OpAdaptor adaptor,
                                 const LLVMTypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kBlock = str_attr("block");
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitWidth = llvmElemTy.getIntOrFloatBitWidth();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    mlir::Type retTy = dstTy;
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
    auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);
    auto paddingShifts = getPaddedSharedShifts(srcTy.getEncoding(), bitWidth,
                                               /*offsetInBytes=*/true);

    auto shape = srcTy.getShape();
    auto ldsParamsVec = targetInfo.queryLDSTransLoadParams(bitWidth);
    if (ldsParamsVec.size() != 1)
      return failure();
    const auto ldsTransLoadParams = &ldsParamsVec[0];
    // FP4 are packed into i8 so the real bitWidth is different
    auto llBitWidth = 4;
    auto ldsTransLayout = triton::gpu::chooseDsReadTrLayout(
        dstTy.getEncoding(), shape, llBitWidth,
        ldsTransLoadParams->instBitWidth,
        ldsTransLoadParams->numLanesInShuffleGroup);

    // Check that we have computed a layout
    if (!ldsTransLayout) {
      return failure();
    }

    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(srcTy.getEncoding());
    LinearLayout cvt = LinearLayout::empty();
    if (paddedEnc) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvt = ldsTransLayout->invertAndCompose(sharedLL);
    } else {
      auto sharedLL = triton::gpu::toLinearLayout(srcTy);
      cvt = ldsTransLayout->invertAndCompose(sharedLL);
    }
    // Check that we will be able to vectorize the load.
    // Need to have exactly ldsTransLoadParams->tileSize,
    // otherwise we can't use ds_read_tr
    auto [elemsPerVec, permutation] =
        largestVectorisation(ctx, cvt, bitWidth, ldsTransLoadParams->tileSize);

    if (paddedEnc)
      elemsPerVec = std::min<int>(elemsPerVec, paddedEnc.getMinInterval());

    if (elemsPerVec != ldsTransLoadParams->tileSize)
      return failure();

    assert(cvt.isTrivialOver({kBlock}) && "NYI");
    auto lowerInst = [&](RewriterBase &rewriter, Location loc,
                         ArrayRef<Value> inVals, Value vecAddr, int idx,
                         VectorType vTy,
                         std::optional<Value> ctaId) -> SmallVector<Value> {
      assert(!ctaId.has_value() && "NYI");
      auto numElemsI32 = (vTy.getNumElements() * bitWidth / 32);
      auto vTyI32 = VectorType::get(numElemsI32, i32_ty);
      Value dsReadTr =
          ROCDL::ds_read_tr4_b64::create(rewriter, loc, vTyI32, vecAddr);
      Value vecVal = b.bitcast(dsReadTr, vTy);
      SmallVector<Value> loadedVals;
      for (int v = 0; v < vTy.getNumElements(); v++) {
        loadedVals.push_back(
            b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
      }

      return loadedVals;
    };

    SmallVector<Value> outVals = lowerLdSt(
        loc, rewriter.getContext(), cvt, {}, // Input for store, output for load
        llvmElemTy, smemObj.getBase(), paddingShifts, affineOffset,
        maskSpanAffineOffset, laneId, warpId, rewriter, targetInfo,
        ldsTransLoadParams->tileSize, lowerInst);
    Value result = packLLElements(loc, typeConverter, outVals, rewriter, retTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

class BarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
public:
  BarrierOpConversion(const LLVMTypeConverter &converter,
                      const AMD::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::BarrierOp>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename triton::gpu::BarrierOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo.getIsaVersion().Major < 9)
      return failure();
    // Check no other memory addrspaces are selected.
    // TensorRead/Write are allowed but noop.
    auto mask = triton::gpu::AddrSpace::Local |
                triton::gpu::AddrSpace::GlobalRead |
                triton::gpu::AddrSpace::GlobalWrite |
                triton::gpu::AddrSpace::TensorRead |
                triton::gpu::AddrSpace::TensorWrite;
    if ((op.getAddrSpace() & ~mask) != triton::gpu::AddrSpace::None)
      return failure();
    // We can lower barrier to MemoryCounterWaitOp + s_barrier
    // - MemoryCounterWaitOp specifies how many operations to
    //   VMEM(Read)/VMEM(Write)/LDS can be outstanding when
    //   the instruction completes.
    // - s_barrier synchronizes the execution for the CTA
    IntegerAttr zero = rewriter.getI32IntegerAttr(0);
    bool localBarrier = op.hasLocal();
    bool globalBarrier = op.hasGlobalRead() || op.hasGlobalWrite();
    if (localBarrier || globalBarrier) {
      amdgpu::MemoryCounterWaitOp::create(
          rewriter, op->getLoc(),
          /* load= */ op.hasGlobalRead() ? zero : nullptr,
          /* store= */ op.hasGlobalWrite() ? zero : nullptr,
          /* ds= */ localBarrier ? zero : nullptr);
    }
    rewriter.replaceOpWithNewOp<ROCDL::SBarrierOp>(op);

    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

/// Encodes the waitcnt value for AMDGPU architectures.
///
/// Note: This function duplicates the bitpacking logic from AMDGPU backend
/// (llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.h), as it's not accessible from
/// llvm/include. The logic handles different encoding schemes across
/// various GPU architecture versions (pre-gfx9 to gfx11).
///
/// The waitcnt encoding uses different bit positions for each counter
/// based on the ISA version:
/// - Vmcnt (vector memory counter): tracks pending vector memory operations
/// - Expcnt (export counter): tracks pending export operations
/// - Lgkmcnt (LDS/GDS/scalar memory counter): tracks pending LDS/GDS/scalar
/// memory ops
///
/// Each architecture version has its own bit layout, Vmcnt, Expcnt and Lgkmcnt
/// are decoded as follows:
///     Vmcnt = Waitcnt[3:0]        (pre-gfx9)
///     Vmcnt = Waitcnt[15:14,3:0]  (gfx9,10)
///     Vmcnt = Waitcnt[15:10]      (gfx11)
///     Expcnt = Waitcnt[6:4]       (pre-gfx11)
///     Expcnt = Waitcnt[2:0]       (gfx11)
///     Lgkmcnt = Waitcnt[11:8]     (pre-gfx10)
///     Lgkmcnt = Waitcnt[13:8]     (gfx10)
///     Lgkmcnt = Waitcnt[9:4]      (gfx11)
static FailureOr<unsigned> encodeWaitcnt(llvm::AMDGPU::IsaVersion isaVersion,
                                         unsigned vmcnt, unsigned lgkmcnt) {
  if (isaVersion.Major == 9) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(15u, lgkmcnt);
    unsigned lowBits = vmcnt & 0xF;
    unsigned highBits = (vmcnt >> 4) << 14;
    unsigned otherCnts = (expcnt << 4) | (lgkmcnt << 8);
    return lowBits | highBits | otherCnts;
  }
  if (isaVersion.Major == 10) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(63u, lgkmcnt);
    unsigned lowBits = vmcnt & 0xF;
    unsigned highBits = (vmcnt >> 4) << 14;
    unsigned otherCnts = (expcnt << 4) | (lgkmcnt << 8);
    return lowBits | highBits | otherCnts;
  }
  if (isaVersion.Major == 11) {
    vmcnt = std::min(63u, vmcnt);
    unsigned expcnt = 0x7;
    lgkmcnt = std::min(63u, lgkmcnt);
    return (vmcnt << 10) | expcnt | (lgkmcnt << 4);
  }
  return failure();
}

struct MemoryCounterWaitOpConversion
    : public ConvertOpToLLVMPattern<amdgpu::MemoryCounterWaitOp> {
  MemoryCounterWaitOpConversion(const LLVMTypeConverter &converter,
                                const AMD::TargetInfo &targetInfo,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(amdgpu::MemoryCounterWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // amdgpu::MemoryCounterWaitOp supports gfx9 onwards
    auto isaVersion = targetInfo.getIsaVersion();

    /// If major version >= gfx12, lower to
    ///   * ROCDL::WaitDscntOp if ds is present
    ///   * ROCDL::WaitLoadcntOp if load is present
    ///   * ROCDL::WaitStorecntOp if store is present
    if (isaVersion.Major >= 12) {
      Location loc = op.getLoc();
      if (std::optional<int> ds = adaptor.getDs())
        ROCDL::WaitDscntOp::create(rewriter, loc, *ds);

      if (std::optional<int> load = adaptor.getLoad())
        ROCDL::WaitLoadcntOp::create(rewriter, loc, *load);

      if (std::optional<int> store = adaptor.getStore())
        ROCDL::WaitStorecntOp::create(rewriter, loc, *store);

      rewriter.eraseOp(op);
      return success();
    }

    /// Otherwise, lower to ROCDL::SWaitcntOp
    auto getVal = [](Attribute attr) -> unsigned {
      if (attr)
        return cast<IntegerAttr>(attr).getInt();

      // This value will be clamped to the maximum value for the target version.
      return 1024;
    };
    unsigned ds = getVal(adaptor.getDsAttr());

    unsigned vmcnt = 1024;
    Attribute load = adaptor.getLoadAttr();
    Attribute store = adaptor.getStoreAttr();
    if (load && store) {
      vmcnt = getVal(load) + getVal(store);
    } else if (load) {
      vmcnt = getVal(load);
    } else if (store) {
      vmcnt = getVal(store);
    }

    FailureOr<unsigned> waitcnt = encodeWaitcnt(isaVersion, vmcnt, ds);
    if (failed(waitcnt))
      return op.emitOpError("unsupported chipset");

    rewriter.replaceOpWithNewOp<ROCDL::SWaitcntOp>(op, *waitcnt);
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  PatternBenefit transBenefit = PatternBenefit(benefit.getBenefit() + 1);
  PatternBenefit barrierBenefit = PatternBenefit(benefit.getBenefit() + 1);

  patterns.add<TransLocalLoadOpConversion>(typeConverter, targetInfo,
                                           transBenefit);
  patterns.add<LocalLoadPackedTransposedOpConversion>(typeConverter, targetInfo,
                                                      benefit);
  patterns.add<BarrierOpConversion, MemoryCounterWaitOpConversion>(
      typeConverter, targetInfo, barrierBenefit);
}
