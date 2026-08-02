#include "AsyncUtility.h"
#include "AtomicRMWOpsEmitter.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include <type_traits>

using mlir::triton::amdgpu::ISAFamily;
using ::mlir::triton::gpu::MemDescType;

namespace {

static LLVM::FenceOp createAMDGPUMemoryFence(OpBuilder &builder, Location loc,
                                             LLVM::AtomicOrdering ordering,
                                             StringRef synchronizeAddrSpace) {
  auto fence =
      LLVM::FenceOp::create(builder, loc, ordering, /*syncscope=*/"workgroup");
  if (!synchronizeAddrSpace.empty()) {
    Attribute mmra = builder.getAttr<LLVM::MMRATagAttr>("amdgpu-synchronize-as",
                                                        synchronizeAddrSpace);
    fence->setDiscardableAttr(LLVM::LLVMDialect::getMmraAttrName(), mmra);
  }
  return fence;
}

// Creates and returns the result Value of a single ds_read_tr* op for the
// given (isaFamily, logicalBitWidth).
static Value createDsReadTr(Operation *op, RewriterBase &rewriter, Location loc,
                            Value vecAddr, VectorType vTy, ISAFamily isaFamily,
                            unsigned logicalBitWidth) {
  // tr16 instructions return vectors of bf16/f16 while tr8 and tr4
  // instructions return vectors of i32. Generate the corresponding i32 vector
  // type.
  const auto physicalBitWidth =
      getIntOrFloatOrPtrBitWidth(vTy.getElementType());
  const auto numElemsI32 = (vTy.getNumElements() * physicalBitWidth / 32);
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
    if (logicalBitWidth == 16)
      return callIntrinsic("llvm.amdgcn.ds.load.tr16.b128", vTy);
    if (logicalBitWidth == 8)
      return callIntrinsic("llvm.amdgcn.ds.load.tr8.b64", vTyI32);
    return {};
  case ISAFamily::CDNA4: {
    Value dsReadTr;
    if (logicalBitWidth == 16)
      dsReadTr = ROCDL::ds_read_tr16_b64::create(rewriter, loc, vTy, vecAddr);
    else if (logicalBitWidth == 8)
      dsReadTr = ROCDL::ds_read_tr8_b64::create(rewriter, loc, vTyI32, vecAddr);
    else if (logicalBitWidth == 4)
      dsReadTr = ROCDL::ds_read_tr4_b64::create(rewriter, loc, vTyI32, vecAddr);
    else
      return {};
    AMD::addLocalLoadNoAliasScope(
        op, cast<LLVM::AliasAnalysisOpInterface>(dsReadTr.getDefiningOp()));
    return dsReadTr;
  }
  default:
    return {};
  }
}

// Emits a single ds_read_tr* operation at `vecAddr` and unpacks the loaded
// vector into individual element Values. Returns an empty vector if the ISA
// family does not support a ds_read_tr* instruction.
SmallVector<Value> emitDsReadTr(Operation *op, Location loc, Value vecAddr,
                                VectorType vTy, Type llvmElemTy,
                                unsigned logicalBitWidth,
                                ConversionPatternRewriter &rewriter,
                                const ::triton::AMD::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  const auto physicalBitWidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
  assert(physicalBitWidth == 16 || physicalBitWidth == 8);

  Value dsReadTr = createDsReadTr(op, rewriter, loc, vecAddr, vTy,
                                  targetInfo.getISAFamily(), logicalBitWidth);
  if (!dsReadTr)
    return {};

  Value vecVal = b.bitcast(dsReadTr, vTy);
  SmallVector<Value> loadedVals;
  for (int v = 0; v < vTy.getNumElements(); v++)
    loadedVals.push_back(b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
  return loadedVals;
}

LogicalResult
lowerDsReadTr(Operation *op,
              ::triton::AMD::TargetInfo::LDSTransLoadParams ldsParams,
              Location loc, LinearLayout cvt, unsigned logicalBitWidth,
              SmallVector<Value> &vals, ArrayRef<Value> smemBases,
              Value affineOffset, uint64_t maskSpanAffineOffset,
              ArrayRef<std::pair<unsigned, unsigned>> paddingShifts,
              Type llvmElemTy, ConversionPatternRewriter &rewriter,
              const ::triton::AMD::TargetInfo &targetInfo) {

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();

  auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kOffset = S("offset");
  auto kBlock = S("block");
  auto kAddr = S("addr");
  auto kPartition = S("partition");
  auto smemPtrTy = ptr_ty(ctx, 3);
  auto bitWidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
  auto logicalMaskSpanAffineOffset =
      maskSpanAffineOffset * (bitWidth / logicalBitWidth);

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

  // A ds_read_trK_bN instruction takes one LDS base address from each of the
  // lanes in a warp, reads N / K contiguous K-bit elements from each base, and
  // distributes the elements along the lanes in a manner that can be described
  // by a bijective linear map
  //
  //                           F: R ⊕ L  ->  C ⊕ A,
  //
  // where R and L are the destination register and lane index spaces, and C and
  // A are the contiguous-offset and address-providing lane index spaces.
  //
  // The LinearLayout `fullTile` describes this map. Its `tile` factor maps the
  // first t := log2(N / K) bases of L to C. The bases mapped to A are an order-
  // preserving interleaving of the bases of R and the remaining bases of L:
  //
  //                   R[0:a], L[t:t + b], R[a:t], L[t + b:p],
  //
  // where p := log2(lanesPerWarp), and a and b are instruction-specific
  // parameters.
  auto tile = LinearLayout::identity1D(ldsParams.tileSize, kLane, kOffset);
  const unsigned numInstrRegBits = llvm::Log2_32(ldsParams.tileSize);
  const unsigned numAddrLaneBits =
      llvm::Log2_32(targetInfo.getWarpSize()) - numInstrRegBits;
  auto fullTile =
      tile *
      LinearLayout::identity1D(1 << ldsParams.leadingRegBases, kReg, kAddr) *
      LinearLayout::identity1D(1 << ldsParams.leadingLaneBases, kLane, kAddr) *
      LinearLayout::identity1D(
          1 << (numInstrRegBits - ldsParams.leadingRegBases), kReg, kAddr) *
      LinearLayout::identity1D(
          1 << (numAddrLaneBits - ldsParams.leadingLaneBases), kLane, kAddr) *
      LinearLayout::identity1D(1, kWarp, kAddr);

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
  assert(fullTile.getInDimSize(kReg) * logicalBitWidth ==
         ldsParams.instBitWidth);

  // If we are lowering a subslice, the subslice offsets shall not touch the
  // contiguous part of the tile
  if (logicalMaskSpanAffineOffset & (tile.getOutDimSize(kOffset) - 1)) {
    return failure();
  }

  // fullTile.invert() is a map from kOffset, kAddr into kReg, kLane, kWarp
  // addrToOffset gives us a map from kAddr into kOffset, which is the map of
  // the addresses each lane should hold
  auto addrToOffset = fullTile.invert().compose(reps);
  // sanity check
  assert(addrToOffset.getInDimSizeLog2(kAddr) >= 3 &&
         addrToOffset.getInDimSizeLog2(kAddr) <= 6);

  // ds_read_tr* shuffles data across lanes so the lane issuing the load
  // matches the kAddr decomposition of fullTile. Using addrToOffset's
  // kAddr bases as the kLane bases of this layout lets us use laneId
  // to get the LDS offset each lane should read.
  LinearLayout addrLayout =
      LinearLayout({{kLane, addrToOffset.getBases().lookup(kAddr)},
                    {kWarp, reps.getBases().lookup(kWarp)}},
                   {{kOffset, reps.getOutDimSize(kOffset)}}, false);

  if (logicalBitWidth == 4) {
    // Writing out the corresponding abstract maps for the above LinearLayouts:
    //
    // `cvtLayout`:                    G    :         D      ->     S
    // `tile * maybeQuot`:          T ⊕ Q  :     L_C ⊕ D'  ->   C ⊕ S'
    // `reps`:                      0 ⊕ Q  :     L_C ⊕ D'  ->   C ⊕ S'
    // `addrLayout`: (0 ⊕ Q) o F^{-1} o i_A:         A      ->   C ⊕ S',
    //
    // we see that `reps` and `addrLayout`, though constructed using nibble
    // coordinates, only have byte-aligned outputs. This allows us to safely
    // convert to byte coordinates by halving the nibble-offset values with
    // `logicalToI8` and dropping the low `kReg` basis in our layouts.
    auto logicalToI8 = LinearLayout::zeros1D(2, kOffset, kOffset) *
                       LinearLayout::identity1D(reps.getOutDimSize(kOffset) / 2,
                                                kOffset, kOffset);
    reps = reps.compose(logicalToI8);
    addrLayout = addrLayout.compose(logicalToI8);

    auto numLogicalRegBases = reps.getInDimSizeLog2(kReg);
    ColumnAction dropNibbleBasis(
        llvm::to_vector(llvm::seq<size_t>(1, numLogicalRegBases)), kReg,
        numLogicalRegBases);
    reps = dropNibbleBasis.apply(reps);
    if (isPartitioned)
      partitionLayout = dropNibbleBasis.apply(partitionLayout);
  }

  // Matrix accesses are CTA-local. Model that with a trivial block output so
  // additive stride analysis always compares (offset, block) components.
  reps =
      reps.reshapeOuts({{kOffset, reps.getOutDimSize(kOffset)}, {kBlock, 1}});
  addrLayout = addrLayout.reshapeOuts(reps.getOutDims());

  // Compute the bits that are moved by one instruction
  // Compute elements for which we can swap the xor by an add
  auto elemsPerInstr = ldsParams.instBitWidth / bitWidth;
  auto [nAdditive, permStrides] =
      actionAdditiveStrides(reps, addrLayout, maskSpanAffineOffset,
                            /*maskSpanBlocks=*/0, elemsPerInstr);
  reps = permStrides.apply(reps);
  if (isPartitioned) {
    partitionLayout = permStrides.apply(partitionLayout);

    // One ds_read_tr* instruction produces `elemsPerInstr` consecutive
    // physical values along kReg from a single LDS base pointer. We only
    // select a partition once per instruction, so all of those register
    // positions must map to the same partition. For a LinearLayout that holds
    // iff the low log2(elemsPerInstr) register bases contribute 0 to
    // kPartition. Bail out if not, so a generic lowering can take over.
    for (unsigned pos = 0; pos < llvm::Log2_32(elemsPerInstr); ++pos) {
      if (partitionLayout.getBasis(kReg, pos, kPartition) != 0)
        return failure();
    }

    // partitionLayout's kLane is the destination lane which is the lane that
    // owns the loaded data in the destination tensor. The laneId is the
    // source lane issuing the load. For ds_read_tr* the hardware shuffles
    // data across lanes, so the two differ: we need to remap.
    //
    // Example: ds_load_tr8_b64 on gfx1250, from the test
    // `ds_transpose_partitioned_remaps_lane`.
    //
    //  fullTile:
    //   - lane=1 -> (1, 0)
    //     lane=2 -> (2, 0)
    //     lane=4 -> (4, 0)
    //     lane=8 -> (0, 4)
    //     lane=16 -> (0, 16)
    //   - register=1 -> (0, 1)
    //     register=2 -> (0, 2)
    //     register=4 -> (0, 8)
    //   where out dims are: [offset (size 8), addr (size 32)]
    //
    // `addr` is the non-contiguous part of the source lane's access.
    // `lane` in the inverse tile is the destination lane after the hardware
    // transpose. `fullTile.invert().sublayout({kAddr}, {kLane})` gives:
    //
    //   - addr=1 -> (0)
    //     addr=2 -> (0)
    //     addr=4 -> (8)
    //     addr=8 -> (0)
    //     addr=16 -> (16)
    //   where out dims are: [lane (size 32)]
    //
    // Then rename the input dimension from `addr` to `lane` so the map can
    // compose with partitionLayout.
    //
    // For this test, partitionLayout would choose the partition from the
    // destination-lane basis `lane=8`:
    //
    //   - register=1 -> (0)
    //     ...
    //     register=32 -> (0)
    //   - lane=1 -> (0)
    //     lane=2 -> (0)
    //     lane=4 -> (0)
    //     lane=8 -> (1)
    //     lane=16 -> (0)
    //   - warp=1 -> (0)
    //     warp=2 -> (0)
    //   where out dims are: [partition (size 2)]
    //
    // Querying this with the runtime source lane asks for the partition of
    // the wrong lane. Composing with laneRemap rewrites the partition basis
    // through the transpose:
    //
    //   - register=1 -> (0)
    //     ...
    //     register=32 -> (0)
    //   - lane=1 -> (0)
    //     lane=2 -> (0)
    //     lane=4 -> (1)
    //     lane=8 -> (0)
    //     lane=16 -> (0)
    //   - warp=1 -> (0)
    //     warp=2 -> (0)
    //   where out dims are: [partition (size 2)]
    //
    // Destination basis `lane=8` is reached from source basis `addr=4`, so
    // each source lane selects the LDS base expected by its destination lane.

    auto regIdentity = LinearLayout::identity1D(
        partitionLayout.getInDimSize(kReg), kReg, kReg);
    auto srcToDstLaneMap =
        fullTile.invert().sublayout({kAddr}, {kLane}).renameInDim(kAddr, kLane);
    auto warpIdentity = LinearLayout::identity1D(
        partitionLayout.getInDimSize(kWarp), kWarp, kWarp);
    auto laneRemap = regIdentity * srcToDstLaneMap * warpIdentity;
    partitionLayout = laneRemap.compose(partitionLayout);
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

  auto vecTy = vec_ty(llvmElemTy, elemsPerInstr);
  for (int i = 0; i < reps.getInDimSize(kReg); i += nAdditive) {
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
                                      logicalBitWidth, rewriter, targetInfo));
    }
  }
  // apply all the inverse permutations in the reverse order
  assert(vals.size() == reps.getInDimSize(kReg));
  vals = permStrides.inverse().apply(vals);

  return success();
}

template <typename OpTy>
class TransLocalLoadOpConversion : public ConvertOpToLLVMPattern<OpTy> {
  static constexpr bool isPackedTransposed =
      std::is_same_v<OpTy, triton::amdgpu::LocalLoadPackedTransposedOp>;

public:
  TransLocalLoadOpConversion(const LLVMTypeConverter &converter,
                             const AMD::TargetInfo &targetInfo,
                             PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<OpTy>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();

    auto typeConverter = this->getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    unsigned bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    unsigned logicalBitWidth = bitWidth;
    if constexpr (isPackedTransposed) {
      // FP4 is represented as packed elements inside i8 values.
      if (bitWidth != 8)
        return failure();
      // FP4 packed along M/N are not supported yet on GFX1250
      if (targetInfo.getISAFamily() == ISAFamily::GFX1250)
        return failure();
      logicalBitWidth = 4;
    } else {
      // FP4 is represented as i8 and, when packed along K, can be
      // transposed using ds_read_tr8 which doesn't change packing.
      if (bitWidth != 16 && bitWidth != 8)
        return failure();
    }
    auto ldsParamsVec = targetInfo.queryLDSTransLoadParams(logicalBitWidth);
    if (ldsParamsVec.empty())
      return failure();
    if (SharedMemoryObject::getMaskSpanOffsetsAndBlocks(srcTy).second != 0)
      return failure();

    auto dstLL = triton::gpu::toLinearLayout(dstTy);
    LinearLayout sharedLL = triton::gpu::toLinearLayoutIgnoringPadding(srcTy);

    if constexpr (isPackedTransposed) {
      // Perform factorization and address routing in logical fp4 coordinates.
      std::optional<StringAttr> srcPackedDim;
      std::optional<StringAttr> dstPackedDim;
      auto srcShape = srcTy.getShape();
      auto dstShape = dstTy.getShape();
      assert(srcShape.size() == dstShape.size());
      auto outDimNames = llvm::to_vector(sharedLL.getOutDimNames());

      for (unsigned dim = 0; dim < srcShape.size(); ++dim) {
        if (srcShape[dim] * 2 == dstShape[dim]) {
          srcPackedDim = outDimNames[dim];
          continue;
        }
        if (dstShape[dim] * 2 == srcShape[dim]) {
          dstPackedDim = outDimNames[dim];
          continue;
        }
        if (srcShape[dim] != dstShape[dim])
          return failure();
      }
      if (!srcPackedDim || !dstPackedDim)
        return failure();

      auto kReg = str_attr("register");
      auto kOffset = str_attr("offset");
      sharedLL = LinearLayout::identity1D(2, kOffset, *srcPackedDim) * sharedLL;
      dstLL = LinearLayout::identity1D(2, kReg, *dstPackedDim) * dstLL;
    }

    auto cvtDstLL = dstLL.invertAndCompose(sharedLL);
    auto kBlock = StringAttr::get(ctx, "block");
    auto maybeSublayout = cvtDstLL.quotient({kBlock});
    if (!maybeSublayout)
      return failure();
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
              ldsParams.instBitWidth / bitWidth) {
        continue;
      }

      SmallVector<Value> values;
      auto result =
          lowerDsReadTr(op, ldsParams, loc, cvtDstLL, logicalBitWidth, values,
                        smemBases, affineOffset, maskSpanAffineOffset,
                        paddingShifts, llvmElemTy, rewriter, targetInfo);
      if (failed(result))
        continue;

      auto value =
          packTensorElements(loc, typeConverter, values, rewriter, dstTy);

      rewriter.replaceOp(op, value);
      return success();
    }
    return failure();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

struct LocalAtomicScatterRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAtomicScatterRMWOp> {

  LocalAtomicScatterRMWOpConversion(const LLVMTypeConverter &converter,
                                    const AMD::TargetInfo &targetInfo,
                                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAtomicScatterRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto lowering = prepareLocalAtomicScatterRMW(
        op, adaptor.getDst(), adaptor.getIndices(), adaptor.getValues(),
        op.getMask() ? adaptor.getMask() : Value(), rewriter, targetInfo,
        getTypeConverter());
    if (failed(lowering))
      return failure();
    LocalAtomicScatterRMWInfo &info = *lowering;

    auto binOp = matchAtomicOp(op.getAtomicRmwOp());
    if (!binOp)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW operation");

    // Lower to per-element llvm.atomicrmw on addrspace(3) with
    // syncscope("workgroup") monotonic.
    const auto memOrder = LLVM::AtomicOrdering::monotonic;
    const StringRef scope = "workgroup";
    LLVM::AMD::AtomicRMWEmitter emitter(targetInfo, *binOp, memOrder, scope);

    bool returnOld = !op.getResult().use_empty();

    if (llvm::any_of(info.addrs, [](const LocalSharedMemoryAddress &addr) {
          return bool(addr.ctaId);
        })) {
      return rewriter.notifyMatchFailure(
          op, "cross-CTA shared atomics are not supported on AMDGPU");
    }

    SmallVector<Value> results;
    if (returnOld)
      results.reserve(info.addrs.size());

    for (auto [i, addrAndValue] :
         llvm::enumerate(llvm::zip(info.addrs, info.values))) {
      auto [addr, value] = addrAndValue;
      Value rmwMask = triton::gpu::maybeAnd(
          rewriter, loc, info.threadPred,
          info.maskValues.empty() ? Value() : info.maskValues[i]);
      // emitAtomicRMW requires a non-null predicate, default to true if null.
      if (!rmwMask)
        rmwMask = b.true_val();

      Value old = emitter.emitAtomicRMW(rewriter, addr.ptr, value, rmwMask,
                                        /*sharedMemBase=*/std::nullopt,
                                        /*enableIntraWaveReduce=*/false);
      if (returnOld)
        results.push_back(old);
    }

    if (!returnOld) {
      rewriter.eraseOp(op);
      return success();
    }

    finalizeTensorAtomicResults(op, info.valuesTy, rewriter, results,
                                info.llvmElemTy, b, info.threadPred, targetInfo,
                                getTypeConverter());
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
    if (!mlir::triton::amdgpu::isCDNA(targetInfo.getISAFamily()))
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
    bool localBarrier = op.hasLocal();
    bool globalBarrier = op.hasGlobalRead() || op.hasGlobalWrite();
    if (localBarrier || globalBarrier) {
      StringRef mmraAddrSpace = "";
      if (localBarrier && !globalBarrier)
        mmraAddrSpace = "local";
      else if (!localBarrier && globalBarrier)
        mmraAddrSpace = "global";

      // Local/global barriers use LLVM fences so the AMDGPU memory legalizer
      // selects target-specific waits. Mixed local+global barriers are left
      // untagged so LLVM conservatively synchronizes every relevant space.
      createAMDGPUMemoryFence(rewriter, op->getLoc(),
                              LLVM::AtomicOrdering::release, mmraAddrSpace);
      ROCDL::SBarrierOp::create(rewriter, op->getLoc());
      createAMDGPUMemoryFence(rewriter, op->getLoc(),
                              LLVM::AtomicOrdering::acquire, mmraAddrSpace);
      rewriter.eraseOp(op);
      return success();
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

  patterns.add<TransLocalLoadOpConversion<triton::gpu::LocalLoadOp>>(
      typeConverter, targetInfo, transBenefit);
  patterns.add<
      TransLocalLoadOpConversion<triton::amdgpu::LocalLoadPackedTransposedOp>>(
      typeConverter, targetInfo, benefit);
  patterns.add<LocalAtomicScatterRMWOpConversion>(typeConverter, targetInfo,
                                                  benefit.getBenefit() + 1);
  patterns.add<BarrierOpConversion, MemoryCounterWaitOpConversion>(
      typeConverter, targetInfo, barrierBenefit);
}
