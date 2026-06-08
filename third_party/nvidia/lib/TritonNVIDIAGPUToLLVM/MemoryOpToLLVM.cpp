#include "Dialect/NVGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/AtomicPTXBuilder.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::NVIDIA;
using namespace mlir::LLVM::NVIDIA;

bool isConstI32OneTensor(Value value) {
  DenseElementsAttr constant;
  return matchPattern(value, m_Constant(&constant)) &&
         constant.getElementType().isInteger(32) &&
         llvm::all_of(constant.getValues<APInt>(),
                      [](const APInt &value) { return value.isOne(); });
}

Value emitSharedInc(ConversionPatternRewriter &rewriter, Location loc,
                    Value ptr, bool returnOld, Value pred = Value()) {
  PTXBuilder ptxBuilder;
  // PTX atom/red.inc resets to 0 only when the old value reaches the bound, so
  // using UINT32_MAX makes it equivalent to a wrapping increment-by-1.
  auto *boundOpr = ptxBuilder.newConstantOperand("0xffffffff");
  if (!returnOld) {
    auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "r");
    auto &red = *ptxBuilder.create("red");
    red.shared().o("cta").o("relaxed").o("inc").o("u32");
    red(ptrOpr, boundOpr).maybePredicate(pred, "b");
    return ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
  }

  auto *dstOpr = ptxBuilder.newOperand("=r", /*init=*/true);
  auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "r");
  auto &atom = *ptxBuilder.create("atom");
  atom.shared().o("cta").o("relaxed").o("inc").o("u32");
  atom(dstOpr, ptrOpr, boundOpr).maybePredicate(pred, "b");
  return ptxBuilder.launch(rewriter, loc, i32_ty);
}

FailureOr<Value> emitSharedAtomicRMW(ConversionPatternRewriter &rewriter,
                                     Location loc, Type valueElemTy, Value ptr,
                                     Value value, RMWOp rmwOp, bool returnOld,
                                     Value pred) {
  SmallVector<Value> vals{value};
  if (!returnOld) {
    auto result = emitPtxAtomicRMW(
        rewriter, loc, valueElemTy, ptr, vals, rmwOp, MemSemantic::RELAXED,
        MemSyncScope::CTA, pred, /*vec=*/1, /*packed=*/1,
        PtxAtomicAddrSpace::Shared, PtxAtomicInstr::Red);
    if (succeeded(result))
      return result;
  }

  return emitPtxAtomicRMW(rewriter, loc, valueElemTy, ptr, vals, rmwOp,
                          MemSemantic::RELAXED, MemSyncScope::CTA, pred,
                          /*vec=*/1, /*packed=*/1, PtxAtomicAddrSpace::Shared,
                          PtxAtomicInstr::Atom);
}

LogicalResult lowerLdStMatrix(
    Location loc, const LinearLayout &regLayout, MemDescType memDescType,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    SharedMemoryObject smemObj, ConversionPatternRewriter &rewriter,
    const NVIDIA::TargetInfo &targetInfo,
    const LLVMTypeConverter *typeConverter) {
  bool isStore = !vals.empty();

  // Remove broadcasting from regLayout
  auto removeBroadcast = actionRemoveBroadcastedRegs(regLayout);
  if (!removeBroadcast.isIdentity()) {
    if (isStore) {
      auto newRegLayout = removeBroadcast.apply(regLayout);
      vals = removeBroadcast.apply(vals);
      return lowerLdStMatrix(loc, newRegLayout, memDescType, vals, smemObj,
                             rewriter, targetInfo, typeConverter);
    } else {
      auto newRegLayout = removeBroadcast.apply(regLayout);
      auto result =
          lowerLdStMatrix(loc, newRegLayout, memDescType, vals, smemObj,
                          rewriter, targetInfo, typeConverter);
      if (succeeded(result)) {
        vals = broadcastAs(vals, regLayout);
      }
      return result;
    }
  }
  if (isa<PaddedSharedEncodingAttr>(memDescType.getEncoding())) {
    return failure();
  }
  auto memLayout = toLinearLayout(memDescType);
  auto cvt = regLayout.invertAndCompose(memLayout);
  auto kBlock = StringAttr::get(loc.getContext(), "block");
  // ldmatrix/stmatrix does not support shared::cluster
  auto maybeSublayout = cvt.quotient({kBlock});
  if (!maybeSublayout) {
    return failure();
  }
  cvt = maybeSublayout.value();
  auto smemBase = smemObj.getBase();
  auto affineOffset = smemObj.getShmemOffset(loc, rewriter, memDescType);
  auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(memDescType);
  auto llvmElemTy = typeConverter->convertType(memDescType.getElementType());
  for (bool transpose : {false, true}) {
    auto result = LLVM::NVIDIA::lowerLdStMatrix(
        loc, cvt, transpose, vals, smemBase, affineOffset, maskSpanAffineOffset,
        llvmElemTy, rewriter, targetInfo);
    if (succeeded(result)) {
      return result;
    }
  }
  return failure();
}

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LocalLoadOpConversion(const LLVMTypeConverter &converter,
                        const NVIDIA::TargetInfo &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    MemDescType memDescType = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Type llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getSrc(), llvmElemTy, rewriter);

    auto *typeConverter = getTypeConverter();
    llvm::SmallVector<Value> values;
    auto regLayout = toLinearLayout(dstTy);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }
    auto structTy = LLVM::LLVMStructType::getLiteral(
        op.getLoc().getContext(), SmallVector<Type>(values.size(), llvmElemTy));
    auto value =
        packLLElements(op.getLoc(), typeConverter, values, rewriter, structTy);
    rewriter.replaceOp(op, value);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    MemDescType memDescType = op.getType();
    RankedTensorType regTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(regTy.getElementType());
    Value smemBase =
        LLVM::getSharedMemoryBase(op.getLoc(), rewriter, targetInfo, op);
    auto smemObj = SharedMemoryObject(
        smemBase, llvmElemTy, memDescType.getRank(), op.getLoc(), rewriter);

    auto regLayout = toLinearLayout(regTy);
    auto values = unpackLLElements(op.getLoc(), adaptor.getSrc(), rewriter);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }

    auto retVal =
        getStructFromSharedMemoryObject(op.getLoc(), smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType memDescType = op.getDst().getType();
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    SharedMemoryObject smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);

    auto regLayout = toLinearLayout(srcTy);
    auto values = unpackLLElements(op.getLoc(), adaptor.getSrc(), rewriter);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

static std::string getAsyncSharedStoreConstraint(unsigned bitwidth) {
  switch (bitwidth) {
  case 32:
    return "r";
  case 64:
    return "l";
  default:
    llvm_unreachable("unsupported st.async.shared bitwidth");
  }
}

static Value normalizeAsyncSharedStoreValue(Value value, Type valueTy,
                                            unsigned bitwidth,
                                            TritonLLVMOpBuilder &b) {
  Type intTy = IntegerType::get(valueTy.getContext(), bitwidth);
  if (isa<LLVM::LLVMPointerType>(valueTy))
    return b.ptrtoint(intTy, value);
  if (!valueTy.isInteger())
    return b.bitcast(value, intTy);
  return value;
}

static Value packAsyncSharedStoreValue(Location loc, ArrayRef<Value> values,
                                       Type elemTy, unsigned storeBitwidth,
                                       TritonLLVMOpBuilder &b,
                                       ConversionPatternRewriter &rewriter) {
  unsigned elemBitwidth = getIntOrFloatOrPtrBitWidth(elemTy);
  assert(values.size() == storeBitwidth / elemBitwidth);
  if (values.size() == 1)
    return normalizeAsyncSharedStoreValue(values.front(), elemTy, storeBitwidth,
                                          b);

  Value packed = packLLVector(loc, values, rewriter);
  return b.bitcast(packed,
                   IntegerType::get(elemTy.getContext(), storeBitwidth));
}

static Value mapSharedToCluster(Location loc, Value ptr, Value ctaId,
                                ConversionPatternRewriter &rewriter) {
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 &&
         "st.async.shared expects a shared-memory pointer");
  return NVVM::MapaOp::create(rewriter, loc, ptr_ty(rewriter.getContext(), 7),
                              ptr, ctaId);
}

static void emitAsyncSharedStore(Location loc, ArrayRef<Value> vals, Value dst,
                                 Value mbarrier, VectorType vecTy,
                                 ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type elemTy = vecTy.getElementType();
  unsigned vec = vecTy.getNumElements();
  unsigned elemBitwidth = getIntOrFloatOrPtrBitWidth(elemTy);
  assert((elemBitwidth == 8 || elemBitwidth == 16 || elemBitwidth == 32 ||
          elemBitwidth == 64) &&
         "st.async.shared only supports packable elements");

  if (vec * elemBitwidth > 128) {
    assert(vec % (128 / elemBitwidth) == 0);
    int maxVec = 128 / elemBitwidth;
    auto splitVecTy = VectorType::get({static_cast<int64_t>(maxVec)}, elemTy);
    for (int i = 0; i < vec / maxVec; i++) {
      auto newDst = b.gep(dst.getType(), elemTy, dst, b.i32_val(i * maxVec),
                          LLVM::GEPNoWrapFlags::inbounds);
      emitAsyncSharedStore(loc, vals.slice(i * maxVec, maxVec), newDst,
                           mbarrier, splitVecTy, rewriter);
    }
    return;
  }
  assert(vec * elemBitwidth >= 32 &&
         "st.async.shared requires at least a 32-bit store");

  unsigned storeBitwidth = elemBitwidth < 32 ? 32u : elemBitwidth;
  unsigned elemsPerStore = storeBitwidth / elemBitwidth;
  assert(vec % elemsPerStore == 0);
  unsigned storeVec = vec / elemsPerStore;
  assert(1 <= storeVec && storeVec <= 4);

  PTXBuilder builder;
  auto st = builder.create("st.async")
                ->o("weak")
                .o("shared::cluster")
                .o("mbarrier::complete_tx::bytes")
                .v(storeVec, /*predicate=*/storeVec > 1)
                .b(storeBitwidth);
  auto *dstOpr = builder.newAddrOperand(dst, "r");
  auto *mbarrierOpr = builder.newAddrOperand(mbarrier, "r");
  auto constraint = getAsyncSharedStoreConstraint(storeBitwidth);
  PTXBuilder::Operand *valueOpr;
  if (storeVec == 1) {
    valueOpr = builder.newOperand(
        packAsyncSharedStoreValue(loc, vals.slice(0, elemsPerStore), elemTy,
                                  storeBitwidth, b, rewriter),
        constraint);
  } else {
    SmallVector<std::pair<Value, std::string>> vecVals;
    vecVals.reserve(storeVec);
    for (unsigned i = 0; i < storeVec; i++) {
      vecVals.push_back({packAsyncSharedStoreValue(
                             loc, vals.slice(i * elemsPerStore, elemsPerStore),
                             elemTy, storeBitwidth, b, rewriter),
                         constraint});
    }
    valueOpr = builder.newListOperand(vecVals);
  }
  st(dstOpr, valueOpr, mbarrierOpr);
  builder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void lowerAsyncSharedStore(Location loc, MLIRContext *ctx,
                                  LinearLayout cvt, ArrayRef<Value> vals,
                                  Type llvmElemTy, MemDescType dstTy,
                                  SharedMemoryObject dstMemObj,
                                  Value mbarrierPtr, MemDescType mbarrierTy,
                                  ConversionPatternRewriter &rewriter,
                                  const NVIDIA::TargetInfo &targetInfo) {
  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(cvt);
  if (!removeBroadcastSrc.isIdentity()) {
    auto prmtCvt = removeBroadcastSrc.apply(cvt);
    auto inVals = removeBroadcastSrc.apply(to_vector(vals));
    lowerAsyncSharedStore(loc, ctx, prmtCvt, inVals, llvmElemTy, dstTy,
                          dstMemObj, mbarrierPtr, mbarrierTy, rewriter,
                          targetInfo);
    return;
  }

  auto affineOffset = dstMemObj.getShmemOffset(loc, rewriter, dstTy);
  auto maskSpanAffineOffset = dstMemObj.getMaskSpanOffsets(dstTy);
  std::optional<int> maybeMaxVecElems;
  SmallVector<std::pair<unsigned, unsigned>> paddingShifts;
  if (triton::gpu::isPaddedEncoding(dstTy.getEncoding())) {
    maybeMaxVecElems = triton::gpu::getMinInterval(dstTy.getEncoding());
    auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
    paddingShifts = getPaddedSharedShifts(dstTy.getEncoding(), bitwidth,
                                          /*offsetInBytes=*/true);
  }

  SmallVector<Value> smemBases(dstMemObj.getBases().begin(),
                               dstMemObj.getBases().end());
  Value currentCTAId = targetInfo.getClusterCTAId(rewriter, loc);
  auto emitSt = [&](RewriterBase &, Location storeLoc, ArrayRef<Value> values,
                    Value shmemAddr, int idx, VectorType vecTy,
                    std::optional<Value> ctaId) -> SmallVector<Value> {
    Value targetCTAId = ctaId.value_or(currentCTAId);
    Value dst = mapSharedToCluster(storeLoc, shmemAddr, targetCTAId, rewriter);
    Value mbarrier =
        mapSharedToCluster(storeLoc, mbarrierPtr, targetCTAId, rewriter);
    mbarrier = LLVM::NVIDIA::getLeaderAddress(storeLoc, rewriter, mbarrier,
                                              mbarrierTy);
    emitAsyncSharedStore(storeLoc, values.slice(idx, vecTy.getNumElements()),
                         dst, mbarrier, vecTy, rewriter);
    return {};
  };
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  lowerLdSt(loc, ctx, cvt, vals, llvmElemTy, smemBases, paddingShifts,
            affineOffset, maskSpanAffineOffset, laneId, warpId, rewriter,
            targetInfo, maybeMaxVecElems, emitSt);
}

struct AsyncSharedStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::AsyncSharedStoreOp> {
  AsyncSharedStoreOpConversion(const LLVMTypeConverter &converter,
                               const NVIDIA::TargetInfo &targetInfo,
                               PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::nvidia_gpu::AsyncSharedStoreOp>(
            converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::AsyncSharedStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!triton::nvidia_gpu::AsyncSharedStoreOp::isSupported(
            targetInfo.getComputeCapability()))
      return op.emitError("requires cluster-capable SM90+");

    auto loc = op.getLoc();
    MemDescType dstTy = op.getDst().getType();
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getDst(), llvmElemTy, rewriter);
    auto mbarrierTy = op.getMbarrier().getType();
    auto mbarrierMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getMbarrier(),
        typeConverter->convertType(mbarrierTy.getElementType()), rewriter);

    auto regLayout = toLinearLayout(srcTy);
    auto sharedLayout = isPaddedEncoding(dstTy.getEncoding())
                            ? paddedLinearLayout(dstTy)
                            : toLinearLayout(dstTy);
    auto cvt = regLayout.invertAndCompose(sharedLayout);
    auto values = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    lowerAsyncSharedStore(loc, op.getContext(), cvt, values, llvmElemTy, dstTy,
                          dstMemObj, mbarrierMemObj.getBase(), mbarrierTy,
                          rewriter, targetInfo);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalAtomicScatterRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAtomicScatterRMWOp> {
public:
  LocalAtomicScatterRMWOpConversion(const LLVMTypeConverter &converter,
                                    const NVIDIA::TargetInfo &targetInfo,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAtomicScatterRMWOp>(converter,
                                                                     benefit),
        targetInfo(targetInfo) {}

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

    RMWOp rmwOp = op.getAtomicRmwOp();
    bool isI32Inc = rmwOp == RMWOp::ADD &&
                    info.valuesTy.getElementType().isInteger(32) &&
                    isConstI32OneTensor(op.getValues());
    bool returnOld = !op.getResult().use_empty();

    SmallVector<Value> results;
    if (returnOld)
      results.reserve(info.ptrs.size());
    for (auto [i, ptrAndValue] :
         llvm::enumerate(llvm::zip(info.ptrs, info.values))) {
      auto [ptr, value] = ptrAndValue;
      Value pred =
          maybeAnd(rewriter, loc, info.threadPred,
                   info.maskValues.empty() ? Value() : info.maskValues[i]);
      if (isI32Inc) {
        Value result = emitSharedInc(rewriter, loc, ptr, returnOld, pred);
        if (returnOld)
          results.push_back(result);
        continue;
      }
      auto old = emitSharedAtomicRMW(rewriter, loc, info.llvmElemTy, ptr, value,
                                     rmwOp, returnOld, pred);
      if (failed(old))
        return failure();
      if (returnOld)
        results.push_back(*old);
    }

    if (!returnOld) {
      rewriter.eraseOp(op);
      return success();
    }

    if (!info.removeBroadcast.isIdentity())
      results = broadcastAs(results, info.regLayout);
    finalizeTensorAtomicResults(op, info.valuesTy, rewriter, results,
                                info.llvmElemTy, b, info.threadPred, targetInfo,
                                getTypeConverter());
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Backend optimized memory ops get higher benefit
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<AsyncSharedStoreOpConversion>(typeConverter, targetInfo,
                                             benefit.getBenefit() + 1);
  patterns.add<LocalAtomicScatterRMWOpConversion>(typeConverter, targetInfo,
                                                  benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo,
                                      benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
}
