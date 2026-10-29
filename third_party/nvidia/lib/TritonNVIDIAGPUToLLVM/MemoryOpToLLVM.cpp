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
                    Value ptr, bool returnOld, bool isCluster,
                    Value pred = Value()) {
  PTXBuilder ptxBuilder;
  // PTX atom/red.inc resets to 0 only when the old value reaches the bound, so
  // using UINT32_MAX makes it equivalent to a wrapping increment-by-1.
  auto *boundOpr = ptxBuilder.newConstantOperand("0xffffffff");
  auto &inc = *ptxBuilder.create(returnOld ? "atom" : "red");
  if (isCluster)
    inc.o("shared::cluster").o("cluster");
  else
    inc.shared().o("cta");
  inc.o("relaxed").o("inc").o("u32");

  if (!returnOld) {
    auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "r");
    inc(ptrOpr, boundOpr).maybePredicate(pred, "b");
    return ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
  }

  auto *dstOpr = ptxBuilder.newOperand("=r", /*init=*/true);
  auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "r");
  inc(dstOpr, ptrOpr, boundOpr).maybePredicate(pred, "b");
  return ptxBuilder.launch(rewriter, loc, i32_ty);
}

FailureOr<Value> emitSharedAtomicRMW(ConversionPatternRewriter &rewriter,
                                     Location loc, Type valueElemTy, Value ptr,
                                     Value value, RMWOp rmwOp, bool returnOld,
                                     bool isCluster, Value pred) {
  SmallVector<Value> vals{value};
  if (!returnOld) {
    auto result =
        emitPtxSharedAtomicRMW(rewriter, loc, valueElemTy, ptr, vals, rmwOp,
                               pred, isCluster, PtxAtomicInstr::Red);
    if (succeeded(result))
      return result;
  }

  return emitPtxSharedAtomicRMW(rewriter, loc, valueElemTy, ptr, vals, rmwOp,
                                pred, isCluster, PtxAtomicInstr::Atom);
}

LogicalResult lowerLdStMatrix(
    Location loc, const LinearLayout &regLayout, MemDescType memDescType,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    SharedMemoryObject smemObj, ConversionPatternRewriter &rewriter,
    const NVIDIA::TargetInfo &targetInfo,
    const LLVMTypeConverter *typeConverter) {
  auto *ctx = loc.getContext();
  assert(regLayout.getFreeVariableMasks().lookup(str_attr("register")) == 0 &&
         "expected register broadcasting to be removed by the caller");
  if (isa<PaddedSharedEncodingAttr>(memDescType.getEncoding())) {
    return failure();
  }
  if (SharedMemoryObject::getMaskSpanOffsetsAndBlocks(memDescType).second !=
      0) {
    return failure();
  }
  auto memLayout = toLinearLayout(memDescType);
  auto cvt = regLayout.invertAndCompose(memLayout);
  auto kBlock = str_attr("block");
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

// TMA expands 16 FP6 codes into a 16-byte atom containing 12 packed
// payload bytes. Locate the code in that payload before reading it.
static SmallVector<Value> loadFP6Codes(Location loc, RankedTensorType regTy,
                                       const LinearLayout &regLayout,
                                       MemDescType memTy,
                                       SharedMemoryObject smem,
                                       RewriterBase &rewriter,
                                       const NVIDIA::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned axis = getOrder(memTy).front();
  SmallVector<Value> bytes, shifts;
  for (auto &index : emitIndices(loc, rewriter, targetInfo, regLayout, regTy,
                                 /*withCTAOffset=*/true)) {
    Value col = index[axis];
    Value bit = b.mul(b.and_(col, b.i32_val(15)), b.i32_val(6));
    bytes.push_back(
        b.add(b.and_(col, b.i32_val(-16)), b.lshr(bit, b.i32_val(3))));
    shifts.push_back(b.and_(bit, b.i32_val(7)));
  }
  auto offsets = computeBlockLocalOffsets(loc, memTy, regLayout, bytes, axis,
                                          rewriter, targetInfo);
  auto addrs =
      materializeLocalAddrs(loc, memTy, smem, i8_ty, offsets, rewriter);
  SmallVector<Value> codes;
  for (auto [addr, shift] : llvm::zip_equal(addrs, shifts)) {
    // Only read a second payload byte when the field crosses a byte boundary.
    Value next = b.zext(i32_ty, b.icmp_ugt(shift, b.i32_val(2)));
    Value hiPtr = b.gep(addr.ptr.getType(), i8_ty, addr.ptr, next);
    Value lo = targetInfo.loadDShared(rewriter, loc, addr.ptr, addr.ctaId,
                                      i8_ty, b.true_val());
    Value hi = targetInfo.loadDShared(rewriter, loc, hiPtr, addr.ctaId, i8_ty,
                                      b.true_val());
    Value pair =
        b.or_(b.zext(i32_ty, lo), b.shl(b.zext(i32_ty, hi), b.i32_val(8)));
    codes.push_back(b.trunc(i8_ty, b.and_(b.lshr(pair, shift), b.i32_val(63))));
  }
  return codes;
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
    auto *ctx = op.getContext();
    MemDescType memDescType = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Type llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getSrc(), llvmElemTy, rewriter);

    auto *typeConverter = getTypeConverter();
    llvm::SmallVector<Value> values;
    auto regLayout =
        toLinearLayout(dstTy).removeZeroBasesAlongDim(str_attr("register"));
    if (NVMMASharedEncodingAttr::getFp6Type(memDescType.getEncoding())) {
      values = loadFP6Codes(op.getLoc(), dstTy, regLayout, memDescType, smemObj,
                            rewriter, targetInfo);
    } else if (failed(lowerLdStMatrix(op.getLoc(), regLayout, memDescType,
                                      values, smemObj, rewriter, targetInfo,
                                      typeConverter))) {
      return failure();
    }
    auto value = packUniqueTensorElements(op.getLoc(), typeConverter, values,
                                          rewriter, dstTy);
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
    auto *ctx = op.getContext();
    MemDescType memDescType = op.getType();
    RankedTensorType regTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(regTy.getElementType());
    Value smemBase =
        LLVM::getSharedMemoryBase(op.getLoc(), rewriter, targetInfo, op);
    auto smemObj = SharedMemoryObject(
        smemBase, llvmElemTy, memDescType.getRank(), op.getLoc(), rewriter);

    auto regLayout =
        toLinearLayout(regTy).removeZeroBasesAlongDim(str_attr("register"));
    auto values =
        unpackUniqueTensorElements(op.getLoc(), adaptor.getSrc(), rewriter);
    if (NVMMASharedEncodingAttr::getFp6Type(memDescType.getEncoding())) {
      // The TMA lowering assigns 16 contiguous codes to each thread. Pack
      // four codes into three bytes, with four padding bytes after each atom.
      auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
      auto reg = str_attr("register");
      auto dims = standardOutDimNames(ctx, memDescType.getRank());
      auto order = applyPermutation(dims, getOrder(memDescType));
      if (regLayout.transposeOuts(order).getNumConsecutiveInOut() < 16)
        return op.emitError(
            "FP6 packing requires 16 contiguous register values");
      SmallVector<Value> packed;
      for (unsigned i = 0; i < values.size(); i += 16) {
        for (unsigned j = 0; j < 16; j += 4) {
          Value word = b.i32_val(0);
          for (unsigned k = 0; k < 4; ++k)
            word = b.or_(word, b.shl(b.zext(i32_ty, values[i + j + k]),
                                     b.i32_val(6 * k)));
          for (unsigned k = 0; k < 3; ++k)
            packed.push_back(b.trunc(i8_ty, b.lshr(word, b.i32_val(8 * k))));
        }
        packed.append(4, b.i8_val(0));
      }
      auto cvt = invertAndComposeStore(toLinearLayout(memDescType), regLayout);
      packed = broadcastAs(packed,
                           regLayout.resizeInDim(reg, cvt.getInDimSize(reg)));
      lowerLocalLdSt(op.getLoc(), ctx, cvt, packed, i8_ty, memDescType, smemObj,
                     rewriter, targetInfo,
                     makeSharedStoreEmitter(targetInfo, b.true_val()));
    } else if (failed(lowerLdStMatrix(op.getLoc(), regLayout, memDescType,
                                      values, smemObj, rewriter, targetInfo,
                                      getTypeConverter()))) {
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
    auto *ctx = op.getContext();
    MemDescType memDescType = op.getDst().getType();
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    SharedMemoryObject smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);

    auto regLayout =
        toLinearLayout(srcTy).removeZeroBasesAlongDim(str_attr("register"));
    auto values =
        unpackUniqueTensorElements(op.getLoc(), adaptor.getSrc(), rewriter);
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
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MemDescType dstTy = op.getDst().getType();
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getDst(), llvmElemTy, rewriter);
    auto mbarrierTy = op.getMbarrier().getType();
    auto mbarrierMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getMbarrier(),
        typeConverter->convertType(mbarrierTy.getElementType()), rewriter);

    auto kBlock = str_attr("block");
    auto regLayout = toLinearLayout(srcTy);
    auto freeVarMasks = regLayout.getFreeVariableMasks();
    freeVarMasks[kBlock] = 0;
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    Value mapPred = threadPred ? threadPred : b.true_val();
    regLayout = regLayout.removeZeroBasesAlongDim(str_attr("register"));
    auto sharedLayout = toLinearLayoutIgnoringPadding(dstTy);
    auto cvt = invertAndComposeLocal(sharedLayout, regLayout, {kBlock});
    auto values = unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
    Value currentCTAId = targetInfo.getClusterCTAId(rewriter, loc);
    Value mbarrierPtr = mbarrierMemObj.getBase();
    auto emitStore = [&](RewriterBase &, Location storeLoc,
                         ArrayRef<Value> values, Value shmemAddr, int idx,
                         VectorType vecTy, Value ctaId) -> SmallVector<Value> {
      Value targetCTAId = ctaId ? ctaId : currentCTAId;
      Value dst = targetInfo.mapDShared(rewriter, storeLoc, shmemAddr,
                                        targetCTAId, mapPred);
      Value mbarrier = targetInfo.mapDShared(rewriter, storeLoc, mbarrierPtr,
                                             targetCTAId, mapPred);
      mbarrier = LLVM::NVIDIA::getLeaderAddress(storeLoc, rewriter, mbarrier,
                                                mbarrierTy);

      Type elemTy = vecTy.getElementType();
      unsigned elemBitwidth = getIntOrFloatOrPtrBitWidth(elemTy);
      unsigned storeBitwidth = std::max(32u, elemBitwidth);
      unsigned elemsPerStore = storeBitwidth / elemBitwidth;
      unsigned storeVec = vecTy.getNumElements() / elemsPerStore;
      std::string constraint = storeBitwidth == 64 ? "l" : "r";

      SmallVector<std::pair<Value, std::string>> storeValues;
      for (unsigned i = 0; i < storeVec; ++i) {
        auto elems = values.slice(idx + i * elemsPerStore, elemsPerStore);
        Value value = elems.front();
        if (elemsPerStore > 1)
          value = b.bitcast(packLLVector(storeLoc, elems, rewriter),
                            int_ty(storeBitwidth));
        else if (isa<LLVM::LLVMPointerType>(elemTy))
          value = b.ptrtoint(int_ty(storeBitwidth), value);
        else if (!elemTy.isInteger())
          value = b.bitcast(value, int_ty(storeBitwidth));
        storeValues.emplace_back(value, constraint);
      }

      PTXBuilder ptxBuilder;
      auto &store = ptxBuilder.create("st.async")
                        ->o("weak")
                        .o("shared::cluster")
                        .o("mbarrier::complete_tx::bytes")
                        .v(storeVec, /*predicate=*/storeVec > 1)
                        .b(storeBitwidth);
      auto *dstOperand = ptxBuilder.newAddrOperand(dst, "r");
      auto *valueOperand =
          storeVec == 1
              ? ptxBuilder.newOperand(storeValues.front().first, constraint)
              : ptxBuilder.newListOperand(storeValues);
      auto *barrierOperand = ptxBuilder.newAddrOperand(mbarrier, "r");
      store(dstOperand, valueOperand, barrierOperand)
          .maybePredicate(threadPred, "b");
      ptxBuilder.launch(rewriter, storeLoc, void_ty(rewriter.getContext()));
      return {};
    };
    lowerLocalLdSt(loc, ctx, cvt, values, llvmElemTy, dstTy, dstMemObj,
                   rewriter, targetInfo, emitStore);
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
      results.reserve(info.addrs.size());
    for (auto [i, addrAndValue] :
         llvm::enumerate(llvm::zip(info.addrs, info.values))) {
      auto [addr, value] = addrAndValue;
      Value pred =
          maybeAnd(rewriter, loc, info.threadPred,
                   info.maskValues.empty() ? Value() : info.maskValues[i]);
      bool isCluster = bool(addr.ctaId);
      Value ptr =
          targetInfo.mapDShared(rewriter, loc, addr.ptr, addr.ctaId, pred);
      if (isI32Inc) {
        Value result =
            emitSharedInc(rewriter, loc, ptr, returnOld, isCluster, pred);
        if (returnOld)
          results.push_back(result);
        continue;
      }
      auto old = emitSharedAtomicRMW(rewriter, loc, info.llvmElemTy, ptr, value,
                                     rmwOp, returnOld, isCluster, pred);
      if (failed(old))
        return failure();
      if (returnOld)
        results.push_back(*old);
    }

    if (!returnOld) {
      rewriter.eraseOp(op);
      return success();
    }

    finalizeAtomicResults(op, rewriter, results, info.threadPred, targetInfo,
                          getTypeConverter());
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
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
  mlir::triton::populateMemoryOpToLLVMPatterns(
      typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);
}
