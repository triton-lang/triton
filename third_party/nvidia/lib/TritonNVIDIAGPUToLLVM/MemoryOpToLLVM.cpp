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
    red.shared().o("inc").o("u32");
    red(ptrOpr, boundOpr).maybePredicate(pred, "b");
    return ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
  }

  auto *dstOpr = ptxBuilder.newOperand("=r", /*init=*/true);
  auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "r");
  auto &atom = *ptxBuilder.create("atom");
  atom.shared().o("inc").o("u32");
  atom(dstOpr, ptrOpr, boundOpr).maybePredicate(pred, "b");
  return ptxBuilder.launch(rewriter, loc, i32_ty);
}

FailureOr<Value> emitSharedAtomicAdd(ConversionPatternRewriter &rewriter,
                                     Location loc, Type valueElemTy, Value ptr,
                                     Value value, bool returnOld,
                                     Value pred) {
  unsigned valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
  if (valueElemNBits != 16 && valueElemNBits != 32 && valueElemNBits != 64)
    return failure();

  std::string tyId =
      getPtxRegisterSizeCode(valueElemNBits, /*isFloat=*/false);
  std::string addOp = "add";
  std::string suffix;
  if (isa<IntegerType>(valueElemTy)) {
    if (valueElemNBits < 32)
      return failure();
    suffix = "u" + std::to_string(valueElemNBits);
  } else if (isa<FloatType>(valueElemTy)) {
    addOp += valueElemNBits == 16 ? ".noftz" : "";
    suffix = (valueElemTy.isBF16() ? "bf" : "f") +
             std::to_string(valueElemNBits);
  } else {
    return failure();
  }

  PTXBuilder ptxBuilder;
  auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "r");
  auto *valOpr = ptxBuilder.newOperand(value, tyId);
  if (!returnOld) {
    auto &red = *ptxBuilder.create("red");
    red.shared().o(addOp).o(suffix);
    red(ptrOpr, valOpr).maybePredicate(pred, "b");
    return ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
  }

  auto *dstOpr = ptxBuilder.newOperand("=" + tyId, /*init=*/true);
  auto &atom = *ptxBuilder.create("atom");
  atom.shared().o(addOp).o(suffix);
  atom(dstOpr, ptrOpr, valOpr).maybePredicate(pred, "b");
  return ptxBuilder.launch(rewriter, loc, valueElemTy);
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

struct LocalAtomicScatterAddOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAtomicScatterAddOp> {
public:
  LocalAtomicScatterAddOpConversion(const LLVMTypeConverter &converter,
                                    const NVIDIA::TargetInfo &targetInfo,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAtomicScatterAddOp>(converter,
                                                                     benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAtomicScatterAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto valuesTy = cast<RankedTensorType>(op.getValues().getType());
    auto memDescTy = cast<MemDescType>(op.getDst().getType());
    if (isa<triton::gpu::PartitionedSharedEncodingAttr>(
            memDescTy.getEncoding())) {
      return failure();
    }

    auto llvmElemTy =
        getTypeConverter()->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);
    SmallVector<Value> idxValues =
        unpackLLElements(op.getLoc(), adaptor.getIndices(), rewriter);
    SmallVector<SmallVector<Value>> srcIndices =
        emitIndices(op.getLoc(), rewriter, targetInfo, valuesTy.getEncoding(),
                    valuesTy, /*withCTAOffset=*/true);
    SmallVector<Value> ptrs =
        computeLocalPtrs(op.getLoc(), memDescTy, smemObj, llvmElemTy, idxValues,
                         srcIndices, op.getAxis(), rewriter);
    SmallVector<Value> values =
        unpackLLElements(op.getLoc(), adaptor.getValues(), rewriter);
    SmallVector<Value> maskValues;
    if (op.getMask())
      maskValues = unpackLLElements(op.getLoc(), adaptor.getMask(), rewriter);
    bool isI32Inc = valuesTy.getElementType().isInteger(32) &&
                    isConstI32OneTensor(op.getValues());

    if (maskValues.empty() && !isI32Inc)
      return failure();

    if (op.getResult().use_empty()) {
      for (auto [i, ptrAndValue] : llvm::enumerate(llvm::zip(ptrs, values))) {
        auto [ptr, value] = ptrAndValue;
        Value pred = maskValues.empty() ? Value() : maskValues[i];
        if (isI32Inc) {
          emitSharedInc(rewriter, op.getLoc(), ptr, /*returnOld=*/false, pred);
          continue;
        }
        if (failed(emitSharedAtomicAdd(rewriter, op.getLoc(), llvmElemTy, ptr,
                                       value, /*returnOld=*/false, pred)))
          return failure();
      }
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<Value> results;
    results.reserve(ptrs.size());
    for (auto [i, ptrAndValue] : llvm::enumerate(llvm::zip(ptrs, values))) {
      auto [ptr, value] = ptrAndValue;
      Value pred = maskValues.empty() ? Value() : maskValues[i];
      if (isI32Inc) {
        results.push_back(emitSharedInc(rewriter, op.getLoc(), ptr,
                                        /*returnOld=*/true, pred));
        continue;
      }
      auto old = emitSharedAtomicAdd(rewriter, op.getLoc(), llvmElemTy, ptr,
                                     value, /*returnOld=*/true, pred);
      if (failed(old))
        return failure();
      results.push_back(*old);
    }

    Value result = packLLElements(op.getLoc(), getTypeConverter(), results,
                                  rewriter, op.getType());
    rewriter.replaceOp(op, result);
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
  patterns.add<LocalAtomicScatterAddOpConversion>(typeConverter, targetInfo,
                                                  benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo,
                                      benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
}
