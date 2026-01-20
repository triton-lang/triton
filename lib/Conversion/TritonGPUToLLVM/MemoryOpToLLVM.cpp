#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Helper for LocalGather/ScatterOpConversion.
// For gather: storeVals is empty, returns loaded values.
// For scatter: storeVals contains values to store, returns empty.
SmallVector<Value> lowerLocalScGt(Location loc, MLIRContext *ctx,
                                  MemDescType memDescTy,
                                  SharedMemoryObject smemObj, Type llvmElemTy,
                                  ArrayRef<Value> idxValues,
                                  ArrayRef<SmallVector<Value>> coords,
                                  unsigned axis, ArrayRef<Value> storeVals,
                                  RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  bool isScatter = !storeVals.empty();

  // Get the shared memory layout (linear component for padded layouts)
  auto sharedEnc =
      cast<triton::gpu::SharedEncodingTrait>(memDescTy.getEncoding());
  auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc);
  LinearLayout sharedLayout;
  if (paddedEnc) {
    sharedLayout = paddedEnc.getLinearComponent();
  } else {
    sharedLayout = toLinearLayout(memDescTy);
  }
  LinearLayout invSharedLayout = sharedLayout.invert();

  // Get layout dimension names for all dims
  SmallVector<StringAttr> allDims;
  for (unsigned dim = 0, rank = memDescTy.getRank(); dim < rank; ++dim) {
    allDims.push_back(str_attr("dim" + Twine(dim)));
  }

  auto kOffset = str_attr("offset");

  // Get the subslice affine offset (non-zero for memdesc subslices)
  Value affineOffset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
  auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);

  SmallVector<Value> results;
  if (!isScatter) {
    results.resize(coords.size());
  }

  for (auto [i, idxVal] : llvm::enumerate(idxValues)) {
    // Convert index to i32 if needed
    Value idx = idxVal;
    unsigned idxWidth = idx.getType().getIntOrFloatBitWidth();
    if (idxWidth > 32) {
      idx = b.trunc(i32_ty, idx);
    } else if (idxWidth < 32) {
      idx = b.zext(i32_ty, idx);
    }

    // Copy coordinates and replace the axis coordinate with the index value
    SmallVector<Value> indices(coords[i]);
    indices[axis] = idx;

    // Apply inverted shared layout to compute offset
    SmallVector<std::pair<StringAttr, Value>> inputs;
    for (unsigned dim = 0; dim < indices.size(); ++dim) {
      inputs.push_back({allDims[dim], indices[dim]});
    }

    auto outputs = applyLinearLayout(loc, rewriter, invSharedLayout, inputs);

    // Extract the offset value
    Value offset = nullptr;
    for (auto [name, value] : outputs) {
      if (name == kOffset) {
        offset = value;
        break;
      }
    }
    assert(offset && "expected offset output from inverted shared layout");

    // For subslices, the physical offset is computed as:
    //   physical_offset = L⁻¹(coords) ⊕ L⁻¹(subslice_logical_offset)
    //
    // We use XOR for consistency with lowerLdSt. MemDescSubsliceOp::verify()
    // enforces:
    // 1. Subslice offsets must be multiples of the tile size
    // 2. Subslice offsets must map to power-of-2 physical offsets
    //
    // These constraints ensure the bit ranges of L⁻¹(coords) and
    // L⁻¹(subslice_offset) are disjoint, so XOR and addition are equivalent.
    offset = b.xor_(offset, affineOffset);

    // Add padding offset for padded layouts (non-linear component)
    Value ptr;
    if (paddedEnc) {
      // Convert offset to bytes for padding calculation
      Value offsetBytes = b.mul(offset, b.i32_val(bitwidth / 8));
      Value padOffset = emitPadding(loc, rewriter, paddedEnc, bitwidth,
                                    offsetBytes, /*offsetInBytes=*/true);
      // GEP in bytes: base + offset*elemSize + padOffset
      Value totalOffset = b.add(offsetBytes, padOffset);
      ptr = b.gep(smemObj.getBase().getType(), i8_ty, smemObj.getBase(),
                  totalOffset);
    } else {
      ptr = b.gep(smemObj.getBase().getType(), llvmElemTy, smemObj.getBase(),
                  offset);
    }

    if (isScatter) {
      b.store(storeVals[i], ptr);
    } else {
      results[i] = b.load(llvmElemTy, ptr);
    }
  }

  return results;
}

LogicalResult lowerLocalStore(Location loc, MLIRContext *ctx, Value regVal,
                              MemDescType memDescTy, SharedMemoryObject smemObj,
                              ArrayRef<Value> inVals,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo) {
  auto regTy = cast<RankedTensorType>(regVal.getType());
  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());

  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kOffset = str_attr("offset");
  auto regLayout = toLinearLayout(regTy);
  auto paddedEnc =
      dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(memDescTy.getEncoding());
  LinearLayout cvt = LinearLayout::empty();
  if (paddedEnc) {
    const auto &sharedLL = paddedEnc.getLinearComponent();
    cvt = regLayout.invertAndCompose(sharedLL);
  } else {
    auto sharedLayout = toLinearLayout(memDescTy);
    cvt = regLayout.invertAndCompose(sharedLayout);
  }
  auto kBlock = str_attr("block");
  // NYI. We would need to emit a map.shared::cluster instruction.
  if (!cvt.isTrivialOver({kBlock})) {
    return failure();
  }
  cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
  lowerLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy, memDescTy, smemObj,
                 rewriter, targetInfo);

  return success();
}

struct GlobalScratchAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::GlobalScratchAllocOp> {
  const TargetInfoBase *targetInfo;

  GlobalScratchAllocOpConversion(LLVMTypeConverter &converter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto opOffsetAttr = op->getAttrOfType<mlir::IntegerAttr>(
        "ttg.global_scratch_memory_offset");
    assert(opOffsetAttr);
    auto opOffset = opOffsetAttr.getValue().getZExtValue();

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp) {
      return failure();
    }
    Value ptr = LLVM::getGlobalScratchPtr(loc, rewriter, *targetInfo, funcOp,
                                          b.i32_val(opOffset));

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto memDescTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, memDescTy.getRank(),
                                      loc, rewriter);
    // If there is an initial tensor, store it into the shared memory.
    if (op.getSrc()) {
      auto *ctx = op.getContext();
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      if (failed(lowerLocalStore(loc, ctx, op.getSrc(), memDescTy, smemObj,
                                 inVals, typeConverter, rewriter,
                                 targetInfo))) {
        return failure();
      }
    }
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalDeallocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalDeallocOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalDeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct LocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescVal = op.getSrc();
    auto regVal = op.getResult();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto regTy = cast<RankedTensorType>(regVal.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    auto sharedEnc =
        cast<triton::gpu::SharedEncodingTrait>(memDescTy.getEncoding());
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    auto regLayout = toLinearLayout(regTy);
    auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc);
    LinearLayout cvt = LinearLayout::empty();
    if (paddedEnc) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvt = regLayout.invertAndCompose(sharedLL);
    } else {
      auto sharedLayout = toLinearLayout(memDescTy);
      cvt = regLayout.invertAndCompose(sharedLayout);
    }
    auto kBlock = str_attr("block");
    // NYI. We would need to emit a map.shared::cluster instruction.
    if (!cvt.isTrivialOver({kBlock})) {
      return failure();
    }
    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});

    auto outVals = lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, memDescTy,
                                  smemObj, rewriter, targetInfo, op);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, regTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalStoreOp>::ConvertOpToLLVMPattern;

  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    Value regVal = op.getSrc();
    Value memDescVal = op.getDst();
    auto typeConverter = getTypeConverter();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                         llvmElemTy, rewriter);
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (failed(lowerLocalStore(loc, ctx, regVal, memDescTy, smemObj, inVals,
                               typeConverter, rewriter, targetInfo))) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

class BarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
public:
  BarrierOpConversion(const LLVMTypeConverter &converter,
                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::BarrierOp>(converter, benefit) {}
  using OpAdaptor = typename triton::gpu::BarrierOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::gpu::BarrierOp>(op);
    return success();
  }
};

struct LocalGatherOpConversion : public ConvertOpToLLVMPattern<LocalGatherOp> {
public:
  LocalGatherOpConversion(LLVMTypeConverter &typeConverter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescTy = cast<MemDescType>(op.getSrc().getType());
    auto regTy = cast<RankedTensorType>(op.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    SmallVector<Value> idxValues =
        unpackLLElements(loc, adaptor.getIndices(), rewriter);
    SmallVector<SmallVector<Value>> dstIndices =
        emitIndices(loc, rewriter, targetInfo, regTy.getEncoding(), regTy,
                    /*withCTAOffset=*/true);

    auto results = lowerLocalScGt(loc, ctx, memDescTy, smemObj, llvmElemTy,
                                  idxValues, dstIndices, op.getAxis(),
                                  /*storeVals=*/{}, rewriter);

    Value result = packLLElements(loc, typeConverter, results, rewriter, regTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalScatterOpConversion
    : public ConvertOpToLLVMPattern<LocalScatterOp> {
public:
  LocalScatterOpConversion(LLVMTypeConverter &typeConverter,
                           const TargetInfoBase &targetInfo,
                           PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescTy = cast<MemDescType>(op.getDst().getType());
    auto valuesTy = cast<RankedTensorType>(op.getValues().getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                         llvmElemTy, rewriter);

    SmallVector<Value> values =
        unpackLLElements(loc, adaptor.getValues(), rewriter);
    SmallVector<Value> idxValues =
        unpackLLElements(loc, adaptor.getIndices(), rewriter);
    SmallVector<SmallVector<Value>> srcIndices =
        emitIndices(loc, rewriter, targetInfo, valuesTy.getEncoding(), valuesTy,
                    /*withCTAOffset=*/true);

    lowerLocalScGt(loc, ctx, memDescTy, smemObj, llvmElemTy, idxValues,
                   srcIndices, op.getAxis(), values, rewriter);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};
} // namespace

void mlir::triton::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo,
                                               benefit);
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalDeallocOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalGatherOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalScatterOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<BarrierOpConversion>(typeConverter, benefit);
}
