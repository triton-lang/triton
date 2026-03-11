#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/AtomicPTXBuilder.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "llvm/ADT/SmallString.h"
#include <limits>

namespace tt = mlir::triton;
namespace tti = mlir::triton::instrument;
namespace ttg = mlir::triton::gpu;

namespace {

static constexpr unsigned kTensorMapStrideWordBase = 3;
static constexpr unsigned kTensorMapShapeWordBase = 8;
static constexpr unsigned kTensorMapScalarWordBase = 2;
static constexpr unsigned kTensorMapNumQwords = 16;

struct GSanSourceLocation {
  Value file;
  Value line;
};

static constexpr StringLiteral kGSanLoadTensorRuntimeFn =
    "__triton_gsan_load_tensor";
static constexpr StringLiteral kGSanStoreTensorRuntimeFn =
    "__triton_gsan_store_tensor";
static constexpr StringLiteral kGSanAtomicBeginRuntimeFn =
    "__triton_gsan_atomic_begin_scalar";
static constexpr StringLiteral kGSanAtomicEndRuntimeFn =
    "__triton_gsan_atomic_end_scalar";
static constexpr StringLiteral kGSanInitRuntimeFn = "__triton_gsan_init";
static constexpr StringLiteral kGSanGlobalStateArgAttr =
    "tti.gsan_global_state";

LLVM::LLVMFuncOp
getOrCreateGSanRuntimeFunction(ConversionPatternRewriter &rewriter,
                               StringRef funcName) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  if (auto funcOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return funcOp;

  auto *ctx = rewriter.getContext();
  SmallVector<Type> argTys;
  if (funcName == kGSanInitRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanLoadTensorRuntimeFn ||
             funcName == kGSanStoreTensorRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i32_ty, i32_ty, ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanAtomicBeginRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i32_ty,      i64_ty, i32_ty,
              i32_ty,      i32_ty,      ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanAtomicEndRuntimeFn) {
    argTys = {ptr_ty(ctx), i32_ty, i32_ty, i32_ty, i32_ty, ptr_ty(ctx), i32_ty};
  } else {
    llvm_unreachable("unexpected GSan runtime symbol");
  }
  auto funcTy = LLVM::LLVMFunctionType::get(void_ty(ctx), argTys);
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  return LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(ctx), funcName,
                                  funcTy);
}

LLVM::LLVMStructType
getGSanAtomicEventStateType(ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  return LLVM::LLVMStructType::getLiteral(
      ctx, {ptr_ty(ctx), array_ty(ptr_ty(ctx), 3), i8_ty});
}

FileLineColLoc extractSourceLocation(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return extractSourceLocation(nameLoc.getChildLoc());
  if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc))
    return extractSourceLocation(opaqueLoc.getFallbackLocation());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
    return extractSourceLocation(fusedLoc.getLocations().front());
  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc))
    return extractSourceLocation(callSiteLoc.getCallee());

  StringAttr unknownFile = StringAttr::get(loc.getContext(), "<unknown>");
  return FileLineColLoc::get(unknownFile, 0, 0);
}

GSanSourceLocation
materializeSourceLocation(ConversionPatternRewriter &rewriter, Location loc) {
  auto fileLoc = extractSourceLocation(loc);
  auto *ctx = rewriter.getContext();
  TritonLLVMOpBuilder b(loc, rewriter);

  llvm::SmallString<64> fileName(fileLoc.getFilename().getValue());
  fileName.push_back('\0');
  Value file = LLVM::addStringToModule(UnknownLoc::get(ctx), rewriter,
                                       "gsanLocation_", fileName);
  return {file, b.i32_val(fileLoc.getLine())};
}

////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////

void emitTensorAccessRuntimeCall(ConversionPatternRewriter &rewriter,
                                 Location loc, Value gsanGlobalStatePtr,
                                 ArrayRef<Value> ptrElems,
                                 ArrayRef<Value> maskElems, uint32_t regMask,
                                 Value threadPred, int32_t bytesPerElem,
                                 bool isStore, unsigned elemIndexStride = 1) {
  if (ptrElems.empty())
    return;

  auto *ctx = rewriter.getContext();
  TritonLLVMOpBuilder b(loc, rewriter);
  Value one = b.i32_val(1);
  Value zero = b.i32_val(0);
  Type i8Ty = rewriter.getI8Type();
  Type i64Ty = rewriter.getI64Type();

  unsigned numElems = ptrElems.size();
  auto ptrArrayTy = array_ty(i64Ty, numElems);
  auto maskArrayTy = array_ty(i8Ty, numElems);
  SmallVector<Type> argsFieldTys = {ptrArrayTy, maskArrayTy};
  auto argsTy = LLVM::LLVMStructType::getLiteral(ctx, argsFieldTys);
  auto argsBuffer = LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx), argsTy,
                                           one, /*alignment=*/0);

  for (unsigned i = 0; i < numElems; ++i) {
    Value idx = b.i32_val(i);
    Value ptrValue = b.ptrtoint(i64_ty, ptrElems[i]);
    Value ptrSlot =
        b.gep(ptr_ty(ctx), argsTy, argsBuffer, ValueRange{zero, zero, idx});
    b.store(ptrValue, ptrSlot);

    Value maskValue = maskElems.empty() ? b.true_val() : maskElems[i];
    if (!isCanonicalIndex(i * elemIndexStride, regMask))
      maskValue = b.false_val();
    maskValue = ttg::maybeAnd(rewriter, loc, maskValue, threadPred);
    Value maskByte = b.zext(i8Ty, maskValue);
    Value maskSlot =
        b.gep(ptr_ty(ctx), argsTy, argsBuffer, ValueRange{zero, one, idx});
    b.store(maskByte, maskSlot);
  }

  StringRef funcName =
      isStore ? kGSanStoreTensorRuntimeFn : kGSanLoadTensorRuntimeFn;
  auto runtimeFunc = getOrCreateGSanRuntimeFunction(rewriter, funcName);
  if (gsanGlobalStatePtr.getType() != ptr_ty(ctx)) {
    gsanGlobalStatePtr = b.addrspacecast(ptr_ty(ctx), gsanGlobalStatePtr);
  }
  Value argsPtr = b.bitcast(argsBuffer, ptr_ty(ctx));
  auto sourceLoc = materializeSourceLocation(rewriter, loc);

  b.call(runtimeFunc,
         ValueRange{gsanGlobalStatePtr, argsPtr, b.i32_val(numElems),
                    b.i32_val(bytesPerElem), sourceLoc.file, sourceLoc.line});
}

void createBarrier(ConversionPatternRewriter &rewriter, Location loc,
                   int numCTAs, const TargetInfoBase &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (numCTAs == 1) {
    b.barrier(ttg::AddrSpace::Local);
  } else {
    targetInfo.clusterBarrier(loc, rewriter);
  }
}

unsigned getCanonicalIndex(unsigned index, unsigned freeVarMask) {
  return index & ~freeVarMask;
}

Value broadcastScalarAtomicResult(Operation *op, Type valueElemTy,
                                  Value resultVal,
                                  ConversionPatternRewriter &rewriter,
                                  TritonLLVMOpBuilder &b, Value threadPred,
                                  const TargetInfoBase &targetInfo) {
  if (!op->hasAttr("allocation.offset"))
    return resultVal;

  auto loc = op->getLoc();
  Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
  targetInfo.storeShared(rewriter, loc, smemBase, resultVal, threadPred);
  b.barrier(ttg::AddrSpace::Local);
  return targetInfo.loadShared(rewriter, loc, smemBase, valueElemTy,
                               b.true_val());
}

Value materializeI32Bool(ConversionPatternRewriter &rewriter,
                         TritonLLVMOpBuilder &b, Value pred) {
  if (!pred)
    return b.i32_val(1);
  return b.zext(i32_ty, pred);
}

void emitGSanAtomicBeginCall(ConversionPatternRewriter &rewriter, Location loc,
                             Value gsanGlobalStatePtr, Value eventStatePtr,
                             Value pred, Value ptr, int32_t bytesPerElem,
                             int32_t sem, int32_t scope,
                             GSanSourceLocation sourceLoc) {
  auto *ctx = rewriter.getContext();
  TritonLLVMOpBuilder b(loc, rewriter);
  if (gsanGlobalStatePtr.getType() != ptr_ty(ctx))
    gsanGlobalStatePtr = b.addrspacecast(ptr_ty(ctx), gsanGlobalStatePtr);
  Value statePtr = b.bitcast(eventStatePtr, ptr_ty(ctx));
  auto runtimeFunc =
      getOrCreateGSanRuntimeFunction(rewriter, kGSanAtomicBeginRuntimeFn);
  b.call(runtimeFunc,
         ValueRange{gsanGlobalStatePtr, statePtr,
                    materializeI32Bool(rewriter, b, pred),
                    b.ptrtoint(i64_ty, ptr), b.i32_val(bytesPerElem),
                    b.i32_val(sem), b.i32_val(scope), sourceLoc.file,
                    sourceLoc.line});
}

void emitGSanAtomicEndCall(ConversionPatternRewriter &rewriter, Location loc,
                           Value eventStatePtr, Value pred, Value didWrite,
                           int32_t sem, int32_t scope,
                           GSanSourceLocation sourceLoc) {
  TritonLLVMOpBuilder b(loc, rewriter);
  auto runtimeFunc =
      getOrCreateGSanRuntimeFunction(rewriter, kGSanAtomicEndRuntimeFn);
  Value statePtr = b.bitcast(eventStatePtr, ptr_ty(rewriter.getContext()));
  b.call(runtimeFunc,
         ValueRange{statePtr, materializeI32Bool(rewriter, b, pred),
                    materializeI32Bool(rewriter, b, didWrite), b.i32_val(sem),
                    b.i32_val(scope), sourceLoc.file, sourceLoc.line});
}

Value bitcastToScalarInt(ConversionPatternRewriter &rewriter, Location loc,
                         Value value) {
  Type ty = value.getType();
  if (ty.isInteger())
    return value;
  auto intTy =
      IntegerType::get(rewriter.getContext(), ty.getIntOrFloatBitWidth());
  TritonLLVMOpBuilder b(loc, rewriter);
  return b.bitcast(value, intTy);
}

Value getGSanGlobalStateArg(FunctionOpInterface funcOp) {
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    if (funcOp.getArgAttr(i, kGSanGlobalStateArgAttr))
      return funcOp.getArgument(i);
  }
  return {};
}

static LLVM::LLVMStructType
getTensorDescStructType(ConversionPatternRewriter &rewriter, Type basePtrTy) {
  SmallVector<Type> fieldTypes;
  fieldTypes.reserve(1 + 2 * (kTensorMapNumQwords - 1));
  fieldTypes.push_back(basePtrTy);
  fieldTypes.append(2 * (kTensorMapNumQwords - 1), i32_ty);
  return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), fieldTypes);
}

static Value extractTensorDescWord(ConversionPatternRewriter &rewriter,
                                   Location loc, Value descStruct,
                                   unsigned word) {
  assert(word >= kTensorMapScalarWordBase && word < 2 * kTensorMapNumQwords &&
         "tensor descriptor word index out of range");
  TritonLLVMOpBuilder b(loc, rewriter);
  Value wordValue =
      b.extract_val(i32_ty, descStruct, word - kTensorMapScalarWordBase + 1);
  return b.zext(i64_ty, wordValue);
}

static SmallVector<Value>
decodeTensorDescShape(ConversionPatternRewriter &rewriter, Location loc,
                      Value descStruct, unsigned rank) {
  TritonLLVMOpBuilder b(loc, rewriter);
  SmallVector<Value> shape;
  shape.reserve(rank);
  for (unsigned dim = 0; dim < rank; ++dim) {
    unsigned packedIdx = rank - 1 - dim;
    Value dimMinusOne = extractTensorDescWord(
        rewriter, loc, descStruct, kTensorMapShapeWordBase + packedIdx);
    shape.push_back(b.add(dimMinusOne, b.i64_val(1)));
  }
  return shape;
}

static SmallVector<Value>
decodeTensorDescStrides(ConversionPatternRewriter &rewriter, Location loc,
                        Value descStruct, unsigned rank, unsigned elemBytes) {
  TritonLLVMOpBuilder b(loc, rewriter);
  SmallVector<Value> strides;
  strides.reserve(rank);
  for (unsigned dim = 0; dim < rank; ++dim) {
    if (dim + 1 == rank) {
      strides.push_back(b.i64_val(1));
      continue;
    }
    unsigned packedIdx = rank - 2 - dim;
    Value strideUnits = extractTensorDescWord(
        rewriter, loc, descStruct, kTensorMapStrideWordBase + packedIdx);
    Value strideBytes = b.mul(strideUnits, b.i64_val(16));
    strides.push_back(b.udiv(strideBytes, b.i64_val(elemBytes)));
  }
  return strides;
}

////////////////////////////////////////////
// Patterns
////////////////////////////////////////////

struct GSanTensorAccessOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanTensorAccessOp> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalGSanTensorAccessOp>::ConvertOpToLLVMPattern;
  const TargetInfoBase *targetInfo;
  ModuleAxisInfoAnalysis *axisInfoAnalysis;

  GSanTensorAccessOpConversion(LLVMTypeConverter &typeConverter,
                               ModuleAxisInfoAnalysis &axisInfoAnalysis,
                               const TargetInfoBase &targetInfo,
                               PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(&targetInfo),
        axisInfoAnalysis(&axisInfoAnalysis) {}

  unsigned getVecSize(tti::ExperimentalGSanTensorAccessOp op) const {
    auto ptrTy = op.getPtr().getType();
    auto contiguity = axisInfoAnalysis->getContiguity(op.getPtr());
    if (!op.getMask()) {
      return contiguity;
    }

    auto maskAlign = axisInfoAnalysis->getMaskAlignment(op.getMask());
    auto bytesPerElem = tt::getPointeeBitWidth(ptrTy) / 8;
    // Round up to at least shadow granularity, and we will or together the
    // masks
    if (bytesPerElem < 4) {
      maskAlign = std::max(maskAlign, 4 / bytesPerElem);
    }
    return std::min(contiguity, maskAlign);
  }

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanTensorAccessOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getContext();
    Location loc = op.getLoc();
    auto func = op->getParentOfType<FunctionOpInterface>();
    Value gsanGlobalStatePtr = getGSanGlobalStateArg(func);
    if (!gsanGlobalStatePtr)
      return emitError(op.getLoc(), "Failed to find pointer to gsan state");

    Value llPtr = adaptor.getPtr();
    auto ptrTy = op.getPtr().getType();
    unsigned numElems = ttg::getTotalElemsPerThread(ptrTy);
    auto bytesPerElem = tt::getPointeeBitWidth(ptrTy) / 8;
    SmallVector<Value> ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems &&
           "Expected pointer element count to match layout");

    SmallVector<Value> maskElems;
    if (Value llMask = adaptor.getMask()) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems &&
             "Expected mask element count to match layout");
    }

    unsigned mergeVec = getVecSize(op);
    if (mergeVec > 1) {
      auto maskAlign =
          op.getMask() ? axisInfoAnalysis->getMaskAlignment(op.getMask()) : 1;
      SmallVector<Value> mergedPtrElems;
      SmallVector<Value> mergedMaskElems;
      mergedPtrElems.reserve(numElems / mergeVec);
      if (!maskElems.empty())
        mergedMaskElems.reserve(numElems / mergeVec);

      for (unsigned i = 0; i < numElems; i += mergeVec) {
        mergedPtrElems.push_back(ptrElems[i]);
        if (maskElems.empty())
          continue;
        Value mergedMask = maskElems[i];
        for (unsigned j = maskAlign; j < mergeVec; j += maskAlign) {
          mergedMask =
              arith::OrIOp::create(rewriter, loc, mergedMask, maskElems[i + j]);
        }
        mergedMaskElems.push_back(mergedMask);
      }

      ptrElems = std::move(mergedPtrElems);
      maskElems = std::move(mergedMaskElems);
      bytesPerElem *= mergeVec;
    }

    auto freeVarMasks = getFreeVariableMasks(ptrTy);
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));
    Value threadPred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter,
                                                         loc, *targetInfo);
    emitTensorAccessRuntimeCall(rewriter, loc, gsanGlobalStatePtr, ptrElems,
                                maskElems, regMask, threadPred, bytesPerElem,
                                op.getIsStore(), mergeVec);

    rewriter.eraseOp(op);
    return success();
  }
};

struct GSanAtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanAtomicRMWOp> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalGSanAtomicRMWOp>::ConvertOpToLLVMPattern;
  const TargetInfoBase *targetInfo;

  GSanAtomicRMWOpConversion(LLVMTypeConverter &typeConverter,
                            const TargetInfoBase &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanAtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Location loc = op.getLoc();
    auto func = op->getParentOfType<FunctionOpInterface>();
    Value gsanGlobalStatePtr = getGSanGlobalStateArg(func);
    if (!gsanGlobalStatePtr)
      return emitError(op.getLoc(), "Failed to find pointer to gsan state");

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for atomic op");
    auto rmwOp = op.getAtomicRmwOp();
    auto sem = op.getSem();
    auto scope = op.getScope();

    TritonLLVMOpBuilder b(loc, rewriter);
    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy = valElements[0].getType();
    unsigned valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    int32_t bytesPerElem = std::max<int32_t>(1, valueElemNBits / 8);
    auto elemsPerThread = ttg::getTotalElemsPerThread(op.getVal().getType());
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter,
                                                         loc, *targetInfo);
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));
    auto sourceLoc = materializeSourceLocation(rewriter, loc);
    auto eventStateTy = getGSanAtomicEventStateType(rewriter);
    Value eventState = LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx),
                                              eventStateTy, b.i32_val(1),
                                              /*alignment=*/0);

    SmallVector<Value> resultVals(elemsPerThread);

    for (size_t i = 0; i < elemsPerThread; ++i) {
      if (auto canonicalIdx = getCanonicalIndex(i, regMask);
          i != canonicalIdx) {
        resultVals[i] = resultVals[canonicalIdx];
        continue;
      }

      Value pred =
          llMask ? ttg::maybeAnd(rewriter, loc, threadPred, maskElements[i])
                 : threadPred;
      Value rmwPtr = ptrElements[i];
      Value rmwVal = valElements[i];

      emitGSanAtomicBeginCall(rewriter, loc, gsanGlobalStatePtr, eventState,
                              pred, rmwPtr, bytesPerElem,
                              static_cast<int32_t>(sem),
                              static_cast<int32_t>(scope), sourceLoc);

      SmallVector<Value> rmwVals{rmwVal};
      auto old = NVIDIA::emitPtxAtomicRMW(rewriter, loc, valueElemTy, rmwPtr,
                                          rmwVals, rmwOp, sem, scope, pred);
      if (failed(old))
        return failure();

      emitGSanAtomicEndCall(rewriter, loc, eventState, pred, pred,
                            static_cast<int32_t>(sem),
                            static_cast<int32_t>(scope), sourceLoc);
      resultVals[i] = *old;
    }

    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    if (!tensorTy) {
      Value scalarResult = broadcastScalarAtomicResult(
          op, valueElemTy, resultVals[0], rewriter, b, threadPred, *targetInfo);
      rewriter.replaceOp(op, {scalarResult});
      return success();
    }

    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, *targetInfo, getTypeConverter());
    return success();
  }
};

struct GSanAtomicCASOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanAtomicCASOp> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalGSanAtomicCASOp>::ConvertOpToLLVMPattern;
  const TargetInfoBase *targetInfo;

  GSanAtomicCASOpConversion(LLVMTypeConverter &typeConverter,
                            const TargetInfoBase &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanAtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Location loc = op.getLoc();
    auto func = op->getParentOfType<FunctionOpInterface>();
    Value gsanGlobalStatePtr = getGSanGlobalStateArg(func);
    if (!gsanGlobalStatePtr)
      return emitError(op.getLoc(), "Failed to find pointer to gsan state");

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for atomic op");
    auto sem = op.getSem();
    auto scope = op.getScope();

    TritonLLVMOpBuilder b(loc, rewriter);
    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy = valElements[0].getType();
    unsigned valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    int32_t bytesPerElem = valueElemNBits / 8;
    auto elemsPerThread = ttg::getTotalElemsPerThread(op.getVal().getType());
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter,
                                                         loc, *targetInfo);
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));
    auto sourceLoc = materializeSourceLocation(rewriter, loc);
    auto eventStateTy = getGSanAtomicEventStateType(rewriter);
    Value eventState = LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx),
                                              eventStateTy, b.i32_val(1),
                                              /*alignment=*/0);

    SmallVector<Value> resultVals(elemsPerThread);

    for (size_t i = 0; i < elemsPerThread; ++i) {
      if (auto canonicalIdx = getCanonicalIndex(i, regMask);
          canonicalIdx != i) {
        resultVals[i] = resultVals[canonicalIdx];
        continue;
      }

      Value pred = threadPred;
      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      Value casVal = valElements[i];

      emitGSanAtomicBeginCall(rewriter, loc, gsanGlobalStatePtr, eventState,
                              pred, casPtr, bytesPerElem,
                              static_cast<int32_t>(sem),
                              static_cast<int32_t>(scope), sourceLoc);

      Value old = NVIDIA::emitPtxAtomicCAS(rewriter, loc, valueElemTy, casPtr,
                                           casCmp, casVal, sem, scope, pred);

      auto oldInt = bitcastToScalarInt(rewriter, loc, old);
      auto cmpInt = bitcastToScalarInt(rewriter, loc, casCmp);
      Value didWrite = LLVM::ICmpOp::create(
          rewriter, loc, i1_ty, LLVM::ICmpPredicate::eq, oldInt, cmpInt);
      didWrite = ttg::maybeAnd(rewriter, loc, pred, didWrite);
      emitGSanAtomicEndCall(rewriter, loc, eventState, pred, didWrite,
                            static_cast<int32_t>(sem),
                            static_cast<int32_t>(scope), sourceLoc);
      resultVals[i] = old;
    }

    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    if (!tensorTy) {
      Value scalarResult = broadcastScalarAtomicResult(
          op, valueElemTy, resultVals[0], rewriter, b, threadPred, *targetInfo);
      rewriter.replaceOp(op, {scalarResult});
      return success();
    }

    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, *targetInfo, getTypeConverter());
    return success();
  }
};

struct GSanTensorDescInfoOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanTensorDescInfoOp> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalGSanTensorDescInfoOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanTensorDescInfoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto descTy = dyn_cast<tt::TensorDescInterface>(op.getDesc().getType());
    if (!descTy)
      return rewriter.notifyMatchFailure(op, "expected tensor descriptor type");

    auto elemTy = descTy.getSignlessBlockType().getElementType();
    if (!elemTy.isIntOrFloat() || (elemTy.getIntOrFloatBitWidth() % 8) != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "expected byte-addressable element");
    }

    unsigned rank = descTy.getBlockType().getRank();
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    if (op->getNumResults() != 1 + 2 * rank) {
      return rewriter.notifyMatchFailure(
          op, "descriptor info result count does not match descriptor rank");
    }

    TritonLLVMOpBuilder b(loc, rewriter);
    Type ptrTy = getTypeConverter()->convertType(op->getResult(0).getType());
    auto structTy = getTensorDescStructType(rewriter, ptrTy);
    Value descStruct = b.load(structTy, adaptor.getDesc());
    SmallVector<Value> decoded;
    decoded.reserve(op->getNumResults());
    decoded.push_back(b.extract_val(ptrTy, descStruct, 0));
    auto shape = decodeTensorDescShape(rewriter, loc, descStruct, rank);
    decoded.append(shape.begin(), shape.end());
    auto strides =
        decodeTensorDescStrides(rewriter, loc, descStruct, rank, elemBytes);
    decoded.append(strides.begin(), strides.end());
    rewriter.replaceOp(op, decoded);
    return success();
  }
};

struct GSanInitOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanInitOp> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalGSanInitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanInitOp op,
                  [[maybe_unused]] OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<FunctionOpInterface>();
    Value gsanGlobalStatePtr = getGSanGlobalStateArg(func);
    if (!gsanGlobalStatePtr)
      return emitError(op.getLoc(), "Failed to find pointer to gsan state");

    auto runtimeFunc =
        getOrCreateGSanRuntimeFunction(rewriter, kGSanInitRuntimeFn);
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    TritonLLVMOpBuilder b(loc, rewriter);
    if (gsanGlobalStatePtr.getType() != ptr_ty(ctx)) {
      gsanGlobalStatePtr = b.addrspacecast(ptr_ty(ctx), gsanGlobalStatePtr);
    }
    auto sourceLoc = materializeSourceLocation(rewriter, loc);
    b.call(runtimeFunc,
           ValueRange{gsanGlobalStatePtr, sourceLoc.file, sourceLoc.line});
    b.barrier(ttg::AddrSpace::Local);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateGSanToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const TargetInfoBase &targetInfo) {
  patterns.add<GSanInitOpConversion>(typeConverter);
  patterns.add<GSanTensorDescInfoOpConversion>(typeConverter);
  patterns.add<GSanAtomicCASOpConversion>(typeConverter, targetInfo);
  patterns.add<GSanAtomicRMWOpConversion>(typeConverter, targetInfo);
  patterns.add<GSanTensorAccessOpConversion>(typeConverter, axisInfoAnalysis,
                                             targetInfo);
}
