#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/AtomicPTXBuilder.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <type_traits>
#include <utility>

namespace tt = mlir::triton;
namespace tti = mlir::triton::instrument;
namespace ttg = mlir::triton::gpu;

namespace {

static constexpr unsigned kGSanShadowGranularityBytes = 4;

struct GSanSourceLocation {
  Value file;
  Value line;
};

static constexpr StringLiteral kGSanLoadTensorRuntimeFn =
    "__triton_gsan_load_tensor";
static constexpr StringLiteral kGSanStoreTensorRuntimeFn =
    "__triton_gsan_store_tensor";
static constexpr StringLiteral kGSanLoadTensorDescRuntimeFn =
    "__triton_gsan_load_tensor_desc";
static constexpr StringLiteral kGSanStoreTensorDescRuntimeFn =
    "__triton_gsan_store_tensor_desc";
static constexpr StringLiteral kGSanLoadIndexedTensorDescRuntimeFn =
    "__triton_gsan_load_indexed_tensor_desc";
static constexpr StringLiteral kGSanStoreIndexedTensorDescRuntimeFn =
    "__triton_gsan_store_indexed_tensor_desc";
static constexpr StringLiteral kGSanAtomicTensorDescRuntimeFn =
    "__triton_gsan_atomic_tensor_desc";
static constexpr StringLiteral kGSanAtomicBeginRuntimeFn =
    "__triton_gsan_atomic_begin_scalar";
static constexpr StringLiteral kGSanAtomicEndRuntimeFn =
    "__triton_gsan_atomic_end_scalar";
static constexpr StringLiteral kGSanInitRuntimeFn = "__triton_gsan_init";
static constexpr StringLiteral kGSanKernelExitRuntimeFn =
    "__triton_gsan_kernel_exit";
static constexpr StringLiteral kGSanGridDependencyWaitRuntimeFn =
    "__triton_gsan_grid_dependency_wait";
static constexpr StringLiteral kGSanClusterBarrierInitRuntimeFn =
    "__triton_gsan_cluster_barrier_init";
static constexpr StringLiteral kGSanClusterBarrierSyncRuntimeFn =
    "__triton_gsan_cluster_barrier_sync";
static constexpr StringLiteral kGSanMBarrierTableInitRuntimeFn =
    "__triton_gsan_mbarrier_table_init";
static constexpr StringLiteral kGSanMBarrierInitRuntimeFn =
    "__triton_gsan_mbarrier_init";
static constexpr StringLiteral kGSanMBarrierArriveRuntimeFn =
    "__triton_gsan_mbarrier_arrive";
static constexpr StringLiteral kGSanMBarrierWaitRuntimeFn =
    "__triton_gsan_mbarrier_wait";
static constexpr StringLiteral kGSanGlobalStateArgAttr =
    "tti.gsan_global_state";
static constexpr StringLiteral kGSanStreamClockArgAttr =
    "tti.gsan_stream_clock";
static constexpr StringLiteral kGSanKernelIdArgAttr = "tti.gsan_kernel_id";

LLVM::LLVMFuncOp
getOrCreateGSanRuntimeFunction(ConversionPatternRewriter &rewriter,
                               StringRef funcName) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  if (auto funcOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return funcOp;

  auto *ctx = rewriter.getContext();
  SmallVector<Type> argTys;
  if (funcName == kGSanInitRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i64_ty,      i32_ty, i32_ty,
              i32_ty,      i32_ty,      ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanKernelExitRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i64_ty,      i32_ty,
              i32_ty,      i32_ty,      ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanGridDependencyWaitRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i64_ty, i32_ty, i32_ty, i32_ty};
  } else if (funcName == kGSanClusterBarrierInitRuntimeFn) {
    argTys = {ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanClusterBarrierSyncRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i32_ty, i32_ty,
              i32_ty,      ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanMBarrierTableInitRuntimeFn) {
    argTys = {ptr_ty(ctx), i32_ty, i32_ty};
  } else if (funcName == kGSanMBarrierInitRuntimeFn) {
    argTys = {ptr_ty(ctx), i32_ty, i32_ty, i32_ty, i32_ty, ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanMBarrierArriveRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i32_ty, i32_ty,      i32_ty,
              i32_ty,      i32_ty,      i32_ty, ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanMBarrierWaitRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i32_ty,      i32_ty,
              i32_ty,      i32_ty,      ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanLoadTensorRuntimeFn ||
             funcName == kGSanStoreTensorRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i32_ty, i32_ty, ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanLoadTensorDescRuntimeFn ||
             funcName == kGSanStoreTensorDescRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), i32_ty,
              ptr_ty(ctx), i32_ty,      i32_ty,      i32_ty,
              i32_ty,      ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanLoadIndexedTensorDescRuntimeFn ||
             funcName == kGSanStoreIndexedTensorDescRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), i32_ty,      i32_ty,
              i32_ty,      i32_ty,      i32_ty,      ptr_ty(ctx), i32_ty};
  } else if (funcName == kGSanAtomicTensorDescRuntimeFn) {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), i32_ty,
              ptr_ty(ctx), i32_ty,      i32_ty,      i32_ty,
              i32_ty,      i32_ty,      ptr_ty(ctx), i32_ty};
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

std::pair<Value, unsigned>
prepareTensorStackArg(ConversionPatternRewriter &rewriter, Location loc,
                      ArrayRef<Value> ptrElems, ArrayRef<Value> maskElems,
                      uint32_t regMask, Value threadPred,
                      unsigned elemIndexStride) {
  auto *ctx = rewriter.getContext();
  TritonLLVMOpBuilder b(loc, rewriter);
  Value one = b.i32_val(1);
  Value zero = b.i32_val(0);
  Type i8Ty = rewriter.getI8Type();
  Type i64Ty = rewriter.getI64Type();

  unsigned numElems = 0;
  for (unsigned i = 0; i < ptrElems.size(); ++i)
    numElems += isCanonicalIndex(i * elemIndexStride, regMask);
  auto ptrArrayTy = array_ty(i64Ty, numElems);
  auto maskArrayTy = array_ty(i8Ty, numElems);
  SmallVector<Type> argsFieldTys = {ptrArrayTy, maskArrayTy};
  auto argsTy = LLVM::LLVMStructType::getLiteral(ctx, argsFieldTys);
  auto argsBuffer = LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx), argsTy,
                                           one, /*alignment=*/0);

  unsigned packedIndex = 0;
  for (unsigned i = 0; i < ptrElems.size(); ++i) {
    if (!isCanonicalIndex(i * elemIndexStride, regMask))
      continue;
    Value idx = b.i32_val(packedIndex++);
    Value ptrValue = b.ptrtoint(i64_ty, ptrElems[i]);
    Value ptrSlot =
        b.gep(ptr_ty(ctx), argsTy, argsBuffer, ValueRange{zero, zero, idx});
    b.store(ptrValue, ptrSlot);

    Value maskValue = maskElems.empty() ? b.true_val() : maskElems[i];
    maskValue = ttg::maybeAnd(rewriter, loc, maskValue, threadPred);
    Value maskByte = b.zext(i8Ty, maskValue);
    Value maskSlot =
        b.gep(ptr_ty(ctx), argsTy, argsBuffer, ValueRange{zero, one, idx});
    b.store(maskByte, maskSlot);
  }

  return {argsBuffer, numElems};
}

void emitTensorAccessRuntimeCall(ConversionPatternRewriter &rewriter,
                                 Location loc, Value gsanGlobalStatePtr,
                                 ArrayRef<Value> ptrElems,
                                 ArrayRef<Value> maskElems, uint32_t regMask,
                                 Value threadPred, int32_t bytesPerElem,
                                 bool isStore, unsigned elemIndexStride = 1) {
  if (ptrElems.empty())
    return;

  TritonLLVMOpBuilder b(loc, rewriter);
  auto [stackPtr, numElems] = prepareTensorStackArg(
      rewriter, loc, ptrElems, maskElems, regMask, threadPred, elemIndexStride);
  StringRef funcName =
      isStore ? kGSanStoreTensorRuntimeFn : kGSanLoadTensorRuntimeFn;
  auto runtimeFunc = getOrCreateGSanRuntimeFunction(rewriter, funcName);
  auto sourceLoc = materializeSourceLocation(rewriter, loc);

  b.call(runtimeFunc,
         ValueRange{gsanGlobalStatePtr, stackPtr, b.i32_val(numElems),
                    b.i32_val(bytesPerElem), sourceLoc.file, sourceLoc.line});
}

unsigned getCanonicalIndex(unsigned index, unsigned freeVarMask) {
  return index & ~freeVarMask;
}

Value materializeI32Bool(ConversionPatternRewriter &rewriter,
                         TritonLLVMOpBuilder &b, Value pred) {
  if (!pred)
    return b.i32_val(1);
  return b.zext(i32_ty, pred);
}

Value castToGenericPointer(ConversionPatternRewriter &rewriter, Location loc,
                           Value ptr) {
  Type genericPtrTy = ptr_ty(rewriter.getContext());
  if (ptr.getType() == genericPtrTy)
    return ptr;
  TritonLLVMOpBuilder b(loc, rewriter);
  return b.addrspacecast(genericPtrTy, ptr);
}

void emitGSanAtomicBeginCall(ConversionPatternRewriter &rewriter, Location loc,
                             Value gsanGlobalStatePtr, Value eventStatePtr,
                             Value pred, Value ptr, int32_t bytesPerElem,
                             int32_t sem, int32_t scope,
                             GSanSourceLocation sourceLoc) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Value statePtr = b.bitcast(eventStatePtr, ptr_ty(rewriter.getContext()));
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

template <typename OpT>
unsigned getTensorAccessVecSize(OpT op,
                                ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                bool keepWithinSingleShadowCell) {
  auto ptrTy = op.getPtr().getType();
  auto bytesPerElem = std::max(8u, tt::getPointeeBitWidth(ptrTy)) / 8;
  auto contiguity = axisInfoAnalysis.getContiguity(op.getPtr());

  if (keepWithinSingleShadowCell) {
    if (bytesPerElem >= kGSanShadowGranularityBytes)
      return 1;
    contiguity =
        std::min(contiguity, kGSanShadowGranularityBytes / bytesPerElem);
  }

  if (!op.getMask())
    return contiguity;

  auto maskAlign = axisInfoAnalysis.getMaskAlignment(op.getMask());
  if (bytesPerElem < kGSanShadowGranularityBytes) {
    maskAlign = std::max(maskAlign, kGSanShadowGranularityBytes / bytesPerElem);
  }
  return std::min(contiguity, maskAlign);
}

void mergeTensorAccessElements(ConversionPatternRewriter &rewriter,
                               Location loc, SmallVector<Value> &ptrElems,
                               SmallVector<Value> &maskElems, unsigned mergeVec,
                               unsigned maskAlign, int32_t &bytesPerElem) {
  if (mergeVec <= 1)
    return;

  SmallVector<Value> mergedPtrElems;
  SmallVector<Value> mergedMaskElems;
  mergedPtrElems.reserve(ptrElems.size() / mergeVec);
  if (!maskElems.empty())
    mergedMaskElems.reserve(ptrElems.size() / mergeVec);

  for (unsigned i = 0; i < ptrElems.size(); i += mergeVec) {
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

FailureOr<Value> getGSanGlobalStateArg(Operation *op,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc) {
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    if (!funcOp.getArgAttr(i, kGSanGlobalStateArgAttr))
      continue;
    Value arg = funcOp.getArgument(i);
    if (arg.getType() == ptr_ty(rewriter.getContext()))
      return arg;
    TritonLLVMOpBuilder b(loc, rewriter);
    arg = b.addrspacecast(ptr_ty(rewriter.getContext()), arg);
    return arg;
  }
  return emitError(loc, "Unable to find gsan global state");
}

FailureOr<Value> getGSanStreamClockArg(Operation *op,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc) {
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    if (!funcOp.getArgAttr(i, kGSanStreamClockArgAttr))
      continue;
    Value arg = funcOp.getArgument(i);
    if (arg.getType() == ptr_ty(rewriter.getContext()))
      return arg;
    TritonLLVMOpBuilder b(loc, rewriter);
    arg = b.addrspacecast(ptr_ty(rewriter.getContext()), arg);
    return arg;
  }
  return emitError(loc, "Unable to find gsan stream clock");
}

FailureOr<Value> getGSanKernelIdArg(Operation *op,
                                    ConversionPatternRewriter &rewriter,
                                    Location loc) {
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    if (funcOp.getArgAttr(i, kGSanKernelIdArgAttr))
      return funcOp.getArgument(i);
  }
  return emitError(loc, "Unable to find gsan kernel ID");
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
    return getTensorAccessVecSize(op, *axisInfoAnalysis,
                                  /*keepWithinSingleShadowCell=*/false);
  }

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanTensorAccessOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto ptrTy = op.getPtr().getType();
    int32_t bytesPerElem = tt::getPointeeBitWidth(ptrTy) / 8;
    auto ptrElems =
        unpackTensorElements(loc, adaptor.getPtr(), rewriter, ptrTy);
    SmallVector<Value> maskElems;
    if (Value llMask = adaptor.getMask()) {
      maskElems =
          unpackTensorElements(loc, llMask, rewriter, op.getMask().getType());
    }

    unsigned mergeVec = getVecSize(op);
    auto maskAlign =
        op.getMask() ? axisInfoAnalysis->getMaskAlignment(op.getMask()) : 1;
    mergeTensorAccessElements(rewriter, loc, ptrElems, maskElems, mergeVec,
                              maskAlign, bytesPerElem);

    auto ctx = op.getContext();
    auto kReg = str_attr("register");
    auto freeVarMasks = getFreeVariableMasks(ptrTy);
    auto threadPred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter,
                                                        loc, *targetInfo);
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr))
      return failure();
    emitTensorAccessRuntimeCall(rewriter, loc, *gsanGlobalStatePtr, ptrElems,
                                maskElems, freeVarMasks.lookup(kReg),
                                threadPred, bytesPerElem, op.getIsStore(),
                                mergeVec);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename AccessOp>
struct GSanTensorDescAccessOpConversion
    : public ConvertOpToLLVMPattern<AccessOp> {
public:
  using ConvertOpToLLVMPattern<AccessOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename AccessOp::Adaptor;
  const TargetInfoBase *targetInfo;

  GSanTensorDescAccessOpConversion(LLVMTypeConverter &typeConverter,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<AccessOp>(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(AccessOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr))
      return failure();

    auto descTy = cast<tt::TensorDescType>(op.getDesc().getType());
    ArrayRef<int64_t> blockShape = descTy.getShape();
    if (blockShape.empty() || blockShape.size() > 5 ||
        blockShape.size() != op.getCoords().size()) {
      return rewriter.notifyMatchFailure(
          op, "expected one to five descriptor coordinates");
    }

    for (int64_t extent : blockShape)
      if (extent < 1 || extent > std::numeric_limits<int16_t>::max())
        return rewriter.notifyMatchFailure(
            op, "descriptor block extent does not fit in int16_t");

    auto elemTy = descTy.getSignlessBlockType().getElementType();
    if (!elemTy.isIntOrFloat() || elemTy.getIntOrFloatBitWidth() % 8 != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "expected byte-addressable element");
    }

    auto *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    unsigned rank = blockShape.size();
    auto coordsTy = array_ty(i32_ty, rank);
    Value coords = LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx), coordsTy,
                                          b.i32_val(1), /*alignment=*/0);
    auto shapeTy = array_ty(i16_ty, rank);
    Value shape = LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx), shapeTy,
                                         b.i32_val(1), /*alignment=*/0);
    for (auto [dim, coord] : llvm::enumerate(adaptor.getCoords())) {
      Value coordSlot = b.gep(ptr_ty(ctx), coordsTy, coords,
                              ValueRange{b.i32_val(0), b.i32_val(dim)});
      b.store(coord, coordSlot);
      Value shapeSlot = b.gep(ptr_ty(ctx), shapeTy, shape,
                              ValueRange{b.i32_val(0), b.i32_val(dim)});
      b.store(b.i16_val(blockShape[dim]), shapeSlot);
    }

    Value warpId = getLaneAndWarpId(rewriter, loc).second;
    int numWarps = ttg::lookupNumWarps(op);
    int numCTAs = ttg::lookupNumCTAs(op);
    if (numCTAs > 1) {
      Value ctaRank = targetInfo->getClusterCTAId(rewriter, loc);
      warpId = b.add(b.mul(warpId, b.i32_val(numCTAs)), ctaRank);
      numWarps *= numCTAs;
    }

    StringRef funcName;
    if constexpr (std::is_same_v<
                      AccessOp,
                      tti::ExperimentalGSanAtomicTensorDescAccessOp>) {
      funcName = kGSanAtomicTensorDescRuntimeFn;
    } else {
      funcName = op.getIsStore() ? kGSanStoreTensorDescRuntimeFn
                                 : kGSanLoadTensorDescRuntimeFn;
    }
    auto runtimeFunc = getOrCreateGSanRuntimeFunction(rewriter, funcName);
    auto sourceLoc = materializeSourceLocation(rewriter, loc);
    SmallVector<Value> args{
        *gsanGlobalStatePtr,
        castToGenericPointer(rewriter, loc, adaptor.getDesc()),
        coords,
        b.i32_val(rank),
        shape,
        b.i32_val(elemTy.getIntOrFloatBitWidth() / 8)};
    if constexpr (std::is_same_v<
                      AccessOp,
                      tti::ExperimentalGSanAtomicTensorDescAccessOp>) {
      args.append({b.i32_val(static_cast<int32_t>(op.getSem())),
                   b.i32_val(static_cast<int32_t>(op.getScope()))});
    } else {
      args.push_back(materializeI32Bool(rewriter, b, adaptor.getPred()));
    }
    args.append({warpId, b.i32_val(numWarps), sourceLoc.file, sourceLoc.line});
    b.call(runtimeFunc, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct GSanIndexedTensorDescAccessOpConversion
    : public ConvertOpToLLVMPattern<
          tti::ExperimentalGSanIndexedTensorDescAccessOp> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalGSanIndexedTensorDescAccessOp>::ConvertOpToLLVMPattern;
  const TargetInfoBase *targetInfo;

  GSanIndexedTensorDescAccessOpConversion(LLVMTypeConverter &typeConverter,
                                          const TargetInfoBase &targetInfo,
                                          PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanIndexedTensorDescAccessOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr))
      return failure();

    auto descTy = cast<tt::TensorDescType>(op.getDesc().getType());
    ArrayRef<int64_t> blockShape = descTy.getShape();
    if (blockShape.size() != 2 || blockShape.front() != 1)
      return rewriter.notifyMatchFailure(op, "expected a 1xN descriptor");

    auto elemTy = descTy.getSignlessBlockType().getElementType();
    if (!elemTy.isIntOrFloat() || elemTy.getIntOrFloatBitWidth() % 8 != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "expected byte-addressable element");
    }

    auto *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> indexElems =
        unpackUniqueTensorElements(loc, adaptor.getIndices(), rewriter);
    auto indicesTy = array_ty(i32_ty, indexElems.size());
    Value indices = LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx),
                                           indicesTy, b.i32_val(1), 0);
    for (auto [index, elem] : llvm::enumerate(indexElems)) {
      Value slot = b.gep(ptr_ty(ctx), indicesTy, indices,
                         ValueRange{b.i32_val(0), b.i32_val(index)});
      b.store(elem, slot);
    }

    auto freeVarMasks = getFreeVariableMasks(op.getIndices().getType());
    freeVarMasks[StringAttr::get(ctx, "lane")] = 0;
    Value pred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter, loc,
                                                   *targetInfo);
    pred = ttg::maybeAnd(rewriter, loc, pred, adaptor.getPred());

    StringRef funcName = op.getIsStore() ? kGSanStoreIndexedTensorDescRuntimeFn
                                         : kGSanLoadIndexedTensorDescRuntimeFn;
    auto runtimeFunc = getOrCreateGSanRuntimeFunction(rewriter, funcName);
    auto sourceLoc = materializeSourceLocation(rewriter, loc);
    b.call(runtimeFunc,
           ValueRange{*gsanGlobalStatePtr,
                      castToGenericPointer(rewriter, loc, adaptor.getDesc()),
                      indices, b.i32_val(indexElems.size()),
                      adaptor.getOffset(), b.i32_val(blockShape.back()),
                      b.i32_val(elemTy.getIntOrFloatBitWidth() / 8),
                      materializeI32Bool(rewriter, b, pred), sourceLoc.file,
                      sourceLoc.line});
    rewriter.eraseOp(op);
    return success();
  }
};

struct GSanAtomicPollOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanAtomicPollOp> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalGSanAtomicPollOp>::ConvertOpToLLVMPattern;
  const TargetInfoBase *targetInfo;

  GSanAtomicPollOpConversion(LLVMTypeConverter &typeConverter,
                             const TargetInfoBase &targetInfo,
                             PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanAtomicPollOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Location loc = op.getLoc();
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr))
      return failure();

    auto ptrs = unpackUniqueTensorElements(loc, adaptor.getPtr(), rewriter);
    auto matched =
        unpackUniqueTensorElements(loc, adaptor.getMatched(), rewriter);
    int32_t bytesPerElem = tt::getPointeeBitWidth(op.getPtr().getType()) / 8;
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter,
                                                         loc, *targetInfo);
    auto sourceLoc = materializeSourceLocation(rewriter, loc);

    TritonLLVMOpBuilder b(loc, rewriter);
    auto eventStateTy = getGSanAtomicEventStateType(rewriter);
    Value eventState = LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx),
                                              eventStateTy, b.i32_val(1),
                                              /*alignment=*/0);
    for (auto [ptr, success] : llvm::zip_equal(ptrs, matched)) {
      Value pred = ttg::maybeAnd(rewriter, loc, threadPred, success);
      emitGSanAtomicBeginCall(rewriter, loc, *gsanGlobalStatePtr, eventState,
                              pred, ptr, bytesPerElem,
                              static_cast<int32_t>(op.getSem()),
                              static_cast<int32_t>(op.getScope()), sourceLoc);
      emitGSanAtomicEndCall(rewriter, loc, eventState, pred, b.false_val(),
                            static_cast<int32_t>(op.getSem()),
                            static_cast<int32_t>(op.getScope()), sourceLoc);
    }

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
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr))
      return failure();

    auto rmwOp = op.getAtomicRmwOp();
    auto sem = op.getSem();
    auto scope = op.getScope();
    insertAtomicOrderingBarriers(op, sem, !atomicResultHasOrderingBarrier(op),
                                 rewriter, *targetInfo);

    TritonLLVMOpBuilder b(loc, rewriter);
    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto ptrElements =
        unpackTensorElements(loc, llPtr, rewriter, op.getPtr().getType());
    auto valElements =
        unpackTensorElements(loc, llVal, rewriter, op.getVal().getType());
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements =
          unpackTensorElements(loc, llMask, rewriter, op.getMask().getType());

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy = valElements[0].getType();
    unsigned valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    int32_t bytesPerElem = std::max<int32_t>(1, valueElemNBits / 8);
    auto elemsPerThread = ttg::getTotalElemsPerThread(op.getVal().getType());
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter,
                                                         loc, *targetInfo);
    uint32_t regMask = freeVarMasks.lookup(str_attr("register"));
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

      emitGSanAtomicBeginCall(rewriter, loc, *gsanGlobalStatePtr, eventState,
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
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr))
      return failure();

    auto sem = op.getSem();
    auto scope = op.getScope();
    insertAtomicOrderingBarriers(op, sem, !atomicResultHasOrderingBarrier(op),
                                 rewriter, *targetInfo);

    TritonLLVMOpBuilder b(loc, rewriter);
    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto ptrElements =
        unpackTensorElements(loc, llPtr, rewriter, op.getPtr().getType());
    auto cmpElements =
        unpackTensorElements(loc, llCmp, rewriter, op.getCmp().getType());
    auto valElements =
        unpackTensorElements(loc, llVal, rewriter, op.getVal().getType());

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy = valElements[0].getType();
    unsigned valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    int32_t bytesPerElem = valueElemNBits / 8;
    auto elemsPerThread = ttg::getTotalElemsPerThread(op.getVal().getType());
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter,
                                                         loc, *targetInfo);
    uint32_t regMask = freeVarMasks.lookup(str_attr("register"));
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

      emitGSanAtomicBeginCall(rewriter, loc, *gsanGlobalStatePtr, eventState,
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

struct GSanInitOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanInitOp> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalGSanInitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanInitOp op,
                  [[maybe_unused]] OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr))
      return failure();
    auto streamClockPtr = getGSanStreamClockArg(op, rewriter, loc);
    if (failed(streamClockPtr))
      return failure();
    auto kernelId = getGSanKernelIdArg(op, rewriter, loc);
    if (failed(kernelId))
      return failure();

    auto runtimeFunc =
        getOrCreateGSanRuntimeFunction(rewriter, kGSanInitRuntimeFn);

    TritonLLVMOpBuilder b(loc, rewriter);
    auto sourceLoc = materializeSourceLocation(rewriter, loc);
    auto threadIdx = mlir::getThreadId(rewriter, loc);
    auto numThreads = b.i32_val(ttg::lookupNumWarps(op) *
                                ttg::lookupThreadsPerWarp(rewriter));
    Value barrierId = tt::nvgpu::WarpGroupBarrierIdOp::create(rewriter, loc);
    b.call(runtimeFunc,
           ValueRange{*gsanGlobalStatePtr, *streamClockPtr, *kernelId,
                      b.i32_val(op.getAcquireStreamClock()), threadIdx,
                      numThreads, barrierId, sourceLoc.file, sourceLoc.line});
    b.barrier(ttg::AddrSpace::Local);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename OpTy>
struct GSanStreamClockOpConversion : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    auto streamClockPtr = getGSanStreamClockArg(op, rewriter, loc);
    auto kernelId = getGSanKernelIdArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr) || failed(streamClockPtr) ||
        failed(kernelId))
      return failure();

    StringRef runtimeFn;
    if constexpr (std::is_same_v<OpTy, tti::ExperimentalGSanKernelExitOp>)
      runtimeFn = kGSanKernelExitRuntimeFn;
    else
      runtimeFn = kGSanGridDependencyWaitRuntimeFn;
    auto runtimeFunc = getOrCreateGSanRuntimeFunction(rewriter, runtimeFn);
    TritonLLVMOpBuilder b(loc, rewriter);
    auto threadIdx = mlir::getThreadId(rewriter, loc);
    auto numThreads = b.i32_val(ttg::lookupNumWarps(op) *
                                ttg::lookupThreadsPerWarp(rewriter));
    Value barrierId = tt::nvgpu::WarpGroupBarrierIdOp::create(rewriter, loc);
    SmallVector<Value> args{*gsanGlobalStatePtr, *streamClockPtr, *kernelId,
                            threadIdx,           numThreads,      barrierId};
    if constexpr (std::is_same_v<OpTy, tti::ExperimentalGSanKernelExitOp>) {
      auto sourceLoc = materializeSourceLocation(rewriter, loc);
      args.append({sourceLoc.file, sourceLoc.line});
    }
    b.call(runtimeFunc, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct GSanClusterBarrierInitOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanClusterBarrierInitOp> {
  const TargetInfoBase &targetInfo;

  GSanClusterBarrierInitOpConversion(LLVMTypeConverter &typeConverter,
                                     const TargetInfoBase &targetInfo)
      : ConvertOpToLLVMPattern<tti::ExperimentalGSanClusterBarrierInitOp>(
            typeConverter),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanClusterBarrierInitOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    Value elect = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);
    Value ctaRank = targetInfo.getClusterCTAId(rewriter, loc);
    Value isCTA0 = b.icmp_eq(ctaRank, b.i32_val(0));
    Value pred = b.and_(elect, isCTA0);
    auto runtimeFunc = getOrCreateGSanRuntimeFunction(
        rewriter, kGSanClusterBarrierInitRuntimeFn);
    Value scratch = castToGenericPointer(rewriter, loc, adaptor.getScratch());
    b.call(runtimeFunc, ValueRange{scratch, b.zext(i32_ty, pred)});
    rewriter.eraseOp(op);
    return success();
  }
};

struct GSanClusterBarrierSyncOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanClusterBarrierSyncOp> {
  const TargetInfoBase &targetInfo;

  GSanClusterBarrierSyncOpConversion(LLVMTypeConverter &typeConverter,
                                     const TargetInfoBase &targetInfo)
      : ConvertOpToLLVMPattern<tti::ExperimentalGSanClusterBarrierSyncOp>(
            typeConverter),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanClusterBarrierSyncOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr))
      return failure();

    TritonLLVMOpBuilder b(loc, rewriter);
    Value elect = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);
    Value ctaRank = targetInfo.getClusterCTAId(rewriter, loc);
    auto sourceLoc = materializeSourceLocation(rewriter, loc);
    auto runtimeFunc = getOrCreateGSanRuntimeFunction(
        rewriter, kGSanClusterBarrierSyncRuntimeFn);
    Value scratch = castToGenericPointer(rewriter, loc, adaptor.getScratch());
    b.call(runtimeFunc,
           ValueRange{*gsanGlobalStatePtr, scratch, b.zext(i32_ty, elect),
                      b.i32_val(ttg::lookupNumCTAs(op)), ctaRank,
                      sourceLoc.file, sourceLoc.line});
    rewriter.eraseOp(op);
    return success();
  }
};

static Value getMBarrierOffset(Location loc, Value barrier,
                               ConversionPatternRewriter &rewriter) {
  auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
      loc, barrier, rewriter.getI64Type(), rewriter);
  TritonLLVMOpBuilder b(loc, rewriter);
  Value address = b.ptrtoint(i32_ty, smemObj.getBase());
  return b.and_(address, b.i32_val(0x00ffffff));
}

static Value
getMBarrierIssuerPredicate(Location loc, ConversionPatternRewriter &rewriter,
                           const TargetInfoBase &targetInfo, Value opPred,
                           ttg::MemDescType barrierTy, bool leaderOnly,
                           uint32_t sourceBroadcastMask) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);
  if (opPred)
    pred = b.and_(pred, opPred);
  if (leaderOnly) {
    if (auto leaderPred =
            LLVM::NVIDIA::getLeaderCTAPredicate(loc, rewriter, barrierTy))
      pred = b.and_(pred, *leaderPred);
  }
  if (sourceBroadcastMask != 0) {
    Value ctaRank = targetInfo.getClusterCTAId(rewriter, loc);
    Value rankInGroup = b.and_(ctaRank, b.i32_val(sourceBroadcastMask));
    pred = b.and_(pred, b.icmp_eq(rankInGroup, b.i32_val(0)));
  }
  return pred;
}

struct GSanMBarrierTableInitOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanMBarrierTableInitOp> {
  const TargetInfoBase &targetInfo;

  GSanMBarrierTableInitOpConversion(LLVMTypeConverter &typeConverter,
                                    const TargetInfoBase &targetInfo)
      : ConvertOpToLLVMPattern<tti::ExperimentalGSanMBarrierTableInitOp>(
            typeConverter),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanMBarrierTableInitOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    Value elect = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);
    Value ctaRank = targetInfo.getClusterCTAId(rewriter, loc);
    Value pred = b.and_(elect, b.icmp_eq(ctaRank, b.i32_val(0)));
    auto runtimeFunc = getOrCreateGSanRuntimeFunction(
        rewriter, kGSanMBarrierTableInitRuntimeFn);
    Value scratch = castToGenericPointer(rewriter, loc, adaptor.getScratch());
    b.call(runtimeFunc, ValueRange{scratch, b.zext(i32_ty, pred),
                                   b.i32_val(op.getCapacity())});
    rewriter.eraseOp(op);
    return success();
  }
};

struct GSanMBarrierInitOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanMBarrierInitOp> {
  const TargetInfoBase &targetInfo;

  GSanMBarrierInitOpConversion(LLVMTypeConverter &typeConverter,
                               const TargetInfoBase &targetInfo)
      : ConvertOpToLLVMPattern<tti::ExperimentalGSanMBarrierInitOp>(
            typeConverter),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanMBarrierInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto barrierTy = op.getBarrier().getType();
    Value pred = getMBarrierIssuerPredicate(
        loc, rewriter, targetInfo, /*opPred=*/{}, barrierTy,
        /*leaderOnly=*/true, /*sourceBroadcastMask=*/0);
    Value ctaRank = targetInfo.getClusterCTAId(rewriter, loc);
    Value offset = getMBarrierOffset(loc, adaptor.getBarrier(), rewriter);
    Value scratch = castToGenericPointer(rewriter, loc, adaptor.getScratch());
    auto sourceLoc = materializeSourceLocation(rewriter, loc);
    auto runtimeFunc =
        getOrCreateGSanRuntimeFunction(rewriter, kGSanMBarrierInitRuntimeFn);
    b.call(runtimeFunc, ValueRange{scratch, offset, b.zext(i32_ty, pred),
                                   ctaRank, b.i32_val(op.getExpectedCount()),
                                   sourceLoc.file, sourceLoc.line});
    rewriter.eraseOp(op);
    return success();
  }
};

struct GSanMBarrierArriveOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanMBarrierArriveOp> {
  const TargetInfoBase &targetInfo;

  GSanMBarrierArriveOpConversion(LLVMTypeConverter &typeConverter,
                                 const TargetInfoBase &targetInfo)
      : ConvertOpToLLVMPattern<tti::ExperimentalGSanMBarrierArriveOp>(
            typeConverter),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanMBarrierArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr))
      return failure();

    TritonLLVMOpBuilder b(loc, rewriter);
    Value ctaRank = targetInfo.getClusterCTAId(rewriter, loc);
    Value pred = getMBarrierIssuerPredicate(
        loc, rewriter, targetInfo, adaptor.getPred(), op.getBarrier().getType(),
        /*leaderOnly=*/false, op.getSourceBroadcastMask());
    Value recipientMask;
    if (!op.getMulticast()) {
      uint32_t broadcastMask =
          LLVM::NVIDIA::getCGABroadcastMask(op.getBarrier().getType());
      Value leaderRank =
          b.and_(ctaRank, b.i32_val(static_cast<uint32_t>(~broadcastMask)));
      recipientMask = b.shl(b.i32_val(1), leaderRank);
    } else {
      for (int32_t mask : op.getMulticastMasks()) {
        Value nextMask = LLVM::NVIDIA::createTMAMulticastMask(
            loc, rewriter, static_cast<uint16_t>(mask), ctaRank);
        recipientMask =
            recipientMask ? b.or_(recipientMask, nextMask) : nextMask;
      }
      assert(recipientMask && "multicast mbarrier arrival needs recipients");
    }

    Value offset = getMBarrierOffset(loc, adaptor.getBarrier(), rewriter);
    Value scratch = castToGenericPointer(rewriter, loc, adaptor.getScratch());
    auto sourceLoc = materializeSourceLocation(rewriter, loc);
    auto runtimeFunc =
        getOrCreateGSanRuntimeFunction(rewriter, kGSanMBarrierArriveRuntimeFn);
    b.call(runtimeFunc, ValueRange{*gsanGlobalStatePtr, scratch, offset,
                                   b.zext(i32_ty, pred), recipientMask,
                                   b.i32_val(op.getCount()), ctaRank,
                                   b.i32_val(op.getPublishClock()),
                                   sourceLoc.file, sourceLoc.line});
    rewriter.eraseOp(op);
    return success();
  }
};

struct GSanMBarrierWaitOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanMBarrierWaitOp> {
  const TargetInfoBase &targetInfo;

  GSanMBarrierWaitOpConversion(LLVMTypeConverter &typeConverter,
                               const TargetInfoBase &targetInfo)
      : ConvertOpToLLVMPattern<tti::ExperimentalGSanMBarrierWaitOp>(
            typeConverter),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanMBarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto gsanGlobalStatePtr = getGSanGlobalStateArg(op, rewriter, loc);
    if (failed(gsanGlobalStatePtr))
      return failure();

    TritonLLVMOpBuilder b(loc, rewriter);
    auto barrierTy = op.getBarrier().getType();
    Value pred = getMBarrierIssuerPredicate(
        loc, rewriter, targetInfo, adaptor.getPred(), barrierTy,
        /*leaderOnly=*/true, /*sourceBroadcastMask=*/0);
    Value ctaRank = targetInfo.getClusterCTAId(rewriter, loc);
    Value offset = getMBarrierOffset(loc, adaptor.getBarrier(), rewriter);
    Value scratch = castToGenericPointer(rewriter, loc, adaptor.getScratch());
    auto sourceLoc = materializeSourceLocation(rewriter, loc);
    auto runtimeFunc =
        getOrCreateGSanRuntimeFunction(rewriter, kGSanMBarrierWaitRuntimeFn);
    b.call(runtimeFunc,
           ValueRange{*gsanGlobalStatePtr, scratch, offset,
                      b.zext(i32_ty, pred), ctaRank, adaptor.getPhase(),
                      sourceLoc.file, sourceLoc.line});
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
  patterns.add<
      GSanStreamClockOpConversion<tti::ExperimentalGSanKernelExitOp>,
      GSanStreamClockOpConversion<tti::ExperimentalGSanGridDependencyWaitOp>>(
      typeConverter);
  patterns.add<GSanClusterBarrierInitOpConversion>(typeConverter, targetInfo);
  patterns.add<GSanClusterBarrierSyncOpConversion>(typeConverter, targetInfo);
  patterns.add<GSanMBarrierTableInitOpConversion>(typeConverter, targetInfo);
  patterns.add<GSanMBarrierInitOpConversion>(typeConverter, targetInfo);
  patterns.add<GSanMBarrierArriveOpConversion>(typeConverter, targetInfo);
  patterns.add<GSanMBarrierWaitOpConversion>(typeConverter, targetInfo);
  patterns.add<
      GSanTensorDescAccessOpConversion<tti::ExperimentalGSanTensorDescAccessOp>,
      GSanTensorDescAccessOpConversion<
          tti::ExperimentalGSanAtomicTensorDescAccessOp>>(typeConverter,
                                                          targetInfo);
  patterns.add<GSanIndexedTensorDescAccessOpConversion>(typeConverter,
                                                        targetInfo);
  patterns.add<GSanAtomicPollOpConversion>(typeConverter, targetInfo);
  patterns.add<GSanAtomicCASOpConversion>(typeConverter, targetInfo);
  patterns.add<GSanAtomicRMWOpConversion>(typeConverter, targetInfo);
  patterns.add<GSanTensorAccessOpConversion>(typeConverter, axisInfoAnalysis,
                                             targetInfo);
}
