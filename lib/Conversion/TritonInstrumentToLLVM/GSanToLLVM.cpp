#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include <limits>

namespace tt = mlir::triton;
namespace tti = mlir::triton::instrument;
namespace ttg = mlir::triton::gpu;

namespace {

static constexpr StringLiteral kGSanLoadTensorRuntimeFn =
    "__triton_gsan_load_tensor";
static constexpr StringLiteral kGSanStoreTensorRuntimeFn =
    "__triton_gsan_store_tensor";
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
    argTys = {ptr_ty(ctx)};
  } else {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i32_ty, i32_ty};
  }
  auto funcTy = LLVM::LLVMFunctionType::get(void_ty(ctx), argTys);
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  return LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(ctx), funcName,
                                  funcTy);
}

////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////

Value maybeAnd(RewriterBase &rewriter, Location loc, Value a, Value b) {
  TritonLLVMOpBuilder tb(loc, rewriter);
  if (a && b)
    return tb.and_(a, b);
  return a ? a : b;
}

void emitTensorAccessRuntimeCall(ConversionPatternRewriter &rewriter,
                                 Location loc, Value gsanGlobalStatePtr,
                                 ArrayRef<Value> ptrElems,
                                 ArrayRef<Value> maskElems, uint32_t regMask,
                                 Value threadPred, int32_t bytesPerElem,
                                 bool isStore) {
  if (ptrElems.empty())
    return;

  auto *ctx = rewriter.getContext();
  TritonLLVMOpBuilder b(loc, rewriter);
  Value one = b.i32_val(1);
  Value zero = b.i32_val(0);
  Type i8Ty = rewriter.getI8Type();
  Type i64Ty = rewriter.getI64Type();

  auto ptrArrayTy = array_ty(i64Ty, ptrElems.size());
  auto maskArrayTy = array_ty(i8Ty, ptrElems.size());
  SmallVector<Type> argsFieldTys = {ptrArrayTy, maskArrayTy};
  auto argsTy = LLVM::LLVMStructType::getLiteral(ctx, argsFieldTys);
  auto argsBuffer = LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx), argsTy,
                                           one, /*alignment=*/0);

  for (unsigned i = 0; i < ptrElems.size(); ++i) {
    Value idx = b.i32_val(i);
    Value ptrValue = b.ptrtoint(i64_ty, ptrElems[i]);
    Value ptrSlot =
        b.gep(ptr_ty(ctx), argsTy, argsBuffer, ValueRange{zero, zero, idx});
    b.store(ptrValue, ptrSlot);

    Value maskValue = maskElems.empty() ? b.true_val() : maskElems[i];
    if (!isCanonicalIndex(i, regMask))
      maskValue = b.false_val();
    maskValue = maybeAnd(rewriter, loc, maskValue, threadPred);
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
  b.call(runtimeFunc,
         ValueRange{gsanGlobalStatePtr, argsPtr, b.i32_val(ptrElems.size()),
                    b.i32_val(bytesPerElem)});
}

Value getGSanGlobalStateArg(FunctionOpInterface funcOp) {
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    if (funcOp.getArgAttr(i, kGSanGlobalStateArgAttr))
      return funcOp.getArgument(i);
  }
  return {};
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

  GSanTensorAccessOpConversion(LLVMTypeConverter &typeConverter,
                               const TargetInfoBase &targetInfo,
                               PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(&targetInfo) {
  }

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanTensorAccessOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    uint32_t regMask =
        freeVarMasks.lookup(StringAttr::get(op.getContext(), "reg"));
    Value threadPred =
        ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, *targetInfo);
    emitTensorAccessRuntimeCall(rewriter, loc, gsanGlobalStatePtr, ptrElems,
                                maskElems, regMask, threadPred,
                                bytesPerElem,
                                op.getIsStore());

    rewriter.eraseOp(op);
    return success();
  }
};

struct GSanInitOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanInitOp> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalGSanInitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanInitOp op, [[maybe_unused]] OpAdaptor adaptor,
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
    b.call(runtimeFunc, ValueRange{gsanGlobalStatePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void mlir::triton::populateGSanToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo) {
  patterns.add<GSanInitOpConversion>(typeConverter);
  patterns.add<GSanTensorAccessOpConversion>(typeConverter, targetInfo);
}
