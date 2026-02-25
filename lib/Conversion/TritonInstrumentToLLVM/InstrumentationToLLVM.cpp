#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <limits>

namespace {

namespace tt = mlir::triton;
namespace ttg = tt::gpu;
namespace tti = mlir::triton::instrument;
namespace ttng = mlir::triton::nvidia_gpu;

////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////

Value maybeAnd(RewriterBase &rewriter, Location loc, Value a, Value b) {
  TritonLLVMOpBuilder tb(loc, rewriter);
  if (a && b)
    return tb.and_(a, b);
  return a ? a : b;
}

Value createMemDescToI32(RewriterBase &rewriter, Location loc,
                         const LLVMTypeConverter *typeConverter,
                         ttg::MemDescType memDescTy, Value sharedMemStruct) {
  TritonLLVMOpBuilder b(loc, rewriter);
  auto i32Ty = rewriter.getIntegerType(32);
  if (isa<ttng::TensorMemorySpaceAttr>(memDescTy.getMemorySpace())) {
    return b.ptrtoint(i32Ty, sharedMemStruct);
  }
  assert(isa<ttg::SharedEncodingTrait>(memDescTy.getEncoding()) &&
         "Unsupported memory encoding");
  Type srcElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, sharedMemStruct,
                                                       srcElemTy, rewriter);
  auto offset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
  auto elemSize = srcElemTy.getIntOrFloatBitWidth() / 8;
  offset = b.mul(offset, b.i32_val(elemSize));
  return b.add(offset, b.ptrtoint(i32Ty, smemObj.getBase()));
}

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

Value castToI1Predicate(Value value, ConversionPatternRewriter &rewriter,
                        Location loc) {
  TritonLLVMOpBuilder b(loc, rewriter);
  if (value.getType().isInteger(1))
    return value;
  if (auto intTy = dyn_cast<IntegerType>(value.getType()))
    return b.icmp_ne(value, b.int_val(intTy.getWidth(), 0));
  if (isa<LLVM::LLVMPointerType>(value.getType())) {
    Value asInt = b.ptrtoint(i64_ty, value);
    return b.icmp_ne(asInt, b.i64_val(0));
  }
  llvm_unreachable("unsupported mask element type for GSan instrumentation");
}

Value castPointerElementToI64(Value value, ConversionPatternRewriter &rewriter,
                              Location loc) {
  TritonLLVMOpBuilder b(loc, rewriter);
  if (isa<LLVM::LLVMPointerType>(value.getType()))
    return b.ptrtoint(i64_ty, value);
  if (value.getType().isInteger(64))
    return value;
  if (auto intTy = dyn_cast<IntegerType>(value.getType())) {
    if (intTy.getWidth() < 64)
      return b.zext(i64_ty, value);
    if (intTy.getWidth() > 64)
      return b.trunc(i64_ty, value);
    return value;
  }
  llvm_unreachable("unsupported pointer element type for GSan instrumentation");
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
    Value ptrValue = castPointerElementToI64(ptrElems[i], rewriter, loc);
    Value ptrSlot =
        b.gep(ptr_ty(ctx), argsTy, argsBuffer, ValueRange{zero, zero, idx});
    b.store(ptrValue, ptrSlot);

    Value maskValue = maskElems.empty() ? b.true_val() : maskElems[i];
    if (!isCanonicalIndex(i, regMask))
      maskValue = b.false_val();
    maskValue = castToI1Predicate(maskValue, rewriter, loc);
    if (threadPred)
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

Value getGSanGlobalStateArgOrNull(ConversionPatternRewriter &rewriter,
                                  Location loc, FunctionOpInterface funcOp) {
  if (Value arg = funcOp ? getGSanGlobalStateArg(funcOp) : Value{})
    return arg;

  TritonLLVMOpBuilder b(loc, rewriter);
  return b.inttoptr(ptr_ty(rewriter.getContext()), b.i64_val(0));
}

Value castGSanGlobalStateArgToGenericPtr(ConversionPatternRewriter &rewriter,
                                         Location loc, Value ptr) {
  auto *ctx = rewriter.getContext();
  TritonLLVMOpBuilder b(loc, rewriter);
  if (ptr.getType() != ptr_ty(ctx))
    return b.addrspacecast(ptr_ty(ctx), ptr);
  return ptr;
}

////////////////////////////////////////////
// Patterns
////////////////////////////////////////////

struct AssertUniformOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalAssertUniformOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(tti::ExperimentalAssertUniformOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TritonLLVMIRRewriter b(op.getLoc(), rewriter);
    Value tid = getThreadId(b, op.getLoc());
    Value threadIdIsZero = b.icmp_eq(tid, b.i32_val(0));

    auto [prevBlock, ifBlock, thenBlock] =
        createIfBlock(rewriter, op.getLoc(), threadIdIsZero);
    rewriter.setInsertionPointToStart(ifBlock);
    AssertOp::create(rewriter, op.getLoc(), adaptor.getCondition(),
                     adaptor.getMessage());
    rewriter.eraseOp(op);
    rewriter.setInsertionPointToStart(thenBlock);
    return success();
  }
};

struct BufferDescriptorsOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalBufferDescriptorsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalBufferDescriptorsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto encoding = cast<ttg::DistributedEncodingTrait>(
        op.getResult().getType().getEncoding());
    auto offsets = adaptor.getOffsets();
    auto lengths = adaptor.getLengths();
    assert(offsets.size() == lengths.size() && "Mismatched descriptor arrays");

    auto tensorType = cast<RankedTensorType>(op.getResult().getType());

    SmallVector<uint64_t> offsetVals;
    offsetVals.reserve(offsets.size());
    for (int32_t offset : offsets)
      offsetVals.push_back(static_cast<uint32_t>(offset));
    Value pointerTensor =
        createInitializedIntArrayTensor(rewriter, loc, encoding, offsetVals);

    TritonLLVMOpBuilder b(loc, rewriter);
    auto i64Ty = rewriter.getIntegerType(64);
    Value baseTensor = nullptr;
    if (op.getMemType() == tti::MemType::SHARED_MEM) {
      auto func = op->getParentOfType<FunctionOpInterface>();
      Value base = getSharedMemoryBase(rewriter, func);
      baseTensor = triton::SplatOp::create(rewriter, loc, tensorType, base);
    } else {
      assert(op.getMemType() == tti::MemType::TENSOR_MEM &&
             "Unsupported memory type");
      Value basePtr = nvgpu::TensorMemoryBaseAddress::create(rewriter, loc);
      Value base = b.ptrtoint(i64Ty, basePtr);
      baseTensor = triton::SplatOp::create(rewriter, loc, tensorType, base);
    }

    pointerTensor = arith::AddIOp::create(
        rewriter, loc, pointerTensor.getType(), pointerTensor, baseTensor);

    SmallVector<uint64_t> maskVals(offsets.size(), 0xffffffffu);
    Value maskTensor =
        createInitializedIntArrayTensor(rewriter, loc, encoding, maskVals);
    Value trimmedPointers = arith::AndIOp::create(
        rewriter, loc, pointerTensor.getType(), pointerTensor, maskTensor);

    SmallVector<uint64_t> lengthVals;
    lengthVals.reserve(lengths.size());
    for (int32_t length : lengths)
      lengthVals.push_back(static_cast<uint64_t>(static_cast<uint32_t>(length))
                           << 32);
    Value lengthTensor =
        createInitializedIntArrayTensor(rewriter, loc, encoding, lengthVals);

    auto bufDescriptors =
        arith::OrIOp::create(rewriter, loc, trimmedPointers.getType(),
                             trimmedPointers, lengthTensor);
    rewriter.replaceOp(op, bufDescriptors);
    return success();
  }

  Value createInitializedIntArrayTensor(OpBuilder &builder, Location loc,
                                        ttg::DistributedEncodingTrait encoding,
                                        ArrayRef<uint64_t> values) const {
    int64_t size = values.size();
    assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
    auto tensorType =
        RankedTensorType::get({size}, builder.getIntegerType(64), encoding);
    SmallVector<APInt> apInts = llvm::to_vector(
        llvm::map_range(values, [](uint64_t v) { return APInt(64, v); }));
    auto denseAttr = DenseElementsAttr::get(tensorType, apInts);
    return arith::ConstantOp::create(builder, loc, tensorType, denseAttr);
  }

  Value getSharedMemoryBase(ConversionPatternRewriter &rewriter,
                            FunctionOpInterface func) const {
    Location loc = func.getLoc();
    Value basePtr = LLVM::getStackPointer(rewriter, func);
    auto i64Ty = rewriter.getIntegerType(64);
    TritonLLVMOpBuilder b(loc, rewriter);
    return b.ptrtoint(i64Ty, basePtr);
  }
};

struct LockAcquireOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalLockAcquireOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(tti::ExperimentalLockAcquireOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    Value lock = op.getLock();

    Type elType = cast<PointerType>(lock.getType()).getPointeeType();
    assert(elType == b.getI32Type() && "Expected i32 lock element type");

    // Build: do { old = atom.global.acquire.cas.b32 [lock], 0, 1; } while (old
    // != 0);
    Block *prevBlock2 = b.getInsertionBlock();
    Block *whileBlock = b.splitBlock(prevBlock2, b.getInsertionPoint());
    Block *endBlock = b.splitBlock(whileBlock, whileBlock->begin());
    b.setInsertionPointToEnd(prevBlock2);
    Value elect = mlir::LLVM::NVIDIA::createElectPredicateWarp0(loc, b);
    if (op.getPred()) {
      elect = arith::AndIOp::create(b, loc, elect, op.getPred());
    }
    LLVM::CondBrOp::create(b, loc, elect, whileBlock, endBlock);

    b.setInsertionPointToEnd(whileBlock);

    auto i32 = b.getI32Type();
    Value zero =
        arith::ConstantOp::create(b, loc, i32, b.getIntegerAttr(i32, 0));
    Value one =
        arith::ConstantOp::create(b, loc, i32, b.getIntegerAttr(i32, 1));

    // Inline PTX CAS: old = atom.global.acquire.gpu.cas.b32 [lock], 0, 1
    // Use converted lock pointer from adaptor for addressing
    PTXBuilder ptx;
    auto *dstOpr = ptx.newOperand("=r", /*init=*/true);
    auto *ptrOpr = ptx.newAddrOperand(adaptor.getLock(), "l");
    auto *cmpOpr = ptx.newOperand(zero, "r");
    auto *valOpr = ptx.newOperand(one, "r");
    auto &atom = *ptx.create("atom");
    atom.global().o("acquire").o("gpu").o("cas").o("b32");
    atom(dstOpr, ptrOpr, cmpOpr, valOpr);
    Value old = ptx.launch(b, loc, i32);

    // while (old != 0) loop
    Value cond =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, old, zero);
    LLVM::CondBrOp::create(b, loc, cond, whileBlock, endBlock);

    b.setInsertionPointToStart(endBlock);
    triton::gpu::BarrierOp::create(b, loc,
                                   triton::gpu::AddrSpace::GlobalRead |
                                       triton::gpu::AddrSpace::GlobalWrite);
    b.eraseOp(op);
    return success();
  }
};

struct LockReleaseOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalLockReleaseOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(tti::ExperimentalLockReleaseOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    Value lock = op.getLock();
    if (op.getPred()) {
      auto [prevBlock, ifBlock, thenBlock] =
          createIfBlock(b, loc, op.getPred());
      b.setInsertionPointToStart(ifBlock);
    }

    Type elType = cast<PointerType>(lock.getType()).getPointeeType();
    assert(elType == b.getI32Type() && "Expected i32 lock element type");

    triton::gpu::BarrierOp::create(b, loc,
                                   triton::gpu::AddrSpace::GlobalRead |
                                       triton::gpu::AddrSpace::GlobalWrite);
    Value zero =
        arith::ConstantOp::create(b, loc, elType, b.getIntegerAttr(elType, 0));
    triton::AtomicRMWOp::create(b, loc, elType, RMWOp::XCHG, lock, zero,
                                nullptr, MemSemantic::ACQUIRE_RELEASE,
                                MemSyncScope::GPU);
    b.eraseOp(op);
    return success();
  }
};

struct MemDescToI32OpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalMemDescToI32Op> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalMemDescToI32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalMemDescToI32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value converted =
        createMemDescToI32(rewriter, loc, getTypeConverter(),
                           op.getMemdesc().getType(), adaptor.getMemdesc());
    rewriter.replaceOp(op, converted);
    return success();
  }
};

struct GSanTensorAccessOpConversion
    : public ConvertOpToLLVMPattern<tti::ExperimentalGSanTensorAccessOp> {
public:
  using ConvertOpToLLVMPattern<
      tti::ExperimentalGSanTensorAccessOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tti::ExperimentalGSanTensorAccessOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto func = op->getParentOfType<FunctionOpInterface>();
    Value gsanGlobalStatePtr = getGSanGlobalStateArgOrNull(rewriter, loc, func);

    Value llPtr = adaptor.getPtr();
    unsigned numElems = ttg::getTotalElemsPerThread(op.getPtr().getType());
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
    Value threadPred;
    if (op.getIsStore()) {
      TritonLLVMOpBuilder b(loc, rewriter);
      auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
      Value laneIsZero = b.icmp_eq(laneId, b.i32_val(0));
      Value warpIsZero = b.icmp_eq(warpId, b.i32_val(0));
      threadPred = b.and_(laneIsZero, warpIsZero);
    }
    emitTensorAccessRuntimeCall(rewriter, loc, gsanGlobalStatePtr, ptrElems,
                                maskElems, regMask, threadPred,
                                static_cast<int32_t>(op.getBytesPerElem()),
                                static_cast<bool>(op.getIsStore()));

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
  matchAndRewrite(tti::ExperimentalGSanInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto func = op->getParentOfType<FunctionOpInterface>();
    Value gsanGlobalStatePtr =
        getGSanGlobalStateArgOrNull(rewriter, op.getLoc(), func);

    auto runtimeFunc =
        getOrCreateGSanRuntimeFunction(rewriter, kGSanInitRuntimeFn);
    auto loc = op.getLoc();
    gsanGlobalStatePtr =
        castGSanGlobalStateArgToGenericPtr(rewriter, loc, gsanGlobalStatePtr);

    TritonLLVMOpBuilder b(loc, rewriter);
    b.call(runtimeFunc, ValueRange{gsanGlobalStatePtr});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateInstrumentationToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<AssertUniformOpConversion>(typeConverter);
  patterns.add<BufferDescriptorsOpConversion>(typeConverter);
  patterns.add<LockAcquireOpConversion>(typeConverter);
  patterns.add<LockReleaseOpConversion>(typeConverter);
  patterns.add<MemDescToI32OpConversion>(typeConverter);
  patterns.add<GSanInitOpConversion>(typeConverter);
  patterns.add<GSanTensorAccessOpConversion>(typeConverter);
}
