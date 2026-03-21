#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
  } else {
    argTys = {ptr_ty(ctx), ptr_ty(ctx), i32_ty, i32_ty, ptr_ty(ctx), i32_ty};
  }
  auto funcTy = LLVM::LLVMFunctionType::get(void_ty(ctx), argTys);
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  return LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(ctx), funcName,
                                  funcTy);
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
  auto sourceLoc = materializeSourceLocation(rewriter, loc);
  b.call(runtimeFunc,
         ValueRange{gsanGlobalStatePtr, argsPtr, b.i32_val(ptrElems.size()),
                    b.i32_val(bytesPerElem), sourceLoc.file, sourceLoc.line});
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

  GSanTensorAccessOpConversion(LLVMTypeConverter &typeConverter,
                               const TargetInfoBase &targetInfo,
                               PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        targetInfo(&targetInfo) {}

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

    auto freeVarMasks = getFreeVariableMasks(ptrTy);
    auto *ctx = getContext();
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));
    Value threadPred = ttg::emitRedundantThreadPredicate(freeVarMasks, rewriter,
                                                         loc, *targetInfo);
    emitTensorAccessRuntimeCall(rewriter, loc, gsanGlobalStatePtr, ptrElems,
                                maskElems, regMask, threadPred, bytesPerElem,
                                op.getIsStore());

    rewriter.eraseOp(op);
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
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateGSanToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo) {
  patterns.add<GSanInitOpConversion>(typeConverter);
  patterns.add<GSanTensorDescInfoOpConversion>(typeConverter);
  patterns.add<GSanTensorAccessOpConversion>(typeConverter, targetInfo);
}
