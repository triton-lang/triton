#include "TypeConverter.h"
#include "Utility.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUOps.h.inc"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DEBUGOPSTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

// TODO: This code is the same as the GPU-backend code. Consider refactoring.
std::string getFormatSubstr(Value value, bool hex = false,
                            std::optional<int> width = std::nullopt) {
  Type type = value.getType();
  if (isa<LLVM::LLVMPointerType>(type)) {
    return "%p";
  }
  // Hex is "0x%0nx" or "0x%0nllx", where n is the number of hex digits in the
  // type (so 4 for fp16, 8 for int32, 16 for int64).
  if (hex) {
    // Ignore `width` for `hex` values, pad to typeWidth.
    std::string ret = "0x%0" + std::to_string(type.getIntOrFloatBitWidth() / 4);
    if (type.getIntOrFloatBitWidth() > 32) {
      ret += "ll";
    }
    ret += "x";
    return ret;
  }

  std::string prefix = "%";
  if (width.has_value()) {
    prefix += std::to_string(*width);
  } else if (hex) {
    prefix += "0";
    prefix += std::to_string(value.getType().getIntOrFloatBitWidth() / 4);
  }

  if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
    return prefix + "f";
  } else if (type.isSignedInteger()) {
    if (type.getIntOrFloatBitWidth() == 64)
      return prefix + "lli";
    else
      return prefix + "i";
  } else if (type.isUnsignedInteger() || type.isSignlessInteger()) {
    if (type.getIntOrFloatBitWidth() == 64)
      return prefix + "llu";
    else
      return prefix + "u";
  }
  assert(false && "not supported type");
  return "";
}

// For printf, need to extend int32 or float64.
Value printfPromoteValue(RewriterBase &rewriter, Value value) {
  auto *context = rewriter.getContext();
  auto type = value.getType();
  auto loc = UnknownLoc::get(context);

  bool isUnsigned = type.isUnsignedInteger();
  if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
    if (isUnsigned) {
      return zext(ui32_ty, value);
    } else {
      return sext(i32_ty, value);
    }
  } else if (type.isBF16() || type.isF16() || type.isF32()) {
    return fpext(f64_ty, value);
  }

  return value;
}

LLVM::LLVMFuncOp getPrintFuncDecl(ConversionPatternRewriter &rewriter,
                                  bool printf) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName = printf ? "printf" : "triton_vector_print";
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *ctx = rewriter.getContext();
  SmallVector<Type> argsType;
  if (printf)
    argsType = {ptr_ty(ctx)};
  else
    argsType = {i32_ty, i32_ty, i32_ty, ptr_ty(ctx), ptr_ty(ctx),
                i32_ty, i32_ty, i32_ty, i64_ty,      i32_ty};

  auto funcType =
      LLVM::LLVMFunctionType::get(i32_ty, argsType, /*isVarArg*/ printf);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                           funcType);
}

static StringRef makeNullTerminatedString(StringRef s) {
  llvm::SmallString<64> ss(s);
  ss.push_back(0);
  return ss;
}

void llPrintf(StringRef prefix, std::array<Value, 3> pid,
              std::optional<Value> arg, ConversionPatternRewriter &rewriter,
              bool hex = false) {
  assert(!prefix.empty() && "printf with empty string not supported");
  auto loc = UnknownLoc::get(rewriter.getContext());

  std::string formatStr;
  llvm::raw_string_ostream os(formatStr);
  os << "(" << getFormatSubstr(pid[0]) << ", " << getFormatSubstr(pid[1])
     << ", " << getFormatSubstr(pid[2]) << ")" << prefix;
  if (arg.has_value())
    os << getFormatSubstr(arg.value(), hex);

  llvm::SmallString<64> formatStrNewline(formatStr);
  formatStrNewline.push_back('\n');
  formatStrNewline.push_back('\0');
  Value formatStrValue =
      LLVM::addStringToModule(loc, rewriter, "printfFormat_", formatStrNewline);

  SmallVector<Value> allArgs{formatStrValue};
  for (auto elem : pid)
    allArgs.push_back(elem);
  if (arg.has_value())
    allArgs.push_back(printfPromoteValue(rewriter, arg.value()));
  call(getPrintFuncDecl(rewriter, true), allArgs);
}

void llVectorPrint(std::array<Value, 3> pid, StringRef prefix, Value ptr,
                   bool isInteger, bool isSigned, uint32_t bitWidth,
                   int64_t numElem, bool hex,
                   ConversionPatternRewriter &rewriter) {
  assert(!prefix.empty());
  auto loc = UnknownLoc::get(rewriter.getContext());

  Value prefixValue = LLVM::addStringToModule(
      loc, rewriter, "vectorPrintPrefix_", makeNullTerminatedString(prefix));

  SmallVector<Value> allArgs;
  for (auto elem : pid)
    allArgs.push_back(elem);
  allArgs.push_back(prefixValue);
  allArgs.push_back(ptr);
  allArgs.push_back(i32_val(isInteger));
  allArgs.push_back(i32_val(isSigned));
  allArgs.push_back(i32_val(bitWidth));
  allArgs.push_back(i64_val(numElem));
  allArgs.push_back(i32_val(hex));
  call(getPrintFuncDecl(rewriter, false), allArgs);
}

bool usePrintf(triton::cpu::PrintOp op) {
  // Simply use printf if no operand or the operand is scalar.
  if (op.getNumOperands() == 0)
    return true;

  // tt.print is already decomposed to triton_cpu.print per value.
  assert(op.getNumOperands() == 1);
  Type oprType = op.getOperands()[0].getType();
  return (oprType.isIntOrIndexOrFloat() || isa<triton::PointerType>(oprType));
}

Value getPid(Operation *op, int axis) {
  return getProgramId(op->getParentOfType<LLVM::LLVMFuncOp>(), axis);
};

struct PrintOpConversion : public ConvertOpToLLVMPattern<triton::cpu::PrintOp> {
  using ConvertOpToLLVMPattern<triton::cpu::PrintOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::cpu::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    std::array<Value, 3> pid = {getPid(op, 0), getPid(op, 1), getPid(op, 2)};

    if (usePrintf(op)) {
      if (op.getNumOperands() == 0) {
        llPrintf(op.getPrefix(), pid, std::nullopt, rewriter);
      } else {
        Value llOpr = adaptor.getOperands()[0];
        llPrintf(op.getPrefix(), pid, llOpr, rewriter, op.getHex());
      }
    } else {
      Value llOpr = adaptor.getOperands()[0];
      auto vecShapedType = cast<ShapedType>(op.getOperands()[0].getType());
      // Currently, we only support 1D vector printing.
      if (vecShapedType.getRank() == 1) {

        // To get the pointer of the vector, create an alloca and store it.
        auto ptrType = ptr_ty(rewriter.getContext());
        auto ptr = rewriter.create<LLVM::AllocaOp>(loc, ptrType,
                                                   llOpr.getType(), i32_val(1));
        rewriter.create<LLVM::StoreOp>(loc, llOpr, ptr);

        // TODO: Consider passing an encoded element type information instead of
        // booleans and separate bit width.
        llVectorPrint(pid, op.getPrefix(), ptr,
                      vecShapedType.getElementType().isInteger(),
                      op.getIsSigned()[0],
                      vecShapedType.getElementTypeBitWidth(),
                      vecShapedType.getNumElements(), op.getHex(), rewriter);
      } else {
        // TODO: support 2D+ vector printing.
        std::string msg{op.getPrefix()};
        llvm::raw_string_ostream os(msg);
        os << "<<not implemented for '" << llOpr.getType() << "'>>";
        llPrintf(msg, pid, std::nullopt, rewriter);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct AssertOpConversion
    : public ConvertOpToLLVMPattern<triton::cpu::AssertOp> {
  using ConvertOpToLLVMPattern<triton::cpu::AssertOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::cpu::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();
    Value message =
        LLVM::addStringToModule(loc, rewriter, "assertMessage_",
                                makeNullTerminatedString(adaptor.getMessage()));

    // Based on lib/Conversion/TritonGPUToLLVM/AssertOpToLLVM.cpp.
    StringRef fileStr = "unknown";
    StringRef funcStr = "unknown";
    int line = 0;
    int col = 0;

    while (auto callLoc = dyn_cast<CallSiteLoc>(loc))
      loc = callLoc.getCallee();

    if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(loc)) {
      fileStr = fileLineColLoc.getFilename();
      line = fileLineColLoc.getLine();
      col = fileLineColLoc.getColumn();
    }

    Value file = LLVM::addStringToModule(loc, rewriter, "assertFile_",
                                         makeNullTerminatedString(fileStr));
    Value func = LLVM::addStringToModule(loc, rewriter, "assertFunc_",
                                         makeNullTerminatedString(funcStr));
    SmallVector<Value> args{getPid(op, 0),     getPid(op, 1), getPid(op, 2),
                            op.getCondition(), message,       file,
                            i32_val(line),     func};
    call(getAssertFuncDecl(rewriter), args);
    rewriter.eraseOp(op);
    return success();
  }

  static LLVM::LLVMFuncOp
  getAssertFuncDecl(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName = "triton_assert";
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto *ctx = rewriter.getContext();
    SmallVector<Type> argsType{i32_ty,      i32_ty,      i32_ty, i1_ty,
                               ptr_ty(ctx), ptr_ty(ctx), i32_ty, ptr_ty(ctx)};

    auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                             funcType);
  }
};

using BarrierOp = mlir::gpu::BarrierOp;

// This is part of the DebugOps pass because gpu::barrier is generated by
// tl.debug_barrier.
struct BarrierOpConversion : public ConvertOpToLLVMPattern<BarrierOp> {
  explicit BarrierOpConversion(LLVMTypeConverter &typeConverter)
      : mlir::ConvertOpToLLVMPattern<BarrierOp>(typeConverter) {}

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Just make it a no-op for now
    rewriter.eraseOp(op);
    return success();
  }
};

struct DebugOpsToLLVM
    : public triton::impl::DebugOpsToLLVMBase<DebugOpsToLLVM> {
  using DebugOpsToLLVMBase::DebugOpsToLLVMBase;

  DebugOpsToLLVM() : DebugOpsToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<PrintOpConversion>(typeConverter);
    patterns.add<AssertOpConversion>(typeConverter);
    patterns.add<BarrierOpConversion>(typeConverter);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // anonymous namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<ModuleOp>> createDebugOpsToLLVMPass() {
  return std::make_unique<DebugOpsToLLVM>();
}

} // namespace mlir::triton::cpu
