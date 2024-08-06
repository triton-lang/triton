#include "TypeConverter.h"
#include "Utility.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUOps.h.inc"

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

// The code for the print is similar to the GPU's TargetInfo.cpp.
LLVM::LLVMFuncOp getPrintfDeclaration(ConversionPatternRewriter &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("printf");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();

  // int printf(char* format, ...)
  SmallVector<Type> argsType{ptr_ty(context)};
  auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType, true);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto op = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context),
                                              funcName, funcType);
  return op;
}

void emitPrintf(ConversionPatternRewriter &rewriter, Value formatStrStart,
                int /*formatStrByteCount*/, ValueRange args) {
  auto loc = UnknownLoc::get(rewriter.getContext());
  SmallVector<Value> formatStrAndArgs{formatStrStart};
  for (auto arg : args) {
    formatStrAndArgs.push_back(arg);
  }
  call(getPrintfDeclaration(rewriter), formatStrAndArgs);
}

Value llPrintf(StringRef msg, ValueRange args,
               ConversionPatternRewriter &rewriter,
               int *formatStrByteCount = nullptr) {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  emitPrintf(rewriter, msgValue, msgNewline.size_in_bytes(), args);
  if (formatStrByteCount)
    *formatStrByteCount = msgNewline.size_in_bytes();
  return msgValue;
}

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

// TritonCPU's device_print prints all values in the same line unlike GPUs
// and interpreter where each value is printed in a separate line.
struct PrintOpConversion : public ConvertOpToLLVMPattern<triton::PrintOp> {
  explicit PrintOpConversion(LLVMTypeConverter &typeConverter)
      : mlir::ConvertOpToLLVMPattern<triton::PrintOp>(typeConverter) {}

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    auto getPid = [&](int axis) {
      return getProgramId(op->getParentOfType<LLVM::LLVMFuncOp>(), axis);
    };
    SmallVector<Value> values = {getPid(0), getPid(1), getPid(2)};

    std::string formatStr;
    llvm::raw_string_ostream os(formatStr);
    os << "(" << getFormatSubstr(values[0]) << ", "
       << getFormatSubstr(values[1]) << ", " << getFormatSubstr(values[2])
       << ")" << op.getPrefix();

    for (size_t i = 0; i < op.getNumOperands(); i++) {
      auto elems = unpackLLElements(loc, adaptor.getOperands()[i], rewriter);
      if (dyn_cast<RankedTensorType>(op.getOperand(i).getType())) {
        llvm_unreachable("Not implemented for tensor types");
      }

      // Only support scalars for now.
      assert(elems.size() == 1);
      if (i != 0) {
        os << ", ";
      }
      os << getFormatSubstr(elems[0], op.getHex());
      values.push_back(elems[0]);
    }

    llPrintf(formatStr, values, rewriter);
    rewriter.eraseOp(op);
    return success();
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
    patterns.add<BarrierOpConversion>(typeConverter);
    // patterns.add<AssertOpConversion>(typeConverter);

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
