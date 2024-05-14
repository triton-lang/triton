#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonCPUToLLVM/CPUTargetInfo.h"
#include "triton/Conversion/TritonCPUToLLVM/PatternTritonCPUOpToLLVM.h"
#include "triton/Conversion/TritonCPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

struct PrintOpConversion : public ConvertOpToLLVMPattern<triton::PrintOp> {
  explicit PrintOpConversion(LLVMTypeConverter &typeConverter,
                             const CPUTargetInfo &targetInfo,
                             PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<triton::PrintOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    auto getPid = [&](int axis) {
      return targetInfo.programId(
          rewriter, loc, op->getParentOfType<LLVM::LLVMFuncOp>(), axis);
    };
    SmallVector<Value> values = {getPid(0), getPid(1), getPid(2)};

    std::string formatStr;
    llvm::raw_string_ostream os(formatStr);
    os << "pid (" << getFormatSubstr(values[0]) << ", "
       << getFormatSubstr(values[1]) << ", " << getFormatSubstr(values[2])
       << ")" << op.getPrefix();

    for (size_t i = 0; i < op.getNumOperands(); i++) {
      auto elems = unpackLLElements(loc, adaptor.getOperands()[i], rewriter);
      if (op.getOperand(i).getType().dyn_cast<RankedTensorType>()) {
        llvm_unreachable("Not implemented for tensor types");
      }

      // Only support scalars for now.
      assert(elems.size() == 1);
      if (i != 0) {
        os << ", ";
      }
      os << getFormatSubstr(elems[0]);
      values.push_back(elems[0]);
    }

    llPrintf(formatStr, values, rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  // TODO: This code is the same as the GPU-backend code. Consider refactoring.
  std::string getFormatSubstr(Value value, bool hex = false,
                              std::optional<int> width = std::nullopt) const {
    Type type = value.getType();
    if (type.isa<LLVM::LLVMPointerType>()) {
      return "%p";
    }
    // Hex is "0x%0nx" or "0x%0nllx", where n is the number of hex digits in the
    // type (so 4 for fp16, 8 for int32, 16 for int64).
    if (hex) {
      // Ignore `width` for `hex` values, pad to typeWidth.
      std::string ret =
          "0x%0" + std::to_string(type.getIntOrFloatBitWidth() / 4);
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

  Value llPrintf(StringRef msg, ValueRange args,
                 ConversionPatternRewriter &rewriter,
                 int *formatStrByteCount = nullptr) const {
    assert(!msg.empty() && "printf with empty string not supported");
    llvm::SmallString<64> msgNewline(msg);
    msgNewline.push_back('\n');
    msgNewline.push_back('\0');
    Value msgValue =
        LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()),
                                rewriter, "printfFormat_", msgNewline);
    targetInfo.printf(rewriter, msgValue, msgNewline.size_in_bytes(), args);
    if (formatStrByteCount)
      *formatStrByteCount = msgNewline.size_in_bytes();
    return msgValue;
  }

protected:
  const CPUTargetInfo &targetInfo;
};

} // namespace

void mlir::triton::cpu::populatePrintOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const CPUTargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<PrintOpConversion>(typeConverter, targetInfo, benefit);
}
