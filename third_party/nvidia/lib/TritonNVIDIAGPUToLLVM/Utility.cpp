#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
// #include "mlir/Dialect/LLVMIR/NVVMDialect.h"
// #include "triton/Dialect/NVGPU/IR/Dialect.h"
namespace mlir {

namespace LLVM {

namespace NVIDIA {
using namespace mlir::triton;

Value storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                  Value val, Value pred) {
  MLIRContext *ctx = rewriter.getContext();
  unsigned bits = std::max(8u, val.getType().getIntOrFloatBitWidth());
  const char *c = bits == 64 ? "l" : (bits == 16 ? "h" : "r");

  PTXBuilder builder;
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto *valOpr = builder.newOperand(val, c);
  auto &st = builder.create<>("st")->shared().b(bits);
  st(ptrOpr, valOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, void_ty(ctx));
}

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Type elemTy, Value pred) {
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = ptr.getType().cast<LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for loadShared");
  unsigned bitwidth = std::max(8u, elemTy.getIntOrFloatBitWidth());

  const char *c = bitwidth == 64 ? "=l" : (bitwidth == 16 ? "=h" : "=r");

  PTXBuilder builder;
  auto *dOpr = builder.newOperand(c);
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto &ld = builder.create<>("ld")->shared().b(bitwidth);
  ld(dOpr, ptrOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, elemTy);
}

Value getSRegValue(OpBuilder &b, Location loc, const std::string &sRegStr) {
  PTXBuilder builder;
  auto &mov = builder.create("mov")->o("u32");
  auto *destOpr = builder.newOperand("=r");
  auto *sRegOpr = builder.newConstantOperand(sRegStr);
  mov(destOpr, sRegOpr);
  Value val = builder.launch(b, loc, b.getIntegerType(32), false);
  return val;
}
} // namespace NVIDIA
} // namespace LLVM
} // namespace mlir
