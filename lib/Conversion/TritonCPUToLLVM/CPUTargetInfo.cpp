#include "triton/Conversion/TritonCPUToLLVM/CPUTargetInfo.h"
#include "triton/Conversion/TritonCPUToLLVM/Utility.h"

namespace {
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

  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                           funcType);
}
} // namespace

namespace mlir::triton::cpu {

Value CPUTargetInfo::programId(ConversionPatternRewriter &rewriter,
                               Location loc, LLVM::LLVMFuncOp funcOp,
                               int axis) const {
  assert(axis >= 0 && axis < 3);

  // program_id for CPU is provided as function arguments. The last three
  // arguments are __grid0 to __grid2 of i32.
  assert(funcOp && funcOp.getArguments().size() >= 3);
  return funcOp.getArgument(funcOp.getArguments().size() - 3 + axis);
}

void CPUTargetInfo::printf(ConversionPatternRewriter &rewriter,
                           Value formatStrStart, int /*formatStrByteCount*/,
                           ValueRange args) const {
  auto loc = UnknownLoc::get(rewriter.getContext());
  SmallVector<Value> formatStrAndArgs{formatStrStart};
  for (auto arg : args) {
    formatStrAndArgs.push_back(arg);
  }
  call(getPrintfDeclaration(rewriter), formatStrAndArgs);
}
} // namespace mlir::triton::cpu
