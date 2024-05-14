#ifndef TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_UTILITY_H

#include "nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators
#define barSync(rewriter, op, bar, numThreads)                                 \
  do {                                                                         \
    ::mlir::triton::PTXBuilder ptxBuilder;                                     \
    auto &barSyncOp = *ptxBuilder.create<>("bar.sync");                        \
    barSyncOp(ptxBuilder.newConstantOperand(bar),                              \
              ptxBuilder.newConstantOperand(numThreads));                      \
    auto voidTy = void_ty(op->getContext());                                   \
    ptxBuilder.launch(rewriter, op->getLoc(), voidTy);                         \
  } while (0)

#define load_dsmem(...)                                                        \
  ::mlir::LLVM::NVIDIA::createLoadDSmem(loc, rewriter, __VA_ARGS__)
#define store_dsmem(...)                                                       \
  ::mlir::LLVM::NVIDIA::createStoreDSmem(loc, rewriter, __VA_ARGS__)

namespace mlir {
namespace LLVM {

namespace NVIDIA {

Value getSRegValue(OpBuilder &b, Location loc, const std::string &sRegStr);
Value shuffleXor(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i);
Value shuffleUp(Location loc, ConversionPatternRewriter &rewriter, Value val,
                int i);
Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i);
Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 Value i);
Value permute(Location loc, ConversionPatternRewriter &rewriter, Value a,
              Value b, Value mask);

Value llGetPid(Location loc, ConversionPatternRewriter &rewriter,
               ModuleOp moduleOp, int axis);

/// Usage of macro load_dsmem
/// (1) load_dsmem(addr, ctaId)
/// (2) load_dsmem(addr, ctaId, vec)
Value createLoadDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Type elemTy);
SmallVector<Value> createLoadDSmem(Location loc, PatternRewriter &rewriter,
                                   Value addr, Value ctaId, unsigned vec,
                                   Type elemTy);

/// Usage of macro store_dsmem
/// (1) store_dsmem(addr, ctaId, value, pred)
/// (2) store_dsmem(addr, ctaId, value)
/// (3) store_dsmem(addr, ctaId, values, pred)
/// (4) store_dsmem(addr, ctaId, values)
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value, Value pred);
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value);
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values, Value pred);
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values);

/// Create a predicate with just single active thread.
Value createElectPredicate(Location loc, PatternRewriter &rewriter);

} // namespace NVIDIA
} // namespace LLVM

} // namespace mlir

#endif
