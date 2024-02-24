#ifndef TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_UTILITY_H

#include "nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
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

namespace mlir {
namespace LLVM {

namespace NVIDIA {

Value storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                  Value val, Value pred);

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Type elemTy, Value pred);

Value getSRegValue(OpBuilder &b, Location loc, const std::string &sRegStr);

static Value llGetPid(int axis, Location loc, ModuleOp moduleOp,
                      ConversionPatternRewriter &rewriter) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);

  // It is not easy to get the compute capability here, so we use numCTAs to
  // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
  // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
  // "%clusterid".
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

  std::string sreg = numCTAs == 1 ? "%ctaid." : "%clusterid.";
  sreg.append(1, 'x' + axis); // 0 -> 'x', 1 -> 'y', 2 -> 'z'
  return getSRegValue(rewriter, loc, sreg);
}
} // namespace NVIDIA
} // namespace LLVM

} // namespace mlir

#endif
