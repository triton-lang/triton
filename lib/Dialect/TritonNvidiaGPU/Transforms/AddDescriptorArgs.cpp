#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace mlir {

namespace ttng = triton::nvidia_gpu;

namespace {

class TritonGPUAddDescriptorArgsPass
    : public TritonGPUAddDescriptorArgsBase<TritonGPUAddDescriptorArgsPass> {
public:
  TritonGPUAddDescriptorArgsPass() = default;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::FuncOp funcOp) {
      // Collect TMA information.
      unsigned numTMALoad = 0;
      funcOp.walk(
          [&numTMALoad](triton::nvidia_gpu::InsertSliceTMAOp insertSliceOp) {
            numTMALoad++;
          });
      unsigned numTMAStore = 0;
      funcOp.walk(
          [&numTMAStore](triton::nvidia_gpu::StoreAsyncTMAOp storeAsyncOp) {
            numTMAStore++;
          });
      unsigned numTMA = numTMALoad + numTMAStore;
      // Append arguments to receive TMADesc in global memory in the runtime
      auto ptrTy = LLVM::LLVMPointerType::get(mod.getContext(), 1);
      auto numArgs = funcOp.getBody().front().getNumArguments();
      auto funcTy = funcOp.getFunctionType().cast<FunctionType>();
      SmallVector<Type> newInputsTy(funcTy.getInputs().begin(),
                                    funcTy.getInputs().end());
      for (unsigned i = 0; i < numTMA; ++i) {
        funcOp.getBody().front().addArgument(ptrTy, funcOp.getLoc());
        newInputsTy.push_back(ptrTy);
      }
      funcOp.setType(FunctionType::get(mod.getContext(), newInputsTy,
                                       funcTy.getResults()));
      Type int32_ty = IntegerType::get(mod.getContext(), 32);
      for (unsigned i = 0; i < numTMA; ++i) {
        funcOp.setArgAttr(numArgs + i, "tt.divisibility",
                          IntegerAttr::get(int32_ty, 1));
      }
      funcOp->setAttr("triton_gpu.num-tma-load",
                      IntegerAttr::get(int32_ty, numTMALoad));
      funcOp->setAttr("triton_gpu.num-tma-store",
                      IntegerAttr::get(int32_ty, numTMAStore));
    });
  }
};

} // namespace

std::unique_ptr<Pass> createTritonNvidiaGPUAddDescriptorArgs() {
  return std::make_unique<TritonGPUAddDescriptorArgsPass>();
}

} // namespace mlir
