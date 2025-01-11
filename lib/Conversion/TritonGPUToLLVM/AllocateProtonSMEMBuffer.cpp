#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ALLOCATEPROTONSMEMBUFFER
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct AllocateProtonSMEMBuffer
    : public mlir::triton::impl::AllocateProtonSMEMBufferBase<
          AllocateProtonSMEMBuffer> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    MLIRContext *context = &getContext();
    auto loc = mod.getLoc();

    //bool hasProtonRecordOp = false;
    //func.walk([&](triton::ProtonRecordOp op) { hasProtonRecordOp = true; });
    //if (!hasProtonRecordOp) {
    //  return;
    //}

    auto elemTy =  IntegerType::get(context, 32);
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 42);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "proton_smem", /*value=*/Attribute(), /*alignment=*/16, /*addrSpace*/ 3);
    ModuleAllocation allocation(mod);

    llvm::errs() << mod  << "\n";
    llvm::errs() <<  "Shared Memory: " << allocation.getSharedMemorySize() << "\n";


  }
};

} // namespace

namespace mlir {

namespace triton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>> createAllocateProtonSMEMBufferPass(StringRef targetArch, int customLDSLimit) {
  return std::make_unique<AllocateProtonSMEMBuffer>();
}

} // namespace gpu

} // namespace triton

} // namespace mlir
