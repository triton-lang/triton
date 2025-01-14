#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ALLOCATEPROTONSMEMBUFFER
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {
//TODO: move this into Proton backend
struct AllocateProtonSMEMBuffer
    : public mlir::triton::impl::AllocateProtonSMEMBufferBase<
          AllocateProtonSMEMBuffer> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod);
    
    OpBuilder b(mod.getBodyRegion());
    MLIRContext *context = &getContext();
    auto loc = mod.getLoc();

    bool hasProtonRecordOp = false;
    FuncOp func = *mod.getOps<triton::FuncOp>().begin();
    func.walk([&](mlir::triton::proton::RecordOp op) { 
	hasProtonRecordOp = true; 
	});
    if(hasProtonRecordOp){
    	b.setInsertionPointToStart(&func.getBody().front());
	//For now just hard code the maximum shared memory we want to use. 
	//TODO: Make this arch specific by querying the targetInfo
	int maxSharedMemoryInBytes = 64;
	int allocatedSharedMemoryInBytes = allocation.getSharedMemorySize();;
    	int bufferSizeInBytes = maxSharedMemoryInBytes-allocatedSharedMemoryInBytes; //bytes
    	auto ptrTy = PointerType::get(IntegerType::get(context, 8), 1);
	auto buffer = b.create<triton::proton::InitLocalBufferOp>(loc, ptrTy, bufferSizeInBytes);

    }

  }
};

} // namespace

namespace mlir {

namespace triton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>> createAllocateProtonSMEMBufferPass() {
  return std::make_unique<AllocateProtonSMEMBuffer>();
}

} // namespace gpu

} // namespace triton

} // namespace mlir
