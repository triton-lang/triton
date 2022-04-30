#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPU.h"
#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

class ConvertTritonToTritonGPU: 
    public ConvertTritonToTritonGPUBase<ConvertTritonToTritonGPU> {

public:
    void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<arith::ArithmeticDialect>();
        registry.insert<StandardOpsDialect>();
        registry.insert<scf::SCFDialect>();
        // LLVM15
        // registry.insert<cf::ControlFlowDialect>()
        // registry.insert<func::FuncDialect>()
    }

    void runOnOperation() override {
        MLIRContext *context = &getContext();
        ConversionTarget target(*context);
        std::cout << "Converting" << std::endl;
    }
};

}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonGPUPass() {
  return std::make_unique<::ConvertTritonToTritonGPU>();
}