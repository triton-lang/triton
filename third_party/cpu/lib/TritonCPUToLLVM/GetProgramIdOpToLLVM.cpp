#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_GETPROGRAMIDOPTOLLVM
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

// TODO: use enums to access struct fields.
struct GetProgramIdOpConversion : public OpConversionPattern<GetProgramIdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    assert(funcOp && "expected LLVM::FuncOp as a parent of GetProgramIdOp");
    auto args = funcOp.getArguments();
    // Last three args are x, y, z program ids.
    auto argIdx = args.size() - 3 + op.getAxisAsInt();
    assert(argIdx < args.size() && "out-of-bounds arg index");
    assert(args[argIdx].getType().isInteger(32) && "unexpected arg type");
    rewriter.replaceOp(op, args[argIdx]);
    return success();
  }
};

struct GetProgramIdOpToLLVM
    : public triton::impl::GetProgramIdOpToLLVMBase<GetProgramIdOpToLLVM> {
  using GetProgramIdOpToLLVMBase::GetProgramIdOpToLLVMBase;

  GetProgramIdOpToLLVM() : GetProgramIdOpToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<GetProgramIdOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createGetProgramIdOpToLLVMPass() {
  return std::make_unique<GetProgramIdOpToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
