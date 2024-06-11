#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ATOMICOPSTOLLVM
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

LLVM::AtomicOrdering getOrdering(MemSemantic sem) {
  switch (sem) {
  case MemSemantic::RELAXED:
    return LLVM::AtomicOrdering::monotonic;
  case MemSemantic::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case MemSemantic::RELEASE:
    return LLVM::AtomicOrdering::release;
  case MemSemantic::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  default:
    llvm_unreachable("Unexpected atomic mem semantic");
  }
}

// TODO: use enums to access struct fields.
struct AtomicRMWOpConversion : public OpConversionPattern<AtomicRMWOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opKind = getAtomicBinOp(op.getAtomicRmwOp(), op.getType());
    auto ptr = rewriter.getRemappedValue(op.getPtr());
    auto val = rewriter.getRemappedValue(op.getVal());
    auto ordering = getOrdering(op.getSem());
    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(op, opKind, ptr, val,
                                                   ordering);
    return success();
  }

  LLVM::AtomicBinOp getAtomicBinOp(RMWOp op, Type type) const {
    switch (op) {
    case RMWOp::AND:
      return LLVM::AtomicBinOp::_and;
    case RMWOp::OR:
      return LLVM::AtomicBinOp::_or;
    case RMWOp::XOR:
      return LLVM::AtomicBinOp::_xor;
    case RMWOp::ADD:
      return LLVM::AtomicBinOp::add;
    case RMWOp::FADD:
      return LLVM::AtomicBinOp::fadd;
    case RMWOp::MAX:
      return type.isIntOrIndex() ? LLVM::AtomicBinOp::max
                                 : LLVM::AtomicBinOp::fmax;
    case RMWOp::MIN:
      return type.isIntOrIndex() ? LLVM::AtomicBinOp::min
                                 : LLVM::AtomicBinOp::fmin;
    case RMWOp::UMAX:
      return LLVM::AtomicBinOp::umax;
    case RMWOp::UMIN:
      return LLVM::AtomicBinOp::umin;
    case RMWOp::XCHG:
      return LLVM::AtomicBinOp::xchg;
    default:
      llvm_unreachable("Unexpected atomic op");
    }
  }
};

struct AtomicCASOpConversion : public OpConversionPattern<AtomicCASOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptr = rewriter.getRemappedValue(op.getPtr());
    auto cmp = rewriter.getRemappedValue(op.getCmp());
    auto val = rewriter.getRemappedValue(op.getVal());
    auto ordering = getOrdering(op.getSem());
    auto failureOrdering = ordering != LLVM::AtomicOrdering::monotonic
                               ? LLVM::AtomicOrdering::acquire
                               : ordering;
    Value cmpXchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
        loc, ptr, cmp, val, ordering, failureOrdering);
    Value oldVal = rewriter.create<LLVM::ExtractValueOp>(loc, cmpXchg, 0);
    rewriter.replaceOp(op, oldVal);
    return success();
  }
};

struct AtomicOpsToLLVM
    : public triton::impl::AtomicOpsToLLVMBase<AtomicOpsToLLVM> {
  using AtomicOpsToLLVMBase::AtomicOpsToLLVMBase;

  AtomicOpsToLLVM() : AtomicOpsToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<AtomicRMWOpConversion>(typeConverter, context);
    patterns.add<AtomicCASOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createAtomicOpsToLLVMPass() {
  return std::make_unique<AtomicOpsToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
